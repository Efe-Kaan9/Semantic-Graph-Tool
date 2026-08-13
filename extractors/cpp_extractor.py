import logging
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter_c as tsc
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Node, Parser

from extractors.base_tree_sitter import AbstractTreeSitterExtractor
from models import EdgeRelation, NodeType, UniversalEdge, UniversalNode

logger = logging.getLogger(__name__)

_C_LANGUAGE = Language(tsc.language())
_CPP_LANGUAGE = Language(tscpp.language())


class CppExtractor(AbstractTreeSitterExtractor):
    """Extracts UniversalNode/UniversalEdge from C and C++ source/header files."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self._c_parser = Parser(_C_LANGUAGE)
        self._cpp_parser = Parser(_CPP_LANGUAGE)

    @property
    def supported_extensions(self) -> List[str]:
        return [".c", ".cpp", ".h", ".hpp", ".cc", ".cxx"]

    def parse_file(
        self,
        file_path: Path,
        code: str,
        *,
        module_id: Optional[str] = None,
    ) -> Tuple[List[UniversalNode], List[UniversalEdge]]:
        file_path = Path(file_path).resolve()
        if module_id is None:
            module_id = self._compute_module_id(file_path)

        parser = self._c_parser if file_path.suffix == ".c" else self._cpp_parser
        code_bytes = code.encode("utf-8")
        root = self._safe_parse(parser, code_bytes)

        if not root:
            return [], []

        nodes: List[UniversalNode] = []
        edges: List[UniversalEdge] = []

        file_node = UniversalNode(
            id=module_id,
            type=NodeType.FILE,
            name=file_path.name,
            file_path=str(file_path),
            lines=(1, max(1, len(code.splitlines()))),
        )
        nodes.append(file_node)

        for node in self._walk_bfs(root):
            if node.type == "preproc_def" or node.type == "preproc_function_def":
                self._extract_macro(node, file_path, module_id, code_bytes, nodes, edges)
            elif node.type in ("class_specifier", "struct_specifier"):
                self._extract_class(node, file_path, module_id, code_bytes, nodes, edges)
            elif node.type == "function_definition":
                self._extract_function(node, file_path, module_id, code_bytes, nodes, edges)
            elif node.type == "call_expression":
                self._extract_call(node, file_path, module_id, code_bytes, edges)

        return nodes, edges

    def _get_node_text(self, node: Node, code_bytes: bytes) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _get_namespaces(self, node: Node, code_bytes: bytes) -> List[str]:
        """Walks up the AST to collect all enclosing namespaces."""
        namespaces = []
        current = node.parent
        while current:
            if current.type == "namespace_definition":
                name_node = current.child_by_field_name("name")
                if name_node:
                    namespaces.insert(0, self._get_node_text(name_node, code_bytes))
            current = current.parent
        return namespaces
        
    def _get_enclosing_class(self, node: Node, code_bytes: bytes) -> Optional[str]:
        current = node.parent
        while current:
            if current.type in ("class_specifier", "struct_specifier"):
                name_node = current.child_by_field_name("name")
                if name_node:
                    return self._get_node_text(name_node, code_bytes)
            current = current.parent
        return None

    def _build_full_scope(self, node: Node, code_bytes: bytes) -> Optional[str]:
        namespaces = self._get_namespaces(node, code_bytes)
        class_name = self._get_enclosing_class(node, code_bytes)
        
        parts = namespaces.copy()
        if class_name:
            parts.append(class_name)
            
        return "::".join(parts) if parts else None

    def _extract_macro(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        name = self._get_node_text(name_node, code_bytes)
        node_id = self._generate_id(module_id, name)
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.FUNCTION, # Map MACRO to FUNCTION
            name=name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={"is_macro": True}
        )
        nodes.append(u_node)
        
        edges.append(UniversalEdge(
            source=module_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))

    def _extract_class(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        name = self._get_node_text(name_node, code_bytes)
        scope = self._build_full_scope(node, code_bytes)
        # For class itself, its own scope is just namespaces
        ns = "::".join(self._get_namespaces(node, code_bytes))
        
        node_id = self._generate_id(module_id, name, ns if ns else None)
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.CLASS,
            name=name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={}
        )
        nodes.append(u_node)
        
        parent_id = self._generate_id(module_id, ns) if ns else module_id
        edges.append(UniversalEdge(
            source=parent_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))

    def _extract_function(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        decl = node.child_by_field_name("declarator")
        if not decl:
            return
            
        # In C/C++, the declarator can be nested (e.g. pointer declarator)
        # We need to dig down to find the function_declarator and its identifier
        func_decl = decl
        while func_decl and func_decl.type != "function_declarator":
            # For simplicity, just search children
            found = False
            for child in func_decl.children:
                if child.type in ("function_declarator", "identifier", "field_identifier", "scoped_identifier"):
                    func_decl = child
                    found = True
                    break
            if not found:
                break
                
        if not func_decl:
            return
            
        name = ""
        # Could be scoped identifier (Class::Method) or simple identifier
        if func_decl.type == "function_declarator":
            decl_name = func_decl.child_by_field_name("declarator")
            if decl_name:
                name = self._get_node_text(decl_name, code_bytes)
        else:
            name = self._get_node_text(func_decl, code_bytes)
            
        if not name:
            return

        # If it's a scoped identifier (e.g. MyClass::Method), extract just the Method part
        # and update scope. For simplicity in this demo, we'll keep the full name as the ID's signature
        
        scope = self._build_full_scope(node, code_bytes)
        
        node_id = self._generate_id(module_id, name, scope)
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.METHOD if scope and "::" in scope else NodeType.FUNCTION, # Approximation
            name=name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={}
        )
        nodes.append(u_node)
        
        parent_id = self._generate_id(module_id, scope) if scope else module_id
        edges.append(UniversalEdge(
            source=parent_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))

    def _extract_call(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, edges: list):
        func_node = node.child_by_field_name("function")
        if not func_node:
            return
            
        callee_name = self._get_node_text(func_node, code_bytes)
        
        # Find caller
        current = node.parent
        caller_id = module_id
        while current:
            if current.type == "function_definition":
                # Rough caller extraction for demonstration
                scope = self._build_full_scope(current, code_bytes)
                # Need to find function name... 
                # This is complex in C++, relying on robust extraction logic
                caller_id = module_id # fallback
                break
            current = current.parent

        edges.append(UniversalEdge(
            source=caller_id,
            target=callee_name,
            relation=EdgeRelation.POTENTIAL_CALL, # Macros act like this
            metadata={"line": node.start_point[0] + 1}
        ))
