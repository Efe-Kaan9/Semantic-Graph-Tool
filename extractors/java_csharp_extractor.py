import logging
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter_java as tsjava
import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Node, Parser

from extractors.base_tree_sitter import AbstractTreeSitterExtractor
from models import EdgeRelation, NodeType, UniversalEdge, UniversalNode

logger = logging.getLogger(__name__)

_JAVA_LANGUAGE = Language(tsjava.language())
_CSHARP_LANGUAGE = Language(tscsharp.language())


class JavaCSharpExtractor(AbstractTreeSitterExtractor):
    """Extracts UniversalNode/UniversalEdge from Java and C# source files."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self._java_parser = Parser(_JAVA_LANGUAGE)
        self._csharp_parser = Parser(_CSHARP_LANGUAGE)

    @property
    def supported_extensions(self) -> List[str]:
        return [".java", ".cs"]

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

        parser = self._java_parser if file_path.suffix == ".java" else self._csharp_parser
        code_bytes = code.encode("utf-8")
        root = self._safe_parse(parser, code_bytes)

        if not root:
            return [], []

        nodes: List[UniversalNode] = []
        edges: List[UniversalEdge] = []

        # Create FILE node
        file_node = UniversalNode(
            id=module_id,
            type=NodeType.FILE,
            name=file_path.name,
            file_path=str(file_path),
            lines=(1, max(1, len(code.splitlines()))),
        )
        nodes.append(file_node)

        # Walk AST using BFS
        for node in self._walk_bfs(root):
            if node.type in ("class_declaration", "interface_declaration"):
                self._extract_class_or_interface(node, file_path, module_id, code_bytes, nodes, edges)
            elif node.type == "method_declaration":
                # Only extract top-level methods if any exist, or handle method calls
                # Usually methods are within classes, so we handle them in the class extraction
                # to keep track of the class namespace. However, BFS processes everything.
                # To avoid duplicating or misassigning, we should probably handle methods
                # when we process the class. But since BFS flattens, we can determine the 
                # parent class by walking up the tree.
                pass
            
            # Extract method calls for edges
            if node.type in ("method_invocation", "invocation_expression"):
                self._extract_method_call(node, file_path, module_id, code_bytes, edges)

        # To properly handle namespacing in BFS, we can find the enclosing class for a method
        for node in self._walk_bfs(root):
            if node.type == "method_declaration":
                self._extract_method(node, file_path, module_id, code_bytes, nodes, edges)

        return nodes, edges

    def _get_enclosing_class(self, node: Node) -> Optional[Node]:
        """Finds the enclosing class or interface for a given node."""
        current = node.parent
        while current:
            if current.type in ("class_declaration", "interface_declaration"):
                return current
            current = current.parent
        return None
        
    def _get_full_class_scope(self, node: Node, code_bytes: bytes) -> str:
        """Builds the full scope string (e.g. OuterClass::InnerClass)."""
        parts = []
        current = node
        while current:
            if current.type in ("class_declaration", "interface_declaration"):
                name_node = current.child_by_field_name("name")
                if name_node:
                    parts.insert(0, self._get_node_text(name_node, code_bytes))
            current = current.parent
        return "::".join(parts)

    def _get_node_text(self, node: Node, code_bytes: bytes) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_class_or_interface(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        class_name = self._get_node_text(name_node, code_bytes)
        
        parent_class = self._get_enclosing_class(node)
        parent_scope = self._get_full_class_scope(parent_class, code_bytes) if parent_class else None
        
        node_id = self._generate_id(module_id, class_name, parent_scope)
        
        is_interface = node.type == "interface_declaration"
        
        metadata = {
            "is_interface": is_interface
        }
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.CLASS,
            name=class_name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata=metadata
        )
        nodes.append(u_node)
        
        # Add CONTAINS edge from parent class or file
        parent_id = self._generate_id(module_id, parent_scope) if parent_scope else module_id
        edges.append(UniversalEdge(
            source=parent_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))

    def _extract_method(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        method_name = self._get_node_text(name_node, code_bytes)
        enclosing_class = self._get_enclosing_class(node)
        
        class_name = ""
        if enclosing_class:
            class_name = self._get_full_class_scope(enclosing_class, code_bytes)
                
        # Extract parameter types for overloading support
        param_types = []
        parameters_node = node.child_by_field_name("parameters") # Java
        if not parameters_node:
            parameters_node = node.child_by_field_name("parameter_list") # C#
            
        if parameters_node:
            for child in parameters_node.children:
                if child.type in ("formal_parameter", "parameter"):
                    type_node = child.child_by_field_name("type")
                    if type_node:
                        param_types.append(self._get_node_text(type_node, code_bytes))

        sig = f"{method_name}({','.join(param_types)})"
        node_id = self._generate_id(module_id, sig, class_name if class_name else None)
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.METHOD,
            name=method_name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={"signature": sig}
        )
        nodes.append(u_node)
        
        if class_name:
            parent_id = self._generate_id(module_id, class_name)
            edges.append(UniversalEdge(
                source=parent_id,
                target=node_id,
                relation=EdgeRelation.CONTAINS
            ))
        else:
            edges.append(UniversalEdge(
                source=module_id,
                target=node_id,
                relation=EdgeRelation.CONTAINS
            ))

    def _extract_method_call(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, edges: list):
        # Determine caller
        enclosing_method = None
        current = node.parent
        while current:
            if current.type == "method_declaration":
                enclosing_method = current
                break
            current = current.parent
            
        if not enclosing_method:
            return
            
        # Get caller ID
        m_name_node = enclosing_method.child_by_field_name("name")
        if not m_name_node:
            return
        caller_name = self._get_node_text(m_name_node, code_bytes)
        
        enclosing_class = self._get_enclosing_class(enclosing_method)
        caller_class = ""
        if enclosing_class:
            caller_class = self._get_full_class_scope(enclosing_class, code_bytes)
                
        # Best effort caller signature (without full type analysis, we might just use name for caller if we can't reconstruct exact signature, 
        # but to match nodes exactly we need the caller's full sig. Let's rebuild it:)
        param_types = []
        parameters_node = enclosing_method.child_by_field_name("parameters") or enclosing_method.child_by_field_name("parameter_list")
        if parameters_node:
            for child in parameters_node.children:
                if child.type in ("formal_parameter", "parameter"):
                    type_node = child.child_by_field_name("type")
                    if type_node:
                        param_types.append(self._get_node_text(type_node, code_bytes))
                        
        caller_sig = f"{caller_name}({','.join(param_types)})"
        caller_id = self._generate_id(module_id, caller_sig, caller_class if caller_class else None)
        
        # Get callee name
        callee_name = ""
        if node.type == "method_invocation": # Java
            name_node = node.child_by_field_name("name")
            if name_node:
                callee_name = self._get_node_text(name_node, code_bytes)
        elif node.type == "invocation_expression": # C#
            expr_node = node.child_by_field_name("function")
            if expr_node:
                # Could be a member access (obj.Method) or identifier (Method)
                if expr_node.type == "member_access_expression":
                    name_node = expr_node.child_by_field_name("name")
                    if name_node:
                        callee_name = self._get_node_text(name_node, code_bytes)
                else:
                    callee_name = self._get_node_text(expr_node, code_bytes)
                    
        if not callee_name:
            return
            
        # We can't know the exact overloaded signature of the callee without a compiler,
        # so we record a POTENTIAL_CALL using just the callee name as a best-effort.
        edges.append(UniversalEdge(
            source=caller_id,
            target=callee_name, # In a real system, we'd try to resolve this to a full ID
            relation=EdgeRelation.POTENTIAL_CALL,
            metadata={"line": node.start_point[0] + 1}
        ))
