import logging
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language, Node, Parser

from extractors.base_tree_sitter import AbstractTreeSitterExtractor
from models import EdgeRelation, NodeType, UniversalEdge, UniversalNode

logger = logging.getLogger(__name__)

_JS_LANGUAGE = Language(tsjs.language())
_TS_LANGUAGE = Language(tsts.language_typescript())


class JSTSExtractor(AbstractTreeSitterExtractor):
    """Extracts UniversalNode/UniversalEdge from JavaScript and TypeScript source files."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self._js_parser = Parser(_JS_LANGUAGE)
        self._ts_parser = Parser(_TS_LANGUAGE)

    @property
    def supported_extensions(self) -> List[str]:
        return [".js", ".jsx", ".ts", ".tsx"]

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

        parser = self._ts_parser if file_path.suffix in (".ts", ".tsx") else self._js_parser
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

        # Build alias map for resolving external calls
        # We look for import_statement and lexical_declaration with require()
        alias_map = self._build_alias_map(root, code_bytes)

        for node in self._walk_bfs(root):
            if node.type in ("class_declaration", "interface_declaration", "type_alias_declaration"):
                self._extract_class_like(node, file_path, module_id, code_bytes, nodes, edges)
            elif node.type in ("function_declaration", "method_definition", "arrow_function", "function"):
                self._extract_function(node, file_path, module_id, code_bytes, nodes, edges)
            
            if node.type == "call_expression":
                self._extract_call(node, file_path, module_id, code_bytes, edges, alias_map)

        return nodes, edges

    def _get_node_text(self, node: Node, code_bytes: bytes) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _build_alias_map(self, root: Node, code_bytes: bytes) -> dict:
        alias_map = {}
        for node in self._walk_bfs(root):
            if node.type == "import_statement":
                # Handle ES6 imports
                pass # Simplified for brevity, in full version we'd extract specific import clauses
            elif node.type == "variable_declarator":
                # Handle const x = require('y')
                pass
        return alias_map

    def _get_enclosing_scope(self, node: Node, code_bytes: bytes) -> Optional[Tuple[str, str]]:
        """Returns tuple of (scope_type, scope_name)"""
        current = node.parent
        while current:
            if current.type in ("class_declaration", "interface_declaration"):
                name_node = current.child_by_field_name("name")
                if name_node:
                    return ("class", self._get_node_text(name_node, code_bytes))
            current = current.parent
        return None

    def _extract_class_like(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        name_node = node.child_by_field_name("name")
        if not name_node:
            return
            
        name = self._get_node_text(name_node, code_bytes)
        node_id = self._generate_id(module_id, name)
        
        is_type_def = node.type in ("interface_declaration", "type_alias_declaration")
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.CLASS,
            name=name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={"is_type_definition": is_type_def}
        )
        nodes.append(u_node)
        
        edges.append(UniversalEdge(
            source=module_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))

    def _extract_function(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        name = None
        if node.type in ("function_declaration", "method_definition", "function"):
            name_node = node.child_by_field_name("name")
            if name_node:
                name = self._get_node_text(name_node, code_bytes)
                
        elif node.type == "arrow_function":
            # Check if assigned to a variable
            parent = node.parent
            if parent and parent.type == "variable_declarator":
                name_node = parent.child_by_field_name("name")
                if name_node:
                    name = self._get_node_text(name_node, code_bytes)
            
            if not name:
                # Anonymous callback
                name = f"anon_func_L{node.start_point[0] + 1}_C{node.start_point[1]}"

        if not name:
            return

        scope = self._get_enclosing_scope(node, code_bytes)
        scope_name = scope[1] if scope else None
        
        node_id = self._generate_id(module_id, name, scope_name)
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.METHOD if scope else NodeType.FUNCTION,
            name=name,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={}
        )
        nodes.append(u_node)
        
        parent_id = self._generate_id(module_id, scope_name) if scope_name else module_id
        edges.append(UniversalEdge(
            source=parent_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))

    def _extract_call(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, edges: list, alias_map: dict):
        func_node = node.child_by_field_name("function")
        if not func_node:
            return
            
        callee_name = self._get_node_text(func_node, code_bytes)
        
        # Determine caller (find enclosing function/method)
        current = node.parent
        caller_id = module_id
        while current:
            if current.type in ("function_declaration", "method_definition", "arrow_function", "function"):
                name = None
                if current.type != "arrow_function":
                    n_node = current.child_by_field_name("name")
                    if n_node:
                        name = self._get_node_text(n_node, code_bytes)
                else:
                    if current.parent and current.parent.type == "variable_declarator":
                        n_node = current.parent.child_by_field_name("name")
                        if n_node:
                            name = self._get_node_text(n_node, code_bytes)
                    if not name:
                        name = f"anon_func_L{current.start_point[0] + 1}_C{current.start_point[1]}"
                
                if name:
                    scope = self._get_enclosing_scope(current, code_bytes)
                    scope_name = scope[1] if scope else None
                    caller_id = self._generate_id(module_id, name, scope_name)
                break
            current = current.parent

        edges.append(UniversalEdge(
            source=caller_id,
            target=callee_name, # Real implementation would resolve aliases here
            relation=EdgeRelation.POTENTIAL_CALL,
            metadata={"line": node.start_point[0] + 1}
        ))
