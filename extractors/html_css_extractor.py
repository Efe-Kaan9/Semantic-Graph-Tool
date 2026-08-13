import logging
from pathlib import Path
from typing import List, Optional, Tuple

import tree_sitter_html as tshtml
import tree_sitter_css as tscss
from tree_sitter import Language, Node, Parser

from extractors.base_tree_sitter import AbstractTreeSitterExtractor
from models import EdgeRelation, NodeType, UniversalEdge, UniversalNode

logger = logging.getLogger(__name__)

_HTML_LANGUAGE = Language(tshtml.language())
_CSS_LANGUAGE = Language(tscss.language())


class HtmlCssExtractor(AbstractTreeSitterExtractor):
    """Extracts structural DOM elements (HTML) and style rules (CSS)."""

    def __init__(self, project_root: Path) -> None:
        super().__init__(project_root)
        self._html_parser = Parser(_HTML_LANGUAGE)
        self._css_parser = Parser(_CSS_LANGUAGE)

    @property
    def supported_extensions(self) -> List[str]:
        return [".html", ".htm", ".css"]

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

        parser = self._html_parser if file_path.suffix in (".html", ".htm") else self._css_parser
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

        # Walk AST using BFS
        if file_path.suffix in (".html", ".htm"):
            for node in self._walk_bfs(root):
                if node.type == "element":
                    self._extract_html_element(node, file_path, module_id, code_bytes, nodes, edges)
        else:
            for node in self._walk_bfs(root):
                if node.type == "rule_set":
                    self._extract_css_rule(node, file_path, module_id, code_bytes, nodes, edges)

        return nodes, edges

    def _get_node_text(self, node: Node, code_bytes: bytes) -> str:
        return code_bytes[node.start_byte:node.end_byte].decode("utf-8")

    def _extract_html_element(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        start_tag = node.children[0] if node.children and node.children[0].type == "start_tag" else None
        if not start_tag:
            return
            
        tag_name_node = None
        for child in start_tag.children:
            if child.type == "tag_name":
                tag_name_node = child
                break
                
        if not tag_name_node:
            return
        tag_name = self._get_node_text(tag_name_node, code_bytes)
        
        # Look for id and class attributes
        node_id = None
        node_classes = []
        
        for child in start_tag.children:
            if child.type == "attribute":
                attr_name_node = None
                for c in child.children:
                    if c.type == "attribute_name":
                        attr_name_node = c
                        break
                if attr_name_node:
                    attr_name = self._get_node_text(attr_name_node, code_bytes)
                    if attr_name == "id":
                        val_node = None
                        for c in child.children:
                            if c.type == "quoted_attribute_value":
                                for cc in c.children:
                                    if cc.type == "attribute_value":
                                        val_node = cc
                                        break
                        if val_node:
                            # Remove quotes and whitespace
                            val = self._get_node_text(val_node, code_bytes).strip("\"' \t\n")
                            node_id = f"#{val}"
                    elif attr_name == "class":
                        val_node = None
                        for c in child.children:
                            if c.type == "quoted_attribute_value":
                                for cc in c.children:
                                    if cc.type == "attribute_value":
                                        val_node = cc
                                        break
                        if val_node:
                            val = self._get_node_text(val_node, code_bytes).strip("\"' \t\n")
                            node_classes.extend([f".{c.strip()}" for c in val.split() if c.strip()])
                            
        # Only extract if it has an ID or class to avoid bloating DB
        identifiers = []
        if node_id:
            identifiers.append(node_id)
        identifiers.extend(node_classes)
        
        if not identifiers:
            return
            
        for identifier in identifiers:
            generated_id = self._generate_id(module_id, identifier)
            u_node = UniversalNode(
                id=generated_id,
                type=NodeType.GLOBAL_VAR, # Using GLOBAL_VAR as a generic DOM element proxy
                name=identifier,
                file_path=str(file_path),
                lines=(node.start_point[0] + 1, node.end_point[0] + 1),
                metadata={"tag": tag_name}
            )
            nodes.append(u_node)
            
            edges.append(UniversalEdge(
                source=module_id,
                target=generated_id,
                relation=EdgeRelation.CONTAINS
            ))

    def _extract_css_rule(self, node: Node, file_path: Path, module_id: str, code_bytes: bytes, nodes: list, edges: list):
        selectors_node = None
        for child in node.children:
            if child.type == "selectors":
                selectors_node = child
                break
                
        if not selectors_node:
            return
            
        # Simplistic selector extraction
        selector = self._get_node_text(selectors_node, code_bytes).replace("\n", " ").strip()
        if not selector:
            return
            
        node_id = self._generate_id(module_id, selector)
        
        u_node = UniversalNode(
            id=node_id,
            type=NodeType.GLOBAL_VAR,
            name=selector,
            file_path=str(file_path),
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata={}
        )
        nodes.append(u_node)
        
        edges.append(UniversalEdge(
            source=module_id,
            target=node_id,
            relation=EdgeRelation.CONTAINS
        ))
