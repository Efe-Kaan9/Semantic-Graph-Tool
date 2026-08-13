import logging
from collections import deque
from pathlib import Path
from typing import Generator, List, Optional, Tuple

from tree_sitter import Node, Parser

from models import BaseExtractor, UniversalEdge, UniversalNode

logger = logging.getLogger(__name__)


class AbstractTreeSitterExtractor(BaseExtractor):
    """Abstract base class for Tree-sitter based language extractors.
    
    Provides shared logic for safe parsing, Breadth-First Search (BFS) AST walking,
    and deterministic ID generation across different languages.
    """

    def __init__(self, project_root: Path) -> None:
        self._project_root = Path(project_root).resolve()

    def _safe_parse(self, parser: Parser, code: bytes) -> Optional[Node]:
        """Graceful degradation: safely parse code and handle tree-sitter errors.
        
        Args:
            parser: The tree-sitter Parser instance.
            code: The source code in bytes.
            
        Returns:
            The root Node of the AST, or None if parsing fails completely.
        """
        try:
            tree = parser.parse(code)
            return tree.root_node
        except Exception as e:
            logger.warning("Tree-sitter encountered a syntax error: %s", e)
            return None

    def _walk_bfs(self, root: Node) -> Generator[Node, None, None]:
        """Breadth-First Search (BFS) AST walk to ensure correct scope assignment.
        
        Using BFS prevents nested scopes (like a function inside a function or class)
        from being incorrectly assigned to the parent's immediate scope when processing.
        
        Args:
            root: The starting AST node.
            
        Yields:
            Nodes in breadth-first order.
        """
        queue = deque([root])
        while queue:
            current = queue.popleft()
            yield current
            for child in current.children:
                queue.append(child)

    def _generate_id(
        self,
        file_path: str,
        symbol_signature: str,
        namespace_class: Optional[str] = None
    ) -> str:
        """Generates a 100% deterministic Node ID.
        
        Convention: <file_path>::<namespace/class>::<symbol_signature>
        
        Args:
            file_path: Relative or absolute file path of the source.
            symbol_signature: The unique name or signature of the symbol.
            namespace_class: Optional namespace and/or class hierarchy.
            
        Returns:
            A deterministic string ID.
        """
        if namespace_class:
            return f"{file_path}::{namespace_class}::{symbol_signature}"
        return f"{file_path}::{symbol_signature}"

    def _compute_module_id(self, file_path: Path) -> str:
        """Helper to compute a stable module ID from a file path."""
        try:
            rel = file_path.relative_to(self._project_root)
        except ValueError:
            rel = file_path
        
        # Strip extension and convert path separators to dots for module format,
        # but for non-Python languages, keeping the path as-is might be preferred.
        # We will just return the posix string of the relative path.
        return rel.with_suffix("").as_posix()
