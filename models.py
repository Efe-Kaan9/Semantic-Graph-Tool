"""
Universal Schema definitions for the Semantic Code Knowledge Graph.

This module defines the language-agnostic data models that all extractors
must produce. The Core Engine operates exclusively on these types, ensuring
that adding a new language only requires writing a new Extractor adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ──────────────────────────────────────────────
#  Enumerations
# ──────────────────────────────────────────────

class NodeType(str, Enum):
    """The kind of symbol a node represents."""

    NAMESPACE = "NAMESPACE"
    FILE = "FILE"
    CLASS = "CLASS"
    METHOD = "METHOD"
    FUNCTION = "FUNCTION"
    GLOBAL_VAR = "GLOBAL_VAR"


class EdgeRelation(str, Enum):
    """The semantic relationship between two nodes."""

    CONTAINS = "CONTAINS"           # Parent → child containment (file→class, class→method)
    CALLS_INTERNAL = "CALLS_INTERNAL"   # Call within the same module / via self
    CALLS_EXTERNAL = "CALLS_EXTERNAL"   # Cross-module call resolved via import alias map
    POTENTIAL_CALL = "POTENTIAL_CALL"    # Ambiguous call (dynamic typing, unresolved target)
    USES = "USES"                   # Read/write access to a variable or attribute
    INHERITS = "INHERITS"           # Class inheritance (child → parent)


# ──────────────────────────────────────────────
#  Universal Node
# ──────────────────────────────────────────────

@dataclass
class UniversalNode:
    """A single symbol extracted from source code.

    Attributes:
        id:        Fully-qualified, deterministic identifier.
                   Convention: ``<dotted.module.path>::<Class>::<member>``
                   Examples:
                     - ``src.api.views``                      (FILE)
                     - ``src.api.views::UserView``            (CLASS)
                     - ``src.api.views::UserView::get``       (METHOD)
                     - ``src.utils.helpers::slugify``          (FUNCTION)
        type:      The kind of symbol (see `NodeType`).
        name:      The short, human-readable symbol name (e.g. ``get``).
        file_path: Absolute path to the source file that contains this node.
        lines:     Inclusive ``[start_line, end_line]`` (1-indexed).
        metadata:  Arbitrary, language-specific extras such as docstrings,
                   decorators, type hints, or local variable names.
    """

    id: str
    type: NodeType
    name: str
    file_path: str
    lines: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ── Convenience helpers ──────────────────

    @property
    def start_line(self) -> int:
        return self.lines[0]

    @property
    def end_line(self) -> int:
        return self.lines[1]

    @property
    def signature(self) -> str:
        """A one-line searchable signature built from available metadata.

        Used by the Vector DB to produce better embeddings than the raw
        symbol name alone.  Falls back to ``<type> <name>`` when no
        richer information is available.
        """
        parts: List[str] = []

        # Decorators (e.g. @staticmethod, @app.route("/users"))
        decorators: List[str] = self.metadata.get("decorators", [])
        for dec in decorators:
            parts.append(f"@{dec}")

        # Base signature: "class Foo" / "def bar(x, y) -> int"
        if self.type in (NodeType.METHOD, NodeType.FUNCTION):
            params = self.metadata.get("parameters", "")
            return_type = self.metadata.get("return_type", "")
            sig = f"def {self.name}({params})"
            if return_type:
                sig += f" -> {return_type}"
            parts.append(sig)
        elif self.type == NodeType.CLASS:
            bases = self.metadata.get("bases", [])
            if bases:
                parts.append(f"class {self.name}({', '.join(bases)})")
            else:
                parts.append(f"class {self.name}")
        else:
            parts.append(f"{self.type.value.lower()} {self.name}")

        return "\n".join(parts)

    @property
    def docstring(self) -> Optional[str]:
        """Shortcut to the docstring stored in metadata, if any."""
        return self.metadata.get("docstring")

    @property
    def embedding_text(self) -> str:
        """Text blob sent to the embedding model for vector search.

        Concatenates the signature and docstring so that semantic search
        can match on *intent* (docstring) as well as *structure* (signature).
        """
        parts = [self.signature]
        if self.docstring:
            parts.append(self.docstring)
        return "\n".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (useful for JSON / ChromaDB metadata)."""
        return {
            "id": self.id,
            "type": self.type.value,
            "name": self.name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────
#  Universal Edge
# ──────────────────────────────────────────────

@dataclass
class UniversalEdge:
    """A directed relationship between two `UniversalNode` instances.

    Attributes:
        source:   ID of the originating node.
        target:   ID of the destination node.
        relation: The semantic type of the relationship.
        metadata: Optional extras (e.g. call-site line number).
    """

    source: str
    target: str
    relation: EdgeRelation
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary."""
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation.value,
            "metadata": self.metadata,
        }


# ──────────────────────────────────────────────
#  Abstract Base Extractor
# ──────────────────────────────────────────────

class BaseExtractor(ABC):
    """Contract that every language-specific extractor must fulfil.

    A concrete extractor receives the path of a source file together with
    its textual content and returns the universal nodes and edges that
    the Core Engine can ingest.

    Subclasses **must** implement `parse_file`.  They *may* override
    `supported_extensions` to declare which file suffixes they handle.
    """

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """File extensions this extractor can process (e.g. ``['.py']``)."""
        ...

    @abstractmethod
    def parse_file(
        self,
        file_path: Path,
        code: str,
        *,
        module_id: Optional[str] = None,
    ) -> Tuple[List[UniversalNode], List[UniversalEdge]]:
        """Parse a single source file and return its nodes and edges.

        Args:
            file_path:  Absolute path to the file (used for line reading and
                        constructing the node ``file_path`` field).
            code:       The full text content of the file.
            module_id:  Optional override for the dotted module path.
                        When *None*, the extractor should derive it from
                        ``file_path`` relative to the project root.

        Returns:
            A tuple of ``(nodes, edges)`` expressed in the universal schema.
        """
        ...
