"""
Graph Visualiser — ``core/visualizer.py``
==========================================

Converts a subgraph dict (from ``CodeGraphDB.get_ego_subgraph``) into
renderable diagram formats.

Currently supported:
  - **Mermaid** (``graph TD``) — automatically rendered by Claude, Cursor,
    and most Markdown-aware UIs.
"""

from __future__ import annotations

import re
from typing import Any, Dict

# ──────────────────────────────────────────────────────────────────────────────
#  Mermaid-safe ID generation
# ──────────────────────────────────────────────────────────────────────────────

_MERMAID_UNSAFE = re.compile(r"[^a-zA-Z0-9_]")

# Monotonic counter ensures uniqueness when different qualified IDs
# collapse to the same Mermaid-safe string.
_id_counter: int = 0
_id_cache: Dict[str, str] = {}


def _mermaid_id(qualified_id: str) -> str:
    """Convert a qualified node ID into a Mermaid-safe identifier.

    ``src.utils.auth::verify``  →  ``src_utils_auth__verify``

    Results are cached per call to ``to_mermaid`` so that the same
    qualified ID always maps to the same short ID within one diagram.
    """
    if qualified_id in _id_cache:
        return _id_cache[qualified_id]

    global _id_counter
    safe = _MERMAID_UNSAFE.sub("_", qualified_id)
    # Mermaid IDs must start with a letter
    if safe and not safe[0].isalpha():
        safe = "n_" + safe
    _id_counter += 1
    safe = f"{safe}_{_id_counter}"
    _id_cache[qualified_id] = safe
    return safe


# ──────────────────────────────────────────────────────────────────────────────
#  Node shape by type
# ──────────────────────────────────────────────────────────────────────────────

# Mermaid shape delimiters per node type (see https://mermaid.js.org/syntax)
_SHAPE_MAP: Dict[str, tuple[str, str]] = {
    "FILE":     ("[", "]"),         # rectangle
    "CLASS":    ("[[", "]]"),       # double-bordered rectangle (subroutine)
    "METHOD":   ("([", "])"),       # stadium / pill
    "FUNCTION": ("([", "])"),       # stadium / pill
}

_DEFAULT_SHAPE = ("(", ")")        # rounded rectangle


# ──────────────────────────────────────────────────────────────────────────────
#  Edge style by relation
# ──────────────────────────────────────────────────────────────────────────────

_EDGE_STYLE: Dict[str, str] = {
    "CONTAINS":       "-->",
    "CALLS_INTERNAL": "-.->",       # dotted arrow
    "CALLS_EXTERNAL": "==>",        # thick arrow
    "POTENTIAL_CALL":  "-. ? .->",  # dotted with label
    "INHERITS":       "-->",
}

_DEFAULT_EDGE = "-->"


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def to_mermaid(subgraph: Dict[str, Any]) -> str:
    """Convert a subgraph dict into a Mermaid flowchart string.

    Parameters
    ----------
    subgraph:
        The dict returned by ``CodeGraphDB.get_ego_subgraph``, with keys
        ``center``, ``nodes``, ``edges``.

    Returns
    -------
    A complete Mermaid ``graph TD`` string, ready to be rendered.
    """
    # Reset the ID cache for each diagram
    global _id_counter
    _id_cache.clear()
    _id_counter = 0

    center_id = subgraph["center"]
    nodes = subgraph["nodes"]
    edges = subgraph["edges"]

    lines: list[str] = ["graph TD"]

    # ── Node declarations ─────────────────────────────────────────────
    for node in nodes:
        mid = _mermaid_id(node["id"])
        ntype = node.get("type", "")
        name = node.get("name", node["id"].split("::")[-1])

        # Build a label:  «TYPE» name
        if ntype:
            label = f"{_type_emoji(ntype)} {name}"
        else:
            label = name

        # Escape quotes inside labels
        label = label.replace('"', "'")

        open_d, close_d = _SHAPE_MAP.get(ntype, _DEFAULT_SHAPE)
        line = f"    {mid}{open_d}\"{label}\"{close_d}"
        lines.append(line)

    # ── Edge declarations ─────────────────────────────────────────────
    for edge in edges:
        src = _mermaid_id(edge["source"])
        tgt = _mermaid_id(edge["target"])
        relation = edge.get("relation", "")
        arrow = _EDGE_STYLE.get(relation, _DEFAULT_EDGE)

        # Add relation as edge label for non-CONTAINS edges
        if relation and relation != "CONTAINS":
            line = f"    {src} {arrow}|{relation}| {tgt}"
        else:
            line = f"    {src} {arrow} {tgt}"
        lines.append(line)

    # ── Highlight the center node ─────────────────────────────────────
    center_mid = _mermaid_id(center_id)
    lines.append(f"    style {center_mid} fill:#f9a825,stroke:#f57f17,stroke-width:3px,color:#000")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _type_emoji(node_type: str) -> str:
    """Return a compact type badge for use inside Mermaid labels."""
    return {
        "FILE":     "📄",
        "CLASS":    "🏛️",
        "METHOD":   "⚙️",
        "FUNCTION": "🔧",
    }.get(node_type, "")
