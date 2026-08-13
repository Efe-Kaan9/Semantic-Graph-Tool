"""
Semantic Code Knowledge Graph — MCP Server
===========================================

Exposes five tools to an LLM via the Model Context Protocol (stdio JSON-RPC):

  search_codebase(query, target_directory, ...)   — semantic vector search
  explore_graph(node_id, target_directory, ...)   — graph neighbourhood / blast-radius
  read_node_code(node_id, target_directory)       — exact source lines for a node
  visualize_graph(node_id, target_directory, ...) — Mermaid diagram of call graph
  get_structural_insights(insight_type, ...)      — dead code, cycles, hot-spots

``target_directory`` MUST be the absolute path to the user's project workspace.
The server is intentionally location-independent: it never assumes its own
directory is the project being indexed.

Usage
-----
Run directly::

    python server.py

Or register in your MCP client config (see ``mcp_config.json``).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mcp.server.mcpserver import MCPServer

from core.database import CodeGraphDB
from core.jit_sync import JITSyncManager, SyncResult
from core.visualizer import to_mermaid

# ──────────────────────────────────────────────────────────────────────────────
#  Logging
# ──────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stderr,   # MCP stdio uses stdout; keep logs on stderr
)
logger = logging.getLogger("semantic_graph")

# ──────────────────────────────────────────────────────────────────────────────
#  Server bootstrap
# ──────────────────────────────────────────────────────────────────────────────

mcp = MCPServer(
    name="semantic-graph",
    instructions=(
        "MANDATORY WORKFLOW — read before every tool call:\n"
        "1. You MUST provide target_directory (the absolute path of the user's "
        "   project workspace) in EVERY tool call. Never omit it.\n"
        "2. Call search_codebase FIRST to obtain exact node_ids. "
        "   NEVER guess or fabricate a node_id.\n"
        "3. Use explore_graph to trace call dependencies (blast radius).\n"
        "4. Use visualize_graph for any diagram or graph request.\n"
        "5. Use read_node_code instead of native file reading — it is faster "
        "   and more precise.\n"
        "6. Use get_structural_insights for codebase-wide queries (dead code, "
        "   hot-spots, cycles) without reading any files.\n"
    ),
)


# ──────────────────────────────────────────────────────────────────────────────
#  Per-directory service cache
#
#  Key insight: the MCP server process CWD is the *server's* install directory,
#  NOT the user's workspace.  We must never use cwd/Path('.') as a project root.
#  Instead, each tool call supplies an explicit target_directory and we maintain
#  a cache of (CodeGraphDB, JITSyncManager) keyed by the resolved path.
# ──────────────────────────────────────────────────────────────────────────────

# Cache: resolved_path_str -> (CodeGraphDB, JITSyncManager)
_manager_cache: Dict[str, Tuple[CodeGraphDB, JITSyncManager]] = {}


def get_or_create_manager(target_directory: str) -> Tuple[CodeGraphDB, JITSyncManager]:
    """Return a (CodeGraphDB, JITSyncManager) pair for *target_directory*.

    On the first call for a given directory the pair is initialised and cached.
    Subsequent calls for the same directory return the cached pair instantly.
    All paths are resolved to absolute before use — CWD is never consulted.
    """
    root = Path(target_directory).resolve()
    key = str(root)
    storage = root / ".code_graph"

    if key not in _manager_cache:
        logger.info("=== NEW PROJECT ROOT ===")
        logger.info("  target_directory arg : %s", target_directory)
        logger.info("  resolved absolute    : %s", root)
        logger.info("  .code_graph location : %s", storage)
        logger.info("  SQLite will be at    : %s", storage / "graph.db")
        logger.info("  ChromaDB will be at  : %s", storage / "vector_db")
        db = CodeGraphDB(storage)
        sync = JITSyncManager(
            project_root=root,
            db=db,
            storage_dir=storage,
        )
        _manager_cache[key] = (db, sync)

    return _manager_cache[key]


def _sync_and_get(target_directory: str) -> Tuple[CodeGraphDB, JITSyncManager]:
    """Run JIT sync then return the (db, sync) pair for *target_directory*."""
    db, sync = get_or_create_manager(target_directory)
    result = sync.sync()
    if result.changed:
        logger.info("JIT sync [%s]: %s", Path(target_directory).name, result)
    return db, sync


# ──────────────────────────────────────────────────────────────────────────────
#  Tool 1 — search_codebase
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def search_codebase(
    query: str,
    target_directory: str,
    node_type: Optional[str] = None,
    n_results: int = 5,
) -> List[Dict[str, Any]]:
    """CRITICAL: ALWAYS use this FIRST. Performs semantic search. Never guess a node_id.

    Returns a list of nodes with their exact ``id`` fields. You MUST pass
    those ids to explore_graph, visualize_graph, or read_node_code.

    Parameters
    ----------
    query:
        Natural-language description of what you are looking for.
        Examples: "tax calculation", "user authentication", "send email".
    target_directory:
        The absolute path of the user's current project workspace.
        YOU MUST PROVIDE THIS. Example: ``C:/Users/alice/projects/MyApp``.
    node_type:
        Optional filter. One of: ``CLASS``, ``METHOD``, ``FUNCTION``.
    n_results:
        Maximum results to return (default 5, max 20).

    Returns
    -------
    List of matching nodes, each with ``id``, ``name``, ``type``,
    ``file_path``, ``lines``, ``score``.  Sorted by relevance.
    """
    db, _ = _sync_and_get(target_directory)

    n_results = min(max(1, n_results), 20)
    results = db.semantic_search(
        query=query,
        n_results=n_results,
        node_type=node_type,
    )

    indexed_root = str(Path(target_directory).resolve())

    if not results:
        return [{"message": f"No results found for query: '{query}'",
                 "indexed_root": indexed_root}]

    # Stamp every result with the indexed root so the LLM can confirm
    # it is querying the correct directory.
    for r in results:
        r["indexed_root"] = indexed_root

    return results


# ──────────────────────────────────────────────────────────────────────────────
#  Tool 2 — explore_graph
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def explore_graph(
    node_id: str,
    target_directory: str,
    depth: int = 1,
) -> Dict[str, Any]:
    """Requires exact node_id. Explores call graph dependencies (blast radius). Use BEFORE reading files manually.

    Parameters
    ----------
    node_id:
        Exact node ID from ``search_codebase`` results.
        Format: ``dotted.module::ClassName::method_name``.
    target_directory:
        The absolute path of the user's current project workspace.
        YOU MUST PROVIDE THIS.
    depth:
        Traversal depth (default 1, max 3). Depth 2 shows callers of callers.
    """
    db, _ = _sync_and_get(target_directory)

    depth = min(max(1, depth), 3)
    context = db.get_node_context(node_id, depth=depth)

    if context is None:
        return {
            "error": f"Node '{node_id}' not found in the graph.",
            "hint": "Use search_codebase to find the correct node ID.",
            "indexed_root": str(Path(target_directory).resolve()),
        }

    context["outgoing_count"] = len(context.get("outgoing_edges", []))
    context["incoming_count"] = len(context.get("incoming_edges", []))
    return context


# ──────────────────────────────────────────────────────────────────────────────
#  Tool 3 — read_node_code
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def read_node_code(
    node_id: str,
    target_directory: str,
) -> Dict[str, Any]:
    """Reads exact source-code bytes of a specific node. Much faster and more precise than native file reading. Prioritize this over standard file read.

    Parameters
    ----------
    node_id:
        Exact node ID from ``search_codebase`` or ``explore_graph``.
    target_directory:
        The absolute path of the user's current project workspace.
        YOU MUST PROVIDE THIS.
    """
    db, _ = _sync_and_get(target_directory)

    node = db.get_node(node_id)
    if node is None:
        return {
            "error": f"Node '{node_id}' not found.",
            "hint": "Use search_codebase to find the correct node ID.",
            "indexed_root": str(Path(target_directory).resolve()),
        }

    file_path = node.get("file_path", "")
    lines: list = node.get("lines", [])

    if not file_path or len(lines) < 2:
        return {
            "error": "Node has incomplete location metadata.",
            "node": node,
        }

    start_line, end_line = int(lines[0]), int(lines[1])

    try:
        source_lines = Path(file_path).read_text(
            encoding="utf-8", errors="replace",
        ).splitlines()
    except OSError as exc:
        return {"error": f"Cannot read file '{file_path}': {exc}"}

    total = len(source_lines)
    start_line = max(1, min(start_line, total))
    end_line   = max(start_line, min(end_line, total))

    snippet = "\n".join(source_lines[start_line - 1 : end_line])

    suffix = Path(file_path).suffix.lower()
    lang_map = {".py": "python", ".js": "javascript", ".ts": "typescript"}
    language = lang_map.get(suffix, suffix.lstrip(".") or "text")

    return {
        "node_id":    node_id,
        "file_path":  file_path,
        "start_line": start_line,
        "end_line":   end_line,
        "language":   language,
        "code":       snippet,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Tool 4 — visualize_graph
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def visualize_graph(
    node_id: str,
    target_directory: str,
    depth: int = 1,
    output_format: str = "mermaid",
) -> Dict[str, Any]:
    """CRITICAL AND MANDATORY: DO NOT attempt to write the Mermaid code manually! 
    You should directly give user this tool's output, because it is guaranteed to be 100% accurate. Do not waste time trying to generate different diagrams or update the output.
    Manual LLM analysis misses deep structural edges and indirect calls. You MUST call this tool to fetch the 100% accurate topology from the Graph Database.
    Requires exact node_id. Returns Mermaid diagram. ALWAYS use this when the user asks for a visual, graph, or diagram.

    Parameters
    ----------
    node_id:
        Exact node ID from ``search_codebase``.
    target_directory:
        The absolute path of the user's current project workspace.
        YOU MUST PROVIDE THIS.
    depth:
        Traversal depth (default 1, max 3). Depth 2+ shows wider blast radius.
    output_format:
        Diagram format.  Currently supported: ``"mermaid"`` (default).

    EXAMPLES OF WHEN TO USE THIS:
        - User: "Draw the call graph for calculate_tax" -> Action: use visualize_graph
        - User: "Visually show me what methods this class contains" -> Action: use visualize_graph
        - User: "Give me a mermaid diagram of X" -> Action: use visualize_graph
        - User: "Draw the method of [...] for me" -> Action: use visualize_graph
    """
    db, _ = _sync_and_get(target_directory)

    depth = min(max(1, depth), 3)

    subgraph = db.get_ego_subgraph(node_id, depth=depth)
    if subgraph is None:
        return {
            "error": f"Node '{node_id}' not found in the graph.",
            "hint": "Use search_codebase to find the correct node ID.",
            "indexed_root": str(Path(target_directory).resolve()),
        }

    fmt = output_format.lower().strip()
    if fmt == "mermaid":
        diagram = to_mermaid(subgraph)
    else:
        return {
            "error": f"Unsupported format: '{output_format}'.",
            "supported_formats": ["mermaid"],
        }

    return {
        "format":     fmt,
        "diagram":    diagram,
        "node_count": len(subgraph["nodes"]),
        "edge_count": len(subgraph["edges"]),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Tool 5 — get_structural_insights
# ──────────────────────────────────────────────────────────────────────────────

@mcp.tool()
def get_structural_insights(
    insight_type: str,
    target_directory: str,
) -> Dict[str, Any]:
    """Use this to find dead code, orphan methods, or architectural flaws instantly using graph topology without reading files.

    Parameters
    ----------
    insight_type:
        The analysis to run:
        - ``"orphan_methods"``: METHODs/FUNCTIONs with 0 incoming calls (dead code).
        - ``"most_connected"``: Top-10 most depended-upon nodes (high refactor risk).
        - ``"circular_deps"``: Circular call chains.
    target_directory:
        The absolute path of the user's current project workspace.
        YOU MUST PROVIDE THIS.
    """
    db, _ = _sync_and_get(target_directory)

    insight = insight_type.lower().strip()

    if insight == "orphan_methods":
        results = db.find_orphan_methods()
    elif insight == "most_connected":
        results = db.find_most_connected(n=10)
    elif insight == "circular_deps":
        results = db.find_circular_deps()
    else:
        return {
            "error": f"Unknown insight_type: '{insight_type}'.",
            "supported_types": ["orphan_methods", "most_connected", "circular_deps"],
        }

    return {
        "insight_type": insight,
        "results":      results,
        "count":        len(results),
        "indexed_root": str(Path(target_directory).resolve()),
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Semantic Code Knowledge Graph — MCP Server",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.getLogger().setLevel(args.log_level)
    logger.info(
        "Server starting — CWD: %s  (project root is resolved per tool call)",
        Path.cwd(),
    )
    mcp.run(transport="stdio")
