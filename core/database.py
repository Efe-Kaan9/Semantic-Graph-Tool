"""
Core Storage Layer — ``CodeGraphDB``
=====================================

Provides a **single class** that unifies three storage backends behind
one API surface:

* **SQLite** — durable on-disk persistence of nodes and edges.
* **NetworkX ``MultiDiGraph``** — in-memory graph loaded from SQLite at
  start-up for lightning-fast traversals and neighbour queries.
* **ChromaDB** — persistent vector store for semantic / embedding search.

The primary mutation entry-point is ``upsert_file_nodes`` which performs
an **atomic, file-level replace**: all nodes and edges belonging to a
given source file are deleted from *every* backend, then the new ones
are inserted.  This is the primitive that the JIT Sync Manager calls.

Query helpers (``semantic_search``, ``get_node_context``, ``get_node``)
are consumed by the MCP tool endpoints.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

import chromadb
import networkx as nx

from models import EdgeRelation, NodeType, UniversalEdge, UniversalNode

logger = logging.getLogger(__name__)


class CodeGraphDB:
    """Unified graph + vector storage for the Semantic Code Knowledge Graph."""

    # Only these node types carry enough semantic weight to embed.
    _EMBEDDABLE_TYPES: FrozenSet[str] = frozenset({
        NodeType.CLASS.value,
        NodeType.METHOD.value,
        NodeType.FUNCTION.value,
    })

    # ══════════════════════════════════════════════════════════════════════
    #  LIFECYCLE
    # ══════════════════════════════════════════════════════════════════════

    def __init__(self, storage_dir: str | Path) -> None:
        # Resolve to absolute immediately — NEVER rely on CWD for storage paths.
        # Path.resolve() calls the OS getcwd() at this point (inside
        # get_or_create_manager, where the caller has already resolved the root),
        # but since we receive an already-absolute path from the server, resolve()
        # is a no-op that also normalises separators.
        self._storage_dir = Path(storage_dir).resolve()
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        # ── SQLite ────────────────────────────────────────────────────
        self._db_path = self._storage_dir / "graph.db"
        logger.info("SQLite: %s", self._db_path)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

        # ── NetworkX (in-memory mirror of SQLite) ─────────────────────
        self._graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self._load_graph_from_sqlite()

        # ── ChromaDB ──────────────────────────────────────────────────
        chroma_path = self._storage_dir / "vector_db"
        chroma_path.mkdir(parents=True, exist_ok=True)
        logger.info("ChromaDB: %s", chroma_path)
        self._chroma_client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = self._chroma_client.get_or_create_collection(
            name="code_nodes",
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "CodeGraphDB ready  (%d nodes, %d edges in SQLite)",
            self.node_count, self.edge_count,
        )

    def close(self) -> None:
        """Flush and close all backend connections."""
        self._conn.close()

    # ── SQLite bootstrap ──────────────────────────────────────────────

    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS nodes (
                id        TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                type      TEXT NOT NULL,
                data      TEXT NOT NULL          -- full JSON blob
            );
            CREATE INDEX IF NOT EXISTS idx_nodes_file
                ON nodes(file_path);

            CREATE TABLE IF NOT EXISTS edges (
                source   TEXT NOT NULL,
                target   TEXT NOT NULL,
                relation TEXT NOT NULL,
                data     TEXT NOT NULL,           -- full JSON blob
                PRIMARY KEY (source, target, relation)
            );
            CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target ON edges(target);
        """)
        self._conn.commit()

    # ── NetworkX bootstrap ────────────────────────────────────────────

    def _load_graph_from_sqlite(self) -> None:
        """Hydrate the in-memory graph from the durable SQLite store."""
        cur = self._conn.cursor()

        for row in cur.execute("SELECT id, data FROM nodes"):
            node_id, data_json = row
            data: Dict[str, Any] = json.loads(data_json)
            self._graph.add_node(
                node_id,
                file_path=data.get("file_path", ""),
                type=data.get("type", ""),
                name=data.get("name", ""),
                lines=data.get("lines", [data.get("start_line"), data.get("end_line")]),
                metadata=data.get("metadata", {}),
            )

        for row in cur.execute("SELECT source, target, relation, data FROM edges"):
            source, target, relation, data_json = row
            # Guard: only materialise edges whose endpoints both exist
            # as real nodes.  Edges to unresolved targets are kept in
            # SQLite and will appear once the target file is parsed.
            if source not in self._graph or target not in self._graph:
                continue
            meta: Dict[str, Any] = json.loads(data_json).get("metadata", {})
            self._graph.add_edge(
                source, target,
                key=relation,
                relation=relation,
                **meta,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  FILE-LEVEL MUTATIONS  (used by JIT Sync)
    # ══════════════════════════════════════════════════════════════════════

    def upsert_file_nodes(
        self,
        file_path: str,
        new_nodes: List[UniversalNode],
        new_edges: List[UniversalEdge],
    ) -> None:
        """Atomically replace every node/edge for *file_path* in all backends.

        1. **Delete** old data for *file_path* from SQLite, NetworkX, ChromaDB.
        2. **Insert** *new_nodes* and *new_edges* into all three stores.
        3. **Restore** cross-file incoming edges that were lost when old
           NetworkX nodes were removed.
        """
        # ── 1. Discover old nodes ─────────────────────────────────────
        old_rows = self._conn.execute(
            "SELECT id, type FROM nodes WHERE file_path = ?", (file_path,),
        ).fetchall()
        old_ids: List[str] = [r[0] for r in old_rows]
        old_embeddable_ids: List[str] = [
            r[0] for r in old_rows if r[1] in self._EMBEDDABLE_TYPES
        ]

        # ── 2. Delete from ChromaDB ───────────────────────────────────
        if old_embeddable_ids:
            try:
                self._collection.delete(ids=old_embeddable_ids)
            except Exception:
                logger.debug("ChromaDB delete (no-op): IDs not in collection")

        # ── 3. Delete + Insert in SQLite (single transaction) ─────────
        with self._conn:
            if old_ids:
                ph = ",".join("?" * len(old_ids))
                # Only delete edges *originating* from this file's nodes.
                # Edges from OTHER files that point INTO this file are kept
                # in SQLite so they can be restored in NetworkX later.
                self._conn.execute(
                    f"DELETE FROM edges WHERE source IN ({ph})", old_ids,
                )
                self._conn.execute(
                    f"DELETE FROM nodes WHERE id IN ({ph})", old_ids,
                )

            for node in new_nodes:
                self._conn.execute(
                    "INSERT OR REPLACE INTO nodes (id, file_path, type, data) "
                    "VALUES (?, ?, ?, ?)",
                    (node.id, node.file_path, node.type.value,
                     json.dumps(node.to_dict())),
                )
            for edge in new_edges:
                self._conn.execute(
                    "INSERT OR REPLACE INTO edges (source, target, relation, data) "
                    "VALUES (?, ?, ?, ?)",
                    (edge.source, edge.target, edge.relation.value,
                     json.dumps(edge.to_dict())),
                )

        # ── 4. Rebuild NetworkX for this file ─────────────────────────
        for nid in old_ids:
            if nid in self._graph:
                self._graph.remove_node(nid)  # also drops all its edges

        for node in new_nodes:
            self._graph.add_node(
                node.id,
                file_path=node.file_path,
                type=node.type.value,
                name=node.name,
                lines=list(node.lines),
                metadata=node.metadata,
            )
        for edge in new_edges:
            # Only add edge if both endpoints exist in the graph.
            # Edges to external / not-yet-parsed targets stay in SQLite.
            if edge.source not in self._graph or edge.target not in self._graph:
                continue
            self._graph.add_edge(
                edge.source, edge.target,
                key=edge.relation.value,
                relation=edge.relation.value,
                **edge.metadata,
            )

        # Restore cross-file incoming edges from SQLite.
        self._restore_incoming_edges({n.id for n in new_nodes})

        # ── 5. Insert into ChromaDB ───────────────────────────────────
        embeddable = [n for n in new_nodes if self._should_embed(n)]
        if embeddable:
            self._collection.upsert(
                ids=[n.id for n in embeddable],
                documents=[n.embedding_text for n in embeddable],
                metadatas=[
                    {
                        "file_path": n.file_path,
                        "type": n.type.value,
                        "name": n.name,
                        "node_id": n.id,
                    }
                    for n in embeddable
                ],
            )

    def delete_file_nodes(self, file_path: str) -> None:
        """Remove all trace of *file_path* from every backend."""
        self.upsert_file_nodes(file_path, [], [])

    # ── Internal helpers for mutations ────────────────────────────────

    def _should_embed(self, node: UniversalNode) -> bool:
        return node.type.value in self._EMBEDDABLE_TYPES

    def _restore_incoming_edges(self, target_ids: Set[str]) -> None:
        """Re-add edges from OTHER files whose targets are in *target_ids*.

        When a node is removed from NetworkX, all its edges vanish.
        Edges *originating* from this file are re-inserted by the caller,
        but edges originating from *other* files must be recovered from
        the SQLite store.
        """
        if not target_ids:
            return

        ph = ",".join("?" * len(target_ids))
        rows = self._conn.execute(
            f"SELECT source, target, relation, data FROM edges "
            f"WHERE target IN ({ph})",
            list(target_ids),
        ).fetchall()

        new_source_ids = target_ids  # edges we just inserted are already there
        for source, target, relation, data_json in rows:
            # Skip edges that were already inserted (same-file edges)
            if source in new_source_ids:
                continue
            # Only restore if the source node actually exists in the graph
            if source not in self._graph:
                continue
            meta: Dict[str, Any] = json.loads(data_json).get("metadata", {})
            self._graph.add_edge(
                source, target,
                key=relation,
                relation=relation,
                **meta,
            )

    # ══════════════════════════════════════════════════════════════════════
    #  QUERY APIs  (used by MCP Tools)
    # ══════════════════════════════════════════════════════════════════════

    def semantic_search(
        self,
        query: str,
        n_results: int = 5,
        node_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Search ChromaDB for nodes whose embedding text is close to *query*.

        Returns a list of dicts, each containing:
        ``id``, ``name``, ``type``, ``file_path``, ``score``, ``lines``.
        """
        where: Optional[Dict[str, str]] = None
        if node_type:
            where = {"type": node_type.upper()}

        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            logger.warning("ChromaDB query error: %s", exc)
            return []

        hits: List[Dict[str, Any]] = []
        if not results or not results["ids"] or not results["ids"][0]:
            return hits

        ids = results["ids"][0]
        metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(ids)
        dists = results["distances"][0] if results.get("distances") else [None] * len(ids)
        docs = results["documents"][0] if results.get("documents") else [""] * len(ids)

        for i, node_id in enumerate(ids):
            graph_data = dict(self._graph.nodes[node_id]) if node_id in self._graph else {}
            hits.append({
                "id": node_id,
                "name": metas[i].get("name", ""),
                "type": metas[i].get("type", ""),
                "file_path": metas[i].get("file_path", ""),
                "score": round(1.0 - dists[i], 4) if dists[i] is not None else None,
                "lines": graph_data.get("lines"),
                "document": docs[i],
            })

        return hits

    def get_node_context(
        self, node_id: str, depth: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Return *node_id*'s data together with its neighbourhood.

        At ``depth=1`` (default) the result includes every directly
        connected incoming and outgoing edge, plus a summary of each
        neighbour node.  Higher depths expand the frontier via BFS.
        """
        if node_id not in self._graph:
            return None

        # ── BFS to collect the neighbourhood ──────────────────────────
        visited: Set[str] = {node_id}
        frontier: Set[str] = {node_id}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                for _, tgt, _ in self._graph.out_edges(nid, data=True):
                    if tgt not in visited:
                        next_frontier.add(tgt)
                for src, _, _ in self._graph.in_edges(nid, data=True):
                    if src not in visited:
                        next_frontier.add(src)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        # ── Assemble the result ───────────────────────────────────────
        node_data = self._node_summary(node_id)

        outgoing: List[Dict[str, Any]] = []
        for _, tgt, data in self._graph.out_edges(node_id, data=True):
            outgoing.append({
                "target_id": tgt,
                "target_name": self._graph.nodes[tgt].get("name", "") if tgt in self._graph else tgt.split("::")[-1],
                "target_type": self._graph.nodes[tgt].get("type", "") if tgt in self._graph else "",
                "relation": data.get("relation", ""),
                "metadata": {k: v for k, v in data.items() if k != "relation"},
            })

        incoming: List[Dict[str, Any]] = []
        for src, _, data in self._graph.in_edges(node_id, data=True):
            incoming.append({
                "source_id": src,
                "source_name": self._graph.nodes[src].get("name", "") if src in self._graph else src.split("::")[-1],
                "source_type": self._graph.nodes[src].get("type", "") if src in self._graph else "",
                "relation": data.get("relation", ""),
                "metadata": {k: v for k, v in data.items() if k != "relation"},
            })

        # Neighbour summaries (depth > 1 may pull in transitive nodes)
        neighbours: Dict[str, Dict[str, Any]] = {}
        for nid in visited:
            if nid != node_id:
                neighbours[nid] = self._node_summary(nid)

        return {
            "node": node_data,
            "outgoing_edges": outgoing,
            "incoming_edges": incoming,
            "neighbours": neighbours,
        }

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return the raw data dict for a single node, or ``None``."""
        if node_id not in self._graph:
            return None
        return self._node_summary(node_id)

    def get_ego_subgraph(
        self, node_id: str, depth: int = 1,
    ) -> Optional[Dict[str, Any]]:
        """Extract an ego subgraph around *node_id* up to *depth* hops.

        Traversal is **undirected** so that both callers (incoming) and
        callees (outgoing) are discovered.  The returned edges are the
        original directed edges restricted to the discovered node set.

        Returns
        -------
        ``None`` if *node_id* is not in the graph, otherwise a dict::

            {
                "center": node_id,
                "nodes": [ {id, name, type, file_path}, ... ],
                "edges": [ {source, target, relation}, ... ],
            }
        """
        if node_id not in self._graph:
            return None

        # Undirected BFS to collect the neighbourhood
        visited: Set[str] = {node_id}
        frontier: Set[str] = {node_id}
        for _ in range(depth):
            next_frontier: Set[str] = set()
            for nid in frontier:
                # outgoing neighbours
                for _, tgt, _ in self._graph.out_edges(nid, data=True):
                    if tgt not in visited:
                        next_frontier.add(tgt)
                # incoming neighbours
                for src, _, _ in self._graph.in_edges(nid, data=True):
                    if src not in visited:
                        next_frontier.add(src)
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        # Collect node summaries
        nodes: List[Dict[str, Any]] = []
        for nid in visited:
            data = dict(self._graph.nodes[nid]) if nid in self._graph else {}
            nodes.append({
                "id": nid,
                "name": data.get("name", nid.split("::")[-1]),
                "type": data.get("type", ""),
                "file_path": data.get("file_path", ""),
            })

        # Collect directed edges restricted to the neighbourhood
        edges: List[Dict[str, Any]] = []
        for nid in visited:
            for _, tgt, data in self._graph.out_edges(nid, data=True):
                if tgt in visited:
                    edges.append({
                        "source": nid,
                        "target": tgt,
                        "relation": data.get("relation", ""),
                    })

        return {
            "center": node_id,
            "nodes": nodes,
            "edges": edges,
        }

    # ══════════════════════════════════════════════════════════════════════
    #  STRUCTURAL INSIGHTS  (used by get_structural_insights tool)
    # ══════════════════════════════════════════════════════════════════════

    #: Methods that are expected entry-points and should NOT be flagged
    #: as orphans even if nothing calls them.
    _ENTRY_POINT_NAMES: FrozenSet[str] = frozenset({
        "main", "__init__", "__str__", "__repr__", "__eq__", "__hash__",
        "__lt__", "__le__", "__gt__", "__ge__", "__len__", "__iter__",
        "__next__", "__enter__", "__exit__", "__call__", "__getattr__",
        "__setattr__", "__delattr__", "__getitem__", "__setitem__",
        "__contains__", "__del__", "__new__", "__post_init__",
        # Common framework entry-points
        "setUp", "tearDown", "setUpClass", "tearDownClass",
        "get", "post", "put", "patch", "delete", "head", "options",  # HTTP methods
    })

    def find_orphan_methods(self) -> List[Dict[str, Any]]:
        """Find METHODs/FUNCTIONs with zero incoming call edges (potential dead code).

        Excludes known entry-point names (``__init__``, ``main``, HTTP verbs, etc.)
        and private methods that start with ``_`` only if they also have zero
        internal callers.
        """
        orphans: List[Dict[str, Any]] = []
        call_relations = {"CALLS_INTERNAL", "CALLS_EXTERNAL", "POTENTIAL_CALL"}

        for nid, data in self._graph.nodes(data=True):
            ntype = data.get("type", "")
            if ntype not in ("METHOD", "FUNCTION"):
                continue

            name = data.get("name", "")
            # Skip known entry-points
            if name in self._ENTRY_POINT_NAMES:
                continue

            # Count incoming call edges (ignore CONTAINS edges)
            incoming_calls = sum(
                1 for _, _, edata in self._graph.in_edges(nid, data=True)
                if edata.get("relation", "") in call_relations
            )

            if incoming_calls == 0:
                orphans.append({
                    "id": nid,
                    "name": name,
                    "type": ntype,
                    "file_path": data.get("file_path", ""),
                })

        return orphans

    def find_most_connected(self, n: int = 10) -> List[Dict[str, Any]]:
        """Find the top-*n* most depended-upon nodes (highest incoming edge count).

        These are the highest-risk refactor targets — changing them has
        the widest blast radius.
        """
        call_relations = {"CALLS_INTERNAL", "CALLS_EXTERNAL", "POTENTIAL_CALL"}
        scored: List[tuple[str, int]] = []

        for nid, data in self._graph.nodes(data=True):
            ntype = data.get("type", "")
            if ntype not in ("METHOD", "FUNCTION", "CLASS"):
                continue

            incoming_calls = sum(
                1 for _, _, edata in self._graph.in_edges(nid, data=True)
                if edata.get("relation", "") in call_relations
            )
            if incoming_calls > 0:
                scored.append((nid, incoming_calls))

        scored.sort(key=lambda x: x[1], reverse=True)

        results: List[Dict[str, Any]] = []
        for nid, count in scored[:n]:
            data = dict(self._graph.nodes[nid])
            results.append({
                "id": nid,
                "name": data.get("name", ""),
                "type": data.get("type", ""),
                "file_path": data.get("file_path", ""),
                "incoming_call_count": count,
            })

        return results

    def find_circular_deps(self) -> List[List[str]]:
        """Detect circular call chains in the graph.

        Returns a list of cycles, where each cycle is a list of node IDs
        forming a loop.  Only considers CALLS_* edges (ignores CONTAINS
        and INHERITS).
        """
        # Build a simple DiGraph with only call edges
        call_graph = nx.DiGraph()
        call_relations = {"CALLS_INTERNAL", "CALLS_EXTERNAL", "POTENTIAL_CALL"}

        for src, tgt, data in self._graph.edges(data=True):
            if data.get("relation", "") in call_relations:
                call_graph.add_edge(src, tgt)

        try:
            cycles = list(nx.simple_cycles(call_graph))
        except Exception:
            cycles = []

        # Return at most 50 cycles to prevent explosion
        return [list(c) for c in cycles[:50]]

    # ══════════════════════════════════════════════════════════════════════
    #  UTILITY / DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════════════

    def get_tracked_files(self) -> Set[str]:
        """Return every ``file_path`` that currently has nodes in the store."""
        rows = self._conn.execute(
            "SELECT DISTINCT file_path FROM nodes"
        ).fetchall()
        return {r[0] for r in rows}

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ── Private helpers ───────────────────────────────────────────────

    def _node_summary(self, node_id: str) -> Dict[str, Any]:
        """Build a flat summary dict for a node in the graph."""
        if node_id not in self._graph:
            return {"id": node_id}
        attrs = dict(self._graph.nodes[node_id])
        return {"id": node_id, **attrs}
