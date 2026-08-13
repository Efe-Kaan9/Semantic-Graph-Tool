"""
JIT Sync Manager — ``JITSyncManager``
=======================================

Sits between the MCP tool layer and the storage layer.  Every time a
tool is called, ``sync()`` is invoked first:

1. Scan the target directory for ``.py`` files (noise dirs ignored).
2. Diff current OS ``mtime`` values against a durable registry.
3. Delete nodes for removed files, upsert nodes for new/changed files.
4. Update the registry.

The no-change fast-path is a pure in-memory dict comparison with no
I/O beyond a single ``os.scandir`` walk — designed to complete in
< 10 ms on typical project sizes.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from core.database import CodeGraphDB
from extractors.python_extractor import PythonExtractor

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────

#: Directory names that are never traversed regardless of nesting depth.
_IGNORED_DIRS: FrozenSet[str] = frozenset({
    ".git", ".hg", ".svn",
    "venv", ".venv", "env", ".env",
    "node_modules",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".tox", "dist", "build", "eggs", ".eggs",
    ".idea", ".vscode",
    ".code_graph",        # our own storage dir
})

#: Extensions that the current extractor fleet can handle.
_SUPPORTED_EXTENSIONS: FrozenSet[str] = frozenset({".py"})


# ──────────────────────────────────────────────────────────────────────────────
#  Sync result (returned for introspection / logging)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SyncResult:
    """Summary of a single ``sync()`` run."""

    added: List[str] = field(default_factory=list)
    modified: List[str] = field(default_factory=list)
    deleted: List[str] = field(default_factory=list)
    errors: List[Tuple[str, str]] = field(default_factory=list)  # (path, msg)
    elapsed_ms: float = 0.0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.modified or self.deleted)

    @property
    def total_changes(self) -> int:
        return len(self.added) + len(self.modified) + len(self.deleted)

    def __str__(self) -> str:
        return (
            f"SyncResult(+{len(self.added)} ~{len(self.modified)} "
            f"-{len(self.deleted)} err={len(self.errors)} "
            f"{self.elapsed_ms:.1f}ms)"
        )


# ──────────────────────────────────────────────────────────────────────────────
#  JIT Sync Manager
# ──────────────────────────────────────────────────────────────────────────────

class JITSyncManager:
    """Orchestrates file-change detection and incremental graph updates.

    Parameters
    ----------
    project_root:
        Absolute path to the codebase directory to watch.
    db:
        An initialised ``CodeGraphDB`` instance.
    storage_dir:
        Directory used by ``CodeGraphDB`` for its own files.
        The mtime registry is stored here as ``mtime_registry.json``.
    """

    def __init__(
        self,
        project_root: str | Path,
        db: CodeGraphDB,
        storage_dir: str | Path,
    ) -> None:
        # Force absolute resolution — never trust that CWD is the user's project.
        self._root = Path(project_root).resolve()
        self._db = db
        self._storage_dir = Path(storage_dir).resolve()
        self._registry_path = self._storage_dir / "mtime_registry.json"
        logger.info("JITSyncManager root: %s", self._root)
        logger.info("JITSyncManager registry: %s", self._registry_path)

        # Extractor fleet — keyed by extension
        self._extractors: Dict[str, PythonExtractor] = {
            ext: PythonExtractor(self._root)
            for ext in _SUPPORTED_EXTENSIONS
        }

        # In-memory registry:  abs_path_str -> mtime (float)
        self._registry: Dict[str, float] = self._load_registry()

    # ══════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════

    def sync(self) -> SyncResult:
        """Detect file-system changes and update all backends accordingly.

        Designed to be called as a pre-hook before every MCP tool
        execution.  On a clean run (nothing changed) the function exits
        in < 10 ms: it performs a directory walk and two dict comparisons,
        with no DB I/O.

        Returns
        -------
        SyncResult
            A lightweight summary of what changed.
        """
        t0 = time.perf_counter()
        result = SyncResult()

        # ── Cold Start Detection ──────────────────────────────────────
        # If no registry AND the DB is empty, this is the very first run.
        # Treat every file as "new" regardless of mtime — full index.
        cold_start = (not self._registry and self._db.node_count == 0)
        if cold_start:
            logger.info("COLD START: empty DB detected — indexing entire project")

        # 1) Snapshot the current on-disk state
        current: Dict[str, float] = self._scan_files()

        # 2) Diff against the registry
        new_or_modified: List[str] = []
        if cold_start:
            # Cold start: treat every file as new — skip mtime comparison
            for path in current:
                result.added.append(path)
                new_or_modified.append(path)
        else:
            for path, mtime in current.items():
                reg_mtime = self._registry.get(path)
                if reg_mtime is None:
                    result.added.append(path)
                    new_or_modified.append(path)
                elif mtime != reg_mtime:
                    result.modified.append(path)
                    new_or_modified.append(path)

        # 3) Deletions: in registry but no longer on disk
        current_set: Set[str] = set(current)
        for path in list(self._registry):
            if path not in current_set:
                result.deleted.append(path)

        # Fast-path: nothing to do
        if not result.changed:
            result.elapsed_ms = (time.perf_counter() - t0) * 1000
            return result

        # 4) Process deletions
        for path in result.deleted:
            try:
                self._db.delete_file_nodes(path)
                del self._registry[path]
                logger.debug("Deleted: %s", path)
            except Exception as exc:
                msg = str(exc)
                result.errors.append((path, msg))
                logger.warning("Error deleting %s: %s", path, msg)

        # 5) Process new / modified files
        for path in new_or_modified:
            self._process_file(path, current[path], result)

        # 6) Persist the updated registry
        self._save_registry()

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info("Sync complete: %s", result)
        return result

    def force_full_reindex(self) -> SyncResult:
        """Invalidate the entire registry and re-parse every file.

        Useful after upgrading the extractor or schema.  Slower than a
        normal ``sync()`` because every file is re-processed regardless
        of mtime.
        """
        logger.info("Starting full reindex of %s", self._root)
        self._registry.clear()
        return self.sync()

    @property
    def registry(self) -> Dict[str, float]:
        """Read-only view of the current mtime registry."""
        return dict(self._registry)

    def registered_file_count(self) -> int:
        return len(self._registry)

    # ══════════════════════════════════════════════════════════════════════
    #  Directory traversal
    # ══════════════════════════════════════════════════════════════════════

    def _scan_files(self) -> Dict[str, float]:
        """Return ``{abs_path: mtime}`` for every supported file under root.

        Uses ``os.scandir`` recursively with ``_IGNORED_DIRS`` pruning for
        maximum speed — measurably faster than ``Path.rglob`` on large
        trees because we skip whole subtrees early.
        """
        result: Dict[str, float] = {}
        self._walk(self._root, result)
        return result

    def _walk(self, directory: Path, out: Dict[str, float]) -> None:
        """Recursive scandir walk that prunes ignored directories."""
        try:
            entries = list(os.scandir(directory))
        except PermissionError:
            logger.debug("Permission denied: %s", directory)
            return

        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                if entry.name not in _IGNORED_DIRS:
                    self._walk(Path(entry.path), out)
            elif entry.is_file(follow_symlinks=False):
                suffix = Path(entry.name).suffix.lower()
                if suffix in _SUPPORTED_EXTENSIONS:
                    try:
                        stat = entry.stat(follow_symlinks=False)
                        out[entry.path] = stat.st_mtime
                    except OSError:
                        pass  # File vanished between scandir and stat

    # ══════════════════════════════════════════════════════════════════════
    #  File processing
    # ══════════════════════════════════════════════════════════════════════

    def _process_file(
        self, path: str, mtime: float, result: SyncResult,
    ) -> None:
        """Parse *path* and upsert its nodes into the DB."""
        suffix = Path(path).suffix.lower()
        extractor = self._extractors.get(suffix)
        if extractor is None:
            # No extractor for this extension — shouldn't happen given _scan_files
            logger.debug("No extractor for %s — skipping", path)
            return

        try:
            code = Path(path).read_text(encoding="utf-8", errors="replace")
            nodes, edges = extractor.parse_file(Path(path), code)
            self._db.upsert_file_nodes(path, nodes, edges)
            self._registry[path] = mtime
            logger.debug(
                "Indexed %s  (%d nodes, %d edges)", path, len(nodes), len(edges),
            )
        except Exception as exc:
            msg = str(exc)
            result.errors.append((path, msg))
            logger.warning("Error indexing %s: %s", path, msg)
            # Do NOT update registry — will be retried on next sync

    # ══════════════════════════════════════════════════════════════════════
    #  Registry persistence
    # ══════════════════════════════════════════════════════════════════════

    def _load_registry(self) -> Dict[str, float]:
        """Load the mtime registry from disk, or return an empty dict."""
        if not self._registry_path.exists():
            return {}
        try:
            raw = self._registry_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                return {str(k): float(v) for k, v in data.items()}
        except (json.JSONDecodeError, ValueError, OSError) as exc:
            logger.warning(
                "Could not load mtime registry (%s) — starting fresh: %s",
                self._registry_path, exc,
            )
        return {}

    def _save_registry(self) -> None:
        """Atomically write the mtime registry to disk.

        Uses a write-then-rename pattern so the file is never left in a
        corrupt state if the process is interrupted mid-write.
        """
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        tmp = self._registry_path.with_suffix(".tmp")
        try:
            tmp.write_text(
                json.dumps(self._registry, indent=2), encoding="utf-8",
            )
            tmp.replace(self._registry_path)
        except OSError as exc:
            logger.error("Failed to save mtime registry: %s", exc)
            tmp.unlink(missing_ok=True)
