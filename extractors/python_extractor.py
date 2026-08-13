"""
Python Extractor — tree-sitter-based parser for Python source files.

Transforms a ``.py`` file into a list of ``UniversalNode`` / ``UniversalEdge``
objects that the language-agnostic Core Engine can ingest.

Key responsibilities
--------------------
* Extract FILE, CLASS, METHOD, FUNCTION nodes with rich metadata
  (docstrings, decorators, parameters, return types, base classes).
* Build a per-file **alias map** from ``import`` / ``from … import`` statements
  so that function calls can be resolved to fully-qualified node IDs.
* Classify every detected call as CALLS_INTERNAL, CALLS_EXTERNAL, or
  POTENTIAL_CALL depending on how much static information is available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

import tree_sitter_python as tspython
from tree_sitter import Language, Node, Parser

from models import (
    BaseExtractor,
    EdgeRelation,
    NodeType,
    UniversalEdge,
    UniversalNode,
)

logger = logging.getLogger(__name__)

# Singleton language / parser ------------------------------------------------
_PY_LANGUAGE = Language(tspython.language())


# ──────────────────────────────────────────────────────────────────────────────
#  Internal helper: lightweight call descriptor
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _RawCall:
    """Lightweight representation of a call-expression found in the AST."""

    name: str                   # The function / method name being called
    receiver: Optional[str]     # Object part: ``self``, ``os``, ``User``, …
    line: int                   # 1-indexed source line of the call


# ──────────────────────────────────────────────────────────────────────────────
#  Python Extractor
# ──────────────────────────────────────────────────────────────────────────────

class PythonExtractor(BaseExtractor):
    """Extract ``UniversalNode`` / ``UniversalEdge`` from Python source."""

    # Python built-in names — calls to these are never interesting edges.
    _BUILTINS: FrozenSet[str] = frozenset({
        "abs", "all", "any", "ascii", "bin", "bool", "breakpoint",
        "bytearray", "bytes", "callable", "chr", "classmethod",
        "compile", "complex", "delattr", "dict", "dir", "divmod",
        "enumerate", "eval", "exec", "filter", "float", "format",
        "frozenset", "getattr", "globals", "hasattr", "hash", "help",
        "hex", "id", "input", "int", "isinstance", "issubclass",
        "iter", "len", "list", "locals", "map", "max", "memoryview",
        "min", "next", "object", "oct", "open", "ord", "pow", "print",
        "property", "range", "repr", "reversed", "round", "set",
        "setattr", "slice", "sorted", "staticmethod", "str", "sum",
        "super", "tuple", "type", "vars", "zip", "__import__",
        # Common typing helpers
        "TypeVar", "Generic", "Protocol", "Union", "Optional",
    })

    # ── Construction ──────────────────────────────────────────────────────

    def __init__(self, project_root: Path) -> None:
        self._project_root = Path(project_root).resolve()
        self._parser = Parser(_PY_LANGUAGE)

    # ── Public interface (BaseExtractor) ──────────────────────────────────

    @property
    def supported_extensions(self) -> List[str]:
        return [".py"]

    def parse_file(
        self,
        file_path: Path,
        code: str,
        *,
        module_id: Optional[str] = None,
    ) -> Tuple[List[UniversalNode], List[UniversalEdge]]:
        """Parse a single ``.py`` file into universal nodes and edges."""
        file_path = Path(file_path).resolve()
        if module_id is None:
            module_id = self._compute_module_id(file_path)

        tree = self._parser.parse(code.encode("utf-8"))
        root = tree.root_node

        nodes: List[UniversalNode] = []
        edges: List[UniversalEdge] = []

        # 1) Build the import alias maps for this file
        symbol_imports, module_imports = self._build_alias_map(root, module_id)

        # 2) Create the FILE node
        file_node = self._make_file_node(module_id, str(file_path), code, root)
        nodes.append(file_node)

        # 3) Pre-scan: collect names of top-level functions (needed for
        #    resolving intra-module calls before the nodes are created).
        local_fn_names: Set[str] = set()
        for child in root.children:
            actual, _ = self._unwrap_decorated(child)
            if actual is not None and actual.type == "function_definition":
                name_node = actual.child_by_field_name("name")
                if name_node:
                    local_fn_names.add(self._text(name_node))

        # 4) Main walk: extract classes & functions, build edges
        for child in root.children:
            actual, decorators = self._unwrap_decorated(child)
            if actual is None:
                continue

            if actual.type == "function_definition":
                func_node = self._make_function_node(
                    actual, module_id, str(file_path),
                    parent_id=None, decorators=decorators,
                )
                nodes.append(func_node)
                edges.append(UniversalEdge(
                    source=module_id, target=func_node.id,
                    relation=EdgeRelation.CONTAINS,
                ))
                # Calls inside this function
                body = actual.child_by_field_name("body")
                if body:
                    edges.extend(self._resolve_calls(
                        self._collect_calls(body),
                        source_id=func_node.id,
                        class_id=None,
                        module_id=module_id,
                        symbol_imports=symbol_imports,
                        module_imports=module_imports,
                        local_names=local_fn_names,
                    ))

            elif actual.type == "class_definition":
                cls_node, method_nodes, cls_edges = self._extract_class(
                    actual, module_id, str(file_path),
                    decorators=decorators,
                    symbol_imports=symbol_imports,
                    module_imports=module_imports,
                    local_names=local_fn_names,
                )
                nodes.append(cls_node)
                nodes.extend(method_nodes)
                edges.append(UniversalEdge(
                    source=module_id, target=cls_node.id,
                    relation=EdgeRelation.CONTAINS,
                ))
                edges.extend(cls_edges)

        return nodes, edges

    # ── Module-ID computation ─────────────────────────────────────────────

    def _compute_module_id(self, file_path: Path) -> str:
        """Derive a dotted module path from a file's position in the project."""
        try:
            rel = file_path.relative_to(self._project_root)
        except ValueError:
            return file_path.stem

        parts = list(rel.with_suffix("").parts)
        # __init__.py represents the *package*, not a module called __init__
        if parts and parts[-1] == "__init__":
            parts.pop()
        return ".".join(parts) if parts else file_path.stem

    # ══════════════════════════════════════════════════════════════════════
    #  IMPORT ALIAS MAP
    # ══════════════════════════════════════════════════════════════════════

    def _build_alias_map(
        self, root: Node, module_id: str,
    ) -> Tuple[Dict[str, str], Dict[str, str]]:
        """Build two look-up dicts from the file's import statements.

        Returns
        -------
        symbol_imports : dict
            ``local_name  ->  "module.path::SymbolName"``
            Populated by ``from X import Y [as Z]``.
        module_imports : dict
            ``local_name  ->  "module.path"``
            Populated by ``import X [as Y]``.
        """
        symbol_imports: Dict[str, str] = {}
        module_imports: Dict[str, str] = {}

        for child in root.children:
            if child.type == "import_statement":
                self._process_import_stmt(child, module_imports)
            elif child.type == "import_from_statement":
                self._process_from_import_stmt(child, module_id, symbol_imports)

        return symbol_imports, module_imports

    # ── import X / import X as Y ──

    def _process_import_stmt(
        self, node: Node, module_imports: Dict[str, str],
    ) -> None:
        for child in node.children:
            if child.type == "dotted_name":
                full_path = self._text(child)
                # ``import os.path`` → local name is "os" (first segment)
                local_name = full_path.split(".")[0]
                module_imports[local_name] = local_name
            elif child.type == "aliased_import":
                name_node = child.child_by_field_name("name")
                alias_node = child.child_by_field_name("alias")
                if name_node:
                    full_path = self._text(name_node)
                    if alias_node:
                        # ``import numpy as np`` → np -> numpy
                        module_imports[self._text(alias_node)] = full_path
                    else:
                        local_name = full_path.split(".")[0]
                        module_imports[local_name] = local_name

    # ── from X import Y / from X import Y as Z ──

    def _process_from_import_stmt(
        self, node: Node, module_id: str, symbol_imports: Dict[str, str],
    ) -> None:
        # 1) Resolve the source module
        mod_name_node = node.child_by_field_name("module_name")
        if mod_name_node is None:
            return
        source_module = self._resolve_module_name(mod_name_node, module_id)
        if source_module is None:
            return

        # 2) Walk children after the ``import`` keyword
        past_import_kw = False
        for child in node.children:
            if not past_import_kw:
                if self._text(child) == "import":
                    past_import_kw = True
                continue

            if child.type == "dotted_name":
                sym = self._text(child)
                symbol_imports[sym] = f"{source_module}::{sym}"

            elif child.type == "aliased_import":
                name_n = child.child_by_field_name("name")
                alias_n = child.child_by_field_name("alias")
                if name_n:
                    sym = self._text(name_n)
                    local = self._text(alias_n) if alias_n else sym
                    symbol_imports[local] = f"{source_module}::{sym}"
            # Silently skip wildcard_import (``from X import *``)

    # ── Relative-import resolution ──

    def _resolve_module_name(
        self, node: Node, current_module_id: str,
    ) -> Optional[str]:
        """Turn a ``module_name`` AST node into a dotted path string."""
        if node.type == "dotted_name":
            return self._text(node)

        if node.type == "relative_import":
            prefix_node: Optional[Node] = None
            dotted_node: Optional[Node] = None
            for child in node.children:
                if child.type == "import_prefix":
                    prefix_node = child
                elif child.type == "dotted_name":
                    dotted_node = child

            dot_count = len(self._text(prefix_node)) if prefix_node else 0
            suffix = self._text(dotted_node) if dotted_node else ""
            return self._resolve_relative_import(
                dot_count, suffix, current_module_id,
            )
        return None

    @staticmethod
    def _resolve_relative_import(
        dot_count: int, suffix: str, module_id: str,
    ) -> str:
        """Resolve ``from ..utils import X`` given the current module ID."""
        parts = module_id.split(".")
        # 1 dot = current package, 2 = parent package, …
        levels = min(dot_count, len(parts))
        base = parts[:-levels] if levels else parts
        if suffix:
            base = base + suffix.split(".")
        return ".".join(base) if base else suffix

    # ══════════════════════════════════════════════════════════════════════
    #  NODE FACTORIES
    # ══════════════════════════════════════════════════════════════════════

    def _make_file_node(
        self, module_id: str, file_path: str, code: str, root: Node,
    ) -> UniversalNode:
        line_count = max(code.count("\n") + (0 if code.endswith("\n") else 1), 1)
        docstring = self._extract_docstring(root)
        meta: Dict[str, Any] = {}
        if docstring:
            meta["docstring"] = docstring
        return UniversalNode(
            id=module_id, type=NodeType.FILE, name=module_id.split(".")[-1],
            file_path=file_path, lines=(1, line_count), metadata=meta,
        )

    def _make_function_node(
        self,
        node: Node,
        module_id: str,
        file_path: str,
        *,
        parent_id: Optional[str] = None,
        decorators: Optional[List[str]] = None,
    ) -> UniversalNode:
        name_node = node.child_by_field_name("name")
        name = self._text(name_node) if name_node else "<anonymous>"

        if parent_id is not None:
            node_id = f"{parent_id}::{name}"
            node_type = NodeType.METHOD
        else:
            node_id = f"{module_id}::{name}"
            node_type = NodeType.FUNCTION

        body = node.child_by_field_name("body")
        meta: Dict[str, Any] = {}

        params = self._extract_parameters(node.child_by_field_name("parameters"))
        if params is not None:
            meta["parameters"] = params

        ret = self._extract_return_type(node)
        if ret:
            meta["return_type"] = ret

        doc = self._extract_docstring(body) if body else None
        if doc:
            meta["docstring"] = doc

        if decorators:
            meta["decorators"] = decorators

        return UniversalNode(
            id=node_id, type=node_type, name=name,
            file_path=file_path,
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata=meta,
        )

    def _extract_class(
        self,
        node: Node,
        module_id: str,
        file_path: str,
        *,
        decorators: Optional[List[str]] = None,
        symbol_imports: Dict[str, str],
        module_imports: Dict[str, str],
        local_names: Set[str],
    ) -> Tuple[UniversalNode, List[UniversalNode], List[UniversalEdge]]:
        """Build a CLASS node, its METHOD children, and all related edges."""

        name_node = node.child_by_field_name("name")
        class_name = self._text(name_node) if name_node else "<anonymous>"
        class_id = f"{module_id}::{class_name}"

        bases = self._extract_bases(node)
        body = node.child_by_field_name("body")
        doc = self._extract_docstring(body) if body else None

        meta: Dict[str, Any] = {}
        if bases:
            meta["bases"] = bases
        if doc:
            meta["docstring"] = doc
        if decorators:
            meta["decorators"] = decorators

        cls_node = UniversalNode(
            id=class_id, type=NodeType.CLASS, name=class_name,
            file_path=file_path,
            lines=(node.start_point[0] + 1, node.end_point[0] + 1),
            metadata=meta,
        )

        method_nodes: List[UniversalNode] = []
        cls_edges: List[UniversalEdge] = []

        # ── Inheritance edges ──
        for base in bases:
            target = self._resolve_base_class(
                base, module_id, symbol_imports, module_imports,
            )
            cls_edges.append(UniversalEdge(
                source=class_id, target=target,
                relation=EdgeRelation.INHERITS,
            ))

        # ── Pre-collect method names for intra-class call resolution ──
        class_method_names: Set[str] = set()
        if body:
            for child in body.children:
                actual, _ = self._unwrap_decorated(child)
                if actual is not None and actual.type == "function_definition":
                    mn = actual.child_by_field_name("name")
                    if mn:
                        class_method_names.add(self._text(mn))

        # ── Extract methods ──
        if body:
            for child in body.children:
                actual, method_decs = self._unwrap_decorated(child)
                if actual is None or actual.type != "function_definition":
                    continue

                method_node = self._make_function_node(
                    actual, module_id, file_path,
                    parent_id=class_id, decorators=method_decs,
                )
                method_nodes.append(method_node)
                cls_edges.append(UniversalEdge(
                    source=class_id, target=method_node.id,
                    relation=EdgeRelation.CONTAINS,
                ))

                # Calls inside the method body
                method_body = actual.child_by_field_name("body")
                if method_body:
                    cls_edges.extend(self._resolve_calls(
                        self._collect_calls(method_body),
                        source_id=method_node.id,
                        class_id=class_id,
                        module_id=module_id,
                        symbol_imports=symbol_imports,
                        module_imports=module_imports,
                        local_names=local_names,
                        class_method_names=class_method_names,
                    ))

        return cls_node, method_nodes, cls_edges

    def _resolve_base_class(
        self,
        base_name: str,
        module_id: str,
        symbol_imports: Dict[str, str],
        module_imports: Dict[str, str],
    ) -> str:
        """Best-effort resolution of a base-class name to a node ID."""
        # Imported symbol: ``from X import Base`` → X::Base
        if base_name in symbol_imports:
            return symbol_imports[base_name]

        # Dotted attribute: ``module.ClassName``
        if "." in base_name:
            parts = base_name.rsplit(".", 1)
            prefix, attr = parts[0], parts[1]
            if prefix in module_imports:
                return f"{module_imports[prefix]}::{attr}"

        # Unresolvable — keep as a bare name so the graph doesn't break
        return base_name

    # ══════════════════════════════════════════════════════════════════════
    #  METADATA HELPERS
    # ══════════════════════════════════════════════════════════════════════

    def _extract_docstring(self, body_node: Optional[Node]) -> Optional[str]:
        """Return the docstring if the first statement is a string literal."""
        if body_node is None:
            return None

        children = body_node.children
        for child in children:
            if child.type == "expression_statement":
                for expr in child.children:
                    if expr.type in ("string", "concatenated_string"):
                        return self._clean_docstring(self._text(expr))
                return None  # first real stmt wasn't a string
            if child.type in ("comment", "newline"):
                continue
            return None  # first real stmt is something else
        return None

    @staticmethod
    def _clean_docstring(raw: str) -> str:
        for delim in ('"""', "'''", '"', "'"):
            if raw.startswith(delim) and raw.endswith(delim) and len(raw) >= 2 * len(delim):
                return raw[len(delim):-len(delim)].strip()
        return raw.strip()

    def _unwrap_decorated(
        self, node: Node,
    ) -> Tuple[Optional[Node], Optional[List[str]]]:
        """If *node* is ``decorated_definition``, return (inner_def, decorators).

        Otherwise return ``(node, None)`` for function/class, ``(None, None)``
        for anything else.
        """
        if node.type == "decorated_definition":
            decs: List[str] = []
            definition: Optional[Node] = None
            for child in node.children:
                if child.type == "decorator":
                    text = self._text(child).lstrip("@").strip()
                    decs.append(text)
                elif child.type in ("function_definition", "class_definition"):
                    definition = child
            return definition, decs or None

        if node.type in ("function_definition", "class_definition"):
            return node, None
        return None, None

    def _extract_parameters(self, params_node: Optional[Node]) -> Optional[str]:
        if params_node is None:
            return None
        text = self._text(params_node)
        # Strip enclosing parentheses
        if text.startswith("(") and text.endswith(")"):
            text = text[1:-1]
        return text

    def _extract_return_type(self, func_node: Node) -> Optional[str]:
        ret = func_node.child_by_field_name("return_type")
        return self._text(ret) if ret else None

    def _extract_bases(self, class_node: Node) -> List[str]:
        bases: List[str] = []
        superclasses = class_node.child_by_field_name("superclasses")
        if superclasses is None:
            return bases
        for child in superclasses.children:
            if child.type in ("identifier", "attribute"):
                bases.append(self._text(child))
            # Skip keyword_argument (``metaclass=…``) and punctuation
        return bases

    # ══════════════════════════════════════════════════════════════════════
    #  CALL EXTRACTION & RESOLUTION
    # ══════════════════════════════════════════════════════════════════════

    def _collect_calls(self, body_node: Node) -> List[_RawCall]:
        """DFS-walk *body_node* and return every call expression found."""
        result: List[_RawCall] = []
        self._walk_calls(body_node, result)
        return result

    def _walk_calls(self, node: Node, out: List[_RawCall]) -> None:
        # Don't descend into nested function / class definitions —
        # they are their own scope and will be processed separately.
        if node.type in ("function_definition", "class_definition"):
            return

        if node.type == "call":
            func = node.child_by_field_name("function")
            if func is not None:
                parsed = self._parse_call_target(func)
                if parsed is not None:
                    receiver, name = parsed
                    out.append(_RawCall(
                        name=name, receiver=receiver,
                        line=node.start_point[0] + 1,
                    ))

        for child in node.children:
            self._walk_calls(child, out)

    def _parse_call_target(
        self, func_node: Node,
    ) -> Optional[Tuple[Optional[str], str]]:
        """Return ``(receiver, name)`` or ``None`` if too complex.

        Examples::

            foo()            → (None,       "foo")
            self.bar()       → ("self",     "bar")
            os.path.join()   → ("os.path",  "join")
            super().save()   → ("super()",  "save")
        """
        if func_node.type == "identifier":
            return None, self._text(func_node)

        if func_node.type == "attribute":
            obj = func_node.child_by_field_name("object")
            attr = func_node.child_by_field_name("attribute")
            if obj is not None and attr is not None:
                return self._flatten_dotted(obj), self._text(attr)

        # Subscript calls, lambda calls, etc. — skip
        return None

    def _flatten_dotted(self, node: Node) -> str:
        """Flatten chained attribute access: ``os.path`` → ``"os.path"``."""
        if node.type == "identifier":
            return self._text(node)
        if node.type == "attribute":
            obj = node.child_by_field_name("object")
            attr = node.child_by_field_name("attribute")
            if obj is not None and attr is not None:
                return f"{self._flatten_dotted(obj)}.{self._text(attr)}"
        # Fallback (call nodes like ``super()``, subscripts, etc.)
        return self._text(node)

    # ── Resolution engine ─────────────────────────────────────────────────

    def _resolve_calls(
        self,
        calls: List[_RawCall],
        source_id: str,
        class_id: Optional[str],
        module_id: str,
        symbol_imports: Dict[str, str],
        module_imports: Dict[str, str],
        local_names: Set[str],
        class_method_names: Optional[Set[str]] = None,
    ) -> List[UniversalEdge]:
        """Convert a list of raw calls into typed, deduplicated edges."""
        edges: List[UniversalEdge] = []
        seen: Set[Tuple[str, str]] = set()   # (target, relation) dedup key

        for call in calls:
            edge = self._resolve_one(
                call, source_id, class_id, module_id,
                symbol_imports, module_imports, local_names,
                class_method_names or set(),
            )
            if edge is None:
                continue
            key = (edge.target, edge.relation.value)
            if key not in seen:
                seen.add(key)
                edges.append(edge)

        return edges

    def _resolve_one(
        self,
        call: _RawCall,
        source_id: str,
        class_id: Optional[str],
        module_id: str,
        symbol_imports: Dict[str, str],
        module_imports: Dict[str, str],
        local_names: Set[str],
        class_method_names: Set[str],
    ) -> Optional[UniversalEdge]:
        meta: Dict[str, Any] = {"call_line": call.line}

        if call.receiver is not None:
            return self._resolve_attribute_call(
                call, source_id, class_id, module_id,
                symbol_imports, module_imports, meta,
            )
        return self._resolve_simple_call(
            call, source_id, module_id,
            symbol_imports, local_names, meta,
        )

    # ── obj.method() ──

    def _resolve_attribute_call(
        self,
        call: _RawCall,
        source_id: str,
        class_id: Optional[str],
        module_id: str,
        symbol_imports: Dict[str, str],
        module_imports: Dict[str, str],
        meta: Dict[str, Any],
    ) -> Optional[UniversalEdge]:
        receiver = call.receiver
        assert receiver is not None

        # 1) self.method() / cls.method()  →  CALLS_INTERNAL
        if receiver in ("self", "cls") and class_id:
            return UniversalEdge(
                source=source_id,
                target=f"{class_id}::{call.name}",
                relation=EdgeRelation.CALLS_INTERNAL,
                metadata=meta,
            )

        # 2) super().method()  →  CALLS_INTERNAL (to own class for now)
        if receiver in ("super()", "super") and class_id:
            return UniversalEdge(
                source=source_id,
                target=f"{class_id}::{call.name}",
                relation=EdgeRelation.CALLS_INTERNAL,
                metadata={**meta, "via_super": True},
            )

        # 3) Module-import receiver:  ``logging.info()``, ``os.path.join()``
        #    Try longest prefix first.
        receiver_parts = receiver.split(".")
        for i in range(len(receiver_parts), 0, -1):
            prefix = ".".join(receiver_parts[:i])
            if prefix in module_imports:
                resolved_mod = module_imports[prefix]
                remaining = receiver_parts[i:]
                if remaining:
                    target = f"{resolved_mod}.{'.'.join(remaining)}::{call.name}"
                else:
                    target = f"{resolved_mod}::{call.name}"
                return UniversalEdge(
                    source=source_id, target=target,
                    relation=EdgeRelation.CALLS_EXTERNAL,
                    metadata=meta,
                )

        # 4) Symbol-import receiver:  ``User.find()``  (User from-imported)
        if receiver in symbol_imports:
            return UniversalEdge(
                source=source_id,
                target=f"{symbol_imports[receiver]}::{call.name}",
                relation=EdgeRelation.CALLS_EXTERNAL,
                metadata=meta,
            )

        # 5) Unknown receiver → POTENTIAL_CALL
        return UniversalEdge(
            source=source_id,
            target=f"?::{receiver}.{call.name}",
            relation=EdgeRelation.POTENTIAL_CALL,
            metadata=meta,
        )

    # ── name() ──

    def _resolve_simple_call(
        self,
        call: _RawCall,
        source_id: str,
        module_id: str,
        symbol_imports: Dict[str, str],
        local_names: Set[str],
        meta: Dict[str, Any],
    ) -> Optional[UniversalEdge]:
        name = call.name

        # Skip builtins
        if name in self._BUILTINS:
            return None

        # 1) From-imported symbol:  ``verify()``
        if name in symbol_imports:
            return UniversalEdge(
                source=source_id, target=symbol_imports[name],
                relation=EdgeRelation.CALLS_EXTERNAL,
                metadata=meta,
            )

        # 2) File-local function:  ``helper()``
        if name in local_names:
            return UniversalEdge(
                source=source_id, target=f"{module_id}::{name}",
                relation=EdgeRelation.CALLS_INTERNAL,
                metadata=meta,
            )

        # 3) Unresolvable → POTENTIAL_CALL
        return UniversalEdge(
            source=source_id, target=f"?::{name}",
            relation=EdgeRelation.POTENTIAL_CALL,
            metadata=meta,
        )

    # ══════════════════════════════════════════════════════════════════════
    #  TREE-SITTER HELPERS
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _text(node: Optional[Node]) -> str:
        """Extract the source text of a node (safe on ``None``)."""
        if node is None:
            return ""
        raw = node.text
        return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
