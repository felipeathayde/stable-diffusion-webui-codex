#!/usr/bin/env python3
"""
Dependency Checker

Runs a static dependency walk starting from apps/server/run_api.py, verifies
that imports resolve, and recursively explores local (repo-internal) modules.

- External (site-packages/stdlib) modules: only verify the top-level module
  resolves; do not descend into their internals.
- Local modules (under this repo root): parse their imports and continue until
  leaves or already-visited modules.

Output: a tree view with per-module status and a summary. Exits 0 regardless,
but highlights missing modules for quick triage.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import sys
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set, Tuple


REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
START_FILE = os.path.join(REPO_ROOT, 'apps', 'server', 'run_api.py')


def ensure_repo_on_path() -> None:
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)


@dataclass
class DepNode:
    name: str
    kind: str  # 'local' | 'external' | 'missing'
    path: Optional[str] = None
    children: List['DepNode'] = field(default_factory=list)


def module_path_from_file(file_path: str) -> str:
    rel = os.path.relpath(os.path.abspath(file_path), REPO_ROOT)
    if rel.endswith('.py'):
        rel = rel[:-3]
    parts = []
    for p in rel.split(os.sep):
        if p == '__init__':
            continue
        parts.append(p)
    return '.'.join(parts)


def resolve_import_base(current_module: str, level: int) -> str:
    """Return base package for a relative import of given level.
    level=1 means current package; level=2 means parent package, etc.
    """
    parts = current_module.split('.')
    # If current is a module (not a package), drop last component
    if parts:
        parts = parts[:-1]
    asc = max(0, level - 1)
    if asc:
        parts = parts[:-asc]
    return '.'.join(parts)


def parse_imports(file_path: str, current_module: str) -> List[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        src = f.read()
    tree = ast.parse(src, filename=file_path)
    imports: Set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name:
                    imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            # Base module path
            if node.level and node.level > 0:
                base = resolve_import_base(current_module, node.level)
                if node.module:
                    mod = f"{base}.{node.module}" if base else node.module
                else:
                    mod = base
            else:
                mod = node.module or ''
            if mod:
                imports.add(mod)
            # Try to detect submodule imports (from X import Y)
            for alias in node.names:
                if alias.name and mod:
                    sub = f"{mod}.{alias.name}"
                    imports.add(sub)

    # Normalize: drop empty strings and builtins
    normed = sorted({m for m in imports if m and not m.startswith('builtins')})
    return normed


def classify_module(name: str) -> Tuple[str, Optional[str]]:
    """Return (kind, path) where kind is 'local'|'external'|'missing'."""
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec is None:
        return 'missing', None
    # builtin/namespace module
    origin = getattr(spec, 'origin', None)
    locations = getattr(spec, 'submodule_search_locations', None)
    def _is_local_path(p: Optional[str]) -> bool:
        return bool(p) and os.path.abspath(str(p)).startswith(REPO_ROOT)
    if origin == 'built-in':
        return 'external', None
    if locations:
        # package
        for loc in locations:
            if _is_local_path(loc):
                return 'local', os.path.join(loc, '__init__.py') if os.path.isfile(os.path.join(loc, '__init__.py')) else loc
        return 'external', str(list(locations)[0])
    # module
    if _is_local_path(origin):
        return 'local', origin
    return 'external', origin


def file_for_module(name: str, hinted_path: Optional[str]) -> Optional[str]:
    if hinted_path and os.path.isfile(hinted_path):
        return hinted_path
    # Best-effort: resolve spec again
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        return None
    if not spec:
        return None
    if spec.origin and os.path.isfile(spec.origin):
        return spec.origin
    if spec.submodule_search_locations:
        init = os.path.join(list(spec.submodule_search_locations)[0], '__init__.py')
        return init if os.path.isfile(init) else None
    return None


def walk_dependencies(start_file: str) -> DepNode:
    ensure_repo_on_path()
    root_mod = module_path_from_file(start_file)
    visited: Set[str] = set()

    def _walk(mod_name: str, src_path_hint: Optional[str]) -> DepNode:
        kind, mod_path = classify_module(mod_name)
        node = DepNode(name=mod_name, kind=kind, path=mod_path)
        if kind != 'local':
            return node
        # Local: parse and descend
        file_path = file_for_module(mod_name, mod_path)
        if not file_path:
            return node
        # Avoid revisiting
        if mod_name in visited:
            return node
        visited.add(mod_name)
        try:
            imports = parse_imports(file_path, current_module=mod_name)
        except Exception as e:
            node.children.append(DepNode(name=f"<parse-error: {e}>", kind='missing', path=file_path))
            return node
        for imp in imports:
            # Stop descending for externals automatically inside recursion
            child = _walk(imp, None)
            node.children.append(child)
        return node

    # Entry node is synthetic: 'apps.server.run_api' local
    entry = _walk(root_mod, start_file)
    return entry


def print_tree(node: DepNode, prefix: str = '') -> Tuple[int, int, int]:
    """Print a tree; returns (total, local, missing)."""
    marker = {
        'local': '[LOCAL]',
        'external': '[EXT]',
        'missing': '[MISSING]'
    }.get(node.kind, '[?]')
    loc = f" ({node.path})" if node.path else ''
    print(f"{prefix}{marker} {node.name}{loc}")
    total = 1
    local = 1 if node.kind == 'local' else 0
    missing = 1 if node.kind == 'missing' else 0
    n = len(node.children)
    for i, ch in enumerate(node.children):
        last = (i == n - 1)
        child_prefix = prefix + ("└─ " if last else "├─ ")
        t, l, m = print_tree(ch, prefix=child_prefix)
        total += t; local += l; missing += m
    return total, local, missing


def main() -> None:
    if not os.path.isfile(START_FILE):
        print(f"[ERROR] start file not found: {START_FILE}")
        sys.exit(1)
    entry = walk_dependencies(START_FILE)
    print("\nDependency tree (starting at apps/server/run_api.py):")
    total, local, missing = print_tree(entry)
    print("\nSummary:")
    print(f"  Total modules visited: {total}")
    print(f"  Local modules:         {local}")
    print(f"  Missing modules:       {missing}")
    if missing:
        print("\n[WARNING] Some modules are missing. See [MISSING] entries above.")


if __name__ == '__main__':
    main()

