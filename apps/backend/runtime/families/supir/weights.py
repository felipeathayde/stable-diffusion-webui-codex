"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SUPIR weights discovery and validation.
Resolves the per-variant SUPIR checkpoint paths from the configured `supir_models` roots (`apps/paths.json`).
Selection is strict and fail-loud to avoid ambiguous “it picked the wrong file” errors.

Symbols (top-level; keep in sync; no ghosts):
- `SupirVariant` (enum): SUPIR checkpoint variant (`v0F` Fidelity, `v0Q` Quality).
- `SupirWeights` (dataclass): Resolved SUPIR weights paths (variant checkpoint).
- `expected_variant_filenames` (function): Canonical filenames for a variant (used for diagnostics).
- `discover_supir_weights` (function): Discover SUPIR ckpts under roots (strict canonical match first).
- `resolve_supir_weights` (function): Resolve required weights or raise a `SupirWeightsError`.
- `supir_weights_diagnostics` (function): Build a JSON-friendly diagnostic payload for `/api/supir/models`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable, Sequence

from .errors import SupirWeightsError


class SupirVariant(str, Enum):
    V0F = "v0F"
    V0Q = "v0Q"


@dataclass(frozen=True)
class SupirWeights:
    variant: SupirVariant
    ckpt_path: Path


def expected_variant_filenames(variant: SupirVariant) -> tuple[str, ...]:
    if variant is SupirVariant.V0F:
        return ("SUPIR-v0F.ckpt",)
    if variant is SupirVariant.V0Q:
        return ("SUPIR-v0Q.ckpt",)
    raise SupirWeightsError(f"Unknown SUPIR variant: {variant!r}")


def _iter_candidate_files(roots: Sequence[Path]) -> Iterable[Path]:
    exts = {".ckpt", ".pt", ".pth", ".safetensors"}
    for root in roots:
        if root.is_file() and root.suffix.lower() in exts:
            yield root
            continue
        if not root.is_dir():
            continue
        try:
            for p in sorted(root.rglob("*"), key=lambda x: str(x).lower()):
                if p.is_file() and p.suffix.lower() in exts:
                    yield p
        except Exception:
            continue


def discover_supir_weights(*, roots: Sequence[Path]) -> dict[SupirVariant, Path]:
    """Return best-effort discoveries under the roots (canonical names first)."""

    found: dict[SupirVariant, Path] = {}
    by_name: dict[str, Path] = {}
    for p in _iter_candidate_files(roots):
        by_name[p.name] = p

    for variant in (SupirVariant.V0F, SupirVariant.V0Q):
        for name in expected_variant_filenames(variant):
            p = by_name.get(name)
            if p is not None:
                found[variant] = p
                break
    return found


def resolve_supir_weights(*, roots: Sequence[Path], variant: SupirVariant) -> SupirWeights:
    """Resolve SUPIR weights for a variant or raise a fail-loud error."""

    roots = [Path(r).expanduser() for r in roots]
    hits = discover_supir_weights(roots=roots)
    path = hits.get(variant)
    if path is None:
        expected = ", ".join(expected_variant_filenames(variant))
        scanned = [str(p) for p in roots]
        found_files = sorted({p.name for p in _iter_candidate_files(roots)})
        raise SupirWeightsError(
            "SUPIR weights missing for variant "
            f"{variant.value!r}. Expected {expected} under supir_models roots: {scanned}. "
            f"Found: {found_files or '[]'}"
        )
    if not path.is_file():
        raise SupirWeightsError(f"SUPIR weights path is not a file: {path}")
    return SupirWeights(variant=variant, ckpt_path=path)


def supir_weights_diagnostics(*, roots: Sequence[Path]) -> dict[str, object]:
    roots = [Path(r).expanduser() for r in roots]
    found = discover_supir_weights(roots=roots)
    expected: dict[str, object] = {}
    for variant in (SupirVariant.V0F, SupirVariant.V0Q):
        exp = list(expected_variant_filenames(variant))
        hit = found.get(variant)
        expected[variant.value] = {
            "expected_filenames": exp,
            "present": bool(hit and hit.is_file()),
            "path": str(hit) if hit else None,
        }

    files = []
    for p in _iter_candidate_files(roots):
        try:
            st = p.stat()
            files.append(
                {
                    "name": p.name,
                    "path": str(p),
                    "bytes": int(st.st_size),
                    "mtime": float(st.st_mtime),
                }
            )
        except Exception:
            files.append({"name": p.name, "path": str(p)})

    return {
        "roots": [str(p) for p in roots],
        "expected": expected,
        "found_files": files,
    }


__all__ = [
    "SupirVariant",
    "SupirWeights",
    "discover_supir_weights",
    "expected_variant_filenames",
    "resolve_supir_weights",
    "supir_weights_diagnostics",
]

