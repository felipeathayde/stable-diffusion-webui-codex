"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vendored “preset” descriptors for the GGUF converter UI.
Scans the local Hugging Face mirror under `apps/backend/huggingface/**` and exposes a small, typed list of
supported config directories (Flux/ZImage transformer configs + LLM CausalLM text encoders).

Symbols (top-level; keep in sync; no ghosts):
- `GGUFConverterPreset` (dataclass): UI preset entry (id/label/config dir + profile hints).
- `list_vendored_gguf_converter_presets` (function): List supported presets from the vendored HF mirror.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from apps.backend.inventory.scanners.vendored_hf import iter_vendored_hf_repos


@dataclass(frozen=True, slots=True)
class GGUFConverterPreset:
    id: str
    label: str
    config_dir: str
    kind: str
    profile_id: str | None = None
    profile_id_comfy: str | None = None
    profile_id_native: str | None = None


def _iter_candidate_config_dirs(repo_dir: str) -> Iterable[tuple[str, str]]:
    yield "", repo_dir
    try:
        entries = sorted(os.listdir(repo_dir), key=lambda s: s.lower())
    except Exception:
        return
    for name in entries:
        full = os.path.join(repo_dir, name)
        if os.path.isdir(full):
            yield name, full


def _has_weights_index(dir_path: str) -> bool:
    try:
        for name in os.listdir(dir_path):
            lower = name.lower()
            if lower.endswith(".safetensors.index.json"):
                return True
            if lower.endswith(".safetensors"):
                return True
    except Exception:
        return False
    return False


def _classify_config(cfg: dict[str, Any]) -> tuple[str, dict[str, str]]:
    class_name = str(cfg.get("_class_name") or "").strip()
    if class_name == "FluxTransformer2DModel":
        return (
            "flux_transformer",
            {
                "profile_id_comfy": "flux_transformer_comfy",
                "profile_id_native": "flux_transformer_native",
            },
        )
    if class_name == "ZImageTransformer2DModel":
        return (
            "zimage_transformer",
            {
                "profile_id_comfy": "zimage_transformer_comfy",
                "profile_id_native": "zimage_transformer_native",
            },
        )

    architectures = cfg.get("architectures")
    if isinstance(architectures, list) and any(str(a).endswith("ForCausalLM") for a in architectures):
        return ("llm_causallm", {"profile_id": "llama_hf_to_gguf"})

    return ("unknown", {})


def list_vendored_gguf_converter_presets(*, codex_root: Path) -> list[GGUFConverterPreset]:
    vendored_root = codex_root / "apps" / "backend" / "huggingface"
    presets: list[GGUFConverterPreset] = []

    for org, repo, repo_dir in iter_vendored_hf_repos(str(vendored_root)):
        for subdir, config_dir in _iter_candidate_config_dirs(repo_dir):
            cfg_path = os.path.join(config_dir, "config.json")
            if not os.path.isfile(cfg_path):
                continue
            if not _has_weights_index(config_dir):
                continue

            try:
                cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
            except Exception:
                continue

            kind, profile_hints = _classify_config(cfg)
            if kind == "unknown":
                continue

            preset_id = f"{org}/{repo}" + (f"/{subdir}" if subdir else "")
            label = f"{org}/{repo}" + (f" ({subdir})" if subdir else "")
            presets.append(
                GGUFConverterPreset(
                    id=preset_id,
                    label=label,
                    config_dir=str(Path(config_dir).resolve()),
                    kind=kind,
                    profile_id=profile_hints.get("profile_id"),
                    profile_id_comfy=profile_hints.get("profile_id_comfy"),
                    profile_id_native=profile_hints.get("profile_id_native"),
                )
            )

    presets.sort(key=lambda p: p.label.lower())
    return presets


__all__ = ["GGUFConverterPreset", "list_vendored_gguf_converter_presets"]

