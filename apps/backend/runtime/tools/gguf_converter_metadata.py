"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GGUF metadata injection helpers for the converter.
Adds provenance/source metadata and minimal architecture keys required by loader tooling.

Symbols (top-level; keep in sync; no ghosts):
- `_is_hf_repo_id` (function): Returns True when a string looks like a Hugging Face repo id (`org/repo`).
- `add_basic_metadata` (function): Adds standard provenance/license metadata keys into the output GGUF.
"""

from __future__ import annotations

import datetime as _dt
import re
from pathlib import Path

from apps.backend.quantization.gguf import GGUFWriter
from apps.backend.infra.config.provenance import CODEX_GENERATED_BY, CODEX_REPO_URL, best_effort_git_commit
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType


def _is_hf_repo_id(value: str) -> bool:
    candidate = str(value or "").strip()
    if not candidate:
        return False
    if candidate.startswith((".", "/", "\\")):
        return False
    return bool(re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", candidate))


def add_basic_metadata(
    writer: GGUFWriter,
    arch: str,
    config: dict,
    quant: QuantizationType,
    *,
    config_path: Path,
    safetensors_path: str,
) -> None:
    # `GGUFWriter` writes `general.architecture` eagerly in `__init__`.
    # Codex uses a custom metadata schema, so remove it from the output.
    try:
        for shard in writer.kv_data:
            if isinstance(shard, dict):
                shard.pop("general.architecture", None)
    except Exception:
        pass

    repo_root = Path(__file__).resolve().parents[4]
    commit = best_effort_git_commit(repo_root)
    writer.add_string("codex.quantized_by", CODEX_GENERATED_BY)
    writer.add_string("codex.repository", CODEX_REPO_URL)
    if commit:
        writer.add_string("codex.commit", commit)

    model_name = str(config.get("_name_or_path") or config.get("name") or "model")
    writer.add_string("model.name", model_name)
    writer.add_string("model.architecture", str(arch))

    upstream = str(config.get("_name_or_path") or "").strip()
    if _is_hf_repo_id(upstream):
        writer.add_string("model.repository", f"https://huggingface.co/{upstream}")

    # Minimal model metadata; loaders in this repo generally key off tensor names and shapes.
    writer.add_uint32("model.context_length", int(config.get("max_position_embeddings", 4096)))
    writer.add_uint32("model.embedding_length", int(config.get("hidden_size", 4096)))
    writer.add_uint32("model.block_count", int(config.get("num_hidden_layers", 32)))
    writer.add_uint32("model.attention.head_count", int(config.get("num_attention_heads", 32)))
    writer.add_uint32("model.attention.head_count_kv", int(config.get("num_key_value_heads", 8)))
    writer.add_float32("model.rope.freq_base", float(config.get("rope_theta", 10000.0)))
    writer.add_float32("model.attention.layer_norm_rms_epsilon", float(config.get("rms_norm_eps", 1e-6)))

    writer.add_string("gguf.quantized_at_utc", _dt.datetime.now(tz=_dt.timezone.utc).isoformat())
    writer.add_string("gguf.quantization", str(quant.value))


__all__ = [
    "add_basic_metadata",
]
