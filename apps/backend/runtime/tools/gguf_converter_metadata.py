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
- `add_basic_metadata` (function): Adds standard provenance/license metadata keys into the output GGUF.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import re
from pathlib import Path

from apps.backend.quantization.gguf import GGMLQuantizationType, GGUFWriter, LlamaFileType
from apps.backend.infra.config.provenance import CODEX_GENERATED_BY, CODEX_REPO_URL, best_effort_git_commit
from apps.backend.runtime.tools import gguf_converter_quantization as _quantization
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType


def _hash_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


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
    name = str(config.get("_name_or_path") or config.get("name") or "model")
    writer.add_name(name)
    writer.add_quantized_by(CODEX_GENERATED_BY)
    writer.add_repo_url(CODEX_REPO_URL)

    repo_root = Path(__file__).resolve().parents[4]
    commit = best_effort_git_commit(repo_root)
    if commit:
        writer.add_version(commit)

    # NOTE: Keep Codex provenance lightweight and prefer stable GGUF keys.
    # - Repo URL/commit are available via `general.repo_url` and `general.version`.
    # - Additional conversion details live under `codex.*` keys below.
    writer.add_string("codex.converted_at_utc", _dt.datetime.now(tz=_dt.timezone.utc).isoformat())
    writer.add_string("codex.quantization", str(quant.value))

    if config_path.is_file():
        writer.add_string("codex.source_config", config_path.name)
        try:
            writer.add_uint64("codex.source_config_bytes", int(config_path.stat().st_size))
        except Exception:
            pass
        try:
            writer.add_string("codex.source_config_sha256", _hash_file(config_path))
        except Exception:
            pass

    source_ref = str(safetensors_path or "").strip()
    if source_ref:
        sp = Path(source_ref).expanduser()
        writer.add_string("codex.source_weights", sp.name)
        kind = "unknown"
        if sp.is_dir():
            kind = "dir"
        elif sp.name.endswith(".safetensors.index.json"):
            kind = "index"
        elif sp.suffix.lower() == ".safetensors":
            kind = "safetensors"
        writer.add_string("codex.source_weights_kind", kind)
        if sp.is_file():
            try:
                writer.add_uint64("codex.source_weights_bytes", int(sp.stat().st_size))
            except Exception:
                pass

    # Best-effort: mark the upstream repo when the config includes a Hugging Face id.
    upstream = str(config.get("_name_or_path") or "").strip()
    if _is_hf_repo_id(upstream):
        writer.add_source_url(f"https://huggingface.co/{upstream}")

    requested = _quantization.requested_ggml_type(quant)
    if requested == GGMLQuantizationType.F16:
        writer.add_file_type(int(LlamaFileType.MOSTLY_F16))
    elif requested == GGMLQuantizationType.F32:
        writer.add_file_type(int(LlamaFileType.ALL_F32))
    elif requested == GGMLQuantizationType.Q8_0:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q8_0))
    elif requested == GGMLQuantizationType.Q6_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q6_K))
    elif requested == GGMLQuantizationType.Q5_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q5_K_M))
    elif requested == GGMLQuantizationType.Q5_1:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q5_1))
    elif requested == GGMLQuantizationType.Q5_0:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q5_0))
    elif requested == GGMLQuantizationType.Q4_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q4_K_M))
    elif requested == GGMLQuantizationType.Q4_1:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q4_1))
    elif requested == GGMLQuantizationType.Q4_0:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q4_0))
    elif requested == GGMLQuantizationType.Q3_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q3_K_M))
    elif requested == GGMLQuantizationType.Q2_K:
        writer.add_file_type(int(LlamaFileType.MOSTLY_Q2_K))
    elif requested == GGMLQuantizationType.IQ4_NL:
        writer.add_file_type(int(LlamaFileType.MOSTLY_IQ4_NL))

    # Minimal arch metadata; loaders in this repo generally key off tensor names and shapes.
    writer.add_uint32(f"{arch}.context_length", int(config.get("max_position_embeddings", 4096)))
    writer.add_uint32(f"{arch}.embedding_length", int(config.get("hidden_size", 4096)))
    writer.add_uint32(f"{arch}.block_count", int(config.get("num_hidden_layers", 32)))
    writer.add_uint32(f"{arch}.attention.head_count", int(config.get("num_attention_heads", 32)))
    writer.add_uint32(f"{arch}.attention.head_count_kv", int(config.get("num_key_value_heads", 8)))
    writer.add_float32(f"{arch}.rope.freq_base", float(config.get("rope_theta", 10000.0)))
    writer.add_float32(f"{arch}.attention.layer_norm_rms_epsilon", float(config.get("rms_norm_eps", 1e-6)))


__all__ = [
    "add_basic_metadata",
]
