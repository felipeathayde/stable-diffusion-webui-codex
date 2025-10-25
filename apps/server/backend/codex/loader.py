from __future__ import annotations

import os
from typing import Optional

from dataclasses import dataclass
from typing import Optional

from apps.server.backend.registry.checkpoints import list_checkpoints
from apps.server.backend.runtime.models.loader import forge_loader, load_engine_from_diffusers
from apps.server.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as _apply_attn
from apps.server.backend.engines.util.accelerator import apply_to_diffusers_pipeline as _apply_accel


def _find_checkpoint_by_name(name: str):
    for entry in list_checkpoints():
        if entry.name == name or entry.title == name:
            return entry
    return None


@dataclass
class EngineLoadOptions:
    device: Optional[str] = None   # 'cuda'|'cpu'|None → auto
    dtype: Optional[str] = None    # 'fp16'|'bf16'|'fp32'|None → default
    attention_backend: Optional[str] = None  # 'torch-sdpa'|'xformers'|'sage'
    accelerator: Optional[str] = None       # 'tensorrt'|'none'
    vae_path: Optional[str] = None          # optional override


def _apply_runtime_options(engine, opts: EngineLoadOptions | None):
    if not opts:
        return engine
    # Apply attention/accelerator to diffusers pipelines when present
    pipe = getattr(getattr(engine, "_comp", None), "pipeline", None)
    if pipe is not None:
        try:
            _apply_attn(pipe, backend=opts.attention_backend)
        except Exception:
            pass
        try:
            _apply_accel(pipe, accelerator=opts.accelerator)
        except Exception:
            pass
    return engine


def load_engine(name_or_path: str, options: EngineLoadOptions | None = None):
    """Load an engine instance by checkpoint name or path.

    - If a diffusers repo directory is detected (has model_index.json), load via native diffusers loader.
    - If a file path to a state dict is given, use forge_loader.
    - Otherwise, resolve by registry name and load accordingly.
    """
    path = name_or_path
    if os.path.isdir(path):
        mi = os.path.join(path, "model_index.json")
        if os.path.isfile(mi):
            return _apply_runtime_options(load_engine_from_diffusers(path), options)
        raise ValueError(f"Not a diffusers repo (missing model_index.json): {path}")
    if os.path.isfile(path):
        return _apply_runtime_options(forge_loader(path), options)

    entry = _find_checkpoint_by_name(name_or_path)
    if entry is None:
        raise ValueError(f"Checkpoint not found: {name_or_path}")
    if entry.metadata.get("format") == "diffusers" or os.path.isfile(os.path.join(entry.path, "model_index.json")):
        return _apply_runtime_options(load_engine_from_diffusers(entry.path), options)
    return _apply_runtime_options(forge_loader(entry.filename), options)


__all__ = ["load_engine", "EngineLoadOptions"]
