from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

from apps.backend.core.registry import create_engine
from apps.backend.engines import register_default_engines
from apps.backend.engines.util.accelerator import apply_to_diffusers_pipeline as _apply_accel
from apps.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as _apply_attn
from apps.backend.runtime.models.loader import (
    DiffusionModelBundle,
    FAMILY_TO_ENGINE_KEY,
    resolve_diffusion_bundle,
)


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


def _options_to_kwargs(opts: EngineLoadOptions | None) -> Dict[str, Any]:
    if opts is None:
        return {}
    payload: Dict[str, Any] = {}
    if opts.device is not None:
        payload["device"] = str(opts.device)
    if opts.dtype is not None:
        payload["dtype"] = str(opts.dtype)
    if opts.vae_path is not None:
        payload["vae_path"] = str(opts.vae_path)
    if opts.attention_backend is not None:
        payload["attention_backend"] = str(opts.attention_backend)
    if opts.accelerator is not None:
        payload["accelerator"] = str(opts.accelerator)
    return payload


def _ensure_registry_ready() -> None:
    register_default_engines(replace=False)


def _instantiate_engine(bundle: DiffusionModelBundle):
    engine_key = FAMILY_TO_ENGINE_KEY.get(bundle.family)
    if engine_key is None:
        raise NotImplementedError(f"Model family {bundle.family.value} is not registered with Codex engines.")
    _ensure_registry_ready()
    return create_engine(engine_key)


def load_engine(name_or_path: str, options: EngineLoadOptions | None = None):
    """Load and initialize a Codex diffusion engine for direct use."""

    bundle = resolve_diffusion_bundle(name_or_path)
    engine = _instantiate_engine(bundle)

    load_kwargs = _options_to_kwargs(options)
    load_kwargs["_bundle"] = bundle

    try:
        engine.load(name_or_path, **load_kwargs)
    except Exception:
        # Ensure partially-initialised engines don't linger loaded.
        with contextlib.suppress(Exception):
            engine.unload()
        raise

    return _apply_runtime_options(engine, options)


__all__ = ["load_engine", "EngineLoadOptions"]
