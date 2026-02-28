"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bundle-aware engine loader for backend use cases.
Resolves a `DiffusionModelBundle`, instantiates the matching engine via the registry, loads the model, and applies runtime options
(attention/accelerator) for engines backed by diffusers pipelines with explicit failures on invalid attention backend configuration.

Symbols (top-level; keep in sync; no ghosts):
- `EngineLoadOptions` (dataclass): Optional engine load overrides (device/dtype/attention backend/accelerator/VAE override).
- `_ensure_registry_ready` (function): Ensures the engine registry has the default engines registered (idempotent).
- `_instantiate_engine` (function): Creates an engine instance for a resolved diffusion bundle (family → engine key).
- `_options_to_kwargs` (function): Converts `EngineLoadOptions` into `engine.load(...)` keyword arguments.
- `_apply_runtime_options` (function): Applies runtime options (attention backend from explicit load options or runtime memory config, plus accelerator) to diffusers-backed engines.
- `load_engine` (function): Loads and initializes a diffusion engine for direct use (best-effort cleanup on failures).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from apps.backend.core.registry import create_engine
from apps.backend.engines import register_default_engines
from apps.backend.engines.util.accelerator import apply_to_diffusers_pipeline as _apply_accel
from apps.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as _apply_attn
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.models.loader import (
    DiffusionModelBundle,
    FAMILY_TO_ENGINE_KEY,
    resolve_diffusion_bundle,
)

_LOG = logging.getLogger("backend.core.engine_loader")


@dataclass
class EngineLoadOptions:
    device: Optional[str] = None  # 'cuda'|'cpu'|None → auto
    dtype: Optional[str] = None  # 'fp16'|'bf16'|'fp32'|None → default
    attention_backend: Optional[str] = None  # 'pytorch'|'xformers'|'split'|'quad'
    accelerator: Optional[str] = None  # 'tensorrt'|'none'
    vae_path: Optional[str] = None  # optional override


def _ensure_registry_ready() -> None:
    register_default_engines(replace=False)


def _instantiate_engine(bundle: DiffusionModelBundle):
    engine_key = FAMILY_TO_ENGINE_KEY.get(bundle.family)
    if engine_key is None:
        raise NotImplementedError(f"Model family {bundle.family.value} is not registered with Codex engines.")
    _ensure_registry_ready()
    return create_engine(engine_key)


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


def _apply_runtime_options(engine: Any, opts: EngineLoadOptions | None) -> Any:
    pipe = getattr(getattr(engine, "_comp", None), "pipeline", None)
    if pipe is None:
        return engine

    attention_backend = None
    if opts and opts.attention_backend is not None:
        attention_backend = opts.attention_backend
    else:
        try:
            attention_backend = str(memory_management.manager.config.attention.backend.value)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to resolve attention_backend from runtime memory config.") from exc

    if attention_backend is not None:
        _apply_attn(pipe, backend=attention_backend)

    if opts and opts.accelerator is not None:
        try:
            _apply_accel(pipe, accelerator=opts.accelerator)
        except Exception as exc:  # noqa: BLE001
            _LOG.warning("Failed to apply accelerator %s: %s", opts.accelerator, exc)

    return engine


def load_engine(name_or_path: str, options: EngineLoadOptions | None = None):
    """Load and initialize a Codex diffusion engine for direct use."""

    bundle = resolve_diffusion_bundle(name_or_path)
    engine = _instantiate_engine(bundle)

    load_kwargs = _options_to_kwargs(options)
    load_kwargs["_bundle"] = bundle

    try:
        engine.load(name_or_path, **load_kwargs)
    except Exception:
        with contextlib.suppress(Exception):
            engine.unload()
        raise

    try:
        return _apply_runtime_options(engine, options)
    except Exception:
        with contextlib.suppress(Exception):
            engine.unload()
        raise


__all__ = ["EngineLoadOptions", "load_engine"]
