"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend package facade with lazy exports.
Re-exports core engine/request types and lazily resolves engines/runtime/patchers/services on first access to avoid importing heavy dependencies during startup.

Symbols (top-level; keep in sync; no ghosts):
- `TaskType` (enum): Canonical backend task types used across requests and orchestrator (re-export).
- `EngineRegistry` (class): Engine registry type (re-export).
- `registry` (constant): Global engine registry instance (re-export).
- `_LAZY` (constant): Lazy export name groups that drive `__getattr__` dispatch.
- `__getattr__` (function): Lazy import hook for heavy backend surfaces and special-cased exports.
- `__all__` (constant): Explicit export list for the backend facade (see list for the full public surface).
"""
# tags: backend, exports, lazy-imports

from .core.engine_interface import BaseInferenceEngine, EngineCapabilities, TaskType
from .core.exceptions import EngineExecutionError, EngineLoadError
from .core.registry import EngineDescriptor, EngineRegistry, registry
from .core.requests import (
    Img2ImgRequest,
    Img2VidRequest,
    InferenceEvent,
    ProgressEvent,
    ResultEvent,
    Txt2ImgRequest,
    Txt2VidRequest,
)

__all__ = [
    "BaseInferenceEngine",
    "EngineCapabilities",
    "EngineDescriptor",
    "EngineExecutionError",
    "EngineLoadError",
    "EngineRegistry",
    "Img2ImgRequest",
    "Img2VidRequest",
    "InferenceEvent",
    "InferenceOrchestrator",
    "ImageService",
    "attention",
    "logging",
    "memory_management",
    "models",
    "nn",
    "ops",
    "stream",
    "text_processing",
    "utils",
    "CLIP",
    "ControlLora",
    "ControlNet",
    "LoraLoader",
    "ModelPatcher",
    "T2IAdapter",
    "UnetPatcher",
    "VAE",
    "apply_controlnet_advanced",
    "clip_preprocess",
    "extra_weight_calculators",
    "load_lora",
    "load_t2i_adapter",
    "merge_lora_to_weight",
    "model_lora_keys_clip",
    "model_lora_keys_unet",
    "set_model_options_patch_replace",
    "set_model_options_post_cfg_function",
    "set_model_options_pre_cfg_function",
    "ClassicTextProcessingEngine",
    "EmbeddingDatabase",
    "T5TextProcessingEngine",
    "embedding_from_b64",
    "embedding_to_b64",
    "text_emphasis",
    "text_parsing",
    "textual_inversion",
    "MediaService",
    "OptionsService",
    "ProgressEvent",
    "ProgressService",
    "ResultEvent",
    "SamplerService",
    "TaskType",
    "Txt2ImgRequest",
    "Txt2VidRequest",
    "Wan2214BEngine",
    "Wan225BEngine",
    "register_default_engines",
    "registry",
]

# Lazy exports to avoid pulling heavy dependencies (WAN engines, torch, HF) during
# package import. Engines/text-processing objects are resolved on first access.
from apps.backend.types.exports import LAZY_EXPORTS as _LAZY


def __getattr__(name: str):  # pragma: no cover - runtime dispatch
    # Engines and registration (WAN heavy deps) are loaded on demand.
    if name in _LAZY.ENGINES:
        from . import engines as _engines
        value = getattr(_engines, name)
        globals()[name] = value
        return value

    # Text processing exports are torch-bound; keep lazy to avoid import errors
    # in environments without torch until explicitly needed.
    if name in _LAZY.TEXT_PROCESSING:
        from .runtime import text_processing as _tp
        value = getattr(_tp, name)
        globals()[name] = value
        return value

    # Lazy-export runtime helpers to avoid circular imports during package init.
    if name in _LAZY.RUNTIME:
        from . import runtime as _runtime
        value = getattr(_runtime, name)
        globals()[name] = value
        return value

    # Patchers
    if name in _LAZY.PATCHERS:
        from . import patchers as _patchers
        value = getattr(_patchers, name)
        globals()[name] = value
        return value

    # Services
    if name in _LAZY.SERVICES:
        from . import services as _services
        value = getattr(_services, name)
        globals()[name] = value
        return value

    if name == "InferenceOrchestrator":
        from .core.orchestrator import InferenceOrchestrator as _InferenceOrchestrator
        globals()[name] = _InferenceOrchestrator
        return _InferenceOrchestrator

    if name == "ensure_repo_minimal_files":
        from .huggingface import ensure_repo_minimal_files as _ermf
        globals()[name] = _ermf
        return _ermf
    raise AttributeError(name)
