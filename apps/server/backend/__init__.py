"""Apps backend façade exposing orchestrator interfaces and engines."""

from .core.engine_interface import BaseInferenceEngine, EngineCapabilities, TaskType
from .core.orchestrator import InferenceOrchestrator
from .core.registry import EngineRegistry, EngineDescriptor, registry
from .core.requests import (
    Img2ImgRequest,
    Img2VidRequest,
    InferenceEvent,
    ProgressEvent,
    ResultEvent,
    Txt2ImgRequest,
    Txt2VidRequest,
)
# Avoid importing heavy runtime modules at package import time to prevent
# circular imports (e.g., runtime.utils -> backend.gguf -> backend.runtime).
# These are exposed lazily via __getattr__ below.
from .huggingface import ensure_repo_minimal_files
from .patchers import (
    CLIP,
    ControlLora,
    ControlNet,
    LoraLoader,
    ModelPatcher,
    T2IAdapter,
    UnetPatcher,
    VAE,
    apply_controlnet_advanced,
    clip_preprocess,
    extra_weight_calculators,
    load_lora,
    load_t2i_adapter,
    lora_collection_priority,
    merge_lora_to_weight,
    model_lora_keys_clip,
    model_lora_keys_unet,
    set_model_options_patch_replace,
    set_model_options_post_cfg_function,
    set_model_options_pre_cfg_function,
)
from .runtime.text_processing import (
    ClassicTextProcessingEngine,
    EmbeddingDatabase,
    T5TextProcessingEngine,
    embedding_from_b64,
    embedding_to_b64,
    emphasis as text_emphasis,
    parsing as text_parsing,
    textual_inversion,
)
from .services import (
    ImageService,
    MediaService,
    OptionsService,
    ProgressService,
    SamplerService,
)
from .engines import (
    EngineExecutionError,
    EngineLoadError,
    WanI2V14BEngine,
    WanT2V14BEngine,
    register_default_engines,
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
    "ensure_repo_minimal_files",
    "attention",
    "logging",
    "memory_management",
    "models",
    "nn",
    "ops",
    "shared",
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
    "lora_collection_priority",
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
    "stream",
    "SamplerService",
    "TaskType",
    "Txt2ImgRequest",
    "Txt2VidRequest",
    "WanI2V14BEngine",
    "WanT2V14BEngine",
    "register_default_engines",
    "registry",
]


def __getattr__(name: str):  # pragma: no cover - runtime dispatch
    # Lazy-export runtime helpers to avoid circular imports during package init.
    if name in {
        "attention",
        "logging",
        "memory_management",
        "models",
        "nn",
        "ops",
        "shared",
        "stream",
        "text_processing",
        "utils",
    }:
        from . import runtime as _runtime
        return getattr(_runtime, name)
    raise AttributeError(name)
