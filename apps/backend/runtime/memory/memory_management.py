"""Compatibility facade exposing Codex memory manager APIs."""

from __future__ import annotations

import contextlib
import logging
from types import SimpleNamespace
from typing import Iterable, Sequence

import torch

from apps.backend.infra.config import args as legacy_args
from .config import DeviceRole, RuntimeMemoryConfig
from .exceptions import MemoryLoadError
from .manager import CodexMemoryManager


logger = logging.getLogger("backend.memory.facade")


class VRAMState(SimpleNamespace):
    DISABLED = "disabled"
    NO_VRAM = "no_vram"
    LOW_VRAM = "low_vram"
    NORMAL_VRAM = "normal_vram"
    HIGH_VRAM = "high_vram"
    SHARED = "shared"


_CONFIG: RuntimeMemoryConfig | None = None
_MANAGER: CodexMemoryManager | None = None


def _bind_config(config: RuntimeMemoryConfig) -> None:
    global _CONFIG, _MANAGER, memory_config, OOM_EXCEPTION, cpu
    old_manager = _MANAGER
    if old_manager is not None:
        with contextlib.suppress(Exception):
            old_manager.unload_all_models()
    _CONFIG = config
    _MANAGER = CodexMemoryManager.create(config)
    memory_config = config
    OOM_EXCEPTION = _MANAGER.oom_exception
    cpu = _MANAGER.cpu_device


def reinitialize(config: RuntimeMemoryConfig) -> None:
    """Replace the active memory manager with a new configuration."""
    logger.info(
        "Reinitializing memory manager (device_backend=%s, gpu_prefer_construct=%s)",
        getattr(config.device_backend, "value", config.device_backend),
        getattr(config, "gpu_prefer_construct", False),
    )
    _bind_config(config)


_bind_config(legacy_args.memory_config)


def _wrap_model_sequence(models: Sequence[object] | Iterable[object]) -> Sequence[object]:
    return list(models)


def get_torch_device() -> torch.device:
    return _MANAGER.get_device(DeviceRole.CORE)


def get_free_memory(dev: torch.device | None = None, torch_free_too: bool = False):
    return _MANAGER.get_free_memory(dev, return_torch_stats=torch_free_too)


def minimum_inference_memory() -> int:
    return _MANAGER.minimum_inference_memory()


def load_models_gpu(models, memory_required: int = 0, hard_memory_preservation: int = 0):
    _MANAGER.load_models(
        _wrap_model_sequence(models),
        memory_required=memory_required,
        hard_memory_preservation=hard_memory_preservation,
    )


def load_model_gpu(model):
    _MANAGER.load_model(model)


def free_memory(memory_required: int, device=None, keep_loaded=(), free_all: bool = False):
    _MANAGER.free_memory(memory_required, device=device, keep_loaded=keep_loaded, free_all=free_all)


def soft_empty_cache(force: bool = False):
    _MANAGER.soft_empty_cache(force=force)


def unload_all_models():
    _MANAGER.unload_all_models()


def cast_to_device(tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None, copy: bool = False) -> torch.Tensor:
    return _MANAGER.cast_to_device(tensor, device, dtype, copy=copy)


def module_size(module, exclude_device=None, include_device=None, return_split: bool = False):
    return _MANAGER.module_size(
        module,
        exclude_device=exclude_device,
        include_device=include_device,
        return_split=return_split,
    )


def get_computation_dtype(inference_device, parameters: int = 0, supported_dtypes=()):
    supported = supported_dtypes if supported_dtypes else (torch.float16, torch.bfloat16, torch.float32)
    return _MANAGER.dtype_for_role(DeviceRole.CORE, supported=supported)


def core_dtype(device=None, model_params: int = 0, supported_dtypes=()):
    return get_computation_dtype(device, model_params, supported_dtypes)


def core_offload_device():
    return _MANAGER.get_offload_device(DeviceRole.CORE)


def core_initial_load_device(parameters, dtype):
    return _MANAGER.get_device(DeviceRole.CORE)


def text_encoder_device():
    return _MANAGER.get_device(DeviceRole.TEXT_ENCODER)


def text_encoder_offload_device():
    return _MANAGER.get_offload_device(DeviceRole.TEXT_ENCODER)


def text_encoder_dtype(device=None, supported_dtypes=()):
    supported = supported_dtypes or (torch.float16, torch.bfloat16, torch.float32)
    return _MANAGER.dtype_for_role(DeviceRole.TEXT_ENCODER, supported=supported)


def vae_device():
    return _MANAGER.get_device(DeviceRole.VAE)


def vae_offload_device():
    return _MANAGER.get_offload_device(DeviceRole.VAE)


def vae_dtype(device=None, allowed_dtypes=()):
    supported = allowed_dtypes or (torch.float16, torch.bfloat16, torch.float32)
    return _MANAGER.dtype_for_role(DeviceRole.VAE, supported=supported)


def current_precision(role: DeviceRole) -> torch.dtype | None:
    return _MANAGER.current_precision(role)


def allow_precision_fallback(role: DeviceRole) -> bool:
    return _MANAGER.allow_precision_fallback(role)


def report_precision_failure(role: DeviceRole, *, location: str, reason: str) -> torch.dtype | None:
    return _MANAGER.report_precision_failure(role, location=location, reason=reason)


def precision_hint(role: DeviceRole) -> str:
    return _MANAGER.precision_hint(role)


def intermediate_device():
    return _MANAGER.get_device(DeviceRole.INTERMEDIATE)


def force_upcast_attention_dtype():
    return _MANAGER.force_upcast_attention_dtype()


def xformers_enabled():
    return _MANAGER.xformers_enabled()


def xformers_enabled_vae():
    return _MANAGER.xformers_enabled_vae()


def pytorch_attention_enabled():
    return _MANAGER.pytorch_attention_enabled()


def should_use_fp16(device=None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False):
    return _MANAGER.should_use_fp16(
        device=device,
        model_params=model_params,
        prioritize_performance=prioritize_performance,
        manual_cast=manual_cast,
    )


def should_use_bf16(device=None, model_params: int = 0, prioritize_performance: bool = True, manual_cast: bool = False):
    return _MANAGER.should_use_bf16(
        device=device,
        model_params=model_params,
        prioritize_performance=prioritize_performance,
        manual_cast=manual_cast,
    )


def dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def state_dict_dtype(state_dict) -> torch.dtype:
    for tensor in state_dict.values():
        if isinstance(tensor, torch.Tensor):
            return tensor.dtype
    return torch.float32


def bake_gguf_model(model):
    if not hasattr(model, "bake"):
        raise MemoryLoadError("Model does not support bake() required for GGUF.")
    return model.bake()


def is_device_cpu(device: torch.device):
    return device.type == "cpu"


def mps_mode():
    return get_torch_device().type == "mps"

def vram_state():
    device = get_torch_device()
    if device.type == "cpu":
        return VRAMState.DISABLED
    return VRAMState.NORMAL_VRAM


class _LoadedModelsProxy:
    __slots__ = ("_manager",)

    def __init__(self, manager: CodexMemoryManager) -> None:
        self._manager = manager

    def __iter__(self):
        return iter(self._manager.loaded_models())

    def __len__(self) -> int:
        return len(self._manager.loaded_models())

    def __getitem__(self, index):
        return self._manager.loaded_models()[index]

    def __contains__(self, item) -> bool:
        return item in self._manager.loaded_models()

    def __repr__(self) -> str:
        return f"LoadedModelsProxy(count={len(self)})"


current_loaded_models = _LoadedModelsProxy(_MANAGER)


def unload_model_clones(model):
    _MANAGER.unload_model_clones(model)


def unload_model(model):
    _MANAGER.unload_model(model)


__all__ = [
    "OOM_EXCEPTION",
    "VRAMState",
    "bake_gguf_model",
    "cast_to_device",
    "current_precision",
    "allow_precision_fallback",
    "report_precision_failure",
    "precision_hint",
    "core_dtype",
    "core_initial_load_device",
    "core_offload_device",
    "cpu",
    "current_loaded_models",
    "dtype_size",
    "force_upcast_attention_dtype",
    "free_memory",
    "get_computation_dtype",
    "get_free_memory",
    "get_torch_device",
    "intermediate_device",
    "is_device_cpu",
    "load_model_gpu",
    "load_models_gpu",
    "minimum_inference_memory",
    "module_size",
    "mps_mode",
    "pytorch_attention_enabled",
    "should_use_bf16",
    "should_use_fp16",
    "signal_empty_cache",
    "soft_empty_cache",
    "state_dict_dtype",
    "text_encoder_device",
    "text_encoder_dtype",
    "text_encoder_offload_device",
    "unload_all_models",
    "unload_model",
    "unload_model_clones",
    "vae_device",
    "vae_dtype",
    "vae_offload_device",
    "vram_state",
    "xformers_enabled",
    "xformers_enabled_vae",
    "args",
    "memory_config",
    "switch_primary_device",
]


def __getattr__(name: str):
    if name == "memory_config":
        return memory_config
    if name == "signal_empty_cache":
        return _MANAGER.signal_empty_cache
    if name == "VAE_ALWAYS_TILED":
        return _MANAGER.vae_always_tiled
    if name == "args":
        return legacy_args.args
    raise AttributeError(f"module {__name__} has no attribute {name!r}")


def __setattr__(name: str, value):
    if name == "signal_empty_cache":
        _MANAGER.signal_empty_cache = bool(value)
        return
    if name == "VAE_ALWAYS_TILED":
        _MANAGER.vae_always_tiled = bool(value)
        return
    globals()[name] = value


def __dir__():
    return sorted(set(globals().keys()) | {"signal_empty_cache", "VAE_ALWAYS_TILED", "memory_config"})


def switch_primary_device(backend: str) -> bool:
    """Switch the primary device backend explicitly.

    Accepts one of: 'cpu' | 'cuda' | 'mps' | 'xpu' | 'directml'.
    Returns True if the configuration changed and a reinitialization occurred.
    """
    from copy import deepcopy
    from .config import DeviceBackend

    normalized = (backend or "").strip().lower()
    mapping = {
        "cpu": DeviceBackend.CPU,
        "cuda": DeviceBackend.CUDA,
        "mps": DeviceBackend.MPS,
        "xpu": DeviceBackend.XPU,
        "directml": DeviceBackend.DIRECTML,
    }
    if normalized not in mapping:
        raise ValueError(f"Invalid device backend '{backend}'. Allowed: cpu, cuda, mps, xpu, directml")

    current = _CONFIG.device_backend if _CONFIG is not None else None
    target = mapping[normalized]
    if current == target:
        return False
    new_cfg = deepcopy(_CONFIG)
    new_cfg.device_backend = target
    reinitialize(new_cfg)
    logger.info("[memory] primary device switched to %s", normalized)
    return True
