"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Compatibility facade exposing Codex memory manager APIs under legacy names.
Provides a stable module-level API (devices/dtypes/load/unload/cache) backed by `CodexMemoryManager`, so older code can call into the
new manager without directly depending on its implementation details.

Symbols (top-level; keep in sync; no ghosts):
- `VRAMState` (class): Legacy VRAM mode labels (`disabled/no_vram/low_vram/normal_vram/high_vram/shared`) used by UI/compat code.
- `_bind_config` (function): Internal initializer; unloads any previous manager, creates a new `CodexMemoryManager`, and wires globals/proxies.
- `reinitialize` (function): Replaces the active memory manager with a new `RuntimeMemoryConfig`.
- `_wrap_model_sequence` (function): Normalizes `models` input to a concrete `Sequence` for downstream loader calls.
- `get_torch_device` (function): Returns the active primary torch device.
- `get_free_memory` (function): Returns free memory for a device (optionally including torch allocator-free memory).
- `minimum_inference_memory` (function): Returns the minimum memory budget required for inference.
- `memory_snapshot` (function): Returns a JSON-serializable snapshot of current memory/config state.
- `load_models_gpu` (function): Loads a sequence of models onto GPU given a required memory budget (compat wrapper around manager load).
- `load_model_gpu` (function): Loads a single model onto GPU (compat wrapper).
- `free_memory` (function): Frees memory to satisfy a requirement (supports keeping some models loaded / freeing all).
- `soft_empty_cache` (function): Best-effort cache emptying (no hard guarantee; wraps manager behavior).
- `unload_all_models` (function): Unloads all currently loaded models/components via the manager.
- `cast_to_device` (function): Casts/moves a tensor to a device/dtype with optional copy.
- `module_size` (function): Estimates module size (bytes) with optional include/exclude device filtering.
- `get_computation_dtype` (function): Chooses computation dtype for a device/model size given supported dtypes.
- `core_dtype` (function): Chooses core model dtype for the current config/device.
- `core_offload_device` (function): Returns the offload device used for core model components.
- `core_initial_load_device` (function): Returns the initial-load device for core model components given params/dtype.
- `text_encoder_device` (function): Returns the active device for text encoder components.
- `text_encoder_offload_device` (function): Returns the offload device for text encoder components.
- `text_encoder_dtype` (function): Chooses dtype for text encoder components.
- `vae_device` (function): Returns the active device for VAE components.
- `vae_offload_device` (function): Returns the offload device for VAE components.
- `vae_dtype` (function): Chooses dtype for VAE components.
- `current_precision` (function): Returns current precision for a `DeviceRole` (core/vae/tenc).
- `allow_precision_fallback` (function): Whether precision fallback is allowed for the role.
- `report_precision_failure` (function): Reports a precision failure and returns a suggested fallback dtype (or None).
- `precision_hint` (function): Human-readable hint string for the current precision decision.
- `intermediate_device` (function): Returns the device used for intermediate tensors (when different from core).
- `force_upcast_attention_dtype` (function): Whether attention should be forced to an upcast dtype for stability.
- `xformers_enabled` (function): Legacy flag: whether xformers attention is enabled.
- `xformers_enabled_vae` (function): Legacy flag: whether xformers is enabled for VAE (if applicable).
- `pytorch_attention_enabled` (function): Whether PyTorch-native attention (SDPA) is enabled.
- `should_use_fp16` (function): Legacy heuristic for selecting FP16 given device/model params and perf/stability flags.
- `should_use_bf16` (function): Legacy heuristic for selecting BF16 given device/model params and perf/stability flags.
- `dtype_size` (function): Returns element size (bytes) for a torch dtype.
- `state_dict_dtype` (function): Best-effort dtype extraction for a state dict (tensor dtype or string).
- `bake_gguf_model` (function): Converts a GGUF-backed model to a baked/dequantized variant (manager-backed helper).
- `is_device_cpu` (function): Returns whether a torch device is CPU.
- `mps_mode` (function): Returns whether MPS mode is enabled (legacy).
- `vram_state` (function): Returns the legacy VRAM state label from current config.
- `_LoadedModelsProxy` (class): Proxy view over loaded models for legacy APIs (delegates to manager loaded-model registry).
- `unload_model_clones` (function): Unloads any clones associated with a given model (legacy helper).
- `unload_model` (function): Unloads a model/component (legacy helper).
- `__getattr__` (function): Dynamic attribute proxy to the underlying manager (compat surface).
- `__setattr__` (function): Dynamic attribute setter proxy to the underlying manager (compat surface).
- `__dir__` (function): Dynamic dir() proxy for the module’s compat surface.
- `switch_primary_device` (function): Changes the primary device backend (cpu/cuda/etc) at runtime (returns success bool).
- `set_component_backend` (function): Sets component backend for a role (core/vae/tenc) at runtime (returns success bool).
- `set_component_dtype` (function): Sets component dtype for a role at runtime (returns success bool).
"""

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


def memory_snapshot() -> dict:
    """Expose a JSON-friendly view of the current runtime memory state.

    This is a thin proxy around ``CodexMemoryManager.memory_snapshot`` so that
    API layers and diagnostics tools can inspect VRAM usage without reaching
    into the manager internals directly.
    """
    return _MANAGER.memory_snapshot()


def load_models_gpu(models, memory_required: int = 0, hard_memory_preservation: int = 0):
    _MANAGER.load_models(
        _wrap_model_sequence(models),
        memory_required=memory_required,
        hard_memory_preservation=hard_memory_preservation,
    )


def load_model_gpu(model):
    try:
        target = getattr(model, "model", model)
        name = target.__class__.__name__
    except Exception:
        name = type(model).__name__
    logger.info("[memory] request load %s", name)
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
    # Return offload device (CPU) for lazy loading - models will be moved to GPU
    # by the memory manager when actually needed
    return _MANAGER.get_offload_device(DeviceRole.CORE)


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


def state_dict_dtype(state_dict) -> torch.dtype | str:
    """Best-effort dtype / quantization hint for a state dict.

    - Returns ``"gguf"`` when the mapping contains CodexParameter packed weights.
    - Otherwise returns the first encountered torch dtype (defaults to fp32).
    """

    # Avoid scanning large lazy safetensors dicts: keep the old "first tensor" behavior.
    materialize = getattr(state_dict, "materialize", None)
    if callable(materialize):
        for tensor in state_dict.values():
            if isinstance(tensor, torch.Tensor):
                return tensor.dtype
        return torch.float32

    from apps.backend.quantization.tensor import CodexParameter

    first_dtype: torch.dtype | None = None
    for idx, tensor in enumerate(state_dict.values()):
        if isinstance(tensor, CodexParameter) and tensor.qtype is not None:
            return "gguf"
        if first_dtype is None and isinstance(tensor, torch.Tensor):
            first_dtype = tensor.dtype
        # Defensive cap: most GGUF state dicts will reveal themselves quickly.
        if idx >= 4096 and first_dtype is not None:
            break
    return first_dtype or torch.float32


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
    "set_component_backend",
    "set_component_dtype",
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


def set_component_backend(role: str, backend: str) -> bool:
    """Update the preferred backend for a component role and reinitialize.

    role: one of core|text_encoder|vae|clip_vision|intermediate
    backend: one of auto|cpu|cuda|mps|xpu|directml
    """
    from copy import deepcopy
    from .config import DeviceBackend

    role_norm = str(role).strip().lower()
    mapping_role = {
        "core": DeviceRole.CORE,
        "text_encoder": DeviceRole.TEXT_ENCODER,
        "te": DeviceRole.TEXT_ENCODER,
        "vae": DeviceRole.VAE,
        "clip_vision": DeviceRole.CLIP_VISION,
        "vision": DeviceRole.CLIP_VISION,
        "intermediate": DeviceRole.INTERMEDIATE,
    }
    if role_norm not in mapping_role:
        raise ValueError(f"Invalid role '{role}'")
    role_enum = mapping_role[role_norm]

    backend_norm = (backend or "").strip().lower()
    mapping_backend = {
        "auto": DeviceBackend.AUTO,
        "cpu": DeviceBackend.CPU,
        "cuda": DeviceBackend.CUDA,
        "mps": DeviceBackend.MPS,
        "xpu": DeviceBackend.XPU,
        "directml": DeviceBackend.DIRECTML,
    }
    if backend_norm not in mapping_backend:
        raise ValueError(f"Invalid backend '{backend}'")

    current_cfg = _CONFIG
    if current_cfg is None:
        raise RuntimeError("Memory manager not initialized")

    new_cfg = deepcopy(current_cfg)
    if role_enum == DeviceRole.CORE:
        new_cfg.device_backend = mapping_backend[backend_norm]
    policy = new_cfg.component_policy(role_enum)
    policy.preferred_backend = mapping_backend[backend_norm]
    reinitialize(new_cfg)
    logger.info("[memory] backend for %s set to %s", role_enum.value, backend_norm)
    return True


def set_component_dtype(role: str, dtype: str) -> bool:
    """Force dtype for a component role. Use 'auto' to clear overrides."""
    from copy import deepcopy

    role_norm = str(role).strip().lower()
    mapping_role = {
        "core": DeviceRole.CORE,
        "text_encoder": DeviceRole.TEXT_ENCODER,
        "te": DeviceRole.TEXT_ENCODER,
        "vae": DeviceRole.VAE,
        "clip_vision": DeviceRole.CLIP_VISION,
        "vision": DeviceRole.CLIP_VISION,
        "intermediate": DeviceRole.INTERMEDIATE,
    }
    if role_norm not in mapping_role:
        raise ValueError(f"Invalid role '{role}'")
    role_enum = mapping_role[role_norm]

    dtype_norm = (dtype or "").strip().lower()
    if dtype_norm not in {"auto", "fp16", "bf16", "fp32"}:
        raise ValueError("dtype must be one of auto|fp16|bf16|fp32")

    if _CONFIG is None:
        raise RuntimeError("Memory manager not initialized")

    new_cfg = deepcopy(_CONFIG)

    # Reset precision flags for the target role
    flags = new_cfg.precision
    if role_enum == DeviceRole.CORE:
        flags.core_fp16 = flags.core_bf16 = flags.core_fp8_e4m3fn = flags.core_fp8_e5m2 = False
    elif role_enum == DeviceRole.TEXT_ENCODER:
        flags.clip_fp16 = flags.clip_fp32 = flags.clip_bf16 = flags.clip_fp8_e4m3fn = flags.clip_fp8_e5m2 = False
    elif role_enum == DeviceRole.VAE:
        flags.vae_fp16 = flags.vae_fp32 = flags.vae_bf16 = False

    policy = new_cfg.component_policy(role_enum)
    policy.forced_dtype = None

    if dtype_norm != "auto":
        torch_name = {
            "fp16": "float16",
            "bf16": "bfloat16",
            "fp32": "float32",
        }[dtype_norm]
        policy.forced_dtype = torch_name
        if role_enum == DeviceRole.CORE:
            flags.core_fp16 = dtype_norm == "fp16"
            flags.core_bf16 = dtype_norm == "bf16"
        elif role_enum == DeviceRole.TEXT_ENCODER:
            flags.clip_fp16 = dtype_norm == "fp16"
            flags.clip_bf16 = dtype_norm == "bf16"
            flags.clip_fp32 = dtype_norm == "fp32"
        elif role_enum == DeviceRole.VAE:
            flags.vae_fp16 = dtype_norm == "fp16"
            flags.vae_bf16 = dtype_norm == "bf16"
            flags.vae_fp32 = dtype_norm == "fp32"

    reinitialize(new_cfg)
    logger.info("[memory] dtype for %s set to %s", role_enum.value, dtype_norm)
    return True
