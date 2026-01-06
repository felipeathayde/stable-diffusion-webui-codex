"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed configuration models for runtime memory management (devices, swap, attention, and budgets).

Symbols (top-level; keep in sync; no ghosts):
- `DeviceBackend` (enum): Execution backend selection (auto/cuda/cpu/mps/xpu/directml).
- `DeviceRole` (enum): Logical component roles managed by the memory manager.
- `SwapPolicy` (enum): Where models are offloaded when VRAM is constrained.
- `SwapMethod` (enum): How tensors are copied during swap operations (blocked vs async).
- `AttentionBackend` (enum): Preferred attention implementation.
- `PrecisionFlags` (dataclass): Fine-grained dtype override flags for core/text/VAE/CLIP components.
- `ComponentPolicy` (dataclass): Device/dtype expectations for a component role.
- `SwapConfig` (dataclass): Swap strategy controlling offload behavior.
- `AttentionConfig` (dataclass): Attention backend preferences and fallback flags.
- `MemoryBudgets` (dataclass): Budget constraints for load/offload heuristics.
- `_default_component_policies` (function): Internal default mapping for `RuntimeMemoryConfig.components`.
- `RuntimeMemoryConfig` (dataclass): High-level configuration consumed by the memory manager.
- `HardwareProbe` (dataclass): Snapshot of detected hardware capabilities (VRAM/RAM/dtype/backend support).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Mapping, MutableMapping, Optional


class DeviceBackend(str, Enum):
    """Execution backend used for model components."""

    AUTO = "auto"
    CUDA = "cuda"
    CPU = "cpu"
    MPS = "mps"
    XPU = "xpu"
    DIRECTML = "directml"


class DeviceRole(str, Enum):
    """Logical roles covered by the memory manager."""

    CORE = "core"
    TEXT_ENCODER = "text_encoder"
    VAE = "vae"
    INTERMEDIATE = "intermediate"
    CLIP_VISION = "clip_vision"


class SwapPolicy(str, Enum):
    """Where models are offloaded when VRAM is constrained."""

    NEVER = "never"
    CPU = "cpu"
    SHARED = "shared"


class SwapMethod(str, Enum):
    """Mechanism used to copy tensors during swap operations."""

    BLOCKED = "blocked"
    ASYNC = "async"


class AttentionBackend(str, Enum):
    """Preferred attention implementation."""

    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    SPLIT = "split"
    QUAD = "quad"


@dataclass(slots=True)
class PrecisionFlags:
    """Fine-grained overrides for dtype preferences."""

    all_fp16: bool = False
    core_fp16: bool = False
    core_bf16: bool = False
    core_fp8_e4m3fn: bool = False
    core_fp8_e5m2: bool = False
    vae_fp16: bool = False
    vae_fp32: bool = False
    vae_bf16: bool = False
    vae_in_cpu: bool = False
    clip_fp16: bool = False
    clip_fp32: bool = False
    clip_bf16: bool = False
    clip_fp8_e4m3fn: bool = False
    clip_fp8_e5m2: bool = False


@dataclass(slots=True)
class ComponentPolicy:
    """Device/dtype expectations for a specific component role."""

    preferred_backend: DeviceBackend = DeviceBackend.AUTO
    forced_dtype: Optional[str] = None
    allow_offload: bool = True
    allow_manual_cast: bool = True


@dataclass(slots=True)
class SwapConfig:
    """Swap strategy controlling offload behaviour."""

    policy: SwapPolicy = SwapPolicy.CPU
    method: SwapMethod = SwapMethod.BLOCKED
    always_offload: bool = False
    pin_shared_memory: bool = False


@dataclass(slots=True)
class AttentionConfig:
    """Configures attention backend preferences."""

    backend: AttentionBackend = AttentionBackend.PYTORCH
    enable_flash: bool = True
    enable_mem_efficient: bool = True
    force_upcast: bool = False
    allow_split_fallback: bool = False
    allow_quad_fallback: bool = False


@dataclass(slots=True)
class MemoryBudgets:
    """Budget constraints used for model loading/offload heuristics."""

    minimum_inference_mb: int = 0
    hard_reservation_mb: int = 0
    safety_margin_mb: int = 0


def _default_component_policies() -> Dict[DeviceRole, ComponentPolicy]:
    return {
        DeviceRole.CORE: ComponentPolicy(),
        DeviceRole.TEXT_ENCODER: ComponentPolicy(),
        DeviceRole.VAE: ComponentPolicy(),
        DeviceRole.INTERMEDIATE: ComponentPolicy(),
        DeviceRole.CLIP_VISION: ComponentPolicy(),
    }


@dataclass(slots=True)
class RuntimeMemoryConfig:
    """High-level configuration consumed by the memory manager."""

    device_backend: DeviceBackend = DeviceBackend.AUTO
    gpu_device_id: Optional[int] = None
    gpu_prefer_construct: bool = False
    precision: PrecisionFlags = field(default_factory=PrecisionFlags)
    swap: SwapConfig = field(default_factory=SwapConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    components: MutableMapping[DeviceRole, ComponentPolicy] = field(
        default_factory=_default_component_policies
    )
    budgets: MemoryBudgets = field(default_factory=MemoryBudgets)
    deterministic_algorithms: bool = False
    disable_xformers: bool = False
    enable_xformers_vae: bool = True
    allow_directml: bool = False

    def component_policy(self, role: DeviceRole) -> ComponentPolicy:
        """Return the policy for the requested component role."""
        if role not in self.components:
            self.components[role] = ComponentPolicy()
        return self.components[role]


@dataclass(slots=True)
class HardwareProbe:
    """Snapshot of detected hardware capabilities."""

    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: Optional[str] = None
    cuda_cc_major: Optional[int] = None
    cuda_cc_minor: Optional[int] = None
    mps_available: bool = False
    xpu_available: bool = False
    directml_available: bool = False
    total_vram_mb: Optional[int] = None
    total_ram_mb: Optional[int] = None
    bf16_support: bool = False
    native_bf16: bool = False
    fp8_support: bool = False
    xformers_available: bool = False
    xformers_version: Optional[str] = None

    def to_dict(self) -> Mapping[str, object]:
        """Return a JSON-friendly representation."""
        return {
            "cuda_available": self.cuda_available,
            "cuda_device_count": self.cuda_device_count,
            "cuda_device_name": self.cuda_device_name,
            "cuda_cc_major": self.cuda_cc_major,
            "cuda_cc_minor": self.cuda_cc_minor,
            "mps_available": self.mps_available,
            "xpu_available": self.xpu_available,
            "directml_available": self.directml_available,
            "total_vram_mb": self.total_vram_mb,
            "total_ram_mb": self.total_ram_mb,
            "bf16_support": self.bf16_support,
            "native_bf16": self.native_bf16,
            "fp8_support": self.fp8_support,
            "xformers_available": self.xformers_available,
            "xformers_version": self.xformers_version,
        }


__all__ = [
    "AttentionBackend",
    "AttentionConfig",
    "ComponentPolicy",
    "DeviceBackend",
    "DeviceRole",
    "HardwareProbe",
    "MemoryBudgets",
    "PrecisionFlags",
    "RuntimeMemoryConfig",
    "SwapConfig",
    "SwapMethod",
    "SwapPolicy",
]
