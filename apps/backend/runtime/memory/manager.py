"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Codex-native runtime memory management service (hardware probe + precision/budget policies + loaded model registry).
Provides a single manager that decides device/precision defaults, tracks loaded components, and applies swap/offload policies during
engine orchestration.

Symbols (top-level; keep in sync; no ghosts):
- `_PrecisionState` (dataclass): Internal precision selection state (derived from hardware + configured flags) used to choose dtypes.
- `_detect_oom_exception` (function): Detects the appropriate OOM exception class for the active backend/runtime.
- `_normalize_device_name` (function): Normalizes device name strings for stable matching and policy decisions.
- `_device_has_native_bf16` (function): Heuristic for whether a device likely supports native BF16 (name + compute capability).
- `_probe_hardware` (function): Performs hardware probing and returns a `HardwareProbe` (raises `HardwareProbeError` on failure).
- `_LoadedModelRecord` (dataclass): Tracks one loaded model/component (name/path/device/dtype) for introspection and unload decisions.
- `CodexMemoryManager` (class): Main memory manager; owns runtime config, budget calculation, model registry, and policy decisions
  (contains many methods for load/unload bookkeeping, swap/offload behavior, and “best defaults” selection).
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, MutableSequence, Optional, Sequence, Set, Tuple

import torch

from .config import (
    AttentionBackend,
    AttentionConfig,
    ComponentPolicy,
    DeviceBackend,
    DeviceRole,
    HardwareProbe,
    MemoryBudgets,
    PrecisionFlags,
    RuntimeMemoryConfig,
    SwapConfig,
    SwapMethod,
    SwapPolicy,
)
from .exceptions import HardwareProbeError, MemoryConfigurationError, MemoryLoadError
from .smart_offload import smart_offload_enabled


logger = logging.getLogger("backend.memory.manager")


_NATIVE_BF16_NAME_FRAGMENTS: Tuple[str, ...] = (
    "geforce rtx 30",
    "geforce rtx 40",
    "geforce rtx 50",
    "nvidia a100",
    "nvidia a10",
    "nvidia a30",
    "nvidia a40",
    "nvidia l4",
    "nvidia l40",
    "nvidia l40s",
    "nvidia h100",
    "nvidia h200",
    "amd instinct mi100",
    "amd instinct mi200",
    "amd instinct mi250",
    "amd instinct mi250x",
    "amd instinct mi300",
    "amd instinct mi300a",
    "amd instinct mi300x",
    "intel arc a",
    "intel data center gpu max",
)

_PRECISION_HINTS: Dict[DeviceRole, str] = {
    DeviceRole.CORE: "--core-dtype fp32",
    DeviceRole.TEXT_ENCODER: "--te-dtype fp32",
    DeviceRole.VAE: "--vae-dtype fp32",
}


@dataclass(slots=True)
class _PrecisionState:
    role: DeviceRole
    ladder: Tuple[torch.dtype, ...]
    manual_override: bool = False
    index: int = 0

    def current(self) -> torch.dtype:
        return self.ladder[min(self.index, len(self.ladder) - 1)]

    def select(self, supported: Sequence[torch.dtype]) -> torch.dtype:
        if not supported:
            raise MemoryConfigurationError(f"No supported dtypes provided for {self.role.value}.")
        if self.manual_override:
            dtype = self.current()
            if dtype not in supported:
                return supported[-1]
            return dtype
        # Prefer staying on current index if valid, otherwise advance within ladder order
        cur = self.current()
        if cur in supported:
            return cur
        # search forward from current index
        for idx in range(self.index + 1, len(self.ladder)):
            dtype = self.ladder[idx]
            if dtype in supported:
                self.index = idx
                return dtype
        # search from start if nothing ahead is supported
        for idx in range(0, self.index):
            dtype = self.ladder[idx]
            if dtype in supported:
                self.index = idx
                return dtype
        return supported[-1]

    def advance(self) -> Optional[torch.dtype]:
        if self.manual_override:
            return None
        next_index = self.index + 1
        if next_index >= len(self.ladder):
            return None
        self.index = next_index
        return self.current()

    def allow_fallback(self) -> bool:
        return not self.manual_override and self.index < len(self.ladder) - 1


def _detect_oom_exception() -> type[BaseException]:
    try:
        return torch.cuda.OutOfMemoryError  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return RuntimeError


def _normalize_device_name(value: Optional[str]) -> str:
    if not value:
        return ""
    return value.strip().lower()


def _device_has_native_bf16(name: Optional[str], cc_major: Optional[int]) -> bool:
    normalized = _normalize_device_name(name)
    if normalized:
        for fragment in _NATIVE_BF16_NAME_FRAGMENTS:
            if fragment in normalized:
                return True
    if cc_major is not None and cc_major >= 9:
        # Hopper/Blackwell class or newer.
        return True
    return False


def _probe_hardware() -> HardwareProbe:
    probe = HardwareProbe()
    try:
        probe.cuda_available = torch.cuda.is_available()
        probe.cuda_device_count = torch.cuda.device_count() if probe.cuda_available else 0
    except Exception:  # pragma: no cover
        probe.cuda_available = False
        probe.cuda_device_count = 0

    props = None

    if probe.cuda_available:
        try:
            current_index = torch.cuda.current_device()
        except Exception:  # pragma: no cover
            current_index = 0
        try:
            props = torch.cuda.get_device_properties(current_index)
            probe.cuda_device_name = getattr(props, "name", None)
            probe.cuda_cc_major = getattr(props, "major", None)
            probe.cuda_cc_minor = getattr(props, "minor", None)
        except Exception:  # pragma: no cover
            props = None

    try:
        probe.mps_available = bool(torch.backends.mps.is_available())
    except Exception:  # pragma: no cover
        probe.mps_available = False

    try:  # pragma: no cover
        probe.xpu_available = bool(getattr(torch, "xpu", None) and torch.xpu.is_available())  # type: ignore[attr-defined]
    except Exception:
        probe.xpu_available = False

    probe.directml_available = bool(os.getenv("DIRECTML_PATH"))  # best effort

    try:
        mem_total = getattr(props, "total_memory", None)
        if mem_total is None and probe.cuda_available:
            fallback_props = torch.cuda.get_device_properties(0)
            mem_total = getattr(fallback_props, "total_memory", None)
        if mem_total is not None:
            probe.total_vram_mb = int(mem_total // (1024 * 1024))
    except Exception:  # pragma: no cover
        probe.total_vram_mb = None

    try:
        import psutil  # type: ignore

        probe.total_ram_mb = int(psutil.virtual_memory().total // (1024 * 1024))
    except Exception:  # pragma: no cover
        probe.total_ram_mb = None

    try:
        probe.bf16_support = bool(torch.cuda.is_bf16_supported()) if probe.cuda_available else False
    except Exception:  # pragma: no cover
        probe.bf16_support = False

    if probe.cuda_available:
        probe.native_bf16 = bool(
            probe.bf16_support and _device_has_native_bf16(probe.cuda_device_name, probe.cuda_cc_major)
        )

    probe.fp8_support = False
    if probe.cuda_available:
        try:
            cc = torch.cuda.get_device_properties(torch.cuda.current_device()).major
            probe.fp8_support = cc >= 9
        except Exception:  # pragma: no cover
            probe.fp8_support = False

    try:
        import xformers  # type: ignore

        probe.xformers_available = True
        probe.xformers_version = getattr(getattr(xformers, "version", None), "__version__", None)
    except Exception:
        probe.xformers_available = False
        probe.xformers_version = None

    return probe


@dataclass
class _LoadedModelRecord:
    """Internal bookkeeping for loaded models."""

    model: object
    loader: object | None
    base_module: torch.nn.Module | None
    load_device: torch.device
    offload_device: torch.device
    storage_dtype: torch.dtype
    inclusive_memory: int = 0
    exclusive_memory: int = 0
    model_accelerated: bool = False

    def __hash__(self) -> int:
        return hash(id(self.model))

    def matches(self, other: object) -> bool:
        return self.model is other


class CodexMemoryManager:
    """Central service coordinating device/dtype selection and model lifecycle."""

    def __init__(
        self,
        config: RuntimeMemoryConfig,
        *,
        probe: HardwareProbe | None = None,
    ) -> None:
        self._config = config
        self._probe = probe or _probe_hardware()
        self._oom_exception: type[BaseException] = _detect_oom_exception()
        self._loaded_models: MutableSequence[_LoadedModelRecord] = []
        self._signal_empty_cache: bool = False
        self._vae_always_tiled: bool = False
        self._primary_device: torch.device = self._resolve_primary_device()
        self._cpu_device = torch.device("cpu")
        self._attention = config.attention
        self._swap = config.swap
        self._budgets = config.budgets
        self._precision_states: Dict[DeviceRole, _PrecisionState] = {}

        self._log_probe()
        self._apply_deterministic_mode()
        self._configure_attention()
        self._initialize_precision_states()

    # --------------------------------------------------------------------- setup helpers
    def _log_probe(self) -> None:
        logger.debug("hardware probe: %s", self._probe.to_dict())

    def _apply_deterministic_mode(self) -> None:
        if self._config.deterministic_algorithms:
            logger.info("Enabling deterministic torch algorithms (warn_only=True).")
            torch.use_deterministic_algorithms(True, warn_only=True)

    def _configure_attention(self) -> None:
        if self._attention.backend == AttentionBackend.PYTORCH and self._probe.cuda_available:
            try:
                torch.backends.cuda.enable_math_sdp(True)
                if self._attention.enable_flash:
                    torch.backends.cuda.enable_flash_sdp(True)
                if self._attention.enable_mem_efficient:
                    torch.backends.cuda.enable_mem_efficient_sdp(True)
            except Exception:  # pragma: no cover
                logger.debug("Failed to enable PyTorch SDP backends.", exc_info=True)

    def _initialize_precision_states(self) -> None:
        self._precision_states.clear()
        for role in DeviceRole:
            ladder, manual = self._precision_policy_for(role)
            if not ladder:
                continue
            state = _PrecisionState(role=role, ladder=tuple(ladder), manual_override=manual)
            self._precision_states[role] = state
            ladder_display = " -> ".join(self._dtype_label(dt) for dt in state.ladder)
            logger.debug(
                "precision ladder[%s]=%s manual=%s",
                role.value,
                ladder_display,
                manual,
            )

    def _precision_policy_for(self, role: DeviceRole) -> Tuple[List[torch.dtype], bool]:
        flags = self._config.precision
        manual = False
        ladder: List[torch.dtype] = []

        global_forced = torch.float16 if flags.all_fp16 else None

        if role == DeviceRole.VAE:
            if flags.vae_fp32:
                ladder = [torch.float32]
                manual = True
            elif flags.vae_fp16:
                ladder = [torch.float16]
                manual = True
            elif flags.vae_bf16:
                ladder = [torch.bfloat16]
                manual = True
            elif global_forced is not None:
                ladder = [global_forced]
                manual = True
            else:
                ladder = self._auto_ladder_vae()
        elif role == DeviceRole.CORE:
            if flags.core_fp16:
                ladder = [torch.float16]
                manual = True
            elif flags.core_bf16:
                ladder = [torch.bfloat16]
                manual = True
            elif flags.core_fp8_e4m3fn or flags.core_fp8_e5m2:
                ladder = [torch.float16]
                manual = True
            elif global_forced is not None:
                ladder = [global_forced]
                manual = True
            else:
                ladder = self._auto_ladder_core()
        elif role == DeviceRole.TEXT_ENCODER:
            if flags.clip_fp32:
                ladder = [torch.float32]
                manual = True
            elif flags.clip_fp16:
                ladder = [torch.float16]
                manual = True
            elif flags.clip_bf16:
                ladder = [torch.bfloat16]
                manual = True
            elif flags.clip_fp8_e4m3fn or flags.clip_fp8_e5m2:
                ladder = [torch.float16]
                manual = True
            elif global_forced is not None:
                ladder = [global_forced]
                manual = True
            else:
                ladder = self._auto_ladder_text_encoder()
        else:
            ladder = [torch.float32]
            manual = True

        policy = self._config.component_policy(role)
        if policy.forced_dtype:
            try:
                forced = getattr(torch, policy.forced_dtype)
            except AttributeError as exc:
                raise MemoryConfigurationError(
                    f"Unsupported dtype '{policy.forced_dtype}' for {role.value}."
                ) from exc
            ladder = [forced]
            manual = True

        return ladder, manual

    def _auto_ladder_vae(self) -> List[torch.dtype]:
        ladder = [torch.float16, torch.float32]
        if self._probe.native_bf16:
            ladder.insert(0, torch.bfloat16)
        return ladder

    def _auto_ladder_core(self) -> List[torch.dtype]:
        if self._probe.native_bf16:
            return [torch.bfloat16, torch.float16]
        return [torch.float16]

    def _auto_ladder_text_encoder(self) -> List[torch.dtype]:
        if self._probe.native_bf16:
            return [torch.bfloat16, torch.float16]
        return [torch.float16]

    @staticmethod
    def _dtype_label(dtype: torch.dtype) -> str:
        text = str(dtype)
        return text.split(".")[-1]

    # --------------------------------------------------------------------- device/dtype
    def _resolve_primary_device(self) -> torch.device:
        cfg = self._config
        backend = cfg.device_backend
        probe = self._probe

        def choose_cuda() -> torch.device:
            if not probe.cuda_available:
                raise MemoryConfigurationError("CUDA requested but not available.")
            index = cfg.gpu_device_id or torch.cuda.current_device()
            return torch.device("cuda", index)

        if backend == DeviceBackend.AUTO:
            if probe.cuda_available:
                device = choose_cuda()
                logger.info(
                    "Device AUTO selected CUDA device %s (%s)",
                    device,
                    probe.cuda_device_name or "unknown",
                )
                return device
            # No CUDA/XPU/MPS/DirectML: fall back to CPU instead of aborting.
            logger.warning(
                "Device AUTO fallback to CPU (no GPU/accelerator detected). "
                "Set CODEX_DIFFUSION_DEVICE=cpu to silence this warning."
            )
            self._config.device_backend = DeviceBackend.CPU
            return torch.device("cpu")
        if backend == DeviceBackend.CUDA:
            return choose_cuda()
        if backend == DeviceBackend.MPS:
            if not probe.mps_available:
                raise MemoryConfigurationError("MPS backend requested but not available.")
            return torch.device("mps")
        if backend == DeviceBackend.XPU:
            if not probe.xpu_available:
                raise MemoryConfigurationError("XPU backend requested but not available.")
            return torch.device("xpu")
        if backend == DeviceBackend.DIRECTML:
            if not cfg.allow_directml:
                raise MemoryConfigurationError("DirectML backend disabled by configuration.")
            return torch.device("dml")
        return torch.device("cpu")

    # --------------------------------------------------------------------- public accessors
    @property
    def config(self) -> RuntimeMemoryConfig:
        """Return the active runtime memory configuration (treat as read-only)."""
        return self._config

    @property
    def oom_exception(self) -> type[BaseException]:
        return self._oom_exception

    @property
    def cpu_device(self) -> torch.device:
        return self._cpu_device

    @property
    def hardware_probe(self) -> HardwareProbe:
        """Expose a read-only view of the hardware probe used for this manager."""
        return self._probe

    @property
    def signal_empty_cache(self) -> bool:
        return self._signal_empty_cache

    @signal_empty_cache.setter
    def signal_empty_cache(self, value: bool) -> None:
        self._signal_empty_cache = bool(value)

    @property
    def vae_always_tiled(self) -> bool:
        return self._vae_always_tiled

    @vae_always_tiled.setter
    def vae_always_tiled(self, value: bool) -> None:
        self._vae_always_tiled = bool(value)

    def primary_device(self) -> torch.device:
        return self._primary_device

    def get_device(self, role: DeviceRole) -> torch.device:
        policy = self._config.component_policy(role)
        backend = policy.preferred_backend
        if backend == DeviceBackend.AUTO:
            if role == DeviceRole.TEXT_ENCODER and self._swap.policy != SwapPolicy.NEVER:
                return self._primary_device if self._primary_device.type != "cpu" else self._cpu_device
            if role == DeviceRole.VAE and policy.allow_offload:
                return self._primary_device if self._primary_device.type != "cpu" else self._cpu_device
            return self._primary_device
        if backend == DeviceBackend.CUDA:
            if not self._probe.cuda_available:
                raise MemoryConfigurationError("CUDA device requested for role but unsupported.")
            return torch.device("cuda", self._primary_device.index if self._primary_device.type == "cuda" else 0)
        if backend == DeviceBackend.MPS:
            if not self._probe.mps_available:
                raise MemoryConfigurationError("MPS device requested for role but unsupported.")
            return torch.device("mps")
        if backend == DeviceBackend.XPU:
            if not self._probe.xpu_available:
                raise MemoryConfigurationError("XPU device requested for role but unsupported.")
            return torch.device("xpu")
        if backend == DeviceBackend.DIRECTML:
            if not self._config.allow_directml:
                raise MemoryConfigurationError("DirectML requested but disabled.")
            return torch.device("dml")
        return self._cpu_device

    def get_offload_device(self, role: DeviceRole) -> torch.device:
        policy = self._config.component_policy(role)
        if not policy.allow_offload or self._swap.policy == SwapPolicy.NEVER:
            return self.get_device(role)
        if self._swap.policy == SwapPolicy.SHARED and self._primary_device.type != "cpu":
            return torch.device("cpu")
        return torch.device("cpu")

    def dtype_for_role(
        self,
        role: DeviceRole,
        *,
        supported: Sequence[torch.dtype] = (torch.float16, torch.bfloat16, torch.float32),
    ) -> torch.dtype:
        if not supported:
            raise MemoryConfigurationError(f"No supported dtypes passed for {role.value}.")

        policy = self._config.component_policy(role)
        device = self.get_device(role)
        if policy.forced_dtype:
            try:
                forced_dtype = getattr(torch, policy.forced_dtype)
            except AttributeError as exc:
                raise MemoryConfigurationError(f"Unsupported dtype '{policy.forced_dtype}' for {role.value}.") from exc
            if device.type == "cpu" and torch.float32 in supported and forced_dtype != torch.float32:
                return torch.float32
            if forced_dtype in supported:
                return forced_dtype
            return supported[-1]

        if device.type == "cpu":
            if torch.float32 in supported:
                return torch.float32
            return supported[-1]
        state = self._precision_states.get(role)
        if state is None:
            return supported[0]
        return state.select(supported)

    def current_precision(self, role: DeviceRole) -> Optional[torch.dtype]:
        state = self._precision_states.get(role)
        if not state:
            return None
        return state.current()

    def allow_precision_fallback(self, role: DeviceRole) -> bool:
        state = self._precision_states.get(role)
        return bool(state and state.allow_fallback())

    def precision_hint(self, role: DeviceRole) -> str:
        return _PRECISION_HINTS.get(role, "set manual precision for the component")

    def report_precision_failure(self, role: DeviceRole, *, location: str, reason: str) -> Optional[torch.dtype]:
        state = self._precision_states.get(role)
        device = None
        try:
            device = self.get_device(role)
        except Exception:
            device = None

        if state is None:
            logger.error(
                "Precision failure for %s at %s (reason=%s) but no ladder is configured.",
                role.value,
                location,
                reason,
            )
            return None

        current = state.current()
        if not state.allow_fallback():
            hint = self.precision_hint(role)
            logger.error(
                "Precision fallback exhausted for %s on %s (dtype=%s). Reason: %s. Manual action required: %s",
                role.value,
                device,
                self._dtype_label(current),
                reason,
                hint,
            )
            return None

        next_dtype = state.advance()
        if next_dtype is None:
            hint = self.precision_hint(role)
            logger.error(
                "Precision fallback unavailable for %s on %s (dtype=%s). Reason: %s. Manual action required: %s",
                role.value,
                device,
                self._dtype_label(current),
                reason,
                hint,
            )
            return None

        logger.warning(
            "Precision fallback triggered for %s at %s on %s: %s -> %s (%s)",
            role.value,
            location,
            device,
            self._dtype_label(current),
            self._dtype_label(next_dtype),
            reason,
        )
        return next_dtype

    # --------------------------------------------------------------------- tensor helpers
    def module_size(
        self,
        module,
        *,
        exclude_device: torch.device | None = None,
        include_device: torch.device | None = None,
        return_split: bool = False,
    ) -> int | Tuple[int, int, int]:
        module_mem = 0
        weight_mem = 0

        state_dict = module.state_dict()
        for key, tensor in state_dict.items():
            if exclude_device is not None and tensor.device == exclude_device:
                continue
            if include_device is not None and tensor.device != include_device:
                continue

            element_size = tensor.element_size()
            module_mem += tensor.nelement() * element_size
            if return_split and key.endswith(("weight", "bias")):
                weight_mem += tensor.nelement() * element_size

        if return_split:
            return module_mem, weight_mem, module_mem - weight_mem
        return module_mem

    def cast_to_device(self, tensor: torch.Tensor, device: torch.device, dtype: torch.dtype | None, *, copy: bool = False) -> torch.Tensor:
        target_dtype = dtype or tensor.dtype
        if tensor.device == device and tensor.dtype == target_dtype and not copy:
            return tensor
        if copy:
            tensor = tensor.clone()
        return tensor.to(device=device, dtype=target_dtype)

    # --------------------------------------------------------------------- memory metrics
    def get_free_memory(self, device: torch.device | None = None, *, return_torch_stats: bool = False) -> int | Tuple[int, int]:
        device = device or self._primary_device

        if device.type == "cpu":
            import psutil  # type: ignore

            virtual = psutil.virtual_memory()
            total = virtual.available
            return (total, total) if return_torch_stats else total

        if device.type == "cuda":
            stats = torch.cuda.memory_stats(device)
            torch_reserved = stats.get("reserved_bytes.all.current", 0)
            free, total = torch.cuda.mem_get_info(device)
            if return_torch_stats:
                return free, max(torch_reserved, free)
            return free

        if device.type == "xpu":
            stats = torch.xpu.memory_stats(device)  # type: ignore[attr-defined]
            torch_reserved = stats.get("reserved_bytes.all.current", 0)
            try:
                free, total = torch.xpu.mem_get_info(device)  # type: ignore[attr-defined]
            except Exception:  # pragma: no cover
                free, total = 0, 0
            if return_torch_stats:
                return free, max(torch_reserved, free)
            return free

        if return_torch_stats:
            return 0, 0
        return 0

    def minimum_inference_memory(self) -> int:
        return max(self._budgets.minimum_inference_mb * 1024 * 1024, 0)

    def memory_snapshot(self) -> Dict[str, object]:
        """Return a JSON-friendly snapshot of current memory usage and managed models.

        The snapshot is intentionally shallow and avoids side effects so it can be
        called from diagnostics endpoints without perturbing runtime behaviour.
        """
        device = self._primary_device

        # Hardware/probe info
        try:
            probe_dict: Dict[str, object] = self._probe.to_dict()
        except Exception:  # pragma: no cover - extremely defensive
            probe_dict = {}

        # Torch-level stats (best-effort; only populated when supported)
        torch_stats: Dict[str, int] = {}
        if device.type == "cuda":
            try:
                stats = torch.cuda.memory_stats(device)
                free_bytes, total_bytes = torch.cuda.mem_get_info(device)
                torch_stats = {
                    "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
                    "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                }
            except Exception:  # pragma: no cover
                torch_stats = {}
        elif device.type == "xpu":
            try:
                stats = torch.xpu.memory_stats(device)  # type: ignore[attr-defined]
                free_bytes, total_bytes = torch.xpu.mem_get_info(device)  # type: ignore[attr-defined]
                torch_stats = {
                    "allocated_bytes": int(stats.get("allocated_bytes.all.current", 0)),
                    "reserved_bytes": int(stats.get("reserved_bytes.all.current", 0)),
                    "free_bytes": int(free_bytes),
                    "total_bytes": int(total_bytes),
                }
            except Exception:  # pragma: no cover
                torch_stats = {}
        elif device.type == "mps":
            try:
                allocated = int(torch.mps.current_allocated_memory())
                total = int(torch.mps.driver_allocated_memory())
                free = max(total - allocated, 0)
                torch_stats = {
                    "allocated_bytes": allocated,
                    "reserved_bytes": allocated,
                    "free_bytes": free,
                    "total_bytes": total,
                }
            except Exception:  # pragma: no cover
                torch_stats = {}

        # Managed model records
        models: List[Dict[str, object]] = []
        total_inclusive = 0
        total_exclusive = 0

        for record in self._loaded_models:
            try:
                module = record.base_module or self._extract_module(record.loader or record.model)
            except Exception:
                module = None

            if module is not None:
                module_name = module.__class__.__name__
            else:
                module_name = type(record.model).__name__

            load_device = getattr(record.load_device, "type", str(record.load_device))
            offload_device = getattr(record.offload_device, "type", str(record.offload_device))

            models.append(
                {
                    "module": module_name,
                    "load_device": load_device,
                    "offload_device": offload_device,
                    "storage_dtype": str(record.storage_dtype).replace("torch.", ""),
                    "inclusive_bytes": int(record.inclusive_memory),
                    "exclusive_bytes": int(record.exclusive_memory),
                    "accelerated": bool(record.model_accelerated),
                }
            )
            total_inclusive += int(record.inclusive_memory)
            total_exclusive += int(record.exclusive_memory)

        budgets = {
            "minimum_inference_mb": int(self._budgets.minimum_inference_mb),
            "hard_reservation_mb": int(self._budgets.hard_reservation_mb),
            "safety_margin_mb": int(self._budgets.safety_margin_mb),
        }

        device_backend = getattr(self._config.device_backend, "value", str(self._config.device_backend))

        return {
            "device_backend": device_backend,
            "primary_device": str(device),
            "probe": probe_dict,
            "budgets": budgets,
            "torch": torch_stats,
            "models": models,
            "totals": {
                "models_inclusive_bytes": total_inclusive,
                "models_exclusive_bytes": total_exclusive,
            },
        }

    # --------------------------------------------------------------------- dtype helpers
    def should_use_fp16(
        self,
        *,
        device: torch.device | None = None,
        model_params: int = 0,
        prioritize_performance: bool = True,
        manual_cast: bool = False,
    ) -> bool:
        device = device or self._primary_device
        if device.type == "cpu":
            return False
        if manual_cast:
            return True
        policy = self._config.component_policy(DeviceRole.CORE)
        if policy.forced_dtype and policy.forced_dtype != "float16":
            return False
        if not prioritize_performance:
            return False
        return True

    def should_use_bf16(
        self,
        *,
        device: torch.device | None = None,
        model_params: int = 0,
        prioritize_performance: bool = True,
        manual_cast: bool = False,
    ) -> bool:
        if not self._probe.bf16_support:
            return False
        if manual_cast:
            return True
        if not prioritize_performance:
            return False
        return True

    # --------------------------------------------------------------------- attention helpers
    def xformers_enabled(self) -> bool:
        if self._config.disable_xformers:
            return False
        return self._probe.xformers_available and self._attention.backend == AttentionBackend.XFORMERS

    def xformers_enabled_vae(self) -> bool:
        return self.xformers_enabled() and self._config.enable_xformers_vae

    def force_upcast_attention_dtype(self) -> torch.dtype:
        if self._attention.force_upcast:
            return torch.float32
        return torch.float16

    def pytorch_attention_enabled(self) -> bool:
        return self._attention.backend == AttentionBackend.PYTORCH

    # --------------------------------------------------------------------- cache helpers
    def soft_empty_cache(self, force: bool = False) -> None:
        if self._primary_device.type == "cuda":
            torch.cuda.empty_cache()
        if force:
            self._signal_empty_cache = False

    def unload_all_models(self) -> None:
        for record in list(self._loaded_models):
            self._unload_record(record, avoid_model_moving=True)
        self._loaded_models.clear()
        logger.info("Unloaded all models, cache cleared.")

    def loaded_models(self) -> Tuple[_LoadedModelRecord, ...]:
        return tuple(self._loaded_models)

    def unload_model_clones(self, model: object) -> None:
        if not hasattr(model, "is_clone"):
            return
        predicate = getattr(model, "is_clone")
        if not callable(predicate):
            return
        for index in range(len(self._loaded_models) - 1, -1, -1):
            record = self._loaded_models[index]
            try:
                matches = predicate(record.model)
            except Exception:  # pragma: no cover
                matches = False
            if matches:
                self._unload_record(record, avoid_model_moving=True)
                self._loaded_models.pop(index)

    def unload_model(self, model: object) -> None:
        record = self._find_loaded_model(model)
        if record is None:
            return
        self._unload_record(record, avoid_model_moving=True)
        try:
            self._loaded_models.remove(record)
        except ValueError:  # pragma: no cover
            pass
        if self._primary_device.type == "cuda":
            torch.cuda.empty_cache()

    # --------------------------------------------------------------------- load/unload
    def load_models(
        self,
        models: Sequence[object],
        *,
        memory_required: int = 0,
        hard_memory_preservation: int = 0,
    ) -> None:
        if not models:
            return

        execution_start = time.perf_counter()
        memory_budget = max(self.minimum_inference_memory(), memory_required) + hard_memory_preservation
        models_to_load: List[_LoadedModelRecord] = []
        already_loaded: List[_LoadedModelRecord] = []

        # DEBUG: Log memory state before loading
        if self._primary_device.type == "cuda":
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._primary_device)
            allocated = torch.cuda.memory_allocated(self._primary_device)
            reserved = torch.cuda.memory_reserved(self._primary_device)
            logger.info(
                "[memory-debug] BEFORE load_models: free=%.2f GB, allocated=%.2f GB, reserved=%.2f GB, total=%.2f GB",
                free_bytes / 1e9, allocated / 1e9, reserved / 1e9, total_bytes / 1e9,
            )
            logger.info(
                "[memory-debug] Currently loaded models (%d): %s",
                len(self._loaded_models),
                [r.base_module.__class__.__name__ if r.base_module else "?" for r in self._loaded_models],
            )

        for model in models:
            record = self._find_loaded_model(model)
            if record:
                already_loaded.append(record)
            else:
                models_to_load.append(self._create_record(model))

        if models_to_load:
            self._allocate_memory(models_to_load, memory_budget, already_loaded)
            for record in models_to_load:
                self._load_record(record)
                self._loaded_models.insert(0, record)
        else:
            self._cleanup_for_loaded_models(already_loaded, memory_budget)

        elapsed = time.perf_counter() - execution_start
        logger.info("Model load completed (%d new, %d existing) in %.2fs.", len(models_to_load), len(already_loaded), elapsed)

    def load_model(self, model: object) -> None:
        self.load_models([model])

    def free_memory(
        self,
        memory_required: int,
        *,
        device: torch.device | None = None,
        keep_loaded: Sequence[_LoadedModelRecord] | None = None,
        free_all: bool = False,
    ) -> None:
        device = device or self._primary_device
        keep_loaded = keep_loaded or ()
        release_candidates = [record for record in self._loaded_models if record not in keep_loaded]
        released = 0

        for record in release_candidates:
            self._unload_record(record, avoid_model_moving=free_all)
            self._loaded_models.remove(record)
            released += record.exclusive_memory
            if not free_all and released >= memory_required:
                break

        if device.type == "cuda":
            torch.cuda.empty_cache()
        logger.debug("Freed %d bytes on %s (required=%d).", released, device, memory_required)

    # --------------------------------------------------------------------- internals
    def _find_loaded_model(self, model: object) -> _LoadedModelRecord | None:
        for record in self._loaded_models:
            if record.matches(model):
                return record
        return None

    # ------------------------------------------------------------------ load target helpers
    def _extract_module(self, obj: object, *, visited: Optional[Set[int]] = None) -> torch.nn.Module | None:
        visited = visited or set()
        stack: List[object] = [obj]
        while stack:
            current = stack.pop()
            if current is None:
                continue
            ident = id(current)
            if ident in visited:
                continue
            visited.add(ident)
            if isinstance(current, torch.nn.Module):
                return current
            for attr in (
                "model",
                "module",
                "_module",
                "target",
                "wrapped",
                "_wrapped",
                "inner",
                "inner_model",
                "base",
                "_base",
                "first_stage_model",
            ):
                if not hasattr(current, attr):
                    continue
                try:
                    value = getattr(current, attr)
                except Exception:
                    continue
                if value is not None and value is not current:
                    stack.append(value)
        return None

    def _resolve_loader(self, model: object) -> tuple[object, torch.nn.Module]:
        candidate_attrs = (
            None,
            "patcher",
            "loader",
            "wrapped",
            "target",
            "model",
            "module",
        )

        for attr in candidate_attrs:
            candidate = model if attr is None else getattr(model, attr, None)
            if candidate is None:
                continue
            loader, module = self._classify_loader(candidate)
            if loader is not None and module is not None:
                return loader, module

        module = self._extract_module(model)
        if module is not None:
            return model, module
        raise MemoryLoadError(
            f"Unable to resolve a loadable module from {type(model).__name__}; "
            "expected a ModelPatcher, torch.nn.Module, or wrapper exposing one."
        )

    def _classify_loader(self, candidate: object) -> tuple[object | None, torch.nn.Module | None]:
        module = self._extract_module(candidate)
        if module is None:
            return None, None
        if hasattr(candidate, "codex_patch_model") or hasattr(candidate, "model_patches_to"):
            return candidate, module
        if isinstance(candidate, torch.nn.Module):
            return candidate, module
        return None, module

    def _create_record(self, model: object) -> _LoadedModelRecord:
        load_device = getattr(model, "load_device", self._primary_device)
        offload_device = getattr(model, "offload_device", self.get_offload_device(DeviceRole.CORE))
        storage_dtype_getter = getattr(model, "model_dtype", None)
        storage_dtype = storage_dtype_getter() if callable(storage_dtype_getter) else torch.float32
        loader, base_module = self._resolve_loader(model)
        return _LoadedModelRecord(
            model=model,
            loader=loader,
            base_module=base_module,
            load_device=load_device,
            offload_device=offload_device,
            storage_dtype=storage_dtype,
        )

    def _load_record(self, record: _LoadedModelRecord) -> None:
        loader = record.loader or record.model
        module = record.base_module or self._extract_module(loader)
        if module is None:
            raise MemoryLoadError(
                f"Failed to resolve base module for {type(record.model).__name__}; cannot compute memory usage."
            )
        # Ensure cache of module for downstream accounting
        record.base_module = module
        target_name = module.__class__.__name__

        # DEBUG: Log module size before loading
        module_size_bytes = self.module_size(module)
        logger.info(
            "[memory-debug] _load_record: module=%s size=%.2f GB target_device=%s",
            target_name, module_size_bytes / 1e9, record.load_device,
        )

        # DEBUG: Log current device of module parameters
        try:
            first_param = next(module.parameters(), None)
            if first_param is not None:
                logger.info(
                    "[memory-debug] module %s current device: %s dtype=%s",
                    target_name, first_param.device, first_param.dtype,
                )
        except Exception:
            pass

        try:
            if smart_offload_enabled() and getattr(record.load_device, "type", "") == "cuda":
                torch.cuda.empty_cache()
                # DEBUG: Log after empty_cache
                if self._primary_device.type == "cuda":
                    free_bytes, _ = torch.cuda.mem_get_info(self._primary_device)
                    logger.info("[memory-debug] AFTER empty_cache: free=%.2f GB", free_bytes / 1e9)

            if hasattr(loader, "model_patches_to"):
                logger.info("[memory-debug] calling model_patches_to(%s)", record.load_device)
                loader.model_patches_to(record.load_device)
                loader.model_patches_to(record.storage_dtype)
            elif hasattr(loader, "to"):
                logger.info("[memory-debug] calling loader.to(device=%s)", record.load_device)
                loader.to(device=record.load_device)
                if record.storage_dtype is not None:
                    loader.to(dtype=record.storage_dtype)
            if hasattr(loader, "codex_patch_model"):
                logger.info("[memory-debug] calling codex_patch_model(%s)", record.load_device)
                loader.codex_patch_model(record.load_device)
            if hasattr(loader, "current_device"):
                setattr(loader, "current_device", record.load_device)
            record.inclusive_memory = self.module_size(module)
            record.exclusive_memory = record.inclusive_memory
            record.model_accelerated = True
            compute_dtype = None
            try:
                dtype_attr = getattr(module, "computation_dtype", None)
                if callable(dtype_attr):
                    compute_dtype = dtype_attr()
                elif dtype_attr is not None:
                    compute_dtype = dtype_attr
                elif hasattr(module, "dtype"):
                    compute_dtype = getattr(module, "dtype")
            except Exception:  # pragma: no cover
                compute_dtype = None

            # DEBUG: Log memory after load
            if self._primary_device.type == "cuda":
                free_bytes, _ = torch.cuda.mem_get_info(self._primary_device)
                logger.info("[memory-debug] AFTER load %s: free=%.2f GB", target_name, free_bytes / 1e9)

            logger.info(
                "[memory] loaded %s to device=%s storage=%s compute=%s mem=%d",
                target_name,
                record.load_device,
                record.storage_dtype,
                compute_dtype,
                record.inclusive_memory,
            )
        except self._oom_exception as exc:
            raise MemoryLoadError(f"OOM while loading {record.model}: {exc}") from exc
        except Exception as exc:
            raise MemoryLoadError(f"Failed to load model {record.model}: {exc}") from exc

    def _unload_record(self, record: _LoadedModelRecord, *, avoid_model_moving: bool = False) -> None:
        try:
            loader = record.loader or record.model
            if hasattr(loader, "codex_unpatch_model"):
                offload_device = record.offload_device if not avoid_model_moving else self._cpu_device
                loader.codex_unpatch_model(offload_device)
            elif hasattr(loader, "to") and not avoid_model_moving:
                loader.to(self._cpu_device)
            record.model_accelerated = False
            logger.debug("Unloaded model %s (avoid_move=%s).", record.model, avoid_model_moving)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to unload model %s: %s", record.model, exc, exc_info=True)

    def _cleanup_for_loaded_models(self, records: Sequence[_LoadedModelRecord], memory_budget: int) -> None:
        devices = {record.load_device for record in records if record.load_device.type != "cpu"}
        for device in devices:
            self.free_memory(memory_budget, device=device, keep_loaded=records)

    def _allocate_memory(
        self,
        records: Sequence[_LoadedModelRecord],
        memory_budget: int,
        already_loaded: Sequence[_LoadedModelRecord],
    ) -> None:
        total_required: dict[torch.device, int] = {}
        for record in records:
            module = record.base_module or self._extract_module(record.loader or record.model)
            if module is None:
                raise MemoryLoadError(
                    f"Unable to resolve module for {type(record.model).__name__} while allocating memory."
                )
            record.base_module = module
            record.inclusive_memory = self.module_size(module)
            record.exclusive_memory = record.inclusive_memory
            total_required[record.load_device] = total_required.get(record.load_device, 0) + record.inclusive_memory

        for device, requirement in total_required.items():
            target = requirement + memory_budget
            self.free_memory(target, device=device, keep_loaded=already_loaded)

    # --------------------------------------------------------------------- factory
    @classmethod
    def create(cls, config: RuntimeMemoryConfig, probe: HardwareProbe | None = None) -> "CodexMemoryManager":
        if probe is None:
            probe = _probe_hardware()
        if config.device_backend == DeviceBackend.CUDA and not probe.cuda_available:
            raise HardwareProbeError("CUDA backend requested but no CUDA device detected.")
        return cls(config, probe=probe)
__all__ = ["CodexMemoryManager"]
