"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima engine specification and runtime assembly (Cosmos Predict2 / Anima adapter).
Assembles a Codex-native runtime from the parsed Anima core bundle and validates sha-selected external assets (Qwen3-0.6B TE + WanVAE-style VAE),
while deferring external TE/VAE/T5 tokenizer materialization to first use to reduce startup latency.
Produces a denoiser patcher suitable for the canonical txt2img/img2img pipelines (Option A).
Sets Anima predictor defaults, including SIMPLE schedule mode selection for ComfyUI sigma-ladder parity.

Symbols (top-level; keep in sync; no ghosts):
- `_parse_to_device_dtype` (function): Extract normalized `device`/`dtype` targets from `Module.to(...)` call arguments.
- `_move_tensors_to_device` (function): Recursively moves tensor-only argument trees onto a target device.
- `_LazyAnimaQwenModel` (class): Lazy Qwen3 model module that loads Anima text-encoder weights on first forward/use.
- `_build_lazy_anima_vae_config` (function): Header-only strict Anima WAN VAE config probe used by lazy VAE wrapper.
- `_LazyAnimaWanVAE` (class): Lazy WAN VAE module that loads safetensors weights on first encode/decode use.
- `_LazyAnimaT5Tokenizer` (class): Lazy tokenizer proxy for Anima T5 tokenization path.
- `AnimaTextPipelines` (dataclass): Text pipeline container (Qwen3 embeddings + offline T5 tokenizer).
- `AnimaEngineRuntime` (dataclass): Assembled runtime container (denoiser + VAE + text pipelines + patchers).
- `AnimaEngineSpec` (dataclass): Engine spec (family defaults + flow shift/multiplier overrides).
- `assemble_anima_runtime` (function): Assemble an Anima runtime from a `DiffusionModelBundle` and engine options.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.patchers.base import ModelPatcher
from apps.backend.runtime.model_registry.family_runtime import FamilyRuntimeSpec, get_family_spec
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.sampling_adapters.prediction import (
    SIMPLE_SCHEDULE_MODE_COMFY_DOWNSAMPLE_SIGMAS,
    PredictionDiscreteFlow,
)

logger = logging.getLogger("backend.engines.anima.spec")


def _torch_dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported torch dtype: {dtype!r}")


def _parse_to_device_dtype(
    *args: Any,
    **kwargs: Any,
) -> tuple[torch.device | None, torch.dtype | None]:
    raw_device: Any = kwargs.get("device")
    raw_dtype: Any = kwargs.get("dtype")

    for arg in args:
        if isinstance(arg, torch.dtype):
            raw_dtype = arg
            continue
        if isinstance(arg, torch.device):
            raw_device = arg
            continue
        if isinstance(arg, str):
            try:
                raw_device = torch.device(arg)
            except Exception:
                continue

    device: torch.device | None = None
    if raw_device is not None:
        try:
            device = torch.device(raw_device)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Anima lazy loader received invalid device target: {raw_device!r}") from exc

    dtype: torch.dtype | None = None
    if raw_dtype is not None:
        if not isinstance(raw_dtype, torch.dtype):
            raise RuntimeError(f"Anima lazy loader received invalid dtype target: {raw_dtype!r}")
        dtype = raw_dtype

    return device, dtype


def _move_tensors_to_device(value: Any, *, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device)
    if isinstance(value, tuple):
        return tuple(_move_tensors_to_device(item, device=device) for item in value)
    if isinstance(value, list):
        return [_move_tensors_to_device(item, device=device) for item in value]
    if isinstance(value, dict):
        return {key: _move_tensors_to_device(item, device=device) for key, item in value.items()}
    return value


class _LazyAnimaQwenModel(torch.nn.Module):
    """Lazy loader for Anima Qwen3 model weights."""

    def __init__(self, *, tenc_path: str, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self._tenc_path = str(tenc_path)
        self._torch_dtype = torch_dtype
        self._lock = threading.Lock()
        self._loaded_model: torch.nn.Module | None = None
        self.preferred_device: torch.device | None = None
        self.preferred_dtype: torch.dtype | None = None

    def _ensure_loaded(self) -> torch.nn.Module:
        model = self._loaded_model
        if model is not None:
            return model

        with self._lock:
            model = self._loaded_model
            if model is not None:
                return model

            loaded_text_encoder = _load_external_text_encoder(tenc_path=self._tenc_path, torch_dtype=self._torch_dtype)
            loaded_model = getattr(loaded_text_encoder, "model", None)
            if not isinstance(loaded_model, torch.nn.Module):
                raise RuntimeError(
                    "Anima lazy Qwen loader expected .model nn.Module, "
                    f"got {type(loaded_model).__name__}."
                )
            if self.preferred_device is not None or self.preferred_dtype is not None:
                loaded_model.to(
                    device=self.preferred_device,
                    dtype=self.preferred_dtype or self._torch_dtype,
                )
            loaded_model.eval()
            self._loaded_model = loaded_model
            logger.info(
                "Anima lazy load: qwen model materialized path=%s device=%s dtype=%s",
                self._tenc_path,
                str(self.preferred_device),
                str(self.preferred_dtype or self._torch_dtype),
            )
            return loaded_model

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        device, dtype = _parse_to_device_dtype(*args, **kwargs)
        if device is not None:
            self.preferred_device = device
        if dtype is not None:
            self.preferred_dtype = dtype
        if self._loaded_model is not None:
            self._loaded_model.to(*args, **kwargs)
        return self

    def state_dict(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        if self._loaded_model is None:
            return {}
        return self._loaded_model.state_dict(*args, **kwargs)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(())
        return self._loaded_model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(())
        return self._loaded_model.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(())
        return self._loaded_model.buffers(recurse=recurse)

    def modules(self):  # type: ignore[override]
        if self._loaded_model is None:
            return iter((self,))
        return self._loaded_model.modules()

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(((prefix, self),))
        return self._loaded_model.named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

    def forward(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        model = self._ensure_loaded()
        target_device = self.preferred_device
        if target_device is None:
            try:
                target_device = next(model.parameters()).device
            except StopIteration:
                target_device = memory_management.manager.cpu_device
        moved_args = _move_tensors_to_device(args, device=target_device)
        moved_kwargs = _move_tensors_to_device(kwargs, device=target_device)
        return model(*moved_args, **moved_kwargs)


def _build_lazy_anima_vae_config(*, vae_path: str):
    from apps.backend.runtime.checkpoint.safetensors_header import read_safetensors_header
    from apps.backend.runtime.families.anima.wan_vae import (
        WanVaeConfig,
        detect_wan_vae_variant_from_header,
        infer_wan_vae_config_from_safetensors_header,
    )
    from apps.backend.runtime.families.wan22.wan_latent_norms import WAN21_LATENTS_MEAN, WAN21_LATENTS_STD

    header = read_safetensors_header(Path(vae_path))
    variant = detect_wan_vae_variant_from_header(header)
    if variant == "2.2":
        raise NotImplementedError(
            "WAN VAE 2.2 detected by safetensors header keys; "
            "Anima v1 supports only WAN 2.1 image-mode assets."
        )
    inferred = infer_wan_vae_config_from_safetensors_header(header)
    if int(inferred.latent_channels) != 16:
        raise RuntimeError(
            f"WAN VAE latent_channels mismatch for Anima: got {inferred.latent_channels}, expected 16."
        )
    return WanVaeConfig(
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        latent_channels=int(inferred.latent_channels),
        scaling_factor=1.0,
        shift_factor=None,
        latents_mean=tuple(float(value) for value in WAN21_LATENTS_MEAN),
        latents_std=tuple(float(value) for value in WAN21_LATENTS_STD),
    )


class _LazyAnimaWanVAE(torch.nn.Module):
    """Lazy loader for Anima WAN VAE weights."""

    def __init__(self, *, vae_path: str, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self._vae_path = str(vae_path)
        self._torch_dtype = torch_dtype
        self._lock = threading.Lock()
        self._loaded_model: torch.nn.Module | None = None
        self.preferred_device: torch.device | None = None
        self.preferred_dtype: torch.dtype | None = None
        self.config = _build_lazy_anima_vae_config(vae_path=self._vae_path)

    def _ensure_loaded(self) -> torch.nn.Module:
        model = self._loaded_model
        if model is not None:
            return model

        with self._lock:
            model = self._loaded_model
            if model is not None:
                return model

            loaded_model = _load_external_vae(vae_path=self._vae_path, torch_dtype=self._torch_dtype)
            if not isinstance(loaded_model, torch.nn.Module):
                raise RuntimeError(
                    "Anima lazy VAE loader returned invalid model type: "
                    f"{type(loaded_model).__name__}"
                )
            if self.preferred_device is not None or self.preferred_dtype is not None:
                loaded_model.to(
                    device=self.preferred_device,
                    dtype=self.preferred_dtype or self._torch_dtype,
                )
            loaded_model.eval()
            self._loaded_model = loaded_model
            logger.info(
                "Anima lazy load: vae model materialized path=%s device=%s dtype=%s",
                self._vae_path,
                str(self.preferred_device),
                str(self.preferred_dtype or self._torch_dtype),
            )
            return loaded_model

    def to(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        device, dtype = _parse_to_device_dtype(*args, **kwargs)
        if device is not None:
            self.preferred_device = device
        if dtype is not None:
            self.preferred_dtype = dtype
        if self._loaded_model is not None:
            self._loaded_model.to(*args, **kwargs)
        return self

    def state_dict(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        if self._loaded_model is None:
            return {}
        return self._loaded_model.state_dict(*args, **kwargs)

    def parameters(self, recurse: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(())
        return self._loaded_model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(())
        return self._loaded_model.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(())
        return self._loaded_model.buffers(recurse=recurse)

    def modules(self):  # type: ignore[override]
        if self._loaded_model is None:
            return iter((self,))
        return self._loaded_model.modules()

    def named_modules(self, memo=None, prefix: str = "", remove_duplicate: bool = True):  # type: ignore[override]
        if self._loaded_model is None:
            return iter(((prefix, self),))
        return self._loaded_model.named_modules(
            memo=memo,
            prefix=prefix,
            remove_duplicate=remove_duplicate,
        )

    def eval(self):  # type: ignore[override]
        if self._loaded_model is not None:
            self._loaded_model.eval()
        return self

    def encode(self, *args: Any, **kwargs: Any):
        model = self._ensure_loaded()
        return model.encode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any):
        model = self._ensure_loaded()
        return model.decode(*args, **kwargs)


class _LazyAnimaT5Tokenizer:
    """Lazy loader for Anima T5 tokenizer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._tokenizer: Any | None = None

    def _ensure_loaded(self) -> Any:
        tokenizer = self._tokenizer
        if tokenizer is not None:
            return tokenizer
        with self._lock:
            tokenizer = self._tokenizer
            if tokenizer is not None:
                return tokenizer
            from apps.backend.runtime.families.anima.text_encoder import load_anima_t5_tokenizer

            tokenizer = load_anima_t5_tokenizer()
            self._tokenizer = tokenizer
            logger.info("Anima lazy load: t5 tokenizer materialized")
            return tokenizer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._ensure_loaded(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        tokenizer = self._ensure_loaded()
        return tokenizer(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class AnimaTextPipelines:
    qwen3_text: Any
    t5_tokenizer: Any


@dataclass(slots=True)
class AnimaEngineRuntime:
    vae: VAE
    denoiser: DenoiserPatcher
    text: AnimaTextPipelines
    qwen: ModelPatcher
    device: torch.device = field(default_factory=lambda: memory_management.manager.mount_device())
    core_storage_dtype: str = "bf16"
    core_compute_dtype: str = "fp32"
    te_storage_dtype: str = "bf16"
    te_compute_dtype: str = "fp32"
    vae_storage_dtype: str = "bf16"
    vae_compute_dtype: str = "fp32"


@dataclass(frozen=True, slots=True)
class AnimaEngineSpec:
    name: str = "anima"
    family: ModelFamily = ModelFamily.ANIMA
    _flow_shift_override: float | None = field(default=None, repr=False)
    _flow_multiplier_override: float | None = field(default=None, repr=False)

    def _get_family_spec(self) -> FamilyRuntimeSpec:
        return get_family_spec(self.family)

    @property
    def flow_shift(self) -> float:
        spec = self._get_family_spec()
        if self._flow_shift_override is not None:
            return float(self._flow_shift_override)
        if spec.flow_shift is None:
            raise RuntimeError("AnimaEngineSpec.flow_shift missing: family runtime spec flow_shift is None.")
        return float(spec.flow_shift)

    @property
    def flow_multiplier(self) -> float:
        if self._flow_multiplier_override is not None:
            return float(self._flow_multiplier_override)
        # Source-of-truth: Anima signature extras declare multiplier=1.0; use-case/engine may override.
        return 1.0


def _predictor(*, spec: AnimaEngineSpec) -> PredictionDiscreteFlow:
    return PredictionDiscreteFlow(
        prediction_type="const",
        shift=float(spec.flow_shift),
        multiplier=float(spec.flow_multiplier),
        timesteps=1000,
        simple_schedule_mode=SIMPLE_SCHEDULE_MODE_COMFY_DOWNSAMPLE_SIGMAS,
    )


def _load_external_text_encoder(*, tenc_path: str, torch_dtype: torch.dtype) -> object:
    from apps.backend.runtime.families.anima.text_encoder import load_anima_qwen3_06b_text_encoder

    return load_anima_qwen3_06b_text_encoder(tenc_path, torch_dtype=torch_dtype)


def _load_external_vae(*, vae_path: str, torch_dtype: torch.dtype) -> object:
    from apps.backend.runtime.families.anima.wan_vae import load_wan_vae_from_safetensors

    return load_wan_vae_from_safetensors(vae_path, torch_dtype=torch_dtype)


def _require_external_asset_path(*, opts: Mapping[str, Any], key: str, label: str) -> str:
    raw = opts.get(key)
    if raw is None:
        raise ValueError(f"Anima requires an external {label} via `{key}` (sha-selected); missing.")
    if not isinstance(raw, str):
        raise TypeError(
            f"Anima option `{key}` must be a non-empty string path (sha-selected); got {type(raw).__name__}."
        )
    value = raw.strip()
    if not value:
        raise ValueError(f"Anima requires an external {label} via `{key}` (sha-selected); missing.")
    return value


def _require_existing_external_asset_path(*, raw_path: str, label: str) -> str:
    resolved = os.path.expanduser(raw_path)
    if not os.path.isfile(resolved):
        raise RuntimeError(f"Anima {label} path not found: {resolved}")
    return resolved


def assemble_anima_runtime(
    *,
    spec: AnimaEngineSpec,
    estimated_config: Any,
    codex_components: Mapping[str, object],
    engine_options: Mapping[str, Any] | None = None,
) -> AnimaEngineRuntime:
    """Assemble Anima runtime from a minimal (state-dict) bundle plus sha-selected external assets."""
    opts = dict(engine_options or {})
    vae_path = _require_external_asset_path(opts=opts, key="vae_path", label="VAE")
    tenc_path = _require_external_asset_path(opts=opts, key="tenc_path", label="text encoder")

    # Core transformer component is provided as a parser-stripped state dict (`net.` removed).
    transformer_sd = codex_components.get("transformer")
    if transformer_sd is None:
        raise RuntimeError("Anima bundle missing required component 'transformer' (state dict).")

    vae_path = _require_existing_external_asset_path(raw_path=vae_path, label="VAE")
    tenc_path = _require_existing_external_asset_path(raw_path=tenc_path, label="text encoder")

    from apps.backend.runtime.families.anima.loader import load_anima_dit_from_state_dict

    native_core_dtype: torch.dtype | None = None
    try:
        first_key = next(iter(transformer_sd.keys()))  # type: ignore[attr-defined]
        first_tensor = transformer_sd[first_key]  # type: ignore[index]
        if isinstance(first_tensor, torch.Tensor):
            native_core_dtype = first_tensor.dtype
    except Exception:
        native_core_dtype = None

    core_storage = memory_management.manager.dtype_for_role(DeviceRole.CORE, native_dtype=native_core_dtype)
    core_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.CORE, storage_dtype=core_storage)
    load_device = memory_management.manager.get_device(DeviceRole.CORE)
    offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
    initial_device = offload_device

    model = load_anima_dit_from_state_dict(
        transformer_sd,  # type: ignore[arg-type]
        device=initial_device,
        dtype=core_storage if isinstance(core_storage, torch.dtype) else None,
    )
    model.storage_dtype = core_storage
    model.computation_dtype = core_compute
    model.load_device = load_device
    model.initial_device = initial_device
    model.offload_device = offload_device

    denoiser = DenoiserPatcher.from_model(
        model=model,
        diffusers_scheduler=None,
        predictor=_predictor(spec=spec),
        config=estimated_config,
    )

    te_storage = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
    te_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.TEXT_ENCODER, storage_dtype=te_storage)
    lazy_qwen_model = _LazyAnimaQwenModel(tenc_path=tenc_path, torch_dtype=te_storage)

    vae_storage = memory_management.manager.dtype_for_role(DeviceRole.VAE)
    vae_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.VAE, storage_dtype=vae_storage)
    lazy_vae_model = _LazyAnimaWanVAE(vae_path=vae_path, torch_dtype=vae_storage)

    # Wrap VAE with shared patcher interface (encode/decode + normalization via family spec fallback).
    vae = VAE(model=lazy_vae_model, family=ModelFamily.ANIMA)

    # Text encoder patcher for memory management integration.
    te_load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
    te_offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
    from apps.backend.runtime.families.anima.text_encoder import AnimaQwenTextEncoder

    text_encoder = AnimaQwenTextEncoder(model=lazy_qwen_model)
    qwen = ModelPatcher(
        lazy_qwen_model,
        load_device=te_load_device,
        offload_device=te_offload_device,
        size=1,
    )

    # Text pipelines: Qwen embeddings + offline T5 tokenizer.
    from apps.backend.runtime.families.anima.text_encoder import AnimaQwenTextProcessingEngine

    qwen_max_length = int(os.getenv("CODEX_ANIMA_QWEN_MAX_LENGTH", "512") or 512)
    if qwen_max_length <= 0:
        raise ValueError("CODEX_ANIMA_QWEN_MAX_LENGTH must be > 0")

    text_engine = AnimaQwenTextProcessingEngine(text_encoder, max_length=qwen_max_length)
    t5_tokenizer = _LazyAnimaT5Tokenizer()
    text_pipelines = AnimaTextPipelines(qwen3_text=text_engine, t5_tokenizer=t5_tokenizer)

    core_dev = memory_management.manager.get_device(DeviceRole.CORE)
    device = core_dev
    logger.debug(
        "Anima runtime assembled: device=%s core_storage=%s core_compute=%s te_storage=%s te_compute=%s vae_storage=%s vae_compute=%s",
        device,
        _torch_dtype_label(core_storage),
        _torch_dtype_label(core_compute),
        _torch_dtype_label(te_storage),
        _torch_dtype_label(te_compute),
        _torch_dtype_label(vae_storage),
        _torch_dtype_label(vae_compute),
    )

    return AnimaEngineRuntime(
        vae=vae,
        denoiser=denoiser,
        text=text_pipelines,
        qwen=qwen,
        device=device,
        core_storage_dtype=_torch_dtype_label(core_storage),
        core_compute_dtype=_torch_dtype_label(core_compute),
        te_storage_dtype=_torch_dtype_label(te_storage),
        te_compute_dtype=_torch_dtype_label(te_compute),
        vae_storage_dtype=_torch_dtype_label(vae_storage),
        vae_compute_dtype=_torch_dtype_label(vae_compute),
    )


ANIMA_SPEC = AnimaEngineSpec()

__all__ = [
    "ANIMA_SPEC",
    "AnimaEngineRuntime",
    "AnimaEngineSpec",
    "AnimaTextPipelines",
    "assemble_anima_runtime",
]
