"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima engine specification and runtime assembly (Cosmos Predict2 / Anima adapter).
Assembles a Codex-native runtime from the parsed Anima core bundle plus sha-selected external assets (Qwen3-0.6B TE + WanVAE-style VAE),
producing a denoiser patcher suitable for the canonical txt2img/img2img pipelines (Option A).

Symbols (top-level; keep in sync; no ghosts):
- `AnimaTextPipelines` (dataclass): Text pipeline container (Qwen3 embeddings + offline T5 tokenizer).
- `AnimaEngineRuntime` (dataclass): Assembled runtime container (denoiser + VAE + text pipelines + patchers).
- `AnimaEngineSpec` (dataclass): Engine spec (family defaults + flow shift/multiplier overrides).
- `assemble_anima_runtime` (function): Assemble an Anima runtime from a `DiffusionModelBundle` and engine options.
"""

from __future__ import annotations

import logging
import os
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
from apps.backend.runtime.sampling_adapters.prediction import PredictionDiscreteFlow

logger = logging.getLogger("backend.engines.anima.spec")


def _torch_dtype_label(dtype: torch.dtype) -> str:
    if dtype == torch.float16:
        return "fp16"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float32:
        return "fp32"
    raise ValueError(f"Unsupported torch dtype: {dtype!r}")


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
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
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
    text_encoder = _load_external_text_encoder(tenc_path=tenc_path, torch_dtype=te_storage)

    vae_storage = memory_management.manager.dtype_for_role(DeviceRole.VAE)
    vae_compute = memory_management.manager.compute_dtype_for_role(DeviceRole.VAE, storage_dtype=vae_storage)
    vae_model = _load_external_vae(vae_path=vae_path, torch_dtype=vae_storage)

    # Wrap VAE with shared patcher interface (encode/decode + normalization via family spec fallback).
    vae = VAE(model=vae_model, family=ModelFamily.ANIMA)

    # Text encoder patcher for memory management integration.
    te_load_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
    te_offload_device = memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER)
    qwen_model = getattr(text_encoder, "model", None)
    if not isinstance(qwen_model, torch.nn.Module):
        raise RuntimeError(f"Anima text encoder wrapper missing .model nn.Module; got {type(qwen_model).__name__}")
    qwen = ModelPatcher(qwen_model, load_device=te_load_device, offload_device=te_offload_device)

    # Text pipelines: Qwen embeddings + offline T5 tokenizer.
    from apps.backend.runtime.families.anima.text_encoder import AnimaQwenTextProcessingEngine, load_anima_t5_tokenizer

    qwen_max_length = int(os.getenv("CODEX_ANIMA_QWEN_MAX_LENGTH", "512") or 512)
    if qwen_max_length <= 0:
        raise ValueError("CODEX_ANIMA_QWEN_MAX_LENGTH must be > 0")

    text_engine = AnimaQwenTextProcessingEngine(text_encoder, max_length=qwen_max_length)
    t5_tokenizer = load_anima_t5_tokenizer()
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
