"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed SUPIR enhance request config.
Defines a dataclass that covers the SUPIR enhance parameter surface and a strict parser that:
- applies defaults,
- validates types/ranges,
- rejects unsupported combinations.

Symbols (top-level; keep in sync; no ghosts):
- `SupirEnhanceConfig` (dataclass): Parsed SUPIR enhance configuration (payload → typed fields).
- `parse_supir_enhance_config` (function): Parse and validate a config from a JSON payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .errors import SupirConfigError
from .weights import SupirVariant


def _as_int(v: Any, *, name: str) -> int:
    if isinstance(v, bool):
        raise SupirConfigError(f"{name} must be an int, not bool")
    try:
        return int(v)
    except Exception as exc:
        raise SupirConfigError(f"{name} must be an int") from exc


def _as_float(v: Any, *, name: str) -> float:
    if isinstance(v, bool):
        raise SupirConfigError(f"{name} must be a float, not bool")
    try:
        return float(v)
    except Exception as exc:
        raise SupirConfigError(f"{name} must be a float") from exc


def _as_bool(v: Any, *, name: str) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "yes", "on"}:
            return True
        if s in {"0", "false", "no", "off"}:
            return False
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    raise SupirConfigError(f"{name} must be a bool")


def _as_str(v: Any, *, name: str) -> str:
    if v is None:
        return ""
    if not isinstance(v, str):
        raise SupirConfigError(f"{name} must be a string")
    return v


@dataclass(frozen=True)
class SupirEnhanceConfig:
    device: str
    base_model: str
    variant: SupirVariant

    # Target sizing
    scale: float = 2.0
    target_width: Optional[int] = None
    target_height: Optional[int] = None

    # Prompts
    prompt: str = ""
    a_prompt: str = ""
    n_prompt: str = ""

    # Core sampling
    num_samples: int = 1
    steps: int = 50
    sampler: str = "Restore Heun EDM (Stable)"
    cfg_scale: float = 4.0
    seed: int = -1

    # Restore/control
    restoration_scale: float = 4.0
    control_scale: float = 1.0
    restore_steps_hint: int = 0
    restore_gain: float = 1.0
    restore_cfg_s_tmin: float = 0.05
    restore_structure_only: bool = False
    restore_lpf_ksize: int = 0
    control_curve_gamma: float = 1.0

    # Linear schedules (advanced)
    use_linear_cfg: bool = False
    use_linear_control_scale: bool = False
    cfg_scale_start: float = 1.0
    control_scale_start: float = 0.0
    linear_cfg_reverse: bool = False
    linear_control_reverse: bool = False

    # Stochasticity / sampler physics (advanced)
    s_churn: float = 0.0
    s_noise: float = 1.003
    s_tmin: float = 0.0
    s_tmax: float = 50.0
    eta: float = 1.0

    # Preprocess
    gamma_correction: float = 1.0
    pre_upscaler: str = "None"

    # Color fix
    color_fix: str = "None"

    # Dtype/offload
    diff_dtype: str = "fp16"
    ae_dtype: str = "bf16"
    low_vram_clip: bool = True
    extreme_offload: bool = False

    # VAE cache
    use_vae_cache: bool = False
    vae_cache_key: Optional[str] = None

    # Tiling
    use_tile_vae: bool = False
    vae_encoder_tile_size: int = 512
    vae_decoder_tile_size: int = 64
    use_tiled_sampler: bool = False
    tile_size: int = 512
    tile_stride: Optional[int] = None
    tile_overlap: int = 32
    tile_batch_size: int = 1

    # Debug
    use_all_samplers: bool = False


def parse_supir_enhance_config(payload: Mapping[str, Any], *, device: str) -> SupirEnhanceConfig:
    if not isinstance(payload, Mapping):
        raise SupirConfigError("payload must be an object")

    device = _as_str(device, name="device").strip().lower()
    if device not in {"cpu", "cuda", "mps", "xpu", "directml"}:
        raise SupirConfigError("device must be one of: cpu|cuda|mps|xpu|directml")

    base_model = _as_str(payload.get("supir_base_model"), name="supir_base_model").strip()
    if not base_model:
        raise SupirConfigError("Missing 'supir_base_model'")

    variant_raw = _as_str(payload.get("supir_variant"), name="supir_variant").strip()
    try:
        variant = SupirVariant(variant_raw)
    except Exception as exc:
        raise SupirConfigError("Invalid 'supir_variant' (allowed: v0F, v0Q)") from exc

    scale = _as_float(payload.get("supir_scale", 2.0), name="supir_scale")
    if scale <= 0:
        raise SupirConfigError("supir_scale must be > 0")

    target_width = payload.get("supir_target_width")
    target_height = payload.get("supir_target_height")
    w = _as_int(target_width, name="supir_target_width") if target_width not in (None, "") else None
    h = _as_int(target_height, name="supir_target_height") if target_height not in (None, "") else None
    if (w is None) ^ (h is None):
        raise SupirConfigError("supir_target_width and supir_target_height must be set together")
    if w is not None and (w <= 0 or h is None or h <= 0):
        raise SupirConfigError("supir_target_width/height must be > 0")

    steps = _as_int(payload.get("supir_steps", 50), name="supir_steps")
    if steps <= 0:
        raise SupirConfigError("supir_steps must be > 0")

    num_samples = _as_int(payload.get("supir_num_samples", 1), name="supir_num_samples")
    if num_samples <= 0:
        raise SupirConfigError("supir_num_samples must be > 0")

    diff_dtype = _as_str(payload.get("supir_diff_dtype", "fp16"), name="supir_diff_dtype").strip().lower()
    if diff_dtype not in {"fp16", "bf16", "fp32"}:
        raise SupirConfigError("supir_diff_dtype must be one of: fp16|bf16|fp32")

    ae_dtype = _as_str(payload.get("supir_ae_dtype", "bf16"), name="supir_ae_dtype").strip().lower()
    if ae_dtype not in {"bf16", "fp32"}:
        raise SupirConfigError("supir_ae_dtype must be one of: bf16|fp32 (fp16 is not supported)")

    return SupirEnhanceConfig(
        device=device,
        base_model=base_model,
        variant=variant,
        scale=scale,
        target_width=w,
        target_height=h,
        prompt=_as_str(payload.get("supir_prompt", ""), name="supir_prompt"),
        a_prompt=_as_str(payload.get("supir_a_prompt", ""), name="supir_a_prompt"),
        n_prompt=_as_str(payload.get("supir_n_prompt", ""), name="supir_n_prompt"),
        num_samples=num_samples,
        steps=steps,
        sampler=_as_str(payload.get("supir_sampler", "Restore Heun EDM (Stable)"), name="supir_sampler"),
        cfg_scale=_as_float(payload.get("supir_cfg_scale", 4.0), name="supir_cfg_scale"),
        seed=_as_int(payload.get("supir_seed", -1), name="supir_seed"),
        restoration_scale=_as_float(payload.get("supir_restoration_scale", 4.0), name="supir_restoration_scale"),
        control_scale=_as_float(payload.get("supir_control_scale", 1.0), name="supir_control_scale"),
        restore_steps_hint=_as_int(payload.get("supir_restore_steps_hint", 0), name="supir_restore_steps_hint"),
        restore_gain=_as_float(payload.get("supir_restore_gain", 1.0), name="supir_restore_gain"),
        restore_cfg_s_tmin=_as_float(payload.get("supir_restore_cfg_s_tmin", 0.05), name="supir_restore_cfg_s_tmin"),
        restore_structure_only=_as_bool(payload.get("supir_restore_structure_only", False), name="supir_restore_structure_only"),
        restore_lpf_ksize=_as_int(payload.get("supir_restore_lpf_ksize", 0), name="supir_restore_lpf_ksize"),
        control_curve_gamma=_as_float(payload.get("supir_control_curve_gamma", 1.0), name="supir_control_curve_gamma"),
        use_linear_cfg=_as_bool(payload.get("supir_use_linear_cfg", False), name="supir_use_linear_cfg"),
        use_linear_control_scale=_as_bool(payload.get("supir_use_linear_control_scale", False), name="supir_use_linear_control_scale"),
        cfg_scale_start=_as_float(payload.get("supir_cfg_scale_start", 1.0), name="supir_cfg_scale_start"),
        control_scale_start=_as_float(payload.get("supir_control_scale_start", 0.0), name="supir_control_scale_start"),
        linear_cfg_reverse=_as_bool(payload.get("supir_linear_cfg_reverse", False), name="supir_linear_cfg_reverse"),
        linear_control_reverse=_as_bool(payload.get("supir_linear_control_reverse", False), name="supir_linear_control_reverse"),
        s_churn=_as_float(payload.get("supir_s_churn", 0.0), name="supir_s_churn"),
        s_noise=_as_float(payload.get("supir_s_noise", 1.003), name="supir_s_noise"),
        s_tmin=_as_float(payload.get("supir_s_tmin", 0.0), name="supir_s_tmin"),
        s_tmax=_as_float(payload.get("supir_s_tmax", 50.0), name="supir_s_tmax"),
        eta=_as_float(payload.get("supir_eta", 1.0), name="supir_eta"),
        gamma_correction=_as_float(payload.get("supir_gamma_correction", 1.0), name="supir_gamma_correction"),
        pre_upscaler=_as_str(payload.get("supir_pre_upscaler", "None"), name="supir_pre_upscaler"),
        color_fix=_as_str(payload.get("supir_color_fix", "None"), name="supir_color_fix"),
        diff_dtype=diff_dtype,
        ae_dtype=ae_dtype,
        low_vram_clip=_as_bool(payload.get("supir_low_vram_clip", True), name="supir_low_vram_clip"),
        extreme_offload=_as_bool(payload.get("supir_extreme_offload", False), name="supir_extreme_offload"),
        use_vae_cache=_as_bool(payload.get("supir_use_vae_cache", False), name="supir_use_vae_cache"),
        vae_cache_key=(_as_str(payload.get("supir_vae_cache_key"), name="supir_vae_cache_key").strip() or None),
        use_tile_vae=_as_bool(payload.get("supir_use_tile_vae", False), name="supir_use_tile_vae"),
        vae_encoder_tile_size=_as_int(payload.get("supir_vae_encoder_tile_size", 512), name="supir_vae_encoder_tile_size"),
        vae_decoder_tile_size=_as_int(payload.get("supir_vae_decoder_tile_size", 64), name="supir_vae_decoder_tile_size"),
        use_tiled_sampler=_as_bool(payload.get("supir_use_tiled_sampler", False), name="supir_use_tiled_sampler"),
        tile_size=_as_int(payload.get("supir_tile_size", 512), name="supir_tile_size"),
        tile_stride=(_as_int(payload.get("supir_tile_stride"), name="supir_tile_stride") if payload.get("supir_tile_stride") not in (None, "") else None),
        tile_overlap=_as_int(payload.get("supir_tile_overlap", 32), name="supir_tile_overlap"),
        tile_batch_size=_as_int(payload.get("supir_tile_batch_size", 1), name="supir_tile_batch_size"),
        use_all_samplers=_as_bool(payload.get("supir_use_all_samplers", False), name="supir_use_all_samplers"),
    )


__all__ = [
    "SupirEnhanceConfig",
    "parse_supir_enhance_config",
]
