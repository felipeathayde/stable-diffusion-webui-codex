"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampling driver (native-only) for diffusion runtimes.
Selects sampler implementations from specs, compiles conditioning, handles cancellation/precision fallback, and runs the sampling loop
while emitting timeline/diagnostic hooks (and optional global profiling sections via `CODEX_PROFILE`), including native ER-SDE stage updates
with strict runtime option validation (`solver_type`, `max_stage`, `eta`, `s_noise`) and optional guidance policy wiring (APG/rescale/trunc/renorm),
and emits explicit runtime telemetry for console block-progress activation state.

Symbols (top-level; keep in sync; no ghosts):
- `_SamplingCancelled` (exception): Raised when an in-flight sampling run is cancelled (checked via backend state).
- `_raise_if_cancelled` (function): Checks cancellation state and raises `_SamplingCancelled` when requested.
- `_PrecisionFallbackRequest` (exception): Signals the caller to retry sampling with a different precision policy.
- `_resolve_guidance_policy` (function): Resolves and validates optional guidance policy overrides (env + request extras) for APG/rescale/trunc/renorm.
- `CodexSampler` (class): Main sampler driver; builds `SamplingContext`, resolves sampler specs, runs the native sampler loop, and integrates
  memory-management/timeline diagnostics.
"""

from __future__ import annotations

# tags: sampling, diagnostics

from typing import Any, Optional, Callable, List, Mapping
import math
import logging
import os

import torch

from apps.backend.core.rng import ImageRNG, NoiseSettings
from apps.backend.infra.config.env_flags import env_flag, env_int
from apps.backend.infra.config.bootstrap_env import get_bootstrap_env

from .inner_loop import sampling_function_inner, sampling_prepare, sampling_cleanup
from .block_progress import (
    BLOCK_PROGRESS_CALLBACK_KEY,
    RichBlockProgressController,
    validate_block_progress_payload,
)
from .condition import compile_conditions
from .context import SamplingContext, build_sampling_context
from .registry import get_sampler_spec
from ...core.state import state as backend_state
from apps.backend.engines.util.schedulers import SamplerKind
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload_invariants import enforce_smart_offload_pre_sampling_residency
from apps.backend.runtime.diagnostics.timeline import timeline
from apps.backend.runtime.diagnostics.profiler import profiler
from apps.backend.runtime.logging import emit_backend_event

class _SamplingCancelled(Exception):
    """Signal that sampling was cancelled externally."""


def _raise_if_cancelled() -> None:
    if backend_state.should_stop:
        raise _SamplingCancelled("cancelled")



_LMS_COEFFS = {
    1: (1.0,),
    2: (3.0 / 2.0, -1.0 / 2.0),
    3: (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0),
    4: (55.0 / 24.0, -59.0 / 24.0, 37.0 / 24.0, -9.0 / 24.0),
}


class _PrecisionFallbackRequest(Exception):
    """Internal control flow exception used to trigger a sampling retry."""


_GUIDANCE_POLICY_KEY = "codex_guidance_policy"
_GUIDANCE_STEP_INDEX_KEY = "codex_guidance_step_index"
_GUIDANCE_TOTAL_STEPS_KEY = "codex_guidance_total_steps"
_GUIDANCE_APG_MOMENTUM_BUFFER_KEY = "codex_guidance_apg_momentum_buffer"
_GUIDANCE_WARNED_SAMPLER_CFG_KEY = "codex_guidance_sampler_cfg_warned"

_GUIDANCE_ALLOWED_KEYS = {
    "apg_enabled",
    "apg_start_step",
    "apg_eta",
    "apg_momentum",
    "apg_norm_threshold",
    "apg_rescale",
    "guidance_rescale",
    "cfg_trunc_ratio",
    "renorm_cfg",
}

_COMFY_DENOISE_FULL_THRESHOLD = 0.9999
_MAX_COMFY_DENOISE_STEPS = 10000


def _read_env_text(name: str) -> str | None:
    value = get_bootstrap_env(name)
    if value is None:
        value = os.getenv(name)
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _read_env_float(name: str, default: float) -> float:
    text = _read_env_text(name)
    if text is None:
        return float(default)
    try:
        value = float(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} must be a float; got: {text!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite; got: {text!r}")
    return value


def _read_env_bool(name: str, default: bool) -> bool:
    text = _read_env_text(name)
    if text is None:
        return bool(default)
    normalized = text.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be a boolean (1/0/true/false/yes/no/on/off); got: {text!r}")


def _read_env_nonnegative_int(name: str, default: int) -> int:
    text = _read_env_text(name)
    if text is None:
        return int(default)
    try:
        value = int(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} must be an integer >= 0; got: {text!r}") from exc
    if value < 0:
        raise RuntimeError(f"{name} must be >= 0; got: {text!r}")
    return value


def _read_env_optional_float(name: str) -> float | None:
    text = _read_env_text(name)
    if text is None:
        return None
    try:
        value = float(text)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"{name} must be a float; got: {text!r}") from exc
    if not math.isfinite(value):
        raise RuntimeError(f"{name} must be finite; got: {text!r}")
    return value


def _resolve_guidance_policy(processing: Any) -> dict[str, Any] | None:
    policy: dict[str, Any] = {
        "apg_enabled": _read_env_bool("CODEX_GUIDANCE_APG_ENABLED", default=False),
        "apg_start_step": _read_env_nonnegative_int("CODEX_GUIDANCE_APG_START_STEP", default=0),
        "apg_eta": _read_env_float("CODEX_GUIDANCE_APG_ETA", default=0.0),
        "apg_momentum": _read_env_float("CODEX_GUIDANCE_APG_MOMENTUM", default=0.0),
        "apg_norm_threshold": _read_env_float("CODEX_GUIDANCE_APG_NORM_THRESHOLD", default=15.0),
        "apg_rescale": _read_env_float("CODEX_GUIDANCE_APG_RESCALE", default=0.0),
        "guidance_rescale": _read_env_float("CODEX_GUIDANCE_RESCALE", default=0.0),
        "cfg_trunc_ratio": _read_env_optional_float("CODEX_GUIDANCE_CFG_TRUNC_RATIO"),
        "renorm_cfg": _read_env_float("CODEX_GUIDANCE_RENORM_CFG", default=0.0),
    }

    overrides = getattr(processing, "override_settings", {})
    if isinstance(overrides, Mapping):
        guidance_override = overrides.get("guidance")
        if guidance_override is not None:
            if not isinstance(guidance_override, Mapping):
                raise RuntimeError(
                    "override_settings.guidance must be an object when provided "
                    f"(got {type(guidance_override).__name__})."
                )
            unknown = sorted(str(key) for key in guidance_override.keys() if str(key) not in _GUIDANCE_ALLOWED_KEYS)
            if unknown:
                raise RuntimeError(
                    "Unexpected override_settings.guidance key(s): "
                    + ", ".join(unknown)
                )
            for key in _GUIDANCE_ALLOWED_KEYS:
                if key not in guidance_override:
                    continue
                raw_value = guidance_override[key]
                if key == "apg_enabled":
                    if not isinstance(raw_value, bool):
                        raise RuntimeError("override_settings.guidance.apg_enabled must be boolean.")
                    policy[key] = raw_value
                    continue
                if key == "apg_start_step":
                    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                        raise RuntimeError("override_settings.guidance.apg_start_step must be an integer >= 0.")
                    if isinstance(raw_value, float) and not raw_value.is_integer():
                        raise RuntimeError("override_settings.guidance.apg_start_step must be an integer >= 0.")
                    value_int = int(raw_value)
                    if value_int < 0:
                        raise RuntimeError("override_settings.guidance.apg_start_step must be >= 0.")
                    policy[key] = value_int
                    continue
                if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                    raise RuntimeError(f"override_settings.guidance.{key} must be numeric.")
                value_float = float(raw_value)
                if not math.isfinite(value_float):
                    raise RuntimeError(f"override_settings.guidance.{key} must be finite.")
                policy[key] = value_float

    apg_enabled = bool(policy.get("apg_enabled", False))
    apg_start_step = int(policy.get("apg_start_step", 0) or 0)
    apg_eta = float(policy.get("apg_eta", 0.0) or 0.0)
    apg_momentum = float(policy.get("apg_momentum", 0.0) or 0.0)
    apg_norm_threshold = float(policy.get("apg_norm_threshold", 0.0) or 0.0)
    apg_rescale = float(policy.get("apg_rescale", 0.0) or 0.0)
    guidance_rescale = float(policy.get("guidance_rescale", 0.0) or 0.0)
    cfg_trunc_ratio = policy.get("cfg_trunc_ratio")
    renorm_cfg = float(policy.get("renorm_cfg", 0.0) or 0.0)

    if apg_start_step < 0:
        raise RuntimeError(f"Invalid guidance apg_start_step={apg_start_step}; expected >= 0.")
    if apg_momentum < 0.0 or apg_momentum >= 1.0:
        raise RuntimeError(f"Invalid guidance apg_momentum={apg_momentum}; expected range [0, 1).")
    if apg_norm_threshold < 0.0:
        raise RuntimeError(f"Invalid guidance apg_norm_threshold={apg_norm_threshold}; expected >= 0.")
    if guidance_rescale < 0.0 or guidance_rescale > 1.0:
        raise RuntimeError(f"Invalid guidance guidance_rescale={guidance_rescale}; expected range [0, 1].")
    if apg_rescale < 0.0 or apg_rescale > 1.0:
        raise RuntimeError(f"Invalid guidance apg_rescale={apg_rescale}; expected range [0, 1].")
    if renorm_cfg < 0.0:
        raise RuntimeError(f"Invalid guidance renorm_cfg={renorm_cfg}; expected >= 0.")
    if cfg_trunc_ratio is not None:
        cfg_trunc_ratio = float(cfg_trunc_ratio)
        if cfg_trunc_ratio < 0.0 or cfg_trunc_ratio > 1.0:
            raise RuntimeError(f"Invalid guidance cfg_trunc_ratio={cfg_trunc_ratio}; expected range [0, 1].")

    active = (
        apg_enabled
        or guidance_rescale > 0.0
        or apg_rescale > 0.0
        or renorm_cfg > 0.0
        or cfg_trunc_ratio is not None
    )
    if not active:
        return None

    return {
        "apg_enabled": apg_enabled,
        "apg_start_step": apg_start_step,
        "apg_eta": apg_eta,
        "apg_momentum": apg_momentum,
        "apg_norm_threshold": apg_norm_threshold,
        "apg_rescale": apg_rescale,
        "guidance_rescale": guidance_rescale,
        "cfg_trunc_ratio": cfg_trunc_ratio,
        "renorm_cfg": renorm_cfg,
    }

class CodexSampler:
    """Minimal native sampler for txt2img using an Euler ODE loop.

    - Uses the model's predictor to derive a sigma schedule from sigma_max→sigma_min.
    - Calls `sampling_function_inner` (CFG and condition assembly) each step.
    - Updates `backend_state` for progress reporting.
    - No ancestral noise yet; samplers other than Euler will be added iteratively.
    """

    def __init__(self, sd_model: Any, *, algorithm: str | None = None) -> None:
        self.sd_model = sd_model
        self.algorithm = (algorithm or "euler a").strip().lower()
        self._logger = logging.getLogger(__name__ + ".CodexSampler")
        self._log_enabled = env_flag("CODEX_LOG_SAMPLER", default=False)
        self._log_sigmas = env_flag("CODEX_LOG_SIGMAS", default=False)

    def _emit_event(self, event: str, /, **fields: object) -> None:
        emit_backend_event(event, logger=self._logger.name, **fields)

    @staticmethod
    def _compact_series(values: list[float]) -> str:
        if not values:
            return "none"
        return "/".join(f"{value:.6g}" for value in values)

    def _summarize_sigmas(self, sigmas: torch.Tensor, *, window: int = 6) -> str:
        try:
            values = [float(v) for v in sigmas.detach().cpu().tolist()]
        except Exception:
            return "<unavailable>"
        if len(values) <= window * 2:
            return ",".join(f"{v:.6g}" for v in values)
        head = ",".join(f"{v:.6g}" for v in values[:window])
        tail = ",".join(f"{v:.6g}" for v in values[-window:])
        return f"{head},...,{tail}"

    @staticmethod
    def _normalize_denoise_strength(denoise_strength: float | None) -> float | None:
        if denoise_strength is None:
            return None
        if isinstance(denoise_strength, bool):
            raise ValueError("denoise_strength must be a float in [0, 1]")
        try:
            value = float(denoise_strength)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"denoise_strength must be numeric; got {type(denoise_strength).__name__}") from exc
        if not math.isfinite(value):
            raise ValueError("denoise_strength must be finite")
        if value < 0.0 or value > 1.0:
            raise ValueError("denoise_strength must be in [0, 1]")
        return value

    def _build_comfy_denoise_sigmas(
        self,
        *,
        processing: Any,
        model: Any,
        active_context: SamplingContext,
        noise: torch.Tensor,
        steps: int,
        denoise_strength: float,
    ) -> torch.Tensor:
        if denoise_strength > _COMFY_DENOISE_FULL_THRESHOLD:
            return active_context.sigmas.to(device=noise.device, dtype=torch.float32)

        new_steps = int(float(steps) / float(denoise_strength))
        if new_steps < 1:
            raise RuntimeError(
                f"Comfy denoise schedule produced invalid new_steps={new_steps} from steps={steps} denoise={denoise_strength}"
            )
        if new_steps > _MAX_COMFY_DENOISE_STEPS:
            raise ValueError(
                f"denoise_strength={denoise_strength} expands schedule to new_steps={new_steps}, "
                f"above safety limit {_MAX_COMFY_DENOISE_STEPS}"
            )
        denoise_context = build_sampling_context(
            self.sd_model,
            sampler_name=self.algorithm,
            scheduler_name=active_context.scheduler_name,
            steps=new_steps,
            noise_source=active_context.noise_settings.source.value,
            eta_noise_seed_delta=active_context.noise_settings.eta_noise_seed_delta,
            height=(int(getattr(processing, "height", 0) or 0) or None),
            width=(int(getattr(processing, "width", 0) or 0) or None),
            device=noise.device,
            dtype=noise.dtype,
            predictor=model,
            is_sdxl=bool(getattr(getattr(self.sd_model, "engine", None), "is_sdxl", False)),
        )
        denoise_sigmas = denoise_context.sigmas.to(device=noise.device, dtype=torch.float32)
        required = int(steps) + 1
        if denoise_sigmas.ndim != 1:
            raise RuntimeError(f"Comfy denoise schedule must be 1D; got shape={tuple(denoise_sigmas.shape)}")
        if int(denoise_sigmas.numel()) < required:
            raise RuntimeError(
                f"Comfy denoise schedule too short: got={int(denoise_sigmas.numel())} required={required} "
                f"(steps={steps} denoise={denoise_strength} new_steps={new_steps})"
            )
        tail = denoise_sigmas[-required:]
        if self._log_enabled:
            self._emit_event(
                "sampling.denoise_schedule",
                steps=int(steps),
                denoise=float(denoise_strength),
                new_steps=int(new_steps),
                selected=int(required),
                first=float(tail[0].item()),
                last=float(tail[-1].item()),
            )
        return tail

    def _rebind_unet_precision(self, dtype: torch.dtype) -> None:
        denoiser = self.sd_model.codex_objects.denoiser
        model = getattr(denoiser, "model", None)
        if model is None:
            return
        previous = getattr(model, "computation_dtype", None)
        if hasattr(model, "computation_dtype"):
            model.computation_dtype = dtype
        diffusion_model = getattr(model, "diffusion_model", None)
        if diffusion_model is not None:
            diffusion_model.to(dtype=dtype)
        self._logger.info(
            "Diffusion core precision updated: %s -> %s",
            str(previous),
            str(dtype),
        )

    @staticmethod
    def _normalize_er_sde_solver_type(value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("ER-SDE option 'solver_type' must be a string")
        normalized = value.strip().lower().replace("-", " ").replace("_", " ")
        solver_map = {
            "er sde": "er_sde",
            "reverse time sde": "reverse_time_sde",
            "ode": "ode",
        }
        solver_type = solver_map.get(normalized)
        if solver_type is None:
            raise ValueError(
                "ER-SDE option 'solver_type' must be one of: ER-SDE, Reverse-time SDE, ODE"
            )
        return solver_type

    @staticmethod
    def _resolve_er_sde_runtime_params(er_sde_options: Any) -> dict[str, Any]:
        allowed_keys = {"solver_type", "max_stage", "eta", "s_noise"}
        if er_sde_options is None:
            payload: dict[str, Any] = {}
        elif isinstance(er_sde_options, dict):
            raw_payload = dict(er_sde_options)
            unknown_keys = sorted(set(raw_payload.keys()) - allowed_keys)
            if unknown_keys:
                raise ValueError(f"Unexpected ER-SDE option key(s): {', '.join(unknown_keys)}")
            payload = {key: value for key, value in raw_payload.items() if value is not None}
        else:
            attr_keys: set[str] = set()
            if hasattr(er_sde_options, "__dict__"):
                attr_keys.update(str(key) for key in vars(er_sde_options).keys())
            slots_value = getattr(type(er_sde_options), "__slots__", ())
            if isinstance(slots_value, str):
                slots_iterable = (slots_value,)
            else:
                slots_iterable = tuple(slots_value) if isinstance(slots_value, (tuple, list)) else ()
            attr_keys.update(str(key) for key in slots_iterable)
            unknown_keys = sorted(
                key for key in attr_keys if key and not key.startswith("_") and key not in allowed_keys
            )
            if unknown_keys:
                raise ValueError(f"Unexpected ER-SDE option key(s): {', '.join(unknown_keys)}")
            payload = {}
            for key in allowed_keys:
                value = getattr(er_sde_options, key, None)
                if value is not None:
                    payload[key] = value

        solver_raw = payload.get("solver_type", "er_sde")
        solver_type = CodexSampler._normalize_er_sde_solver_type(solver_raw)

        max_stage_raw = payload.get("max_stage", 3)
        if isinstance(max_stage_raw, bool) or not isinstance(max_stage_raw, (int, float)):
            raise ValueError("ER-SDE option 'max_stage' must be an integer in [1, 3]")
        if isinstance(max_stage_raw, float) and not max_stage_raw.is_integer():
            raise ValueError("ER-SDE option 'max_stage' must be an integer in [1, 3]")
        max_stage = int(max_stage_raw)
        if max_stage < 1 or max_stage > 3:
            raise ValueError("ER-SDE option 'max_stage' must be in [1, 3]")

        eta_raw = payload.get("eta", 1.0)
        if isinstance(eta_raw, bool) or not isinstance(eta_raw, (int, float)):
            raise ValueError("ER-SDE option 'eta' must be numeric")
        eta = float(eta_raw)
        if not math.isfinite(eta):
            raise ValueError("ER-SDE option 'eta' must be finite")
        if eta < 0.0:
            raise ValueError("ER-SDE option 'eta' must be >= 0")

        s_noise_raw = payload.get("s_noise", 1.0)
        if isinstance(s_noise_raw, bool) or not isinstance(s_noise_raw, (int, float)):
            raise ValueError("ER-SDE option 's_noise' must be numeric")
        s_noise = float(s_noise_raw)
        if not math.isfinite(s_noise):
            raise ValueError("ER-SDE option 's_noise' must be finite")
        if s_noise < 0.0:
            raise ValueError("ER-SDE option 's_noise' must be >= 0")

        if solver_type == "ode" or (solver_type == "reverse_time_sde" and eta == 0.0):
            eta = 0.0
            s_noise = 0.0

        return {
            "solver_type": solver_type,
            "max_stage": max_stage,
            "eta": eta,
            "s_noise": s_noise,
        }

    @staticmethod
    def _er_sde_noise_scaler(
        values: torch.Tensor,
        *,
        solver_type: str,
        eta: float,
    ) -> torch.Tensor:
        if solver_type == "er_sde":
            return values * (torch.exp(values.pow(0.3)) + 10.0)
        return values.pow(eta + 1.0)

    @staticmethod
    def _compute_er_sde_lambdas(sigmas: torch.Tensor, *, prediction_type: str | None) -> torch.Tensor:
        if sigmas.ndim != 1:
            raise RuntimeError(f"ER-SDE expects a 1D sigma schedule, got shape={tuple(sigmas.shape)}")
        sigmas_fp32 = sigmas.to(dtype=torch.float32)
        if prediction_type == "const":
            sigma_safe = sigmas_fp32.clamp(min=1e-6, max=1.0 - 1e-6)
            half_log_snr = -torch.logit(sigma_safe)
        else:
            sigma_safe = sigmas_fp32.clamp(min=1e-12)
            half_log_snr = -torch.log(sigma_safe)
        return torch.exp(-half_log_snr)

    @staticmethod
    def _clone_noise_settings(settings: NoiseSettings) -> NoiseSettings:
        return NoiseSettings(
            source=settings.source,
            eta_noise_seed_delta=int(settings.eta_noise_seed_delta or 0),
            force_device=settings.force_device,
        )

    @staticmethod
    def _resolve_processing_seed_list(processing: Any, *, batch_size: int) -> list[int]:
        seed_values = list(getattr(processing, "all_seeds", []) or []) or list(getattr(processing, "seeds", []) or [])
        if not seed_values:
            seed_value = int(getattr(processing, "seed", -1))
            if seed_value < 0:
                raise RuntimeError(
                    "Deterministic ancestral noise requires explicit per-sample seeds; processing is missing seeds."
                )
            if batch_size != 1:
                raise RuntimeError(
                    "Deterministic ancestral noise requires per-sample seeds for batched runs; "
                    f"got batch_size={batch_size} with only processing.seed."
                )
            seed_values = [seed_value]
        normalized = [int(value) for value in seed_values]
        if len(normalized) != batch_size:
            raise RuntimeError(
                "Deterministic ancestral noise seed count mismatch: "
                f"got seeds={len(normalized)} batch_size={batch_size}."
            )
        return normalized

    def _build_seeded_ancestral_rng(
        self,
        *,
        processing: Any,
        active_context: SamplingContext,
        noise: torch.Tensor,
    ) -> ImageRNG:
        batch_size = int(noise.shape[0])
        latent_shape = tuple(int(dim) for dim in noise.shape[1:])
        template_rng = getattr(processing, "rng", None)
        rng_target_device = noise.device
        if isinstance(template_rng, ImageRNG):
            template_shape = tuple(int(dim) for dim in template_rng.shape)
            if template_shape != latent_shape:
                raise RuntimeError(
                    "processing.rng shape mismatch for deterministic ancestral noise: "
                    f"rng_shape={template_shape} noise_shape={latent_shape}."
                )
            seeds = [int(seed) for seed in template_rng.seeds]
            if len(seeds) != batch_size:
                raise RuntimeError(
                    "processing.rng seed count mismatch for deterministic ancestral noise: "
                    f"rng_seeds={len(seeds)} batch_size={batch_size}."
                )
            subseeds = [int(seed) for seed in template_rng.subseeds]
            subseed_strength = float(template_rng.subseed_strength)
            seed_resize_from_h = int(template_rng.seed_resize_from_h)
            seed_resize_from_w = int(template_rng.seed_resize_from_w)
            settings = self._clone_noise_settings(template_rng.settings)
            rng_target_device = template_rng.device
        else:
            seeds = self._resolve_processing_seed_list(processing, batch_size=batch_size)
            subseeds = [int(seed) for seed in (getattr(processing, "all_subseeds", []) or []) or (getattr(processing, "subseeds", []) or [])]
            subseed_strength = float(getattr(processing, "subseed_strength", 0.0) or 0.0)
            seed_resize_from_h = int(getattr(processing, "seed_resize_from_h", 0) or 0)
            seed_resize_from_w = int(getattr(processing, "seed_resize_from_w", 0) or 0)
            settings = self._clone_noise_settings(active_context.noise_settings)

        # Clone the shared ImageRNG policy; `core.rng` drives determinism through
        # `torch.Generator(...).manual_seed(...)` / Philox and applies `eta_noise_seed_delta`
        # immediately after the initial latent noise draw.
        step_rng = ImageRNG(
            latent_shape,
            seeds,
            subseeds=subseeds,
            subseed_strength=subseed_strength,
            seed_resize_from_h=seed_resize_from_h,
            seed_resize_from_w=seed_resize_from_w,
            settings=settings,
            device=rng_target_device,
        )
        initial_noise = step_rng.next()
        if tuple(initial_noise.shape) != tuple(noise.shape):
            raise RuntimeError(
                "Deterministic ancestral noise bootstrap shape mismatch: "
                f"bootstrap={tuple(initial_noise.shape)} noise={tuple(noise.shape)}."
            )
        return step_rng

    @staticmethod
    def _next_seeded_ancestral_noise(step_rng: ImageRNG, reference: torch.Tensor) -> torch.Tensor:
        sampled = step_rng.next()
        if tuple(sampled.shape) != tuple(reference.shape):
            raise RuntimeError(
                "Deterministic ancestral noise shape mismatch: "
                f"sampled={tuple(sampled.shape)} reference={tuple(reference.shape)}."
            )
        return sampled.to(device=reference.device, dtype=reference.dtype)

    @torch.no_grad()
    def sample(
        self,
        processing: Any,
        noise: torch.Tensor,
        cond: Any,
        uncond: Optional[Any],
        image_conditioning: Optional[torch.Tensor] = None,
        *,
        init_latent: Optional[torch.Tensor] = None,
        start_at_step: int | None = None,
        denoise_strength: float | None = None,
        pre_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]] = None,
        post_denoiser_hook: Optional[Callable[[torch.Tensor, torch.Tensor, int, int], torch.Tensor]] = None,
        preview_callback: Optional[Callable[[torch.Tensor, int, int], None]] = None,
        post_step_hook: Optional[Callable[[torch.Tensor, int, int], None]] = None,
        post_sample_hook: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        context: SamplingContext | None = None,
        er_sde_options: Any = None,
    ) -> torch.Tensor:
        base_noise = noise.detach()
        base_context = context
        warned_full_preview = False

        spec = get_sampler_spec(self.algorithm)

        while True:
            denoiser = self.sd_model.codex_objects.denoiser
            model = denoiser.model

            steps = int(getattr(processing, "steps", 20))
            cfg_scale = float(getattr(processing, "cfg_scale", 7.0))
            
            # Flux Dev uses distilled CFG via guidance embedding - disable traditional CFG
            # to avoid double scaling which corrupts the output (sand texture artifact)
            if hasattr(self.sd_model, "use_distilled_cfg_scale") and self.sd_model.use_distilled_cfg_scale:
                if cfg_scale != 1.0:
                    self._logger.info("[flux] Distilled CFG active: forcing cfg_scale=1.0 (was %.2f)", cfg_scale)
                cfg_scale = 1.0

            if steps <= 0:
                raise ValueError("steps must be >= 1")
            if noise.ndim != 4:
                raise ValueError(f"noise must be BCHW; got shape={tuple(noise.shape)}")

            target_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE)
            noise = base_noise.to(dtype=target_dtype)

            if init_latent is not None and init_latent.dtype != noise.dtype:
                init_latent = init_latent.to(dtype=noise.dtype)
            normalized_denoise = self._normalize_denoise_strength(denoise_strength)
            if normalized_denoise is not None and math.isclose(normalized_denoise, 0.0):
                if self._log_enabled:
                    self._emit_event("sampling.denoise_noop", reason="denoise_zero", has_init_latent=init_latent is not None)
                samples_noop = init_latent if init_latent is not None else torch.zeros_like(noise)
                if post_sample_hook is not None:
                    samples_noop = post_sample_hook(samples_noop)
                return samples_noop

            block_progress_controller: RichBlockProgressController | None = None
            retry = False
            prepared = False
            state_started = False
            active_context = base_context
            guidance_policy: dict[str, Any] | None = None

            try:
                allow_vae_resident = False
                preview_interval = 0
                try:
                    if base_context is not None:
                        preview_interval = max(0, int(getattr(base_context, "preview_interval", 0) or 0))
                    else:
                        from apps.backend.runtime.live_preview import preview_interval_steps

                        preview_interval = preview_interval_steps(default=0)
                except Exception:
                    preview_interval = 0

                if preview_callback is not None and preview_interval > 0:
                    try:
                        from apps.backend.runtime.live_preview import LivePreviewMethod, live_preview_method

                        method = live_preview_method(default=LivePreviewMethod.FULL)
                        allow_vae_resident = method == LivePreviewMethod.FULL
                        if allow_vae_resident and not warned_full_preview:
                            self._logger.warning(
                                "Live preview FULL uses VAE decode during sampling. "
                                "This increases VRAM usage and can reduce generation performance; "
                                "prefer 'Approx cheap' when possible."
                            )
                            warned_full_preview = True
                    except Exception:
                        allow_vae_resident = False

                enforce_smart_offload_pre_sampling_residency(
                    self.sd_model,
                    stage="sampling.prepare",
                    allow_vae_resident=allow_vae_resident,
                )

                sampling_prepare(denoiser, noise)
                prepared = True

                scheduler_name = getattr(processing, "scheduler", None)
                if scheduler_name in (None, ""):
                    scheduler_name = spec.default_scheduler
                if not isinstance(scheduler_name, str) or not scheduler_name:
                    raise ValueError("Scheduler name must be a non-empty string")
                if not spec.is_supported_scheduler(scheduler_name):
                    raise ValueError(
                        f"Scheduler '{scheduler_name}' is not supported by sampler '{spec.name}'. "
                        f"Allowed: {sorted(spec.allowed_schedulers)}"
                    )
                if active_context is None:
                    active_context = build_sampling_context(
                        self.sd_model,
                        sampler_name=self.algorithm,
                        scheduler_name=scheduler_name,
                        steps=steps,
                        noise_source=getattr(processing, "noise_source", None),
                        eta_noise_seed_delta=int(getattr(processing, "eta_noise_seed_delta", 0) or 0),
                        height=(int(getattr(processing, "height", 0) or 0) or None),
                        width=(int(getattr(processing, "width", 0) or 0) or None),
                        device=noise.device,
                        dtype=noise.dtype,
                        predictor=model,
                        is_sdxl=bool(getattr(getattr(self.sd_model, "engine", None), "is_sdxl", False)),
                    )

                # Keep sigma ladder in fp32 for numeric stability; casting to bf16/fp16
                # quantizes the schedule and can produce severe quality regressions.
                steps = int(active_context.steps)
                if normalized_denoise is None:
                    sigmas = active_context.sigmas.to(device=noise.device, dtype=torch.float32)
                else:
                    sigmas = self._build_comfy_denoise_sigmas(
                        processing=processing,
                        model=model,
                        active_context=active_context,
                        noise=noise,
                        steps=steps,
                        denoise_strength=normalized_denoise,
                    )

                if sigmas.ndim != 1 or int(sigmas.numel()) < 2:
                    raise RuntimeError(f"sigma schedule must be 1D with at least 2 entries; got shape={tuple(sigmas.shape)}")
                if int(sigmas.numel()) != steps + 1:
                    raise RuntimeError(
                        f"sigma schedule length mismatch: got {int(sigmas.numel())}, expected {steps + 1} (steps={steps})"
                    )

                if self._log_sigmas or self._log_enabled:
                    schedule_first = float(sigmas[0]) if len(sigmas) > 0 else float("nan")
                    schedule_last = float(sigmas[-1]) if len(sigmas) > 0 else float("nan")
                    schedule_summary = self._summarize_sigmas(sigmas)
                    sigma_min_val = float("nan") if active_context.sigma_min is None else float(active_context.sigma_min)
                    sigma_max_val = float("nan") if active_context.sigma_max is None else float(active_context.sigma_max)
                    self._emit_event(
                        "sampling.sigma_schedule",
                        length=len(sigmas) - 1,
                        predict_min=sigma_min_val,
                        predict_max=sigma_max_val,
                        first=schedule_first,
                        last=schedule_last,
                        ladder=schedule_summary,
                    )

                start_idx = int(start_at_step or 0)
                start_idx = max(0, min(start_idx, steps - 1))
                sigmas_run = sigmas[start_idx:]
                if init_latent is not None:
                    x = init_latent + float(sigmas_run[0]) * noise
                else:
                    # Keep x in the core dtype (bf16/fp16) while preserving the sigma ladder precision.
                    sigma0 = sigmas_run[:1].to(dtype=noise.dtype)
                    x = model.predictor.noise_scaling(sigma0, noise, torch.zeros_like(noise))

                if self._log_enabled:
                    try:
                        smax = float(sigmas[0].item()) if hasattr(sigmas[0], "item") else float(sigmas[0])
                        smin = float(sigmas[-1].item()) if hasattr(sigmas[-1], "item") else float(sigmas[-1])
                        head = [float(v) for v in sigmas[: min(4, len(sigmas))].detach().cpu().tolist()]
                    except Exception:
                        smax = float("nan")
                        smin = float("nan")
                        head = []
                    pred_type = getattr(model.predictor, "prediction_type", None)
                    sigma_data = getattr(model.predictor, "sigma_data", None)
                    self._emit_event(
                        "sampling.plan.prepare",
                        algorithm=self.algorithm,
                        scheduler=active_context.scheduler_name,
                        steps=steps,
                        cfg_scale=float(cfg_scale),
                        prediction=pred_type or getattr(active_context, "prediction_type", None) or "<unknown>",
                        sigma_max=smax,
                        sigma_min=smin,
                        sigma_data=float(sigma_data) if sigma_data is not None else "n/a",
                        head=self._compact_series(head),
                    )

                compiled_cond = compile_conditions(cond)
                compiled_uncond = compile_conditions(uncond) if uncond is not None else None
                log_cfg_delta = False
                cfg_delta_steps = 0
                if self._log_enabled:
                    log_cfg_delta = env_flag("CODEX_LOG_CFG_DELTA", default=False)
                    if log_cfg_delta:
                        cfg_delta_steps = env_int("CODEX_LOG_CFG_DELTA_N", default=2, min_value=0)

                if isinstance(image_conditioning, torch.Tensor):
                    if (
                        image_conditioning.shape[0] == noise.shape[0]
                        and image_conditioning.shape[2] == noise.shape[2]
                        and image_conditioning.shape[3] == noise.shape[3]
                    ):
                        from .condition import Condition

                        for entry in compiled_cond:
                            entry["model_conds"]["c_concat"] = Condition(image_conditioning)
                        if compiled_uncond is not None:
                            for entry in compiled_uncond:
                                entry["model_conds"]["c_concat"] = Condition(image_conditioning)

                run_total_steps = steps - start_idx
                guidance_policy = _resolve_guidance_policy(processing)
                if guidance_policy is None:
                    denoiser.model_options.pop(_GUIDANCE_POLICY_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_STEP_INDEX_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_TOTAL_STEPS_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
                    denoiser.model_options.pop(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, None)
                else:
                    denoiser.model_options[_GUIDANCE_POLICY_KEY] = guidance_policy
                    denoiser.model_options[_GUIDANCE_TOTAL_STEPS_KEY] = run_total_steps
                    denoiser.model_options.pop(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, None)
                    self._emit_event(
                        "guidance.policy",
                        apg_enabled=bool(guidance_policy.get("apg_enabled", False)),
                        start_step=int(guidance_policy.get("apg_start_step", 0) or 0),
                        cfg_trunc_ratio=guidance_policy.get("cfg_trunc_ratio"),
                        guidance_rescale=float(guidance_policy.get("guidance_rescale", 0.0) or 0.0),
                        apg_rescale=float(guidance_policy.get("apg_rescale", 0.0) or 0.0),
                        renorm_cfg=float(guidance_policy.get("renorm_cfg", 0.0) or 0.0),
                    )
                backend_state.start(job_count=1, sampling_steps=run_total_steps)
                state_started = True
                transformer_options = denoiser.model_options.get("transformer_options", None)
                if not isinstance(transformer_options, dict):
                    raise RuntimeError(
                        "denoiser.model_options['transformer_options'] must be a dict for block progress wiring "
                        f"(got {type(transformer_options).__name__})."
                    )
                block_progress_controller = RichBlockProgressController(enabled=active_context.enable_progress)
                console_block_progress_active = bool(getattr(block_progress_controller, "is_active", False))
                if self._log_enabled:
                    self._emit_event(
                        "sampling.block_progress.console",
                        enabled=console_block_progress_active,
                        env_flag="CODEX_PROGRESS_BAR",
                    )

                def _on_block_progress(block_index: int, total_blocks: int) -> None:
                    normalized_index, normalized_total = validate_block_progress_payload(
                        block_index=block_index,
                        total_blocks=total_blocks,
                    )

                    backend_state.update_sampling_block(
                        block_index=normalized_index,
                        total_blocks=normalized_total,
                    )
                    if block_progress_controller is not None:
                        block_progress_controller.update(
                            block_index=normalized_index,
                            total_blocks=normalized_total,
                        )

                transformer_options[BLOCK_PROGRESS_CALLBACK_KEY] = _on_block_progress
                backend_state.reset_sampling_blocks()

                strict = True
                import time as _time

                preview_interval = active_context.preview_interval
                t0 = _time.perf_counter()

                sampler_kind = active_context.sampler_kind
                prediction_type = getattr(active_context, "prediction_type", None)
                if prediction_type is None:
                    prediction_type = getattr(getattr(model, "predictor", None), "prediction_type", None)
                if isinstance(prediction_type, str):
                    prediction_type = prediction_type.lower()
                profile_meta = {
                    "algorithm": self.algorithm,
                    "sampler_kind": sampler_kind.value,
                    "scheduler": active_context.scheduler_name,
                    "steps": run_total_steps,
                    "cfg_scale": float(cfg_scale),
                    "device": str(noise.device),
                    "noise_dtype": str(noise.dtype),
                    "x_dtype": str(x.dtype),
                    "model_compute_dtype": str(getattr(model, "computation_dtype", None)),
                }
                profile_name = f"sampling-{sampler_kind.value}-{active_context.scheduler_name}"

                if self._log_enabled:
                    head = []
                    try:
                        head = [float(v) for v in sigmas_run[: min(4, len(sigmas_run))].detach().cpu().tolist()]
                    except Exception:
                        head = []
                    self._emit_event(
                        "sampling.plan.run",
                        algorithm=sampler_kind.value,
                        scheduler=active_context.scheduler_name,
                        steps=run_total_steps,
                        cfg_scale=float(cfg_scale),
                        head=self._compact_series(head),
                    )

                old_denoised: Optional[torch.Tensor] = None
                old_denoised_d: Optional[torch.Tensor] = None
                t_prev: float | None = None
                h_prev: float | None = None
                eps_history: List[torch.Tensor] = []
                er_sde_params: dict[str, Any] | None = None
                er_sde_lambdas: torch.Tensor | None = None
                er_sde_point_indices: torch.Tensor | None = None
                ancestral_step_rng: ImageRNG | None = None
                if sampler_kind is SamplerKind.ER_SDE:
                    er_sde_params = self._resolve_er_sde_runtime_params(er_sde_options)
                    er_sde_lambdas = self._compute_er_sde_lambdas(
                        sigmas_run,
                        prediction_type=getattr(active_context, "prediction_type", None),
                    )
                    er_sde_point_indices = torch.arange(
                        0.0,
                        200.0,
                        dtype=torch.float32,
                        device=x.device,
                    )
                if sampler_kind is SamplerKind.EULER_A:
                    ancestral_step_rng = self._build_seeded_ancestral_rng(
                        processing=processing,
                        active_context=active_context,
                        noise=base_noise,
                    )
                    for skip_index in range(start_idx):
                        if float(sigmas[skip_index + 1]) > 0.0:
                            ancestral_step_rng.next()

                with profiler.profile_run(profile_name, meta=profile_meta):
                    for i in range(start_idx, steps):
                        if backend_state.should_stop:
                            raise _SamplingCancelled("cancelled")
                        step_index = i - start_idx
                        backend_state.reset_sampling_blocks()
                        if guidance_policy is not None:
                            denoiser.model_options[_GUIDANCE_STEP_INDEX_KEY] = step_index
                        with profiler.section(f"sampling.step/{step_index + 1}"):
                            sigma = sigmas[i]
                            sigma_next = sigmas[i + 1]
                            sigma_batch = torch.full((x.shape[0],), float(sigma), device=x.device, dtype=torch.float32)
                            current_step = step_index + 1

                            if pre_denoiser_hook is not None:
                                x = pre_denoiser_hook(x, sigma_batch, current_step, run_total_steps)
                                if not isinstance(x, torch.Tensor):
                                    raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                if tuple(x.shape) != tuple(noise.shape):
                                    raise RuntimeError(
                                        "pre_denoiser_hook returned unexpected shape "
                                        f"{tuple(x.shape)}; expected {tuple(noise.shape)}"
                                    )

                            if log_cfg_delta and (i - start_idx) < cfg_delta_steps:
                                denoised, cond_pred, uncond_pred = sampling_function_inner(
                                    model,
                                    x,
                                    sigma_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=True,
                                )
                                cfg1_optimization = math.isclose(cfg_scale, 1.0) and not denoiser.model_options.get(
                                    "disable_cfg1_optimization", False
                                )
                                if compiled_uncond is None or cfg1_optimization:
                                    self._emit_event(
                                        "sampling.cfg_delta",
                                        step=i + 1,
                                        total_steps=steps,
                                        sigma=float(sigma),
                                        cfg_scale=float(cfg_scale),
                                        uncond_used=False,
                                    )
                                else:
                                    try:
                                        delta_abs_mean = float((cond_pred - uncond_pred).detach().float().abs().mean().item())
                                    except Exception:
                                        delta_abs_mean = float("nan")
                                    self._emit_event(
                                        "sampling.cfg_delta",
                                        step=i + 1,
                                        total_steps=steps,
                                        sigma=float(sigma),
                                        cfg_scale=float(cfg_scale),
                                        delta_abs_mean=delta_abs_mean,
                                    )
                            else:
                                denoised = sampling_function_inner(
                                    model,
                                    x,
                                    sigma_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=False,
                                )

                            if post_denoiser_hook is not None:
                                denoised = post_denoiser_hook(denoised, sigma_batch, current_step, run_total_steps)
                                if not isinstance(denoised, torch.Tensor):
                                    raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                if tuple(denoised.shape) != tuple(x.shape):
                                    raise RuntimeError(
                                        "post_denoiser_hook returned unexpected shape "
                                        f"{tuple(denoised.shape)}; expected {tuple(x.shape)}"
                                    )

                            eps = (x - denoised) / max(float(sigma), 1e-8)
                            if strict and (torch.isnan(eps).any() or torch.isnan(denoised).any()):
                                reason = f"NaN detected at sampling step {i + 1}"
                                self._logger.warning(
                                    "NaN encountered at step %d with dtype=%s; attempting precision fallback.",
                                    i + 1,
                                    str(getattr(model, "computation_dtype", x.dtype)),
                                )
                                next_dtype = memory_management.manager.report_precision_failure(
                                    DeviceRole.CORE,
                                    location=f"sampler.step_{i + 1}",
                                    reason=reason,
                                )
                                if next_dtype is None:
                                    hint = memory_management.manager.precision_hint(DeviceRole.CORE)
                                    raise RuntimeError(
                                        f"Diffusion core produced NaNs at step {i + 1} on {noise.device} with dtype {getattr(model, 'computation_dtype', x.dtype)}. {hint}"
                                    )
                                self._rebind_unet_precision(next_dtype)
                                retry = True
                                raise _PrecisionFallbackRequest

                            eps_history.append(eps.detach())
                            if len(eps_history) > 4:
                                eps_history.pop(0)

                            if sampler_kind is SamplerKind.EULER:
                                x = x - (float(sigma) - float(sigma_next)) * eps
                            elif sampler_kind is SamplerKind.EULER_A:
                                sigma = float(sigma)
                                sigma_next = float(sigma_next)
                                if prediction_type == "const":
                                    if sigma_next <= 0.0:
                                        x = denoised
                                    else:
                                        if sigma <= 0.0:
                                            raise RuntimeError(
                                                "Euler ancestral RF/CONST requires sigma > 0 before the terminal step."
                                            )
                                        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * 1.0
                                        sigma_down = sigma_next * downstep_ratio
                                        alpha_ip1 = 1.0 - sigma_next
                                        alpha_down = 1.0 - sigma_down
                                        if abs(alpha_down) <= 1e-12:
                                            raise RuntimeError(
                                                "Euler ancestral RF/CONST produced alpha_down=0; cannot compute renoise term."
                                            )
                                        renoise_sq = sigma_next**2 - sigma_down**2 * alpha_ip1**2 / alpha_down**2
                                        if renoise_sq < -1e-12:
                                            raise RuntimeError(
                                                "Euler ancestral RF/CONST produced negative renoise variance."
                                            )
                                        renoise_coeff = max(renoise_sq, 0.0) ** 0.5
                                        sigma_down_i_ratio = sigma_down / sigma
                                        x = sigma_down_i_ratio * x + (1.0 - sigma_down_i_ratio) * denoised
                                        if ancestral_step_rng is None:
                                            raise RuntimeError("Euler ancestral RF/CONST missing deterministic noise RNG.")
                                        x = (
                                            (alpha_ip1 / alpha_down) * x
                                            + self._next_seeded_ancestral_noise(ancestral_step_rng, x) * 1.0 * renoise_coeff
                                        )
                                elif sigma_next <= 0.0:
                                    x = denoised
                                else:
                                    sigma_up_sq = max(sigma_next**2 * (sigma**2 - sigma_next**2) / max(sigma**2, 1e-8), 0.0)
                                    sigma_up = sigma_up_sq ** 0.5
                                    sigma_down = (max(sigma_next**2 - sigma_up_sq, 0.0)) ** 0.5
                                    x = denoised + sigma_down * eps
                                    if ancestral_step_rng is None:
                                        raise RuntimeError("Euler ancestral missing deterministic noise RNG.")
                                    noise = self._next_seeded_ancestral_noise(ancestral_step_rng, x)
                                    x = x + sigma_up * noise
                            elif sampler_kind is SamplerKind.DPM2M:
                                # DPM-Solver++(2M) in log-sigma time (reference update form).
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                    t_prev = None
                                else:
                                    t = -math.log(max(sigma_f, 1e-12))
                                    t_next = -math.log(max(sigma_next_f, 1e-12))
                                    h = t_next - t
                                    if old_denoised is None or t_prev is None:
                                        x = (sigma_next_f / sigma_f) * x - math.expm1(-h) * denoised
                                    else:
                                        h_last = t - t_prev
                                        r = h_last / h if abs(h) > 1e-12 else 1.0
                                        denoised_d = (1.0 + 1.0 / (2.0 * r)) * denoised - (1.0 / (2.0 * r)) * old_denoised
                                        x = (sigma_next_f / sigma_f) * x - math.expm1(-h) * denoised_d
                                    old_denoised = denoised.detach()
                                    t_prev = t
                            elif sampler_kind is SamplerKind.DPM2M_SDE:
                                # DPM-Solver++(2M) SDE (midpoint) in log-sigma time.
                                # This is a conservative native approximation (no BrownianTree),
                                # but matches the core update form.
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                    h_prev = None
                                else:
                                    t = -math.log(max(sigma_f, 1e-12))
                                    s = -math.log(max(sigma_next_f, 1e-12))
                                    h = s - t
                                    eta = 1.0
                                    s_noise = 1.0
                                    eta_h = eta * h
                                    phi = -math.expm1(-h - eta_h)  # 1 - exp(-h-eta*h)
                                    x = (sigma_next_f / sigma_f) * math.exp(-eta_h) * x + phi * denoised
                                    if old_denoised is not None and h_prev is not None:
                                        r = h_prev / h if abs(h) > 1e-12 else 1.0
                                        x = x + 0.5 * phi * (1.0 / r) * (denoised - old_denoised)
                                    if eta != 0.0:
                                        noise_scale = sigma_next_f * math.sqrt(max(-math.expm1(-2.0 * eta_h), 0.0)) * s_noise
                                        x = x + torch.randn_like(x) * noise_scale
                                    old_denoised = denoised.detach()
                                    h_prev = h
                            elif sampler_kind is SamplerKind.ER_SDE:
                                if er_sde_params is None or er_sde_lambdas is None or er_sde_point_indices is None:
                                    raise RuntimeError("ER-SDE runtime state is not initialized")
                                sigma_f = float(sigma)
                                sigma_next_f = float(sigma_next)
                                if sigma_next_f <= 0.0:
                                    x = denoised
                                    old_denoised = denoised.detach()
                                else:
                                    solver_type = str(er_sde_params["solver_type"])
                                    eta = float(er_sde_params["eta"])
                                    s_noise = float(er_sde_params["s_noise"])
                                    max_stage = int(er_sde_params["max_stage"])
                                    stage_used = min(max_stage, step_index + 1)

                                    er_lambda_s = er_sde_lambdas[step_index]
                                    er_lambda_t = er_sde_lambdas[step_index + 1]
                                    er_lambda_s_f = float(er_lambda_s)
                                    er_lambda_t_f = float(er_lambda_t)
                                    if (
                                        not math.isfinite(er_lambda_s_f)
                                        or not math.isfinite(er_lambda_t_f)
                                        or er_lambda_s_f <= 0.0
                                        or er_lambda_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE received invalid lambda values: {er_lambda_s_f}, {er_lambda_t_f}"
                                        )

                                    alpha_s = sigma_f / er_lambda_s_f
                                    alpha_t = sigma_next_f / er_lambda_t_f
                                    if (
                                        not math.isfinite(alpha_s)
                                        or not math.isfinite(alpha_t)
                                        or alpha_s <= 0.0
                                        or alpha_t <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE received invalid alpha values: {alpha_s}, {alpha_t}"
                                        )

                                    r_alpha = alpha_t / alpha_s
                                    noise_scale_s = self._er_sde_noise_scaler(
                                        er_lambda_s,
                                        solver_type=solver_type,
                                        eta=eta,
                                    )
                                    noise_scale_t = self._er_sde_noise_scaler(
                                        er_lambda_t,
                                        solver_type=solver_type,
                                        eta=eta,
                                    )
                                    noise_scale_s_f = float(noise_scale_s)
                                    noise_scale_t_f = float(noise_scale_t)
                                    if (
                                        not math.isfinite(noise_scale_s_f)
                                        or not math.isfinite(noise_scale_t_f)
                                        or noise_scale_s_f <= 0.0
                                        or noise_scale_t_f <= 0.0
                                    ):
                                        raise RuntimeError(
                                            f"ER-SDE noise scaler returned invalid values: {noise_scale_s_f}, {noise_scale_t_f}"
                                        )
                                    r = noise_scale_t_f / noise_scale_s_f
                                    if not math.isfinite(r):
                                        raise RuntimeError(f"ER-SDE produced non-finite ratio r={r}")

                                    # Stage 1 (Euler)
                                    x = r_alpha * r * x + alpha_t * (1.0 - r) * denoised

                                    if stage_used >= 2:
                                        if old_denoised is None:
                                            raise RuntimeError("ER-SDE stage-2 requires previous denoised state")
                                        dt = er_lambda_t_f - er_lambda_s_f
                                        lambda_step_size = -dt / 200.0
                                        lambda_pos = er_lambda_t + er_sde_point_indices * lambda_step_size
                                        scaled_pos = self._er_sde_noise_scaler(
                                            lambda_pos,
                                            solver_type=solver_type,
                                            eta=eta,
                                        )
                                        if not bool(torch.all(torch.isfinite(scaled_pos))):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-finite scaled positions at step={step_index + 1}"
                                            )
                                        if bool(torch.any(scaled_pos <= 0.0)):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-positive scaled positions at step={step_index + 1}"
                                            )
                                        s_term = float(torch.sum(1.0 / scaled_pos) * lambda_step_size)
                                        if not math.isfinite(s_term):
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 produced non-finite integral term at step={step_index + 1}"
                                            )
                                        prev_gap = er_lambda_s_f - float(er_sde_lambdas[step_index - 1])
                                        if abs(prev_gap) <= 1e-12:
                                            raise RuntimeError(
                                                f"ER-SDE stage-2 denominator collapsed at step={step_index + 1}"
                                            )
                                        denoised_d = (denoised - old_denoised) / prev_gap
                                        x = x + alpha_t * (dt + s_term * noise_scale_t_f) * denoised_d

                                        if stage_used >= 3:
                                            if old_denoised_d is None:
                                                raise RuntimeError("ER-SDE stage-3 requires previous stage-2 state")
                                            s_u = float(
                                                torch.sum((lambda_pos - er_lambda_s) / scaled_pos) * lambda_step_size
                                            )
                                            if not math.isfinite(s_u):
                                                raise RuntimeError(
                                                    f"ER-SDE stage-3 produced non-finite integral term at step={step_index + 1}"
                                                )
                                            stage3_gap = (er_lambda_s_f - float(er_sde_lambdas[step_index - 2])) / 2.0
                                            if abs(stage3_gap) <= 1e-12:
                                                raise RuntimeError(
                                                    f"ER-SDE stage-3 denominator collapsed at step={step_index + 1}"
                                                )
                                            denoised_u = (denoised_d - old_denoised_d) / stage3_gap
                                            x = x + alpha_t * ((dt**2) / 2.0 + s_u * noise_scale_t_f) * denoised_u
                                        old_denoised_d = denoised_d.detach()

                                    if s_noise > 0.0:
                                        noise_term = er_lambda_t_f**2 - er_lambda_s_f**2 * (r**2)
                                        if noise_term < -1e-8:
                                            raise RuntimeError(
                                                f"ER-SDE produced negative stochastic term at step={step_index + 1}: {noise_term}"
                                            )
                                        noise_scale = math.sqrt(max(noise_term, 0.0))
                                        if not math.isfinite(noise_scale):
                                            raise RuntimeError(
                                                f"ER-SDE produced non-finite stochastic scale at step={step_index + 1}"
                                            )
                                        x = x + (
                                            alpha_t
                                            * torch.randn_like(x)
                                            * s_noise
                                            * noise_scale
                                        )
                                    old_denoised = denoised.detach()
                            elif sampler_kind is SamplerKind.DDIM:
                                x = denoised + float(sigma_next) * eps
                            elif sampler_kind in (SamplerKind.PLMS, SamplerKind.PNDM):
                                order = min(len(eps_history), 4)
                                coeffs = _LMS_COEFFS[order]
                                derivative = torch.zeros_like(x)
                                for idx_coeff, coeff in enumerate(coeffs):
                                    derivative = derivative + coeff * eps_history[-(idx_coeff + 1)]
                                delta = float(sigma) - float(sigma_next)
                                x = x - delta * derivative
                            elif sampler_kind is SamplerKind.UNI_PC:
                                delta = float(sigma) - float(sigma_next)
                                x_pred = x - delta * eps
                                sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=torch.float32)
                                if pre_denoiser_hook is not None:
                                    x_pred = pre_denoiser_hook(x_pred, sigma_next_batch, current_step, run_total_steps)
                                    if not isinstance(x_pred, torch.Tensor):
                                        raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                    if tuple(x_pred.shape) != tuple(x.shape):
                                        raise RuntimeError(
                                            "pre_denoiser_hook returned unexpected shape "
                                            f"{tuple(x_pred.shape)}; expected {tuple(x.shape)}"
                                        )
                                denoised_next = sampling_function_inner(
                                    model,
                                    x_pred,
                                    sigma_next_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=False,
                                )
                                if post_denoiser_hook is not None:
                                    denoised_next = post_denoiser_hook(denoised_next, sigma_next_batch, current_step, run_total_steps)
                                    if not isinstance(denoised_next, torch.Tensor):
                                        raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                    if tuple(denoised_next.shape) != tuple(x_pred.shape):
                                        raise RuntimeError(
                                            "post_denoiser_hook returned unexpected shape "
                                            f"{tuple(denoised_next.shape)}; expected {tuple(x_pred.shape)}"
                                        )
                                eps_next = (x_pred - denoised_next) / max(float(sigma_next), 1e-8)
                                x = x - delta * 0.5 * (eps + eps_next)
                            elif sampler_kind is SamplerKind.UNI_PC_BH2:
                                # Reuse the UniPC two-stage update as a BH2 variant placeholder.
                                delta = float(sigma) - float(sigma_next)
                                x_pred = x - delta * eps
                                sigma_next_batch = torch.full((x.shape[0],), float(sigma_next), device=x.device, dtype=torch.float32)
                                if pre_denoiser_hook is not None:
                                    x_pred = pre_denoiser_hook(x_pred, sigma_next_batch, current_step, run_total_steps)
                                    if not isinstance(x_pred, torch.Tensor):
                                        raise RuntimeError("pre_denoiser_hook must return a torch.Tensor")
                                    if tuple(x_pred.shape) != tuple(x.shape):
                                        raise RuntimeError(
                                            "pre_denoiser_hook returned unexpected shape "
                                            f"{tuple(x_pred.shape)}; expected {tuple(x.shape)}"
                                        )
                                denoised_next = sampling_function_inner(
                                    model,
                                    x_pred,
                                    sigma_next_batch,
                                    compiled_uncond,
                                    compiled_cond,
                                    cfg_scale,
                                    denoiser.model_options,
                                    seed=None,
                                    return_full=False,
                                )
                                if post_denoiser_hook is not None:
                                    denoised_next = post_denoiser_hook(denoised_next, sigma_next_batch, current_step, run_total_steps)
                                    if not isinstance(denoised_next, torch.Tensor):
                                        raise RuntimeError("post_denoiser_hook must return a torch.Tensor")
                                    if tuple(denoised_next.shape) != tuple(x_pred.shape):
                                        raise RuntimeError(
                                            "post_denoiser_hook returned unexpected shape "
                                            f"{tuple(denoised_next.shape)}; expected {tuple(x_pred.shape)}"
                                        )
                                eps_next = (x_pred - denoised_next) / max(float(sigma_next), 1e-8)
                                x = x - delta * 0.5 * (eps + eps_next)
                            else:
                                raise NotImplementedError(f"Sampler '{sampler_kind.value}' is not implemented natively yet")

                            if post_step_hook is not None:
                                post_step_hook(x, current_step, run_total_steps)

                            if preview_callback is not None and (
                                (preview_interval > 0 and (current_step % preview_interval == 0))
                                or current_step == run_total_steps
                            ):
                                try:
                                    preview_callback(denoised.detach(), current_step, run_total_steps)
                                except Exception:
                                    pass

                            if self._log_enabled and (
                                i == 0 or (i + 1) == steps or (i + 1) % max(1, steps // 5) == 0
                            ):
                                eps_norm = float(eps.norm().item()) if hasattr(eps, "norm") else float("nan")
                                den_norm = float(denoised.norm().item()) if hasattr(denoised, "norm") else float("nan")
                                self._emit_event(
                                    "sampling.step",
                                    step=i + 1,
                                    total_steps=steps,
                                    sigma=float(sigma),
                                    sigma_next=float(sigma_next),
                                    norm_x=float(x.norm().item()),
                                    norm_eps=eps_norm,
                                    norm_den=den_norm,
                                    dt_ms=(_time.perf_counter() - t0) * 1000.0,
                                )
                                t0 = _time.perf_counter()

                            backend_state.tick(sampling_step=current_step)
                            backend_state.reset_sampling_blocks()
                        profiler.step()

                sampling_cleanup(denoiser)
                prepared = False

                backend_state.end()
                state_started = False

                if post_sample_hook is not None:
                    x = post_sample_hook(x)

                return x
            except _PrecisionFallbackRequest:
                self._logger.warning("Precision fallback requested for diffusion core; retrying with next dtype.")
            except _SamplingCancelled:
                self._logger.info("Sampling cancelled by request; aborting current run.")
                raise RuntimeError("cancelled")
            finally:
                if block_progress_controller is not None:
                    block_progress_controller.close()
                    block_progress_controller = None
                if prepared:
                    sampling_cleanup(denoiser)
                if state_started:
                    backend_state.end()
                denoiser.model_options.pop(_GUIDANCE_POLICY_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_STEP_INDEX_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_TOTAL_STEPS_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_APG_MOMENTUM_BUFFER_KEY, None)
                denoiser.model_options.pop(_GUIDANCE_WARNED_SAMPLER_CFG_KEY, None)
                transformer_options = denoiser.model_options.get("transformer_options", None)
                if isinstance(transformer_options, dict):
                    transformer_options.pop(BLOCK_PROGRESS_CALLBACK_KEY, None)
                backend_state.clear_flags()

            if retry:
                memory_management.manager.soft_empty_cache(force=True)
                continue



__all__ = ["CodexSampler"]
