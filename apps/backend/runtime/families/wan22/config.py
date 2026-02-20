"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 GGUF runtime config types and small parsing helpers.
Defines the dataclasses used by the WAN22 GGUF runners (RunConfig/StageConfig) and small env-driven knobs, including
geometry validation (e.g. `height/width % 16 == 0`), metadata-derived sampler/scheduler defaults, and strict WAN VAE
config-source contract checks (bundle dir or file+config), plus strict `gguf_sdpa_policy` validation.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_FLOW_MULTIPLIER` (constant): Multiplier applied to shifted sigma to build the model timestep input.
- `StageConfig` (dataclass): Stage-level configuration (stage model selection + sampler/scheduler/steps/cfg/flow_shift + optional LoRA).
- `RunConfig` (dataclass): Full run configuration (geometry, prompts, devices/dtypes, assets, and both stages).
- `_coerce_int` (function): Best-effort coercion of optional values to `int` (returns `None` on failure).
- `_coerce_float` (function): Best-effort coercion of optional values to `float` (returns `None` on failure).
- `_coerce_bool` (function): Best-effort coercion of optional values to `bool` (returns `None` on failure).
- `as_torch_dtype` (function): Parses dtype strings into torch dtypes (with validation).
- `resolve_device_name` (function): Normalizes device names (`cuda`/`cpu`/etc) into runtime-compatible values.
- `resolve_i2v_order` (function): Resolves the image-to-video conditioning channel order policy.
- `resolve_wan_flow_multiplier` (function): Resolves WAN timestep multiplier from scheduler metadata (`num_train_timesteps`).
- `build_wan22_gguf_run_config` (function): Builds a validated GGUF `RunConfig` from a request-like object and its extras mapping (including strict VAE path + config-source validation).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Mapping, Optional

import torch

from .paths import normalize_win_path

WAN_FLOW_MULTIPLIER = 1000.0


@dataclass(frozen=True)
class StageConfig:
    model_dir: str
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]
    flow_shift: float
    lora_path: Optional[str] = None
    lora_weight: Optional[float] = None


@dataclass(frozen=True)
class RunConfig:
    width: int
    height: int
    fps: int
    num_frames: int
    guidance_scale: Optional[float]
    dtype: str
    device: str
    seed: Optional[int] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    init_image: Optional[object] = None
    vae_dir: Optional[str] = None
    vae_config_dir: Optional[str] = None
    text_encoder_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    metadata_dir: Optional[str] = None
    wan_engine_variant: Optional[str] = None  # '5b' | '14b' when provided by API dispatch
    high: Optional[StageConfig] = None
    low: Optional[StageConfig] = None
    # Memory/attention controls (optional)
    sdpa_policy: Optional[str] = None  # 'mem_efficient' | 'flash' | 'math'
    attention_mode: str = "global"  # 'global' | 'sliding'
    attn_chunk_size: Optional[int] = None  # split attention along sequence if set (>0)
    gguf_cache_policy: Optional[str] = None  # 'none' | 'cpu_lru'
    gguf_cache_limit_mb: Optional[int] = None  # MB limit for cpu_lru cache
    log_mem_interval: Optional[int] = None  # log CUDA mem every N steps if >0
    # Aggressive offload controls
    aggressive_offload: bool = True  # legacy switch; see offload_level
    te_device: Optional[str] = None  # 'cuda' | 'cpu' (None = follow cfg.device)
    # New: coarse-grained offload profile (takes precedence over aggressive_offload if provided)
    # 0 = off (keep resident), 1 = light (offload TE/VAE only), 2 = balanced (also clear between stages), 3 = aggressive (current behavior)
    offload_level: Optional[int] = None


def as_torch_dtype(dtype: str) -> torch.dtype:
    key = str(dtype or "").strip().lower()
    if key in {"fp16", "float16", "f16"}:
        return torch.float16
    if key in {"bf16", "bfloat16"}:
        return getattr(torch, "bfloat16", torch.float16)
    if key in {"fp32", "float32", "f32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype!r} (expected fp16/bf16/fp32)")


def resolve_device_name(name: str) -> str:
    raw = str(name or "auto").strip()
    s = raw.lower()

    if s in {"cpu"}:
        return "cpu"

    if s in {"auto", ""}:
        if torch.cuda.is_available():
            return "cuda"
        raise RuntimeError("WAN22: CUDA is not available; set device='cpu' explicitly to force CPU.")

    # Accept explicit CUDA device strings (cuda, cuda:0, etc).
    if s == "gpu" or s.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda" if s == "gpu" else s
        raise RuntimeError(f"WAN22: device={raw!r} requested but CUDA is not available; set device='cpu' explicitly.")

    raise ValueError(f"Unsupported device: {raw!r} (expected 'auto', 'cpu', or 'cuda').")


def resolve_i2v_order() -> str:
    """Return channel order for I2V concatenation.

    - 'lat_first': latents(16) then cond extras (mask4+img16).
    - 'lat_last' : cond extras first then latents(16).
    Defaults to 'lat_first'. (Env overrides removed; payload-driven only.)
    """
    return "lat_first"


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return None
        return float(value)
    except Exception:
        return None


def _coerce_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return None


def resolve_wan_flow_multiplier(metadata_dir: str) -> float:
    from .scheduler import load_wan_scheduler_config

    vendor_dir = str(metadata_dir or "").strip()
    if not vendor_dir:
        raise RuntimeError("WAN22 GGUF: cannot resolve flow multiplier without metadata_dir.")
    vendor_dir = os.path.expanduser(vendor_dir)
    if not os.path.isdir(os.path.join(vendor_dir, "scheduler")):
        parent = os.path.dirname(vendor_dir)
        if parent and os.path.isdir(os.path.join(parent, "scheduler")):
            vendor_dir = parent
    cfg = load_wan_scheduler_config(vendor_dir)
    raw = cfg.get("num_train_timesteps")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise RuntimeError(
            "WAN22 GGUF: scheduler_config.json must define integer 'num_train_timesteps' for flow multiplier."
        )
    if raw <= 0:
        raise RuntimeError(
            f"WAN22 GGUF: scheduler_config.json has invalid num_train_timesteps={raw!r} (expected > 0)."
        )
    return float(raw)


def build_wan22_gguf_run_config(
    *,
    request: Any,
    device: str,
    dtype: str,
    logger: Any = None,
) -> RunConfig:
    """Build a validated WAN22 GGUF RunConfig from a request-like object.

    Contract: this is a pure mapping layer (no implicit fallbacks, no filesystem guessing).

    Expected `request` attrs (via getattr):
    - prompt / negative_prompt
    - width / height / fps / num_frames / steps / guidance_scale / seed
    - sampler / scheduler
    - init_image (img2vid only)
    - extras: mapping that includes WAN GGUF asset paths and stage overrides
    """
    ex_raw = getattr(request, "extras", {}) or {}
    extras: dict[str, Any] = dict(ex_raw) if isinstance(ex_raw, Mapping) else {}

    raw_wan_engine_variant = extras.get("wan_engine_variant")
    wan_engine_variant: str | None = None
    if raw_wan_engine_variant is not None:
        if not isinstance(raw_wan_engine_variant, str):
            raise RuntimeError(
                "WAN22 GGUF: 'wan_engine_variant' must be a string when provided, "
                f"got {type(raw_wan_engine_variant).__name__}.",
            )
        normalized_variant = raw_wan_engine_variant.strip().lower()
        variant_map = {
            "5b": "5b",
            "14b": "14b",
            "wan22_5b": "5b",
            "wan22_14b": "14b",
            "wan22_14b_animate": "14b",
        }
        wan_engine_variant = variant_map.get(normalized_variant)
        if wan_engine_variant is None:
            allowed = ", ".join(sorted(variant_map))
            raise RuntimeError(
                "WAN22 GGUF: invalid 'wan_engine_variant'="
                f"{raw_wan_engine_variant!r}. Allowed: {allowed}.",
            )

    vae_path = str(extras.get("wan_vae_path") or "").strip() or None

    if extras.get("wan_text_encoder_dir"):
        raise ValueError("WAN22: 'wan_text_encoder_dir' is unsupported in sha-only mode; provide 'wan_text_encoder_path' instead.")

    te_path = str(extras.get("wan_text_encoder_path") or "").strip() or None

    meta_dir = None
    if extras.get("wan_metadata_dir"):
        meta_dir = str(extras.get("wan_metadata_dir") or "").strip() or None
    elif extras.get("wan_tokenizer_dir"):
        # Allow providing tokenizer dir; scheduler_config resolution supports parent fallback.
        meta_dir = str(extras.get("wan_tokenizer_dir") or "").strip() or None

    if not te_path:
        raise RuntimeError(
            "WAN22 GGUF requires a text encoder weights file; provide 'wan_text_encoder_path' (resolved from sha selection)."
        )
    if not vae_path:
        raise RuntimeError(
            "WAN22 GGUF requires a VAE bundle directory; provide 'wan_vae_path' (resolved from sha selection)."
        )
    if not meta_dir:
        raise RuntimeError("WAN22 GGUF requires tokenizer metadata; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'.")

    te_path = os.path.expanduser(te_path)
    te_lower = te_path.lower()
    if not (te_lower.endswith(".safetensors") or te_lower.endswith(".gguf")):
        raise RuntimeError("WAN22 GGUF: 'wan_text_encoder_path' must be a '.safetensors' or '.gguf' file, got: %s" % te_path)
    if not os.path.isfile(te_path):
        raise RuntimeError(f"WAN22 GGUF: text encoder weights not found: {te_path}")

    vae_path = os.path.expanduser(vae_path)
    vae_config_dir: str | None = None
    if os.path.isdir(vae_path):
        config_path = os.path.join(vae_path, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError(f"WAN22 GGUF: VAE bundle is missing config.json: {vae_path}")
        weights_candidates = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model.bin",
        )
        if not any(os.path.isfile(os.path.join(vae_path, name)) for name in weights_candidates):
            raise RuntimeError(
                "WAN22 GGUF: VAE bundle is missing weights file "
                f"(expected one of {weights_candidates}) under: {vae_path}"
            )
        vae_config_dir = vae_path
    elif os.path.isfile(vae_path):
        sibling_dir = os.path.dirname(vae_path)
        sibling_config = os.path.join(sibling_dir, "config.json")
        metadata_root = os.path.expanduser(str(meta_dir or ""))
        metadata_candidates = (
            os.path.join(metadata_root, "vae"),
            os.path.join(os.path.dirname(metadata_root), "vae"),
        )
        if os.path.isfile(sibling_config):
            vae_config_dir = sibling_dir
        else:
            for candidate in metadata_candidates:
                if os.path.isfile(os.path.join(candidate, "config.json")):
                    vae_config_dir = candidate
                    break
            if not vae_config_dir:
                raise RuntimeError(
                    "WAN22 GGUF: file VAE path requires config.json at sibling path or metadata repo "
                    f"(missing for VAE file: {vae_path}; checked metadata candidates: {metadata_candidates})."
                )
    else:
        raise RuntimeError(f"WAN22 GGUF: VAE path not found: {vae_path}")

    wh_raw = extras.get("wan_high") if isinstance(extras.get("wan_high"), dict) else None
    wl_raw = extras.get("wan_low") if isinstance(extras.get("wan_low"), dict) else None

    forbidden = ("lightning", "lora_path")
    for stage_name, stage_cfg in (("wan_high", wh_raw), ("wan_low", wl_raw)):
        if not isinstance(stage_cfg, dict):
            continue
        for key in forbidden:
            if stage_cfg.get(key) not in (None, ""):
                if key == "lora_path":
                    raise RuntimeError(
                        f"WAN22 GGUF: '{stage_name}.lora_path' is not supported (use '{stage_name}.lora_sha')."
                    )
                raise RuntimeError(f"WAN22 GGUF: '{stage_name}.{key}' is not supported (use Diffusers path).")

    total_steps = int(getattr(request, "steps", 12) or 12)
    if total_steps < 2:
        raise RuntimeError(f"WAN22 GGUF requires steps >= 2, got: {total_steps}")
    default_cfg = getattr(request, "guidance_scale", None)

    vendor_dir = str(meta_dir or "").strip()
    vendor_dir = os.path.expanduser(vendor_dir)
    if not os.path.isdir(os.path.join(vendor_dir, "scheduler")):
        parent = os.path.dirname(vendor_dir)
        if parent and os.path.isdir(os.path.join(parent, "scheduler")):
            vendor_dir = parent
    model_index_path = os.path.join(vendor_dir, "model_index.json")
    if not os.path.isfile(model_index_path):
        raise RuntimeError(f"WAN22 GGUF: missing model_index.json under: {vendor_dir!r}")
    try:
        model_index = json.loads(open(model_index_path, encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001 - strict decode
        raise RuntimeError(f"WAN22 GGUF: invalid model_index.json under {vendor_dir!r}: {exc}") from exc
    if not isinstance(model_index, dict):
        raise RuntimeError(f"WAN22 GGUF: model_index.json must be a JSON object: {model_index_path}")
    boundary_ratio_raw = model_index.get("boundary_ratio")
    if boundary_ratio_raw is None:
        raise RuntimeError(f"WAN22 GGUF: model_index.json missing boundary_ratio: {model_index_path}")
    try:
        boundary_ratio = float(boundary_ratio_raw)
    except Exception as exc:  # noqa: BLE001 - strict parsing
        raise RuntimeError(f"WAN22 GGUF: invalid boundary_ratio={boundary_ratio_raw!r} in {model_index_path}") from exc
    if not (0.0 < boundary_ratio < 1.0):
        raise RuntimeError(
            f"WAN22 GGUF: boundary_ratio must be in (0,1), got {boundary_ratio} in {model_index_path}"
        )

    from apps.backend.runtime.model_registry.flow_shift import flow_shift_spec_from_repo_dir

    default_flow_shift = flow_shift_spec_from_repo_dir(vendor_dir).resolve()
    hi_flow_shift_override = None
    if isinstance(wh_raw, dict) and wh_raw.get("flow_shift") is not None:
        hi_flow_shift_override = _coerce_float(wh_raw.get("flow_shift"))
        if hi_flow_shift_override is None:
            raise RuntimeError(
                f"WAN22 GGUF: wan_high.flow_shift must be a float, got: {wh_raw.get('flow_shift')!r}"
            )
    lo_flow_shift_override = None
    if isinstance(wl_raw, dict) and wl_raw.get("flow_shift") is not None:
        lo_flow_shift_override = _coerce_float(wl_raw.get("flow_shift"))
        if lo_flow_shift_override is None:
            raise RuntimeError(
                f"WAN22 GGUF: wan_low.flow_shift must be a float, got: {wl_raw.get('flow_shift')!r}"
            )
    if hi_flow_shift_override is not None and lo_flow_shift_override is not None:
        if float(hi_flow_shift_override) != float(lo_flow_shift_override):
            raise RuntimeError(
                "WAN22 GGUF: high/low flow_shift mismatch. "
                f"wan_high.flow_shift={hi_flow_shift_override} wan_low.flow_shift={lo_flow_shift_override}. "
                "Schedule must be continuous."
            )
    if hi_flow_shift_override is not None:
        effective_flow_shift = float(hi_flow_shift_override)
    elif lo_flow_shift_override is not None:
        effective_flow_shift = float(lo_flow_shift_override)
    else:
        effective_flow_shift = float(default_flow_shift)

    hi_steps_override = None
    if isinstance(wh_raw, dict) and wh_raw.get("steps") is not None:
        hi_steps_override = _coerce_int(wh_raw.get("steps"))
        if hi_steps_override is None:
            raise RuntimeError(f"WAN22 GGUF: wan_high.steps must be an int, got: {wh_raw.get('steps')!r}")
    lo_steps_override = None
    if isinstance(wl_raw, dict) and wl_raw.get("steps") is not None:
        lo_steps_override = _coerce_int(wl_raw.get("steps"))
        if lo_steps_override is None:
            raise RuntimeError(f"WAN22 GGUF: wan_low.steps must be an int, got: {wl_raw.get('steps')!r}")

    if hi_steps_override is not None and lo_steps_override is not None:
        default_steps_high = int(hi_steps_override)
        default_steps_low = int(lo_steps_override)
        if default_steps_high < 1 or default_steps_low < 1:
            raise RuntimeError(
                f"WAN22 GGUF: stage steps must be >= 1 (wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
            )
        if (default_steps_high + default_steps_low) != int(total_steps):
            raise RuntimeError(
                "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                f"(request.steps={total_steps} wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
            )
    elif hi_steps_override is not None:
        default_steps_high = int(hi_steps_override)
        default_steps_low = int(total_steps - default_steps_high)
        if default_steps_high < 1 or default_steps_low < 1:
            raise RuntimeError(
                "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                f"(request.steps={total_steps} wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
            )
    elif lo_steps_override is not None:
        default_steps_low = int(lo_steps_override)
        default_steps_high = int(total_steps - default_steps_low)
        if default_steps_high < 1 or default_steps_low < 1:
            raise RuntimeError(
                "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
                f"(request.steps={total_steps} wan_high.steps={default_steps_high} wan_low.steps={default_steps_low})."
            )
    else:
        from .scheduler import infer_high_steps_from_boundary_ratio

        hi_steps = infer_high_steps_from_boundary_ratio(
            total_steps=total_steps,
            boundary_ratio=boundary_ratio,
            vendor_dir=vendor_dir,
            flow_shift=float(effective_flow_shift),
        )
        default_steps_high = int(hi_steps)
        default_steps_low = int(total_steps - hi_steps)

    def _stage_opts(
        raw: dict | None,
        *,
        stage: str,
        default_steps: int,
    ) -> tuple[str, int, Optional[float], Optional[str], Optional[str], Optional[float], Optional[int], Optional[str], Optional[float]]:
        if not isinstance(raw, dict):
            raise RuntimeError(f"WAN22 GGUF requires {stage}.model_dir (resolved from model_sha).")
        model_dir = str(raw.get("model_dir") or "").strip()
        if not model_dir:
            raise RuntimeError(f"WAN22 GGUF requires {stage}.model_dir (resolved from model_sha).")
        model_dir = normalize_win_path(os.path.expanduser(model_dir))
        if not model_dir.lower().endswith(".gguf"):
            raise RuntimeError(f"WAN22 GGUF: {stage} model must be a .gguf file, got: {model_dir}")
        if not os.path.isfile(model_dir):
            raise RuntimeError(f"WAN22 GGUF: {stage} model not found: {model_dir}")

        raw_steps = raw.get("steps")
        steps = _coerce_int(raw_steps)
        if raw_steps is not None and steps is None:
            raise RuntimeError(f"WAN22 GGUF: {stage}.steps must be an int, got: {raw_steps!r}")
        steps = int(steps) if steps is not None else int(default_steps)
        if int(steps) < 1:
            raise RuntimeError(f"WAN22 GGUF: {stage}.steps must be >= 1, got: {steps}")

        raw_cfg_scale = raw.get("cfg_scale")
        if raw_cfg_scale is None:
            cfg_scale = default_cfg
        else:
            cfg_scale = _coerce_float(raw_cfg_scale)
            if cfg_scale is None:
                raise RuntimeError(f"WAN22 GGUF: {stage}.cfg_scale must be a float, got: {raw_cfg_scale!r}")

        raw_sampler = raw.get("sampler")
        if raw_sampler is None:
            sampler = None
        elif isinstance(raw_sampler, str):
            sampler = raw_sampler.strip() or None
        else:
            raise RuntimeError(f"WAN22 GGUF: {stage}.sampler must be a string, got: {raw_sampler!r}")

        raw_scheduler = raw.get("scheduler")
        if raw_scheduler is None:
            scheduler = None
        elif isinstance(raw_scheduler, str):
            scheduler = raw_scheduler.strip() or None
        else:
            raise RuntimeError(f"WAN22 GGUF: {stage}.scheduler must be a string, got: {raw_scheduler!r}")

        raw_flow_shift = raw.get("flow_shift")
        if raw_flow_shift is None:
            flow_shift = None
        else:
            flow_shift = _coerce_float(raw_flow_shift)
            if flow_shift is None:
                raise RuntimeError(f"WAN22 GGUF: {stage}.flow_shift must be a float, got: {raw_flow_shift!r}")

        raw_seed = raw.get("seed")
        if raw_seed is None:
            seed = None
        else:
            seed = _coerce_int(raw_seed)
            if seed is None:
                raise RuntimeError(f"WAN22 GGUF: {stage}.seed must be an int, got: {raw_seed!r}")

        lora_sha = str(raw.get("lora_sha") or "").strip().lower() or None
        lora_weight = _coerce_float(raw.get("lora_weight")) if raw.get("lora_weight") is not None else None
        if lora_weight is not None and not lora_sha:
            raise RuntimeError(f"WAN22 GGUF: {stage}.lora_weight requires {stage}.lora_sha.")
        lora_path = None
        if lora_sha:
            if not re.fullmatch(r"[0-9a-f]{64}", lora_sha):
                raise RuntimeError(f"WAN22 GGUF: {stage}.lora_sha must be sha256 (64 lowercase hex).")
            from apps.backend.inventory.cache import resolve_asset_by_sha

            resolved = resolve_asset_by_sha(lora_sha)
            if not resolved:
                raise RuntimeError(f"WAN22 GGUF: {stage}.lora_sha not found in inventory: {lora_sha}")
            lora_path = normalize_win_path(os.path.expanduser(str(resolved)))
            if not lora_path.lower().endswith(".safetensors"):
                raise RuntimeError(f"WAN22 GGUF: {stage}.lora_sha must resolve to a .safetensors file: {lora_sha}")
            if not os.path.isfile(lora_path):
                raise RuntimeError(f"WAN22 GGUF: {stage} LoRA file not found: {lora_path}")
        return model_dir, steps, cfg_scale, sampler, scheduler, flow_shift, seed, lora_path, lora_weight

    hi_dir, hi_steps, hi_cfg, hi_sampler, hi_scheduler, hi_flow_shift, hi_seed, hi_lora_path, hi_lora_weight = _stage_opts(
        wh_raw, stage="wan_high", default_steps=default_steps_high
    )
    lo_dir, lo_steps, lo_cfg, lo_sampler, lo_scheduler, lo_flow_shift, _lo_seed, lo_lora_path, lo_lora_weight = _stage_opts(
        wl_raw, stage="wan_low", default_steps=default_steps_low
    )

    explicit_stage_steps = bool((wh_raw and wh_raw.get("steps") is not None) or (wl_raw and wl_raw.get("steps") is not None))
    if explicit_stage_steps and (int(hi_steps) + int(lo_steps)) != int(total_steps):
        raise RuntimeError(
            "WAN22 GGUF: stage steps must sum to request.steps for schedule continuity "
            f"(request.steps={total_steps} wan_high.steps={hi_steps} wan_low.steps={lo_steps})."
        )

    hi_flow_shift = effective_flow_shift
    lo_flow_shift = effective_flow_shift

    seed = getattr(request, "seed", None)
    if hi_seed is not None:
        seed = hi_seed

    def _metadata_sampler_scheduler_defaults() -> tuple[str, str]:
        scheduler_dir = os.path.join(vendor_dir, "scheduler")
        config_path = None
        for filename in ("scheduler_config.json", "config.json"):
            candidate = os.path.join(scheduler_dir, filename)
            if os.path.isfile(candidate):
                config_path = candidate
                break
        if not config_path:
            raise RuntimeError(
                "WAN22 GGUF: missing scheduler config under metadata dir "
                f"(expected '{os.path.join(scheduler_dir, 'scheduler_config.json')}' "
                f"or '{os.path.join(scheduler_dir, 'config.json')}')."
            )
        try:
            scheduler_config = json.loads(open(config_path, encoding="utf-8").read())
        except Exception as exc:  # noqa: BLE001 - strict decode
            raise RuntimeError(f"WAN22 GGUF: invalid scheduler config JSON: {config_path}: {exc}") from exc
        if not isinstance(scheduler_config, dict):
            raise RuntimeError(f"WAN22 GGUF: scheduler config must be a JSON object: {config_path}")
        class_name = str(scheduler_config.get("_class_name") or "").strip()
        if class_name == "UniPCMultistepScheduler":
            return ("uni-pc", "simple")
        if not class_name:
            raise RuntimeError(f"WAN22 GGUF: scheduler config missing _class_name: {config_path}")
        raise RuntimeError(
            f"WAN22 GGUF: unsupported metadata scheduler {class_name!r} in {config_path}; "
            "expected UniPCMultistepScheduler."
        )

    metadata_sampler_default, metadata_scheduler_default = _metadata_sampler_scheduler_defaults()

    request_sampler = getattr(request, "sampler", None)
    if request_sampler in (None, ""):
        sampler_fallback = metadata_sampler_default
    elif isinstance(request_sampler, str):
        sampler_fallback = request_sampler.strip() or metadata_sampler_default
    else:
        raise RuntimeError(f"WAN22 GGUF: request.sampler must be a string when provided, got: {request_sampler!r}")
    request_scheduler = getattr(request, "scheduler", None)
    if request_scheduler in (None, ""):
        scheduler_fallback = metadata_scheduler_default
    elif isinstance(request_scheduler, str):
        scheduler_fallback = request_scheduler.strip() or metadata_scheduler_default
    else:
        raise RuntimeError(
            f"WAN22 GGUF: request.scheduler must be a string when provided, got: {request_scheduler!r}"
        )

    tokenizer_dir = str(extras.get("wan_tokenizer_dir") or "").strip() or None

    offload_level_raw = extras.get("gguf_offload_level")
    if offload_level_raw is None:
        offload_level = None
    else:
        offload_level = _coerce_int(offload_level_raw)
        if offload_level is None:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_offload_level' must be an integer when provided, "
                f"got {offload_level_raw!r}."
            )
        if offload_level < 0:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_offload_level' must be >= 0 when provided, "
                f"got {offload_level!r}."
            )

    if logger is not None:
        try:
            logger.info(
                "[wan22.gguf] assets: metadata=%s te=%s vae=%s",
                os.path.basename(str(meta_dir)) if meta_dir else None,
                os.path.basename(str(te_path)) if te_path else None,
                os.path.basename(str(vae_path)) if vae_path else None,
            )
        except Exception:
            pass

    width = int(getattr(request, "width", 768) or 768)
    height = int(getattr(request, "height", 432) or 432)
    if height % 16 != 0 or width % 16 != 0:
        raise RuntimeError(f"WAN22 GGUF: height and width have to be divisible by 16 but are {height} and {width}.")

    aggressive_offload_raw = extras.get("gguf_offload", True)
    aggressive_offload = _coerce_bool(aggressive_offload_raw)
    if aggressive_offload is None:
        raise RuntimeError(
            f"WAN22 GGUF: 'gguf_offload' must be a boolean when provided, got {aggressive_offload_raw!r}."
        )

    if "gguf_te_impl" in extras:
        raise RuntimeError(
            "WAN22 GGUF: 'gguf_te_impl' was removed. WAN22 text-encoder execution is GGUF-only."
        )
    if "gguf_te_kernel_required" in extras:
        raise RuntimeError(
            "WAN22 GGUF: 'gguf_te_kernel_required' was removed. WAN22 text-encoder execution is GGUF-only."
        )

    attention_mode_raw = extras.get("gguf_attention_mode")
    attention_mode: str = "global"
    if attention_mode_raw is not None:
        attention_mode = str(attention_mode_raw).strip().lower()
        if attention_mode not in {"global", "sliding"}:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_attention_mode' must be 'global' or 'sliding' when provided, "
                f"got {attention_mode_raw!r}."
            )

    attn_chunk_size = (
        int(extras.get("gguf_attn_chunk", 0))
        if extras.get("gguf_attn_chunk") not in (None, "", 0)
        else None
    )
    if attention_mode == "sliding" and attn_chunk_size is None:
        attn_chunk_size = 1024

    sdpa_policy_raw = extras.get("gguf_sdpa_policy")
    sdpa_policy: str | None = None
    if sdpa_policy_raw is not None:
        sdpa_policy = str(sdpa_policy_raw).strip().lower()
        if sdpa_policy not in {"mem_efficient", "flash", "math"}:
            raise RuntimeError(
                "WAN22 GGUF: 'gguf_sdpa_policy' must be one of "
                "'mem_efficient', 'flash', or 'math' when provided, "
                f"got {sdpa_policy_raw!r}."
            )

    return RunConfig(
        width=width,
        height=height,
        fps=int(getattr(request, "fps", 24) or 24),
        num_frames=int(getattr(request, "num_frames", 17) or 17),
        guidance_scale=getattr(request, "guidance_scale", None),
        dtype=str(dtype or "fp16"),
        device=str(device or "cuda"),
        seed=seed,
        prompt=getattr(request, "prompt", None),
        negative_prompt=getattr(request, "negative_prompt", None),
        init_image=getattr(request, "init_image", None),
        vae_dir=vae_path,
        vae_config_dir=vae_config_dir,
        text_encoder_dir=te_path,
        tokenizer_dir=tokenizer_dir,
        metadata_dir=meta_dir,
        wan_engine_variant=wan_engine_variant,
        sdpa_policy=sdpa_policy,
        attention_mode=attention_mode,
        attn_chunk_size=attn_chunk_size,
        gguf_cache_policy=(extras.get("gguf_cache_policy") if extras.get("gguf_cache_policy") is not None else None),
        gguf_cache_limit_mb=(
            int(extras.get("gguf_cache_limit_mb", 0)) if extras.get("gguf_cache_limit_mb") not in (None, "", 0) else None
        ),
        log_mem_interval=(
            int(extras.get("gguf_log_mem_interval", 0)) if extras.get("gguf_log_mem_interval") not in (None, "", 0) else None
        ),
        aggressive_offload=aggressive_offload,
        offload_level=offload_level,
        te_device=(str(extras.get("gguf_te_device")).lower() if extras.get("gguf_te_device") is not None else None),
        high=StageConfig(
            model_dir=hi_dir,
            sampler=str(hi_sampler or sampler_fallback),
            scheduler=str(hi_scheduler or scheduler_fallback),
            steps=max(1, int(hi_steps)),
            cfg_scale=hi_cfg,
            flow_shift=float(hi_flow_shift),
            lora_path=hi_lora_path,
            lora_weight=hi_lora_weight,
        ),
        low=StageConfig(
            model_dir=lo_dir,
            sampler=str(lo_sampler or sampler_fallback),
            scheduler=str(lo_scheduler or scheduler_fallback),
            steps=max(1, int(lo_steps)),
            cfg_scale=lo_cfg,
            flow_shift=float(lo_flow_shift),
            lora_path=lo_lora_path,
            lora_weight=lo_lora_weight,
        ),
    )
