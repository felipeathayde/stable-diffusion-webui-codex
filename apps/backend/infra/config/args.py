"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend CLI argument parsing and runtime memory config bootstrap.
Builds the argparse schema for runtime flags (devices/dtypes/attention/swap/smart offload) and turns argv/env into a `RuntimeMemoryConfig`.
Supports separate storage vs compute dtype overrides for core/text encoder/VAE (e.g., `--core-dtype` vs `--core-compute-dtype`) for stability and tuning.
Also parses diagnostics bootstrap toggles (`--trace-contract`, `--trace-profiler`) for runtime trace/profiler activation.

Symbols (top-level; keep in sync; no ghosts):
- `_build_parser` (function): Defines the argparse schema for backend runtime flags (devices/dtypes/attention/swap/etc).
- `_truthy` (function): Parses a string env/arg into a boolean (truthy/falsey).
- `_has_value` (function): Checks whether a parsed CLI option has a meaningful value (vs unset/default).
- `_apply_source_overrides` (function): Applies overrides from a source mapping onto the argparse namespace.
- `_validate_runtime_flags` (function): Validates behavior-changing runtime flag combinations (GGUF exec modes, online LoRA math).
- `_validate_required_devices` (function): Ensures device defaults are configured (prompts in foreground TTY; raises in non-interactive).
- `_normalize_device_choice` (function): Normalizes device choice strings (e.g., cpu/cuda/directml) into canonical form.
- `_normalize_dtype_choice` (function): Normalizes dtype choice strings (fp32/fp16/bf16/fp8) into canonical form.
- `_torch_dtype_for_choice` (function): Maps a dtype choice string to a torch dtype name (string form used across config objects).
- `_apply_component_device_overrides` (function): Applies per-component device overrides (core/vae/text encoders) to `RuntimeMemoryConfig`.
- `_apply_env_overrides` (function): Applies environment-variable overrides onto parsed args.
- `_resolve_attention_backend` (function): Resolves attention backend selection into `AttentionBackend`.
- `build_runtime_memory_config` (function): Builds a `RuntimeMemoryConfig` from parsed args (includes validation + defaults).
- `initialize` (function): Entry-point helper; parses argv/env and returns the built runtime config (used by launchers).
"""

import argparse
import logging
import os
import sys
from typing import Mapping, MutableMapping, Sequence

from .lora_apply_mode import DEFAULT_LORA_APPLY_MODE, ENV_LORA_APPLY_MODE, LoraApplyMode, parse_lora_apply_mode
from .gguf_exec_mode import DEFAULT_GGUF_EXEC_MODE, GgufExecMode
from .lora_online_math import DEFAULT_LORA_ONLINE_MATH, LoraOnlineMath

from apps.backend.runtime.memory.config import (
    AttentionBackend,
    AttentionConfig,
    DeviceBackend,
    DeviceRole,
    MemoryBudgets,
    PrecisionFlags,
    RuntimeMemoryConfig,
    SwapConfig,
    SwapMethod,
    SwapPolicy,
)

_LOG = logging.getLogger("backend.infra.config.args")
TRACE_DEBUG_DEFAULT = 10


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-device-id", type=int, default=None, metavar="DEVICE_ID")

    fp_group = parser.add_mutually_exclusive_group()
    fp_group.add_argument("--all-in-fp16", action="store_true")

    fpcore_group = parser.add_mutually_exclusive_group()
    fpcore_group.add_argument("--core-in-bf16", action="store_true")
    fpcore_group.add_argument("--core-in-fp16", action="store_true")
    fpcore_group.add_argument("--core-in-fp8-e4m3fn", action="store_true")
    fpcore_group.add_argument("--core-in-fp8-e5m2", action="store_true")

    fpvae_group = parser.add_mutually_exclusive_group()
    fpvae_group.add_argument("--vae-in-fp16", action="store_true")
    fpvae_group.add_argument("--vae-in-fp32", action="store_true")
    fpvae_group.add_argument("--vae-in-bf16", action="store_true")

    parser.add_argument("--vae-in-cpu", action="store_true")

    fpte_group = parser.add_mutually_exclusive_group()
    fpte_group.add_argument("--clip-in-fp8-e4m3fn", action="store_true")
    fpte_group.add_argument("--clip-in-fp8-e5m2", action="store_true")
    fpte_group.add_argument("--clip-in-fp16", action="store_true")
    fpte_group.add_argument("--clip-in-fp32", action="store_true")

    attn_group = parser.add_mutually_exclusive_group()
    attn_group.add_argument("--attention-split", action="store_true")
    attn_group.add_argument("--attention-quad", action="store_true")
    attn_group.add_argument("--attention-pytorch", action="store_true")
    attn_group.add_argument(
        "--attention-backend",
        choices=["pytorch", "xformers", "split", "quad"],
        default=None,
        help=(
            "Attention backend selection (restart required). "
            "Use 'pytorch' for Torch SDPA, 'xformers' for xFormers attention, "
            "'split' for chunked attention, or 'quad' for sub-quadratic attention."
        ),
    )

    upcast = parser.add_mutually_exclusive_group()
    upcast.add_argument("--force-upcast-attention", action="store_true")
    upcast.add_argument("--disable-attention-upcast", action="store_true")

    parser.add_argument("--disable-xformers", action="store_true")
    parser.add_argument(
        "--smart-offload",
        action="store_true",
        help="Load TE/UNet/VAE to GPU only for the active stage, offloading between steps.",
    )

    parser.add_argument("--directml", type=int, nargs="?", metavar="DIRECTML_DEVICE", const=-1)
    parser.add_argument("--disable-ipex-hijack", action="store_true")

    vram_group = parser.add_mutually_exclusive_group()
    vram_group.add_argument("--always-gpu", action="store_true")
    vram_group.add_argument("--always-high-vram", action="store_true")
    vram_group.add_argument("--always-normal-vram", action="store_true")
    vram_group.add_argument("--always-low-vram", action="store_true")
    vram_group.add_argument("--always-no-vram", action="store_true")
    vram_group.add_argument("--always-cpu", action="store_true")

    parser.add_argument("--always-offload-from-vram", action="store_true")
    parser.add_argument("--pytorch-deterministic", action="store_true")

    parser.add_argument("--cuda-malloc", action="store_true")
    parser.add_argument("--cuda-stream", action="store_true")
    parser.add_argument("--pin-shared-memory", action="store_true")

    parser.add_argument("--disable-gpu-warning", action="store_true")

    parser.add_argument("--disable-online-tokenizer", action="store_true")

    parser.add_argument(
        "--gguf-exec",
        choices=[m.value for m in GgufExecMode],
        default=None,
        help=(
            "GGUF execution mode: "
            "'dequant_forward' (default) keeps GGUF tensors byte-packed and dequantizes on demand during forward; "
            "'dequant_upfront' dequantizes GGUF tensors to float at load time (uses more RAM/VRAM, can improve speed); "
            "'cuda_pack' is reserved for Codex packed GGUF execution via fused CUDA kernels (NVIDIA-only)."
        ),
    )

    parser.add_argument(
        "--gguf-dequant-cache",
        choices=["off", "lvl1", "lvl2"],
        default=None,
        help=(
            "Enable run-scoped caching for GGUF dequant_forward execution: "
            "'off' disables caching (default). "
            "'lvl1' caches moved+baked GGUF parameters per sampling run (reduces repeated bake/move overhead). "
            "'lvl2' also caches dequantized float weights per sampling run (more speed, more memory). "
            "Only valid with '--gguf-exec=dequant_forward'."
        ),
    )
    parser.add_argument(
        "--gguf-dequant-cache-limit-mb",
        type=int,
        default=None,
        metavar="MB",
        help=(
            "Optional memory cap (MB) for GGUF dequant cache. When unset, a heuristic cap is chosen from free VRAM/RAM at sampling start. "
            "Ignored when '--gguf-dequant-cache=off'."
        ),
    )
    parser.add_argument(
        "--gguf-dequant-cache-ratio",
        type=float,
        default=None,
        metavar="RATIO",
        help=(
            "Optional ratio used to compute the heuristic GGUF dequant cache budget from free VRAM/RAM at sampling start when "
            "'--gguf-dequant-cache-limit-mb' is unset. Must be >0 and <=1. Applies to both lvl1 and lvl2. "
            "Ignored when '--gguf-dequant-cache=off' or when an explicit limit is provided."
        ),
    )

    parser.add_argument(
        "--lora-apply-mode",
        choices=[m.value for m in LoraApplyMode],
        default=None,
        help=(
            "Global LoRA application mode: "
            "'merge' rewrites weights (default), "
            "'online' applies patches on-the-fly during forward. "
            "Changing this requires restarting the backend process."
        ),
    )

    parser.add_argument(
        "--lora-online-math",
        choices=[m.value for m in LoraOnlineMath],
        default=None,
        help=(
            "Online LoRA math mode when '--lora-apply-mode online' is active: "
            "'weight_merge' (default) materializes patched weights per forward; "
            "'activation' applies LoRA as an activation-side delta (reserved for packed GGUF kernels)."
        ),
    )

    parser.add_argument(
        "--debug-conditioning",
        action="store_true",
        help="Emit verbose conditioning diagnostics during diffusion runs.",
    )

    parser.add_argument(
        "--debug-preview-factors",
        action="store_true",
        help="Emit a best-fit latent→RGB preview matrix (for building Approx-cheap live previews).",
    )

    parser.add_argument(
        "--trace-debug",
        action="store_true",
        help="Enable global function-call trace (logger.debug for every Python call).",
    )
    parser.add_argument(
        "--trace-debug-max-per-func",
        type=int,
        default=TRACE_DEBUG_DEFAULT,
        metavar="N",
        help="Maximum call logs per function when trace debug is enabled (<=0 disables limit).",
    )
    parser.add_argument(
        "--trace-contract",
        action="store_true",
        help="Enable contract-trace logging (JSONL events under logs/contract-trace).",
    )
    parser.add_argument(
        "--trace-profiler",
        action="store_true",
        help="Enable runtime torch profiler tracing for generation runs.",
    )

    parser.add_argument(
        "--swap-policy",
        choices=[p.value for p in SwapPolicy],
        default=SwapPolicy.CPU.value,
        help="Offload policy when VRAM is insufficient.",
    )
    parser.add_argument(
        "--swap-method",
        choices=[m.value for m in SwapMethod],
        default=SwapMethod.BLOCKED.value,
        help="Data transfer mode for swap operations.",
    )
    parser.add_argument(
        "--gpu-prefer-construct",
        action="store_true",
        help="Prefer constructing models directly on GPU (no implicit fallback).",
    )

    device_choices = ["auto", "cuda", "cpu", "mps", "xpu", "directml"]
    parser.add_argument(
        "--core-device",
        choices=device_choices,
        default=None,
        help="Explicit device for diffusion core (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--te-device",
        choices=device_choices,
        default=None,
        help="Explicit device for text encoder (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--vae-device",
        choices=device_choices,
        default=None,
        help="Explicit device for VAE (overrides saved WebUI settings).",
    )

    dtype_choices = ["auto", "fp16", "bf16", "fp32", "fp8_e4m3fn", "fp8_e5m2"]
    parser.add_argument(
        "--core-dtype",
        choices=dtype_choices,
        default=None,
        help="Preferred dtype for diffusion core (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--core-compute-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help=(
            "Compute dtype for diffusion core activations (distinct from --core-dtype storage). "
            "Default is fp32 for stability."
        ),
    )
    parser.add_argument(
        "--te-dtype",
        choices=dtype_choices,
        default=None,
        help="Preferred dtype for text encoder (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--te-compute-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help=(
            "Compute dtype for text encoder activations (distinct from --te-dtype storage). "
            "Default is fp32 for stability."
        ),
    )
    parser.add_argument(
        "--vae-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help="Preferred dtype for VAE (overrides saved WebUI settings).",
    )
    parser.add_argument(
        "--vae-compute-dtype",
        choices=["auto", "fp16", "bf16", "fp32"],
        default=None,
        help=(
            "Compute dtype for VAE activations (distinct from --vae-dtype storage). "
            "Default is fp32 for stability."
        ),
    )

    return parser


_DEVICE_DIRECTIVES = (
    ("core_device", "codex_core_device"),
    ("te_device", "codex_te_device"),
    ("vae_device", "codex_vae_device"),
)

_DTYPE_DIRECTIVES = (
    ("core_dtype", "codex_core_dtype"),
    ("core_compute_dtype", "codex_core_compute_dtype"),
    ("te_dtype", "codex_te_dtype"),
    ("te_compute_dtype", "codex_te_compute_dtype"),
    ("vae_dtype", "codex_vae_dtype"),
    ("vae_compute_dtype", "codex_vae_compute_dtype"),
)


def _truthy(value: str | None) -> bool:
    if not value:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _has_value(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def _apply_source_overrides(
    ns: argparse.Namespace,
    env_map: MutableMapping[str, str],
    settings: Mapping[str, object] | None,
) -> None:
    settings = settings or {}

    def _setting_value(key: str) -> str | None:
        if key not in settings:
            return None
        value = settings[key]
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    for flag_attr, settings_key in _DEVICE_DIRECTIVES + _DTYPE_DIRECTIVES:
        raw = getattr(ns, flag_attr, None)
        if raw is not None:
            text = str(raw).strip().lower()
            setattr(ns, settings_key, text or None)
            continue

        setting_val = _setting_value(settings_key)
        if setting_val:
            setattr(ns, settings_key, setting_val)

    # Attention backend is a runtime-wide policy (not per-request). Allow saved settings
    # to seed the initial runtime config when no CLI override was provided.
    attention_cli_override = any(
        bool(getattr(ns, name, False))
        for name in (
            "attention_split",
            "attention_quad",
            "attention_pytorch",
        )
    ) or _has_value(getattr(ns, "attention_backend", None))
    if not attention_cli_override:
        raw_backend = _setting_value("codex_attention_backend")
        if raw_backend:
            mapped = raw_backend.strip().lower()
            if mapped == "torch-sdpa":
                mapped = "pytorch"
            if mapped in {"pytorch", "xformers", "split", "quad"}:
                ns.attention_backend = mapped

    if getattr(ns, "debug_conditioning", False):
        env_map["CODEX_DEBUG_COND"] = "1"

    if getattr(ns, "debug_preview_factors", False):
        env_map["CODEX_DEBUG_PREVIEW_FACTORS"] = "1"

    _ = env_map  # env_map only carries debug/log vars now (settings are payload/options-driven)


def _validate_runtime_flags(ns: argparse.Namespace) -> None:
    gguf_exec = str(getattr(ns, "gguf_exec", DEFAULT_GGUF_EXEC_MODE.value))
    gguf_dequant_cache = str(getattr(ns, "gguf_dequant_cache", "off") or "off").strip().lower()
    gguf_dequant_cache_limit_mb = getattr(ns, "gguf_dequant_cache_limit_mb", None)
    gguf_dequant_cache_ratio = getattr(ns, "gguf_dequant_cache_ratio", None)
    lora_apply_mode = str(getattr(ns, "lora_apply_mode", DEFAULT_LORA_APPLY_MODE.value))
    lora_online_math = str(getattr(ns, "lora_online_math", DEFAULT_LORA_ONLINE_MATH.value))

    if lora_online_math == LoraOnlineMath.ACTIVATION.value:
        if lora_apply_mode != LoraApplyMode.ONLINE.value:
            raise RuntimeError("--lora-online-math=activation requires '--lora-apply-mode online'.")
        if gguf_exec != GgufExecMode.CUDA_PACK.value:
            raise RuntimeError("--lora-online-math=activation is reserved for '--gguf-exec=cuda_pack'.")

    if gguf_exec == GgufExecMode.CUDA_PACK.value:
        if lora_apply_mode != LoraApplyMode.ONLINE.value:
            raise RuntimeError("--gguf-exec=cuda_pack requires '--lora-apply-mode online'.")
        if lora_online_math != LoraOnlineMath.ACTIVATION.value:
            raise RuntimeError("--gguf-exec=cuda_pack requires '--lora-online-math activation'.")
        raise RuntimeError(
            "--gguf-exec=cuda_pack is reserved and not implemented yet in this build. "
            "Use 'dequant_forward' or 'dequant_upfront' for now.",
        )

    if gguf_dequant_cache != "off":
        if gguf_exec != GgufExecMode.DEQUANT_FORWARD.value:
            raise RuntimeError("--gguf-dequant-cache requires '--gguf-exec=dequant_forward'.")
        if gguf_dequant_cache not in {"lvl1", "lvl2"}:
            raise RuntimeError("--gguf-dequant-cache must be one of: off, lvl1, lvl2.")
        if gguf_dequant_cache_limit_mb is not None and int(gguf_dequant_cache_limit_mb) <= 0:
            raise RuntimeError("--gguf-dequant-cache-limit-mb must be > 0 when provided.")
    if gguf_dequant_cache_ratio is not None:
        try:
            ratio = float(gguf_dequant_cache_ratio)
        except Exception as exc:
            raise RuntimeError("--gguf-dequant-cache-ratio must be a float when provided.") from exc
        if ratio <= 0.0 or ratio > 1.0:
            raise RuntimeError("--gguf-dequant-cache-ratio must be > 0 and <= 1 when provided.")

    attn_backend = _resolve_attention_backend(ns)
    if attn_backend == AttentionBackend.XFORMERS:
        if getattr(ns, "disable_xformers", False):
            raise RuntimeError("xformers attention backend is incompatible with --disable-xformers.")
        try:
            import xformers  # type: ignore # noqa: F401
        except Exception as exc:
            raise RuntimeError(f"xformers attention backend requested, but xformers is not available: {exc}") from exc


def _normalize_device_choice(value: str | None) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v:
        return None
    if v == "auto":
        return "auto"
    if v in {"cuda", "gpu"}:
        return "cuda"
    if v == "cpu":
        return "cpu"
    if v == "mps":
        return "mps"
    if v == "xpu":
        return "xpu"
    if v in {"directml", "dml"}:
        return "directml"
    raise ValueError(f"Unsupported device option '{value}'. Allowed: auto, cuda, cpu, mps, xpu, directml")


def _normalize_dtype_choice(value: str | None, *, allow_fp8: bool = False) -> str | None:
    if value is None:
        return None
    v = value.strip().lower()
    if not v or v == "auto":
        return None
    mapping = {
        "fp16": "fp16",
        "float16": "fp16",
        "half": "fp16",
        "bf16": "bf16",
        "bfloat16": "bf16",
        "fp32": "fp32",
        "float32": "fp32",
        "float": "fp32",
        "single": "fp32",
    }
    if allow_fp8:
        mapping.update(
            {
                "fp8_e4m3fn": "fp8_e4m3fn",
                "fp8-e4m3fn": "fp8_e4m3fn",
                "fp8_e4": "fp8_e4m3fn",
                "fp8_e5m2": "fp8_e5m2",
                "fp8-e5m2": "fp8_e5m2",
                "fp8_e5": "fp8_e5m2",
            }
        )
    result = mapping.get(v)
    if result is None:
        allowed = ", ".join(sorted(set(mapping.values())))
        raise ValueError(f"Unsupported dtype option '{value}'. Allowed: {allowed}")
    return result


def _validate_required_devices(ns: argparse.Namespace) -> None:
    """Ensure device defaults are explicitly configured (no silent fallbacks).

    In interactive terminals, missing device values are prompted from the user.
    In non-interactive contexts, we fail loud with a clear error message.
    """

    required: list[tuple[str, str]] = [
        ("codex_core_device", "diffusion core"),
        ("codex_te_device", "text encoder"),
        ("codex_vae_device", "VAE"),
    ]

    missing = [(attr, label) for attr, label in required if getattr(ns, attr, None) is None]
    if not missing:
        return

    choices = ["auto", "cuda", "cpu", "mps", "xpu", "directml"]

    def _interactive() -> bool:
        try:
            if not (sys.stdin.isatty() and sys.stdout.isatty()):
                return False
            # Avoid prompting when the process is in the background (would SIGTTIN / hang).
            return os.getpgrp() == os.tcgetpgrp(sys.stdin.fileno())
        except Exception:
            return False

    if not _interactive():
        keys = ", ".join(attr for attr, _label in missing)
        raise RuntimeError(
            "Device configuration is required but missing: "
            f"{keys}. Provide it via startup flags "
            "(`--core-device`, `--te-device`, `--vae-device`) or set persisted defaults in "
            "`apps/settings_values.json` (keys `codex_core_device`, `codex_te_device`, `codex_vae_device`)."
        )

    def _prompt(attr: str, label: str) -> str:
        sys.stdout.write("\n")
        sys.stdout.write(f"[config] Missing {attr} ({label}). Choose one:\n")
        for idx, option in enumerate(choices, start=1):
            sys.stdout.write(f"  {idx}) {option}\n")
        sys.stdout.flush()
        while True:
            try:
                raw = input(f"Select {attr} (1-{len(choices)} or value, 'q' to abort): ").strip()
            except EOFError as exc:
                raise RuntimeError(f"Missing {attr} and no input available (stdin closed).") from exc
            if not raw:
                continue
            lowered = raw.strip().lower()
            if lowered in {"q", "quit", "exit"}:
                raise RuntimeError("Aborted by user.")
            if lowered.isdigit():
                n = int(lowered)
                if 1 <= n <= len(choices):
                    return choices[n - 1]
            if lowered in choices:
                return lowered
            sys.stdout.write(f"Invalid choice {raw!r}. Allowed: {', '.join(choices)}\n")
            sys.stdout.flush()

    for attr, label in missing:
        setattr(ns, attr, _prompt(attr, label))


def _torch_dtype_for_choice(choice: str | None) -> str | None:
    if choice is None:
        return None
    mapping = {
        "fp16": "float16",
        "bf16": "bfloat16",
        "fp32": "float32",
        "fp8_e4m3fn": "float16",
        "fp8_e5m2": "float16",
    }
    return mapping.get(choice)


def _apply_component_device_overrides(config: RuntimeMemoryConfig, ns: argparse.Namespace) -> None:
    role_choices = (
        (DeviceRole.CORE, getattr(ns, "codex_core_device", None), getattr(ns, "codex_core_dtype", None)),
        (DeviceRole.TEXT_ENCODER, getattr(ns, "codex_te_device", None), getattr(ns, "codex_te_dtype", None)),
        (DeviceRole.VAE, getattr(ns, "codex_vae_device", None), getattr(ns, "codex_vae_dtype", None)),
    )
    for role, device_choice, dtype_choice in role_choices:
        policy = config.component_policy(role)
        if device_choice == "cuda":
            policy.preferred_backend = DeviceBackend.CUDA
        elif device_choice == "cpu":
            policy.preferred_backend = DeviceBackend.CPU
        elif device_choice == "mps":
            policy.preferred_backend = DeviceBackend.MPS
        elif device_choice == "xpu":
            policy.preferred_backend = DeviceBackend.XPU
        elif device_choice == "directml":
            policy.preferred_backend = DeviceBackend.DIRECTML

        forced = _torch_dtype_for_choice(dtype_choice)
        if device_choice == "cpu":
            forced = "float32"
        if forced:
            policy.forced_dtype = forced

        compute_key = {
            DeviceRole.CORE: "codex_core_compute_dtype",
            DeviceRole.TEXT_ENCODER: "codex_te_compute_dtype",
            DeviceRole.VAE: "codex_vae_compute_dtype",
        }.get(role)
        if compute_key:
            compute_choice = getattr(ns, compute_key, None)
            forced_compute = _torch_dtype_for_choice(compute_choice)
            if device_choice == "cpu" and forced_compute and forced_compute != "float32":
                forced_compute = "float32"
            if forced_compute:
                policy.forced_compute_dtype = forced_compute


def _apply_env_overrides(ns: argparse.Namespace, env: Mapping[str, str]) -> None:
    def _set_core_dtype(val: str | None) -> None:
        ns.core_in_fp16 = False
        ns.core_in_bf16 = False
        ns.core_in_fp8_e4m3fn = False
        ns.core_in_fp8_e5m2 = False
        if not val:
            return
        v = val.strip().lower()
        ns.core_in_bf16 = False
        ns.core_in_fp16 = False
        ns.core_in_fp8_e4m3fn = False
        ns.core_in_fp8_e5m2 = False
        if v in {"bf16", "bfloat16"}:
            ns.core_in_bf16 = True
        elif v in {"fp16", "half"}:
            ns.core_in_fp16 = True
        elif v in {"fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"}:
            ns.core_in_fp8_e4m3fn = True
        elif v in {"fp8_e5m2", "fp8-e5m2", "fp8_e5"}:
            ns.core_in_fp8_e5m2 = True

    def _set_vae_dtype(val: str | None) -> None:
        if not val:
            return
        v = val.strip().lower()
        if v in {"bf16", "bfloat16"}:
            ns.vae_in_bf16 = True
            ns.vae_in_fp16 = False
            ns.vae_in_fp32 = False
        elif v in {"fp16", "half"}:
            ns.vae_in_bf16 = False
            ns.vae_in_fp16 = True
            ns.vae_in_fp32 = False
        elif v in {"fp32", "float", "single"}:
            ns.vae_in_bf16 = False
            ns.vae_in_fp16 = False
            ns.vae_in_fp32 = True

    def _set_te_dtype(val: str | None) -> None:
        for attr in ("clip_in_fp16", "clip_in_fp32", "clip_in_fp8_e4m3fn", "clip_in_fp8_e5m2", "clip_in_bf16"):
            setattr(ns, attr, False)
        if not val:
            return
        v = val.strip().lower()
        if v in {"fp16", "half", "float16"}:
            ns.clip_in_fp16 = True
        elif v in {"fp32", "float32", "float", "single"}:
            ns.clip_in_fp32 = True
        elif v in {"bf16", "bfloat16"}:
            ns.clip_in_bf16 = True
        elif v in {"fp8_e4m3fn", "fp8-e4m3fn", "fp8_e4"}:
            ns.clip_in_fp8_e4m3fn = True
        elif v in {"fp8_e5m2", "fp8-e5m2", "fp8_e5"}:
            ns.clip_in_fp8_e5m2 = True

    core_device_choice = _normalize_device_choice(getattr(ns, "codex_core_device", None))
    core_dtype_raw = getattr(ns, "codex_core_dtype", None)
    core_dtype_choice = _normalize_dtype_choice(core_dtype_raw, allow_fp8=True)
    if core_device_choice == "cpu" and core_dtype_choice not in (None, "fp32"):
        core_dtype_choice = "fp32"
    _set_core_dtype(core_dtype_choice or core_dtype_raw)
    ns.codex_core_device = core_device_choice
    ns.codex_core_dtype = core_dtype_choice

    core_compute_dtype_raw = getattr(ns, "codex_core_compute_dtype", None)
    core_compute_dtype_choice = _normalize_dtype_choice(core_compute_dtype_raw, allow_fp8=False)
    if core_device_choice == "cpu" and core_compute_dtype_choice not in (None, "fp32"):
        core_compute_dtype_choice = "fp32"
    ns.codex_core_compute_dtype = core_compute_dtype_choice

    vae_device_choice = _normalize_device_choice(getattr(ns, "codex_vae_device", None))
    vae_dtype_raw = getattr(ns, "codex_vae_dtype", None)
    vae_dtype_choice = _normalize_dtype_choice(vae_dtype_raw, allow_fp8=False)
    if vae_device_choice == "cpu":
        ns.vae_in_cpu = True
        if vae_dtype_choice not in (None, "fp32"):
            vae_dtype_choice = "fp32"
    _set_vae_dtype(vae_dtype_choice or vae_dtype_raw)
    ns.codex_vae_device = vae_device_choice
    ns.codex_vae_dtype = vae_dtype_choice

    vae_compute_dtype_raw = getattr(ns, "codex_vae_compute_dtype", None)
    vae_compute_dtype_choice = _normalize_dtype_choice(vae_compute_dtype_raw, allow_fp8=False)
    if vae_device_choice == "cpu" and vae_compute_dtype_choice not in (None, "fp32"):
        vae_compute_dtype_choice = "fp32"
    ns.codex_vae_compute_dtype = vae_compute_dtype_choice

    te_device_choice = _normalize_device_choice(getattr(ns, "codex_te_device", None))
    te_dtype_raw = getattr(ns, "codex_te_dtype", None)
    te_dtype_choice = _normalize_dtype_choice(te_dtype_raw, allow_fp8=True)
    if te_device_choice == "cpu" and te_dtype_choice not in (None, "fp32"):
        te_dtype_choice = "fp32"
    _set_te_dtype(te_dtype_choice or te_dtype_raw)
    ns.codex_te_device = te_device_choice
    ns.codex_te_dtype = te_dtype_choice

    te_compute_dtype_raw = getattr(ns, "codex_te_compute_dtype", None)
    te_compute_dtype_choice = _normalize_dtype_choice(te_compute_dtype_raw, allow_fp8=False)
    if te_device_choice == "cpu" and te_compute_dtype_choice not in (None, "fp32"):
        te_compute_dtype_choice = "fp32"
    ns.codex_te_compute_dtype = te_compute_dtype_choice

    if te_device_choice == "cpu" and not getattr(ns, "clip_in_fp32", False):
        ns.clip_in_fp32 = True

    if te_device_choice == "cpu" and getattr(ns, "clip_in_fp16", False):
        ns.clip_in_fp16 = False

    if _truthy(env.get("CODEX_DEBUG_COND")):
        ns.debug_conditioning = True

    if _truthy(env.get("CODEX_DEBUG_PREVIEW_FACTORS")):
        ns.debug_preview_factors = True

    # Global call tracing (function-level). This toggles a runtime hook in
    # the API entrypoint; we keep the flag for visibility in the parsed args.
    if _truthy(env.get("CODEX_TRACE_DEBUG")):
        ns.trace_debug = True

    # Honour max-calls-per-func trace limit from env (used by BIOS/TUI)
    raw_trace_max = env.get("CODEX_TRACE_DEBUG_MAX_PER_FUNC")
    if raw_trace_max is not None:
        try:
            ns.trace_debug_max_per_func = max(0, int(raw_trace_max))
        except Exception:
            ns.trace_debug_max_per_func = TRACE_DEBUG_DEFAULT

    if _truthy(env.get("CODEX_TRACE_CONTRACT")):
        ns.trace_contract = True

    if _truthy(env.get("CODEX_TRACE_PROFILER")) or _truthy(env.get("CODEX_PROFILE")):
        ns.trace_profiler = True

    # LoRA apply mode (global): honour env only when CLI arg is unset.
    raw_lora_mode = env.get(ENV_LORA_APPLY_MODE)
    if raw_lora_mode is not None and not _has_value(getattr(ns, "lora_apply_mode", None)):
        try:
            ns.lora_apply_mode = parse_lora_apply_mode(raw_lora_mode).value
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc


def _resolve_attention_backend(ns: argparse.Namespace) -> AttentionBackend:
    explicit = getattr(ns, "attention_backend", None)
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip().lower()
        mapping = {
            "pytorch": AttentionBackend.PYTORCH,
            "xformers": AttentionBackend.XFORMERS,
            "split": AttentionBackend.SPLIT,
            "quad": AttentionBackend.QUAD,
        }
        resolved = mapping.get(normalized)
        if resolved is not None:
            return resolved
        _LOG.warning("Unsupported attention backend '%s'; falling back to PyTorch.", explicit)
        return AttentionBackend.PYTORCH

    if ns.attention_split:
        return AttentionBackend.SPLIT
    if ns.attention_quad:
        return AttentionBackend.QUAD
    if ns.attention_pytorch:
        return AttentionBackend.PYTORCH
    return AttentionBackend.PYTORCH


def build_runtime_memory_config(ns: argparse.Namespace) -> RuntimeMemoryConfig:
    precision = PrecisionFlags(
        all_fp16=ns.all_in_fp16,
        core_fp16=ns.core_in_fp16,
        core_bf16=ns.core_in_bf16,
        core_fp8_e4m3fn=ns.core_in_fp8_e4m3fn,
        core_fp8_e5m2=ns.core_in_fp8_e5m2,
        vae_fp16=ns.vae_in_fp16,
        vae_fp32=ns.vae_in_fp32,
        vae_bf16=ns.vae_in_bf16,
        vae_in_cpu=ns.vae_in_cpu,
        clip_fp16=getattr(ns, "clip_in_fp16", False),
        clip_fp32=getattr(ns, "clip_in_fp32", False),
        clip_bf16=getattr(ns, "clip_in_bf16", False),
        clip_fp8_e4m3fn=getattr(ns, "clip_in_fp8_e4m3fn", False),
        clip_fp8_e5m2=getattr(ns, "clip_in_fp8_e5m2", False),
    )

    attention_backend = _resolve_attention_backend(ns)
    force_upcast = bool(ns.force_upcast_attention)
    if getattr(ns, "disable_attention_upcast", False):
        force_upcast = False

    attention = AttentionConfig(
        backend=attention_backend,
        enable_flash=attention_backend == AttentionBackend.PYTORCH,
        enable_mem_efficient=attention_backend == AttentionBackend.PYTORCH,
        force_upcast=force_upcast,
        allow_split_fallback=True,
        allow_quad_fallback=True,
    )

    swap = SwapConfig(
        policy=SwapPolicy(ns.swap_policy),
        method=SwapMethod(ns.swap_method),
        always_offload=ns.always_offload_from_vram,
        pin_shared_memory=ns.pin_shared_memory,
    )

    config = RuntimeMemoryConfig(
        device_backend=DeviceBackend.AUTO,
        gpu_device_id=ns.gpu_device_id,
        gpu_prefer_construct=ns.gpu_prefer_construct,
        precision=precision,
        swap=swap,
        attention=attention,
        budgets=MemoryBudgets(),
        deterministic_algorithms=ns.pytorch_deterministic,
        disable_xformers=ns.disable_xformers,
        enable_xformers_vae=not ns.disable_xformers,
    )

    if ns.always_gpu:
        config.device_backend = DeviceBackend.CUDA
    elif ns.always_cpu:
        config.device_backend = DeviceBackend.CPU
    elif ns.directml is not None:
        config.device_backend = DeviceBackend.DIRECTML
        config.allow_directml = True

    if ns.vae_in_cpu:
        config.component_policy(DeviceRole.VAE).preferred_backend = DeviceBackend.CPU

    _apply_component_device_overrides(config, ns)

    return config


_PARSER = _build_parser()
_args: argparse.Namespace | None = None
_memory_config: RuntimeMemoryConfig | None = None
_UNKNOWN: list[str] = []

def initialize(
    argv: Sequence[str] | None = None,
    env: Mapping[str, str] | None = None,
    settings: Mapping[str, object] | None = None,
    *,
    strict: bool = True,
) -> tuple[argparse.Namespace, RuntimeMemoryConfig]:
    """Parse runtime arguments applying CLI/env/settings precedence.

    Returns the parsed namespace and freshly built RuntimeMemoryConfig.
    When ``strict`` is True, raises RuntimeError for unknown CLI arguments and
    if required device flags are missing after applying overrides.
    """
    global _args, _memory_config, _UNKNOWN

    argv_list = list(argv) if argv is not None else sys.argv[1:]
    namespace, unknown = _PARSER.parse_known_args(argv_list)
    _UNKNOWN = list(unknown)

    deprecated = [arg for arg in unknown if arg.startswith("--unet-in-")]
    if deprecated:
        raise RuntimeError(
            "Deprecated precision flag(s) detected: "
            + ", ".join(deprecated)
            + ". Use '--core-in-*' variants instead."
        )
    if strict and unknown:
        raise RuntimeError(
            "Unknown CLI argument(s): "
            + ", ".join(unknown)
            + ". Remove unsupported flags from the runtime bootstrap command."
        )

    source_env = env if env is not None else os.environ
    env_map: MutableMapping[str, str] = {}
    for key, value in source_env.items():
        if value is None:
            continue
        env_map[key] = str(value)

    _apply_source_overrides(namespace, env_map, settings)

    if getattr(namespace, "trace_debug_max_per_func", None) is None:
        namespace.trace_debug_max_per_func = TRACE_DEBUG_DEFAULT
    elif namespace.trace_debug_max_per_func < 0:
        namespace.trace_debug_max_per_func = 0
    _apply_env_overrides(namespace, env_map)
    if getattr(namespace, "lora_apply_mode", None) is None:
        namespace.lora_apply_mode = DEFAULT_LORA_APPLY_MODE.value
    if getattr(namespace, "gguf_exec", None) is None:
        namespace.gguf_exec = DEFAULT_GGUF_EXEC_MODE.value
    if getattr(namespace, "gguf_dequant_cache", None) is None:
        namespace.gguf_dequant_cache = "off"
    if getattr(namespace, "lora_online_math", None) is None:
        namespace.lora_online_math = DEFAULT_LORA_ONLINE_MATH.value

    if strict:
        _validate_runtime_flags(namespace)
        _validate_required_devices(namespace)
    config = build_runtime_memory_config(namespace)

    _args = namespace
    _memory_config = config
    return namespace, config


# Initialise module defaults with non-strict semantics so early imports don't abort.
args, memory_config = initialize(strict=False)

dynamic_args = {
    "embedding_dir": "./embeddings",
    "emphasis_name": "original",
}


__all__ = [
    "args",
    "memory_config",
    "dynamic_args",
    "build_runtime_memory_config",
    "initialize",
]
