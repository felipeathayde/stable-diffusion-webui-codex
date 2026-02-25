"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN fused-attention V1 contract and extension bridge.
Provides strict fail-loud validators, mode resolution, and optional CUDA extension dispatch for WAN fused
self/cross attention (`QKV + RoPE + attention + out-proj`) in inference mode.

Symbols (top-level; keep in sync; no ghosts):
- `WanFusedMode` (enum): Runtime mode for fused attention dispatch (`off|auto|force`).
- `WanFusedContractError` (class): Fail-loud contract error with stable `code` field.
- `WanFusedAttemptResult` (dataclass): Result envelope for non-forced attempts (`output` or `reason_code`).
- `parse_wan_fused_mode` (function): Parses and validates a fused mode string.
- `resolve_effective_wan_fused_mode` (function): Resolves fused mode from override/env.
- `is_extension_available` (function): Returns whether WAN fused CUDA ops are available.
- `last_extension_error` (function): Returns the last extension load/build error details.
- `try_fused_self_attention` (function): Attempts fused self-attention dispatch and returns output or reason.
- `try_fused_cross_attention` (function): Attempts fused cross-attention dispatch and returns output or reason.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch

logger = logging.getLogger("backend.runtime.attention.wan_fused_v1")


_MODE_ENV_KEY = "CODEX_WAN22_FUSED_ATTN_V1_MODE"
_JIT_ENV_KEY = "CODEX_WAN_FUSED_V1_JIT"


E_WAN_FUSED_DISABLED = "E_WAN_FUSED_DISABLED"
E_WAN_FUSED_EXTENSION_UNAVAILABLE = "E_WAN_FUSED_EXTENSION_UNAVAILABLE"
E_WAN_FUSED_KERNEL_RUNTIME_ERROR = "E_WAN_FUSED_KERNEL_RUNTIME_ERROR"
E_WAN_FUSED_INVALID_MODE = "E_WAN_FUSED_INVALID_MODE"
E_WAN_FUSED_DROPOUT_UNSUPPORTED = "E_WAN_FUSED_DROPOUT_UNSUPPORTED"
E_WAN_FUSED_DEVICE_UNSUPPORTED = "E_WAN_FUSED_DEVICE_UNSUPPORTED"
E_WAN_FUSED_DTYPE_UNSUPPORTED = "E_WAN_FUSED_DTYPE_UNSUPPORTED"
E_WAN_FUSED_HEAD_DIM_UNSUPPORTED = "E_WAN_FUSED_HEAD_DIM_UNSUPPORTED"
E_WAN_FUSED_UNSUPPORTED_ARCH = "E_WAN_FUSED_UNSUPPORTED_ARCH"
E_WAN_FUSED_INVALID_SHAPE = "E_WAN_FUSED_INVALID_SHAPE"
E_WAN_FUSED_MISSING_ROPE = "E_WAN_FUSED_MISSING_ROPE"
E_WAN_FUSED_NONCONTIGUOUS = "E_WAN_FUSED_NONCONTIGUOUS"


@dataclass(frozen=True, slots=True)
class _AttemptError:
    stage: str
    message: str


@dataclass(frozen=True, slots=True)
class WanFusedAttemptResult:
    output: torch.Tensor | None
    reason_code: str | None
    reason_detail: str | None = None


class WanFusedMode(str, Enum):
    OFF = "off"
    AUTO = "auto"
    FORCE = "force"


class WanFusedContractError(RuntimeError):
    def __init__(self, *, code: str, message: str):
        self.code = str(code)
        super().__init__(f"{self.code}: {message}")


_ext = None
_last_error: str | None = None
_attempt_errors: list[_AttemptError] = []
_load_attempted = False
_last_attempt_with_build = False


def _set_attempt_error(stage: str, ex: Exception) -> None:
    global _last_error
    message = f"{type(ex).__name__}: {ex}"
    _attempt_errors.append(_AttemptError(stage=stage, message=message))
    _last_error = "\n".join(f"{entry.stage}: {entry.message}" for entry in _attempt_errors)


def _has_ops() -> bool:
    ops = getattr(torch, "ops", None)
    if ops is None or not hasattr(ops, "wan_fused_v1"):
        return False
    wan_ops = ops.wan_fused_v1
    return hasattr(wan_ops, "self_fwd") and hasattr(wan_ops, "cross_fwd")


def _try_load_ext(*, build: bool) -> None:
    global _ext
    global _last_error
    global _load_attempted
    global _last_attempt_with_build

    if _ext is not None and _has_ops():
        return
    if _load_attempted and _ext is None and (_last_attempt_with_build or build == _last_attempt_with_build):
        return

    _load_attempted = True
    _last_attempt_with_build = bool(build)
    _attempt_errors.clear()
    _last_error = None

    try:
        import wan_fused_v1_cuda as loaded

        _ext = loaded
        if not _has_ops():
            raise RuntimeError("module loaded but torch.ops.wan_fused_v1.{self_fwd,cross_fwd} is missing")
        logger.info("loaded wan_fused_v1_cuda extension (prebuilt)")
        return
    except Exception as ex:
        _ext = None
        _set_attempt_error("prebuilt", ex)
        logger.info("wan_fused_v1_cuda prebuilt not available: %s", ex)

    try:
        this_dir = os.path.dirname(__file__)
        ext_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "kernels", "wan_fused_v1"))
        if os.path.isdir(ext_dir) and ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)

        import wan_fused_v1_cuda as loaded

        _ext = loaded
        if not _has_ops():
            raise RuntimeError("in-place module loaded but torch.ops.wan_fused_v1.{self_fwd,cross_fwd} is missing")
        logger.info("loaded wan_fused_v1_cuda extension from in-place build (%s)", ext_dir)
        return
    except Exception as ex:
        _ext = None
        _set_attempt_error("in_place", ex)
        logger.info("wan_fused_v1_cuda in-place module not available: %s", ex)

    if not build:
        return

    try:
        from torch.utils.cpp_extension import load

        this_dir = os.path.dirname(__file__)
        src_dir = os.path.normpath(os.path.join(this_dir, "..", "..", "kernels", "wan_fused_v1"))

        def _src(path: str) -> str:
            return os.path.join(src_dir, path)

        sources = [
            _src("wan_fused_v1_binding.cpp"),
            _src("wan_fused_v1_kernels.cu"),
        ]

        loaded = load(
            name="wan_fused_v1_cuda_jit",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-DUSE_CUDA"],
        )
        _ext = loaded
        if not _has_ops():
            raise RuntimeError("JIT module loaded but torch.ops.wan_fused_v1.{self_fwd,cross_fwd} is missing")
        logger.info("built wan_fused_v1_cuda extension via JIT")
    except Exception as ex:
        _set_attempt_error("jit", ex)
        _ext = None
        logger.error("failed to build wan_fused_v1_cuda via JIT: %s", ex)


def is_extension_available() -> bool:
    build = str(os.environ.get(_JIT_ENV_KEY, "") or "").strip().lower() in {"1", "true", "yes", "on"}
    _try_load_ext(build=build)
    return _ext is not None and _has_ops()


def last_extension_error() -> str | None:
    return _last_error


def parse_wan_fused_mode(value: str, *, field_name: str) -> WanFusedMode:
    normalized = str(value).strip().lower()
    aliases = {
        "0": "off",
        "false": "off",
        "no": "off",
        "off": "off",
        "1": "auto",
        "true": "auto",
        "yes": "auto",
        "on": "auto",
        "auto": "auto",
        "force": "force",
        "required": "force",
    }
    mapped = aliases.get(normalized)
    if mapped is None:
        allowed = "off|auto|force"
        raise WanFusedContractError(
            code=E_WAN_FUSED_INVALID_MODE,
            message=f"{field_name} must be one of {allowed}; got {value!r}.",
        )
    return WanFusedMode(mapped)


def resolve_effective_wan_fused_mode(mode_override: str | WanFusedMode | None) -> WanFusedMode:
    if isinstance(mode_override, WanFusedMode):
        return mode_override
    if mode_override is not None:
        return parse_wan_fused_mode(mode_override, field_name="wan_fused_mode")
    env_value = os.environ.get(_MODE_ENV_KEY, "off")
    return parse_wan_fused_mode(env_value, field_name=_MODE_ENV_KEY)


def _fail(*, code: str, message: str) -> None:
    raise WanFusedContractError(code=code, message=message)


def _resolve_head_count(*, channels: int, head_dim: int, field_name: str) -> int:
    if head_dim <= 0:
        _fail(code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED, message=f"{field_name} must be > 0; got {head_dim}.")
    if channels % head_dim != 0:
        _fail(
            code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
            message=(
                f"channels/head_dim mismatch for {field_name}: channels={channels} head_dim={head_dim} "
                "(channels must be divisible by head_dim)."
            ),
        )
    return channels // head_dim


def _validate_arch_dtype_head_dim(*, device: torch.device, dtype: torch.dtype, head_dim: int) -> None:
    if device.type != "cuda":
        _fail(code=E_WAN_FUSED_DEVICE_UNSUPPORTED, message=f"WAN fused V1 requires CUDA tensors; got device={device}.")

    if dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        _fail(code=E_WAN_FUSED_DTYPE_UNSUPPORTED, message=f"WAN fused V1 supports fp16/bf16/fp32; got dtype={dtype}.")

    if head_dim % 8 != 0:
        _fail(code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED, message=f"head_dim must be a multiple of 8; got head_dim={head_dim}.")

    major, minor = torch.cuda.get_device_capability(device)
    capability = (int(major), int(minor))

    if capability < (7, 5):
        _fail(
            code=E_WAN_FUSED_UNSUPPORTED_ARCH,
            message=f"WAN fused V1 requires SM75+; got compute_capability={capability}.",
        )

    if capability == (7, 5):
        if dtype == torch.bfloat16:
            _fail(
                code=E_WAN_FUSED_DTYPE_UNSUPPORTED,
                message="WAN fused V1 does not support bf16 on SM75.",
            )
        if head_dim > 256:
            _fail(
                code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
                message=f"WAN fused V1 supports head_dim<=256 on SM75; got head_dim={head_dim}.",
            )
        return

    if capability in {(8, 0), (8, 6), (8, 9), (9, 0)}:
        if dtype in {torch.float16, torch.bfloat16} and head_dim > 512:
            _fail(
                code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
                message=(
                    "WAN fused V1 supports fp16/bf16 head_dim<=512 on SM80/86/89/90; "
                    f"got head_dim={head_dim}."
                ),
            )
        if dtype == torch.float32 and head_dim > 256:
            _fail(
                code=E_WAN_FUSED_HEAD_DIM_UNSUPPORTED,
                message=(
                    "WAN fused V1 supports fp32 head_dim<=256 on SM80/86/89/90; "
                    f"got head_dim={head_dim}."
                ),
            )
        return

    _fail(
        code=E_WAN_FUSED_UNSUPPORTED_ARCH,
        message=(
            "WAN fused V1 has no declared architecture contract for this GPU; "
            f"got compute_capability={capability}."
        ),
    )


def _resolve_linear_weight_bias(linear: Any, *, expected_out: int, expected_in: int, label: str) -> tuple[torch.Tensor, torch.Tensor | None]:
    weight = getattr(linear, "weight", None)
    bias = getattr(linear, "bias", None)
    if not torch.is_tensor(weight):
        _fail(code=E_WAN_FUSED_INVALID_SHAPE, message=f"{label}.weight is missing or invalid.")
    if tuple(weight.shape) != (expected_out, expected_in):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label}.weight shape mismatch: got {tuple(weight.shape)} expected {(expected_out, expected_in)}.",
        )
    if bias is not None:
        if not torch.is_tensor(bias):
            _fail(code=E_WAN_FUSED_INVALID_SHAPE, message=f"{label}.bias is not a tensor.")
        if tuple(bias.shape) != (expected_out,):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=f"{label}.bias shape mismatch: got {tuple(bias.shape)} expected {(expected_out,)}.",
            )
    return weight, bias


def _validate_common_inputs(*, x: torch.Tensor, dropout_p: float) -> None:
    if x.ndim != 3:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"WAN fused V1 expects x with shape [B,L,C]; got shape={tuple(x.shape)}.",
        )
    if not x.is_floating_point():
        _fail(code=E_WAN_FUSED_DTYPE_UNSUPPORTED, message=f"WAN fused V1 expects floating x; got dtype={x.dtype}.")
    if abs(float(dropout_p)) > 0.0:
        _fail(
            code=E_WAN_FUSED_DROPOUT_UNSUPPORTED,
            message=f"WAN fused V1 only supports dropout=0.0; got dropout_p={float(dropout_p)}.",
        )
    if not x.is_contiguous():
        _fail(
            code=E_WAN_FUSED_NONCONTIGUOUS,
            message=f"WAN fused V1 requires contiguous x tensor; got stride={tuple(x.stride())}.",
        )


def _validate_rope_tensor(*, tensor: torch.Tensor | None, expected_len: int, label: str) -> None:
    if tensor is None:
        _fail(code=E_WAN_FUSED_MISSING_ROPE, message=f"missing required RoPE tensor: {label}.")
    if tensor.ndim != 4:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} must be [1,S,1,D]; got shape={tuple(tensor.shape)}.",
        )
    if int(tensor.shape[0]) != 1 or int(tensor.shape[2]) != 1:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} must have shape prefix [1,*,1,*]; got shape={tuple(tensor.shape)}.",
        )
    if int(tensor.shape[1]) != int(expected_len):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"{label} sequence mismatch: got S={int(tensor.shape[1])} expected={int(expected_len)}.",
        )
    if not tensor.is_contiguous():
        _fail(
            code=E_WAN_FUSED_NONCONTIGUOUS,
            message=f"{label} must be contiguous; got stride={tuple(tensor.stride())}.",
        )


def _maybe_return_unavailable(*, mode: WanFusedMode) -> WanFusedAttemptResult | None:
    if is_extension_available():
        return None
    detail = last_extension_error()
    if mode is WanFusedMode.FORCE:
        _fail(
            code=E_WAN_FUSED_EXTENSION_UNAVAILABLE,
            message=(
                "WAN fused V1 forced mode requested but extension/ops are unavailable. "
                f"details={detail!r}"
            ),
        )
    return WanFusedAttemptResult(
        output=None,
        reason_code=E_WAN_FUSED_EXTENSION_UNAVAILABLE,
        reason_detail=detail,
    )


def try_fused_self_attention(
    *,
    mode: str | WanFusedMode | None,
    x: torch.Tensor,
    q_proj: Any,
    k_proj: Any,
    v_proj: Any,
    o_proj: Any,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    rope_cos_qk: torch.Tensor | None,
    rope_sin_qk: torch.Tensor | None,
    dropout_p: float,
) -> WanFusedAttemptResult:
    fused_mode = resolve_effective_wan_fused_mode(mode)
    if fused_mode is WanFusedMode.OFF:
        return WanFusedAttemptResult(output=None, reason_code=E_WAN_FUSED_DISABLED)

    unavailable = _maybe_return_unavailable(mode=fused_mode)
    if unavailable is not None:
        return unavailable

    _validate_common_inputs(x=x, dropout_p=dropout_p)
    bsz, seq_len, channels = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))

    _validate_rope_tensor(tensor=rope_cos_qk, expected_len=seq_len, label="rope_cos_qk")
    _validate_rope_tensor(tensor=rope_sin_qk, expected_len=seq_len, label="rope_sin_qk")

    head_dim = int(rope_cos_qk.shape[-1])
    num_heads = _resolve_head_count(channels=channels, head_dim=head_dim, field_name="self")
    _validate_arch_dtype_head_dim(device=x.device, dtype=x.dtype, head_dim=head_dim)

    w_q, b_q = _resolve_linear_weight_bias(q_proj, expected_out=channels, expected_in=channels, label="q_proj")
    w_k, b_k = _resolve_linear_weight_bias(k_proj, expected_out=channels, expected_in=channels, label="k_proj")
    w_v, b_v = _resolve_linear_weight_bias(v_proj, expected_out=channels, expected_in=channels, label="v_proj")
    w_o, b_o = _resolve_linear_weight_bias(o_proj, expected_out=channels, expected_in=channels, label="o_proj")

    if tuple(norm_q_weight.shape) != (channels,) or tuple(norm_k_weight.shape) != (channels,):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=(
                "WAN fused self expects norm weights shaped [C]. "
                f"got norm_q={tuple(norm_q_weight.shape)} norm_k={tuple(norm_k_weight.shape)} expected={(channels,)}."
            ),
        )

    w_qkv = torch.stack(
        [
            w_q.t().contiguous().view(channels, num_heads, head_dim),
            w_k.t().contiguous().view(channels, num_heads, head_dim),
            w_v.t().contiguous().view(channels, num_heads, head_dim),
        ],
        dim=1,
    ).contiguous()

    b_qkv: torch.Tensor | None = None
    if b_q is not None and b_k is not None and b_v is not None:
        b_qkv = torch.stack(
            [
                b_q.contiguous().view(num_heads, head_dim),
                b_k.contiguous().view(num_heads, head_dim),
                b_v.contiguous().view(num_heads, head_dim),
            ],
            dim=0,
        ).contiguous()

    w_out = w_o.t().contiguous().view(num_heads, head_dim, channels)

    try:
        out = torch.ops.wan_fused_v1.self_fwd(
            x,
            w_qkv,
            b_qkv,
            norm_q_weight,
            norm_k_weight,
            rope_cos_qk,
            rope_sin_qk,
            w_out,
            b_o,
        )
        if tuple(out.shape) != (bsz, seq_len, channels):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused self returned unexpected shape: "
                    f"got {tuple(out.shape)} expected {(bsz, seq_len, channels)}."
                ),
            )
        return WanFusedAttemptResult(output=out, reason_code=None)
    except WanFusedContractError:
        raise
    except Exception as ex:
        if fused_mode is WanFusedMode.FORCE:
            _fail(
                code=E_WAN_FUSED_KERNEL_RUNTIME_ERROR,
                message=f"WAN fused self kernel failed: {type(ex).__name__}: {ex}",
            )
        return WanFusedAttemptResult(
            output=None,
            reason_code=E_WAN_FUSED_KERNEL_RUNTIME_ERROR,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )


def try_fused_cross_attention(
    *,
    mode: str | WanFusedMode | None,
    x: torch.Tensor,
    context: torch.Tensor,
    q_proj: Any,
    k_proj: Any,
    v_proj: Any,
    o_proj: Any,
    norm_q_weight: torch.Tensor,
    norm_k_weight: torch.Tensor,
    rope_cos_q: torch.Tensor | None,
    rope_sin_q: torch.Tensor | None,
    rope_cos_k: torch.Tensor | None,
    rope_sin_k: torch.Tensor | None,
    dropout_p: float,
) -> WanFusedAttemptResult:
    fused_mode = resolve_effective_wan_fused_mode(mode)
    if fused_mode is WanFusedMode.OFF:
        return WanFusedAttemptResult(output=None, reason_code=E_WAN_FUSED_DISABLED)

    unavailable = _maybe_return_unavailable(mode=fused_mode)
    if unavailable is not None:
        return unavailable

    _validate_common_inputs(x=x, dropout_p=dropout_p)
    if context.ndim != 3:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=f"WAN fused cross expects context [B,S,Cctx]; got shape={tuple(context.shape)}.",
        )
    if int(context.shape[0]) != int(x.shape[0]):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=(
                "WAN fused cross batch mismatch between x and context: "
                f"x.B={int(x.shape[0])} context.B={int(context.shape[0])}."
            ),
        )

    bsz, q_len, channels = (int(x.shape[0]), int(x.shape[1]), int(x.shape[2]))
    kv_len = int(context.shape[1])
    ctx_dim = int(context.shape[2])

    _validate_rope_tensor(tensor=rope_cos_q, expected_len=q_len, label="rope_cos_q")
    _validate_rope_tensor(tensor=rope_sin_q, expected_len=q_len, label="rope_sin_q")
    _validate_rope_tensor(tensor=rope_cos_k, expected_len=kv_len, label="rope_cos_k")
    _validate_rope_tensor(tensor=rope_sin_k, expected_len=kv_len, label="rope_sin_k")

    head_dim = int(rope_cos_q.shape[-1])
    num_heads = _resolve_head_count(channels=channels, head_dim=head_dim, field_name="cross")
    if int(rope_cos_k.shape[-1]) != head_dim or int(rope_sin_k.shape[-1]) != head_dim:
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=(
                "WAN fused cross requires matching RoPE head_dim between query/key tensors. "
                f"got q={head_dim} k_cos={int(rope_cos_k.shape[-1])} k_sin={int(rope_sin_k.shape[-1])}."
            ),
        )

    _validate_arch_dtype_head_dim(device=x.device, dtype=x.dtype, head_dim=head_dim)

    w_q, b_q = _resolve_linear_weight_bias(q_proj, expected_out=channels, expected_in=channels, label="q_proj")
    w_k, b_k = _resolve_linear_weight_bias(k_proj, expected_out=channels, expected_in=ctx_dim, label="k_proj")
    w_v, b_v = _resolve_linear_weight_bias(v_proj, expected_out=channels, expected_in=ctx_dim, label="v_proj")
    w_o, b_o = _resolve_linear_weight_bias(o_proj, expected_out=channels, expected_in=channels, label="o_proj")

    if tuple(norm_q_weight.shape) != (channels,) or tuple(norm_k_weight.shape) != (channels,):
        _fail(
            code=E_WAN_FUSED_INVALID_SHAPE,
            message=(
                "WAN fused cross expects norm weights shaped [C]. "
                f"got norm_q={tuple(norm_q_weight.shape)} norm_k={tuple(norm_k_weight.shape)} expected={(channels,)}."
            ),
        )

    w_q_contract = w_q.t().contiguous().view(channels, num_heads, head_dim)
    w_k_contract = w_k.t().contiguous().view(ctx_dim, num_heads, head_dim)
    w_v_contract = w_v.t().contiguous().view(ctx_dim, num_heads, head_dim)
    w_out = w_o.t().contiguous().view(num_heads, head_dim, channels)

    b_q_contract = b_q.contiguous().view(num_heads, head_dim) if b_q is not None else None
    b_k_contract = b_k.contiguous().view(num_heads, head_dim) if b_k is not None else None
    b_v_contract = b_v.contiguous().view(num_heads, head_dim) if b_v is not None else None

    try:
        out = torch.ops.wan_fused_v1.cross_fwd(
            x,
            context,
            w_q_contract,
            b_q_contract,
            norm_q_weight,
            rope_cos_q,
            rope_sin_q,
            w_k_contract,
            b_k_contract,
            norm_k_weight,
            rope_cos_k,
            rope_sin_k,
            w_v_contract,
            b_v_contract,
            w_out,
            b_o,
        )
        if tuple(out.shape) != (bsz, q_len, channels):
            _fail(
                code=E_WAN_FUSED_INVALID_SHAPE,
                message=(
                    "WAN fused cross returned unexpected shape: "
                    f"got {tuple(out.shape)} expected {(bsz, q_len, channels)}."
                ),
            )
        return WanFusedAttemptResult(output=out, reason_code=None)
    except WanFusedContractError:
        raise
    except Exception as ex:
        if fused_mode is WanFusedMode.FORCE:
            _fail(
                code=E_WAN_FUSED_KERNEL_RUNTIME_ERROR,
                message=f"WAN fused cross kernel failed: {type(ex).__name__}: {ex}",
            )
        return WanFusedAttemptResult(
            output=None,
            reason_code=E_WAN_FUSED_KERNEL_RUNTIME_ERROR,
            reason_detail=f"{type(ex).__name__}: {ex}",
        )
