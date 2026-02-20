"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime “operations” layer for model execution (weight fetch/casting, GGUF dequantization, streaming/swap hooks).
Provides `CodexOperations` implementations used by runtimes to run common ops while supporting manual casting, memory streaming,
and GGUF-specific paths.

Symbols (top-level; keep in sync; no ghosts):
- `_parse_positive_int` (function): Parses env/config ints with a non-negative clamp and default fallback.
- `_log_weight_fetch` (function): Debug logger for repeated weight/bias fetch events (with per-layer mute limit).
- `_codexpack_require_sm86_cuda` (function): Validates CUDA + SM86+ prerequisites for CodexPack packed-kernel execution.
- `_codexpack_q4k_tilepack_linear` (function): Calls the CodexPack packed linear kernel op (fail loud when unavailable).
- `OperationContext` (dataclass): Thread-local-ish operation config (device/dtype + manual cast + weight-format hints).
- `StreamStashEntry` (dataclass): One stashed streaming entry (weight/bias + bookkeeping for swap/stream state).
- `StreamStash` (dataclass): Tracks streaming stash state across ops (used to coordinate stream workers and cleanup).
- `get_operation_context` (function): Returns the active `OperationContext` (resolved from env/defaults).
- `_resolve_device` (function): Resolves a default device for operations (explicit override vs runtime defaults).
- `_resolve_dtype` (function): Resolves a default dtype for operations (explicit override vs runtime defaults).
- `get_weight_and_bias` (function): Fetches weight/bias tensors for a layer, applying patches and optional streaming hooks.
- `weights_manual_cast` (function): Fetches weight/bias for a layer, moving/casting to the requested device/dtype (optionally streamed).
- `main_stream_worker` (function): Stream worker used to stage weights/biases for streamed execution.
- `cleanup_cache` (function): Best-effort cleanup of internal caches/stashes (used between runs/model switches).
- `_select_operations_class` (function): Chooses the correct operations implementation (vanilla vs GGUF) from context/override.
- `CodexOperations` (class): Base operations implementation used by runtimes (contains many op methods: linear/conv/norm/attention helpers).
- `CodexOperationsGGUF` (class): GGUF-aware operations implementation (handles GGUF parameter containers and dequantization paths).
- `using_codex_operations` (function): Context manager installing an operations instance + operation context for a block of execution.
- `shift_manual_cast` (function): Applies manual-cast toggles to a model/module tree for runtime execution.
- `automatic_memory_management` (function): Context manager enabling/disabling automatic memory management policies for a block.
- `DynamicSwapInstaller` (class): Helper for installing dynamic swap hooks/policies into model execution.
"""

from __future__ import annotations

import contextlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch
from apps.backend.runtime import utils
from apps.backend.runtime.memory import memory_management, stream
from apps.backend.runtime.misc.autocast import autocast_disabled
from .operations_gguf import (
    CodexPackLinearQ4KTilepackV1Parameter,
    CodexParameter,
    dequantize_tensor,
    dequantize_tensor_for_forward,
    is_dequant_forward_cache_enabled,
)

logger = logging.getLogger("backend.runtime.ops.operations")


def _codexpack_require_sm86_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise RuntimeError(f"CodexPack packed-kernel execution requires CUDA; got device={device}.")
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) < (8, 6):
        raise RuntimeError(
            "CodexPack packed-kernel execution requires NVIDIA SM86+ (RTX 30xx baseline). "
            f"got compute_capability={(major, minor)}."
        )


def _codexpack_q4k_tilepack_linear(
    x: torch.Tensor,
    *,
    weight: CodexPackLinearQ4KTilepackV1Parameter,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    if x.dtype != torch.float16:
        raise RuntimeError(
            "CodexPack packed-kernel execution v1 only supports fp16 activations. "
            f"got x.dtype={x.dtype}."
        )
    _codexpack_require_sm86_cuda(x.device)

    if x.shape[-1] != weight.in_features:
        raise RuntimeError(
            "CodexPack packed linear input feature mismatch: "
            f"x.shape[-1]={x.shape[-1]} weight.in_features={weight.in_features}."
        )

    import importlib

    _codexpack_cuda = importlib.import_module("apps.backend.runtime.ops.codexpack_cuda")
    _codexpack_cuda.available()
    ops = getattr(torch, "ops", None)
    if ops is None or not hasattr(ops, "codexpack") or not hasattr(ops.codexpack, "q4k_tilepack_linear"):
        hint = (
            "Build in-place:\n"
            "  CODEX_ROOT=\"$(git rev-parse --show-toplevel)\"\n"
            "  cd \"$CODEX_ROOT/apps/backend/runtime/kernels/codexpack\"\n"
            "  PYTHONPATH=\"$CODEX_ROOT\" \"$CODEX_ROOT/.venv/bin/python\" setup.py build_ext --inplace\n"
        )
        last = _codexpack_cuda.last_error()
        suffix = f"\nLast error: {last}" if last else ""
        raise RuntimeError(
            "CodexPack packed kernel extension is not available (missing `torch.ops.codexpack.q4k_tilepack_linear`)."
            f"{suffix}\n{hint}"
        )

    if bias is not None:
        if bias.dtype != torch.float16:
            raise RuntimeError(
                "CodexPack packed linear v1 requires fp16 bias when present. "
                f"got bias.dtype={bias.dtype}."
            )
        if bias.numel() != weight.out_features:
            raise RuntimeError(
                "CodexPack packed linear bias size mismatch: "
                f"bias.numel()={bias.numel()} weight.out_features={weight.out_features}."
            )

    return torch.ops.codexpack.q4k_tilepack_linear(x, weight, weight.out_features, weight.in_features, bias)


def _parse_positive_int(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        numeric = int(value)
    except Exception:
        return default
    return max(0, numeric)


_FETCH_LOG_LIMIT = _parse_positive_int(os.getenv("CODEX_WEIGHT_FETCH_LOG_LIMIT"), 10)
_fetch_log_counts: Dict[str, int] = defaultdict(int)


def _log_weight_fetch(layer_name: str, has_weight: bool, has_bias: bool, has_patches: bool) -> None:
    if _FETCH_LOG_LIMIT <= 0:
        return
    count = _fetch_log_counts[layer_name]
    _fetch_log_counts[layer_name] = count + 1
    if count < _FETCH_LOG_LIMIT:
        logger.debug(
            "Fetched weight/bias for %s (weight=%s, bias=%s, patches=%s)",
            layer_name,
            "yes" if has_weight else "no",
            "yes" if has_bias else "no",
            has_patches,
        )
    elif count == _FETCH_LOG_LIMIT:
        logger.debug(
            "Muted weight/bias fetch logs for %s after %d occurrences",
            layer_name,
            _FETCH_LOG_LIMIT,
        )


@dataclass
class OperationContext:
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None
    manual_cast_enabled: bool = False
    weight_format: Optional[str] = None

    def describe(self) -> str:
        return (
            f"device={self.device}, dtype={self.dtype}, manual_cast={self.manual_cast_enabled}, "
            f"weight_format={self.weight_format}"
        )


@dataclass
class StreamStashEntry:
    weight: Optional[torch.Tensor]
    bias: Optional[torch.Tensor]
    event: torch.cuda.Event


@dataclass
class StreamStash:
    entries: Dict[int, StreamStashEntry] = field(default_factory=dict)

    def add(self, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor], event: torch.cuda.Event) -> None:
        if event is None:
            return
        self.entries[id(event)] = StreamStashEntry(weight=weight, bias=bias, event=event)

    def collect_finished(self) -> None:
        finished = [key for key, entry in self.entries.items() if entry.event.query()]
        for key in finished:
            self.entries.pop(key, None)

    def clear(self) -> None:
        self.entries.clear()


_operation_context = OperationContext()
_stream_stash = StreamStash()


def get_operation_context() -> OperationContext:
    return _operation_context


def _resolve_device(default: Optional[torch.device] = None) -> Optional[torch.device]:
    ctx = get_operation_context()
    return ctx.device if ctx.device is not None else default


def _resolve_dtype(default: torch.dtype = torch.float32) -> torch.dtype:
    ctx = get_operation_context()
    return ctx.dtype if ctx.dtype is not None else default


def get_weight_and_bias(
    layer,
    weight_args: Optional[Dict[str, object]] = None,
    bias_args: Optional[Dict[str, object]] = None,
    weight_fn=None,
    bias_fn=None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    scale_weight = getattr(layer, "scale_weight", None)
    patches = getattr(layer, "codex_online_loras", None)

    weight_patches = patches.get("weight") if patches is not None else None
    bias_patches = patches.get("bias") if patches is not None else None

    weight = None
    if layer.weight is not None:
        weight = layer.weight
        apply_weight_args = weight_args is not None
        if weight_fn is not None:
            if (
                weight_fn is dequantize_tensor
                and is_dequant_forward_cache_enabled()
                and weight_args is not None
                and (device := weight_args.get("device")) is not None
            ):
                source_is_quantized = isinstance(weight, CodexParameter) and weight.qtype is not None
                weight = dequantize_tensor_for_forward(
                    weight,
                    target_device=device,
                    target_dtype=weight_args.get("dtype") if isinstance(weight_args.get("dtype"), torch.dtype) else None,
                    non_blocking=bool(weight_args.get("non_blocking", False)),
                )
                if source_is_quantized:
                    apply_weight_args = False
            else:
                if weight_args is not None and (device := weight_args.get("device")) is not None:
                    weight = weight.to(device=device)
                weight = weight_fn(weight)
        if apply_weight_args and weight_args is not None:
            weight = weight.to(**weight_args)
        if scale_weight is not None:
            weight = weight * scale_weight.to(device=weight.device, dtype=weight.dtype)
        if weight_patches is not None:
            # Local import to avoid circular imports during package init
            from apps.backend.patchers.lora import merge_lora_to_weight
            weight = merge_lora_to_weight(
                patches=weight_patches,
                weight=weight,
                key="online weight lora",
                computation_dtype=weight.dtype,
            )

    bias = None
    if layer.bias is not None:
        bias = layer.bias
        apply_bias_args = bias_args is not None
        if bias_fn is not None:
            if (
                bias_fn is dequantize_tensor
                and is_dequant_forward_cache_enabled()
                and bias_args is not None
                and (device := bias_args.get("device")) is not None
            ):
                source_is_quantized = isinstance(bias, CodexParameter) and bias.qtype is not None
                bias = dequantize_tensor_for_forward(
                    bias,
                    target_device=device,
                    target_dtype=bias_args.get("dtype") if isinstance(bias_args.get("dtype"), torch.dtype) else None,
                    non_blocking=bool(bias_args.get("non_blocking", False)),
                )
                if source_is_quantized:
                    apply_bias_args = False
            else:
                if bias_args is not None and (device := bias_args.get("device")) is not None:
                    bias = bias.to(device=device)
                bias = bias_fn(bias)
        if apply_bias_args and bias_args is not None:
            bias = bias.to(**bias_args)
        if bias_patches is not None:
            from apps.backend.patchers.lora import merge_lora_to_weight
            bias = merge_lora_to_weight(
                patches=bias_patches,
                weight=bias,
                key="online bias lora",
                computation_dtype=bias.dtype,
            )

    _log_weight_fetch(
        layer.__class__.__name__,
        has_weight=weight is not None,
        has_bias=bias is not None,
        has_patches=patches is not None,
    )
    return weight, bias


def weights_manual_cast(
    layer,
    x: torch.Tensor,
    skip_weight_dtype: bool = False,
    skip_bias_dtype: bool = False,
    weight_fn=None,
    bias_fn=None,
    *,
    target_dtype: Optional[torch.dtype] = None,
):
    weight, bias, signal = None, None, None
    non_blocking = getattr(x.device, "type", None) != "mps"

    target_dtype = x.dtype if target_dtype is None else target_dtype
    target_device = x.device

    if skip_weight_dtype:
        weight_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        weight_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if skip_bias_dtype:
        bias_args = dict(device=target_device, non_blocking=non_blocking)
    else:
        bias_args = dict(device=target_device, dtype=target_dtype, non_blocking=non_blocking)

    if stream.should_use_stream():
        with stream.stream_context()(stream.mover_stream):
            weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)
            signal = stream.mover_stream.record_event()
    else:
        weight, bias = get_weight_and_bias(layer, weight_args, bias_args, weight_fn=weight_fn, bias_fn=bias_fn)

    return weight, bias, signal


@contextlib.contextmanager
def main_stream_worker(weight, bias, signal):
    if signal is None or not stream.should_use_stream():
        yield
        return

    with stream.stream_context()(stream.current_stream):
        stream.current_stream.wait_event(signal)
        yield
        finished_signal = stream.current_stream.record_event()
        _stream_stash.add(weight, bias, finished_signal)

    _stream_stash.collect_finished()


def cleanup_cache():
    if not stream.should_use_stream():
        return

    stream.current_stream.synchronize()
    stream.mover_stream.synchronize()
    _stream_stash.clear()


def _select_operations_class(context: OperationContext, override=None):
    if override is not None:
        return override
    if context.weight_format is None:
        return CodexOperations
    if context.weight_format == "gguf":
        return CodexOperationsGGUF
    raise NotImplementedError(
        "Unsupported pre-quant weight format (NF4/FP4 is not supported). "
        f"Got weight_format={context.weight_format!r}. Convert the model to GGUF or use a safetensors fp16/bf16/fp32 checkpoint."
    )


class CodexOperations:
    class Linear(torch.nn.Module):
        def __init__(self, in_features, out_features, *args, **kwargs):
            super().__init__()
            ctx = get_operation_context()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.has_bias = bool(kwargs.get("bias", True))

            device = ctx.device
            dtype = ctx.dtype or torch.float32
            weight = torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
            torch.nn.init.xavier_uniform_(weight)
            self.weight = torch.nn.Parameter(weight)
            self.bias = (
                torch.nn.Parameter(torch.zeros((self.out_features,), device=device, dtype=dtype))
                if self.has_bias
                else None
            )
            self.scale_weight = None
            self.parameters_manual_cast = ctx.manual_cast_enabled

        def _ensure_params(self, device, dtype):
            if self.weight is not None:
                return
            weight = torch.empty((self.out_features, self.in_features), device=device, dtype=dtype)
            torch.nn.init.xavier_uniform_(weight)
            self.weight = torch.nn.Parameter(weight)
            if self.has_bias:
                self.bias = torch.nn.Parameter(torch.zeros((self.out_features,), device=device, dtype=dtype))
            else:
                self.bias = None

        def forward(self, x):
            self._ensure_params(x.device, x.dtype)
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.linear(x, weight, bias)
            weight, bias = get_weight_and_bias(self)
            return torch.nn.functional.linear(x, weight, bias)

    class _BaseConvMixin:
        def _init_common(self):
            ctx = get_operation_context()
            self.parameters_manual_cast = ctx.manual_cast_enabled

        def _forward_conv(self, x, forward_fn):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return forward_fn(x, weight, bias)
            weight, bias = get_weight_and_bias(self)
            return forward_fn(x, weight, bias)

    class Conv2d(_BaseConvMixin, torch.nn.Conv2d):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self._init_common()

        def reset_parameters(self):
            return None

        def forward(self, x):
            return self._forward_conv(x, super()._conv_forward)

    class Conv3d(_BaseConvMixin, torch.nn.Conv3d):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self._init_common()

        def reset_parameters(self):
            return None

        def forward(self, x):
            return self._forward_conv(x, super()._conv_forward)

    class Conv1d(_BaseConvMixin, torch.nn.Conv1d):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self._init_common()

        def reset_parameters(self):
            return None

        def forward(self, x):
            return self._forward_conv(x, super()._conv_forward)

    class ConvTranspose2d(_BaseConvMixin, torch.nn.ConvTranspose2d):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self._init_common()

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            def fn(_x, weight, bias):
                output_padding = self._output_padding(
                    _x, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation
                )
                return torch.nn.functional.conv_transpose2d(
                    _x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
                )

            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return fn(x, weight, bias)
            weight, bias = get_weight_and_bias(self)
            return fn(x, weight, bias)

    class ConvTranspose1d(_BaseConvMixin, torch.nn.ConvTranspose1d):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self._init_common()

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            def fn(_x, weight, bias):
                output_padding = self._output_padding(
                    _x, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation
                )
                return torch.nn.functional.conv_transpose1d(
                    _x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
                )

            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return fn(x, weight, bias)
            weight, bias = get_weight_and_bias(self)
            return fn(x, weight, bias)

    class ConvTranspose3d(_BaseConvMixin, torch.nn.ConvTranspose3d):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self._init_common()

        def reset_parameters(self):
            return None

        def forward(self, x, output_size=None):
            def fn(_x, weight, bias):
                output_padding = self._output_padding(
                    _x, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation
                )
                return torch.nn.functional.conv_transpose3d(
                    _x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
                )

            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return fn(x, weight, bias)
            weight, bias = get_weight_and_bias(self)
            return fn(x, weight, bias)

    class GroupNorm(torch.nn.GroupNorm):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = ctx.manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if not x.is_floating_point():
                raise TypeError(f"GroupNorm expects a floating-point input tensor; got dtype={x.dtype}.")
            norm_dtype = x.dtype
            weight_args = {"dtype": norm_dtype}
            bias_args = {"dtype": norm_dtype}

            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x, target_dtype=norm_dtype)
                with main_stream_worker(weight, bias, signal):
                    with autocast_disabled(x.device.type):
                        return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)

            weight, bias = get_weight_and_bias(self, weight_args, bias_args)
            with autocast_disabled(x.device.type):
                return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)

    class LayerNorm(torch.nn.LayerNorm):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            if ctx.dtype is not None:
                kwargs["dtype"] = ctx.dtype
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = ctx.manual_cast_enabled

        def reset_parameters(self):
            return None

        def forward(self, x):
            if not x.is_floating_point():
                raise TypeError(f"LayerNorm expects a floating-point input tensor; got dtype={x.dtype}.")
            norm_dtype = x.dtype
            weight_args = {"dtype": norm_dtype}
            bias_args = {"dtype": norm_dtype}

            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x, target_dtype=norm_dtype)
                with main_stream_worker(weight, bias, signal):
                    with autocast_disabled(x.device.type):
                        return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)

            weight, bias = get_weight_and_bias(self, weight_args, bias_args)
            with autocast_disabled(x.device.type):
                return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)

    class Embedding(torch.nn.Embedding):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = ctx.manual_cast_enabled
            self.bias = None

        def reset_parameters(self):
            self.bias = None
            return None

        def forward(self, x):
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(
                    self,
                    x,
                    skip_weight_dtype=True,
                    skip_bias_dtype=True,
                )
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.embedding(
                        x,
                        weight,
                        self.padding_idx,
                        self.max_norm,
                        self.norm_type,
                        self.scale_grad_by_freq,
                        self.sparse,
                    )
            return super().forward(x)


class CodexOperationsGGUF(CodexOperations):
    class _GGUFMixin:
        def _gguf_init_dummy(self) -> None:
            ctx = get_operation_context()
            dtype = ctx.dtype or torch.float32
            device = ctx.device
            self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
            self.parameters_manual_cast = ctx.manual_cast_enabled

        def _gguf_load_params(self, state_dict, prefix: str) -> None:
            if not hasattr(self, "dummy"):
                return
            computation_dtype = self.dummy.dtype
            if computation_dtype not in (torch.float16, torch.bfloat16):
                computation_dtype = torch.float16
            if prefix + "weight" in state_dict:
                self.weight = utils.tensor2parameter(state_dict[prefix + "weight"].to(device=self.dummy.device))
                if hasattr(self.weight, "computation_dtype"):
                    self.weight.computation_dtype = computation_dtype
            if prefix + "bias" in state_dict:
                self.bias = utils.tensor2parameter(state_dict[prefix + "bias"].to(device=self.dummy.device))
                if hasattr(self.bias, "computation_dtype"):
                    self.bias.computation_dtype = computation_dtype
            del self.dummy

        def _apply(self, fn, recurse=True):
            for name, param in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, name, utils.tensor2parameter(fn(param)))
            return self

    class Linear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            ctx = get_operation_context()
            dtype = ctx.dtype or torch.float32
            device = ctx.device
            self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
            self.weight = None
            self.bias = None
            self.parameters_manual_cast = ctx.manual_cast_enabled

        @staticmethod
        def _normalize_loaded_linear_tensor(
            tensor_obj: torch.Tensor,
            *,
            target_device: torch.device,
            computation_dtype: torch.dtype,
            is_weight: bool,
        ) -> torch.Tensor:
            loaded = tensor_obj.to(device=target_device)
            if isinstance(loaded, CodexPackLinearQ4KTilepackV1Parameter):
                return loaded
            if isinstance(loaded, CodexParameter):
                loaded.computation_dtype = computation_dtype
                # Transformers UMT5 casts FFN activations to `wo.weight.dtype` unless it is int8.
                # Keep packed GGUF bytes but expose int8 storage so activations stay floating.
                if is_weight and loaded.dtype == torch.uint8:
                    loaded = loaded.copy_with_data(loaded.data.view(torch.int8))
                    loaded.computation_dtype = computation_dtype
                return loaded
            if not loaded.is_floating_point():
                loaded = loaded.to(dtype=computation_dtype)
            return loaded

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                computation_dtype = self.dummy.dtype
                if computation_dtype not in (torch.float16, torch.bfloat16):
                    computation_dtype = torch.float16
                if prefix + "weight" in state_dict:
                    self.weight = utils.tensor2parameter(
                        self._normalize_loaded_linear_tensor(
                            state_dict[prefix + "weight"],
                            target_device=self.dummy.device,
                            computation_dtype=computation_dtype,
                            is_weight=True,
                        )
                    )
                    if hasattr(self.weight, "computation_dtype"):
                        self.weight.computation_dtype = computation_dtype
                if prefix + "bias" in state_dict:
                    self.bias = utils.tensor2parameter(
                        self._normalize_loaded_linear_tensor(
                            state_dict[prefix + "bias"],
                            target_device=self.dummy.device,
                            computation_dtype=computation_dtype,
                            is_weight=False,
                        )
                    )
                    if hasattr(self.bias, "computation_dtype"):
                        self.bias.computation_dtype = computation_dtype
                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    target_device = state_dict[prefix + "weight"].device
                    computation_dtype = getattr(self.weight, "computation_dtype", torch.float16)
                    if computation_dtype not in (torch.float16, torch.bfloat16):
                        computation_dtype = torch.float16
                    self.weight = self._normalize_loaded_linear_tensor(
                        state_dict[prefix + "weight"],
                        target_device=target_device,
                        computation_dtype=computation_dtype,
                        is_weight=True,
                    )
                if prefix + "bias" in state_dict:
                    target_device = state_dict[prefix + "bias"].device
                    computation_dtype = getattr(self.weight, "computation_dtype", torch.float16)
                    if computation_dtype not in (torch.float16, torch.bfloat16):
                        computation_dtype = torch.float16
                    self.bias = self._normalize_loaded_linear_tensor(
                        state_dict[prefix + "bias"],
                        target_device=target_device,
                        computation_dtype=computation_dtype,
                        is_weight=False,
                    )

        def _apply(self, fn, recurse=True):
            for name, param in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, name, utils.tensor2parameter(fn(param)))
            return self

        def forward(self, x):
            if not torch.is_floating_point(x):
                raise RuntimeError(
                    "GGUF Linear received non-floating activations "
                    f"(dtype={x.dtype}). Expected dequantized floating activations before dense matmul."
                )
            if isinstance(self.weight, CodexPackLinearQ4KTilepackV1Parameter):
                if self.bias is not None:
                    self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(device=x.device, dtype=x.dtype))
                return _codexpack_q4k_tilepack_linear(x, weight=self.weight, bias=self.bias)

            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))
            if (
                self.weight is not None
                and self.weight.dtype != x.dtype
                and not isinstance(self.weight, (CodexParameter, CodexPackLinearQ4KTilepackV1Parameter))
            ):
                self.weight = utils.tensor2parameter(self.weight.to(x.dtype))
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=None,
                skip_bias_dtype=True,
            )
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.linear(x, weight, bias)

    class Embedding(torch.nn.Embedding):
        def __init__(self, *args, **kwargs):
            ctx = get_operation_context()
            supplied_weight = kwargs.get("_weight", None)
            if supplied_weight is None and len(args) > 7:
                supplied_weight = args[7]
            # Keep GGUF Embedding construction lazy: build placeholder metadata on
            # `meta` so we avoid allocating a full real tensor before state_dict load.
            placeholder_mode = supplied_weight is None
            if placeholder_mode:
                kwargs["device"] = torch.device("meta")
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = ctx.manual_cast_enabled
            if placeholder_mode:
                dtype = ctx.dtype or torch.float32
                device = ctx.device
                self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
            self.bias = None

        def reset_parameters(self):
            self.bias = None
            return None

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                key = prefix + "weight"
                computation_dtype = self.dummy.dtype
                if computation_dtype not in (torch.float16, torch.bfloat16):
                    computation_dtype = torch.float16
                if key in state_dict:
                    self.weight = utils.tensor2parameter(state_dict[key].to(device=self.dummy.device))
                    if hasattr(self.weight, "computation_dtype"):
                        self.weight.computation_dtype = computation_dtype
                elif strict:
                    missing_keys.append(key)
                del self.dummy
            else:
                key = prefix + "weight"
                if key in state_dict:
                    self.weight = state_dict[key]
                elif strict:
                    missing_keys.append(key)

        def _apply(self, fn, recurse=True):
            for name, param in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, name, utils.tensor2parameter(fn(param)))
            return self

        def forward(self, x):
            if hasattr(self, "dummy"):
                raise RuntimeError(
                    "GGUF Embedding forward called before weight load. "
                    "Call load_state_dict(...) before forward."
                )
            if self.weight is None or getattr(self.weight, "is_meta", False):
                raise RuntimeError(
                    "GGUF Embedding weight is missing after load. "
                    "Ensure state_dict contains 'weight'."
                )
            target_compute_dtype = getattr(self.weight, "computation_dtype", None)
            if not isinstance(target_compute_dtype, torch.dtype) or not torch.empty((), dtype=target_compute_dtype).is_floating_point():
                target_compute_dtype = torch.float16 if x.device.type == "cuda" else torch.float32
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                skip_weight_dtype=False,
                skip_bias_dtype=True,
                target_dtype=target_compute_dtype,
            )
            with main_stream_worker(weight, bias, signal):
                return torch.nn.functional.embedding(
                    x,
                    weight,
                    self.padding_idx,
                    self.max_norm,
                    self.norm_type,
                    self.scale_grad_by_freq,
                    self.sparse,
                )

    class Conv1d(_GGUFMixin, CodexOperations.Conv1d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
            )
            with main_stream_worker(weight, bias, signal):
                return super()._conv_forward(x, weight, bias)

    class Conv2d(_GGUFMixin, CodexOperations.Conv2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
            )
            with main_stream_worker(weight, bias, signal):
                return super()._conv_forward(x, weight, bias)

    class Conv3d(_GGUFMixin, CodexOperations.Conv3d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
            )
            with main_stream_worker(weight, bias, signal):
                return super()._conv_forward(x, weight, bias)

    class ConvTranspose1d(_GGUFMixin, CodexOperations.ConvTranspose1d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x, output_size=None):
            def fn(_x, weight, bias):
                output_padding = self._output_padding(
                    _x, output_size, self.stride, self.padding, self.kernel_size, 1, self.dilation
                )
                return torch.nn.functional.conv_transpose1d(
                    _x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
                )

            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
            )
            with main_stream_worker(weight, bias, signal):
                return fn(x, weight, bias)

    class ConvTranspose2d(_GGUFMixin, CodexOperations.ConvTranspose2d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x, output_size=None):
            def fn(_x, weight, bias):
                output_padding = self._output_padding(
                    _x, output_size, self.stride, self.padding, self.kernel_size, 2, self.dilation
                )
                return torch.nn.functional.conv_transpose2d(
                    _x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
                )

            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
            )
            with main_stream_worker(weight, bias, signal):
                return fn(x, weight, bias)

    class ConvTranspose3d(_GGUFMixin, CodexOperations.ConvTranspose3d):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x, output_size=None):
            def fn(_x, weight, bias):
                output_padding = self._output_padding(
                    _x, output_size, self.stride, self.padding, self.kernel_size, 3, self.dilation
                )
                return torch.nn.functional.conv_transpose3d(
                    _x, weight, bias, self.stride, self.padding, output_padding, self.groups, self.dilation
                )

            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
            )
            with main_stream_worker(weight, bias, signal):
                return fn(x, weight, bias)

    class GroupNorm(_GGUFMixin, CodexOperations.GroupNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x):
            if not x.is_floating_point():
                raise TypeError(f"GroupNorm expects a floating-point input tensor; got dtype={x.dtype}.")
            norm_dtype = x.dtype
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
                target_dtype=norm_dtype,
            )
            with main_stream_worker(weight, bias, signal):
                with autocast_disabled(x.device.type):
                    return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)

    class LayerNorm(_GGUFMixin, CodexOperations.LayerNorm):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gguf_init_dummy()

        def _load_from_state_dict(
            self,
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        ):
            if hasattr(self, "dummy"):
                self._gguf_load_params(state_dict, prefix)
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def forward(self, x):
            if not x.is_floating_point():
                raise TypeError(f"LayerNorm expects a floating-point input tensor; got dtype={x.dtype}.")
            norm_dtype = x.dtype
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
                bias_fn=dequantize_tensor,
                target_dtype=norm_dtype,
            )
            with main_stream_worker(weight, bias, signal):
                with autocast_disabled(x.device.type):
                    return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)


@contextlib.contextmanager
def using_codex_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, weight_format=None):
    if weight_format is not None:
        if not isinstance(weight_format, str):
            raise TypeError(f"weight_format must be a string or None; got {type(weight_format).__name__}.")
        if weight_format != "gguf":
            raise NotImplementedError(
                "NF4/FP4 is not supported. "
                f"Got weight_format={weight_format!r}. Convert the model to GGUF or use a safetensors fp16/bf16/fp32 checkpoint."
            )

    global _operation_context
    previous_context = _operation_context
    _operation_context = OperationContext(
        device=device,
        dtype=dtype,
        manual_cast_enabled=manual_cast_enabled,
        weight_format=weight_format,
    )

    operations_class = _select_operations_class(_operation_context, operations)
    op_names = [
        "Linear",
        "Conv1d",
        "Conv2d",
        "Conv3d",
        "ConvTranspose1d",
        "ConvTranspose2d",
        "ConvTranspose3d",
        "GroupNorm",
        "LayerNorm",
        "Embedding",
    ]
    backups = {name: getattr(torch.nn, name) for name in op_names}

    try:
        for name in op_names:
            setattr(torch.nn, name, getattr(operations_class, name))
        logger.debug("Installed Codex operations (%s) with context %s", operations_class.__name__, _operation_context.describe())
        yield
    finally:
        for name, original in backups.items():
            setattr(torch.nn, name, original)
        logger.debug("Restored torch.nn operations to originals")
        _operation_context = previous_context


def shift_manual_cast(model, enabled):
    for module in model.modules():
        if hasattr(module, "parameters_manual_cast"):
            module.parameters_manual_cast = enabled
    return


@contextlib.contextmanager
def automatic_memory_management():
    memory_management.manager.free_memory(
        memory_required=3 * 1024 * 1024 * 1024,
        device=memory_management.manager.primary_device(),
    )

    module_list = []

    original_init = torch.nn.Module.__init__
    original_to = torch.nn.Module.to

    def patched_init(self, *args, **kwargs):
        module_list.append(self)
        return original_init(self, *args, **kwargs)

    def patched_to(self, *args, **kwargs):
        module_list.append(self)
        return original_to(self, *args, **kwargs)

    try:
        torch.nn.Module.__init__ = patched_init
        torch.nn.Module.to = patched_to
        yield
    finally:
        torch.nn.Module.__init__ = original_init
        torch.nn.Module.to = original_to

    start = time.perf_counter()
    module_list = set(module_list)

    for module in module_list:
        module.cpu()

    memory_management.manager.soft_empty_cache()
    elapsed = time.perf_counter() - start
    logger.info("Automatic Memory Management: %d Modules in %.2f seconds.", len(module_list), elapsed)


class DynamicSwapInstaller:
    @staticmethod
    def _install_module(module: torch.nn.Module, target_device: torch.device):
        original_class = module.__class__
        module.__dict__["codex_backup_original_class"] = original_class

        def hacked_get_attr(self, name: str):
            if "_parameters" in self.__dict__:
                parameters = self.__dict__["_parameters"]
                if name in parameters:
                    param = parameters[name]
                    if param is None:
                        return None
                    if isinstance(param, torch.nn.Parameter):
                        return torch.nn.Parameter(param.to(target_device), requires_grad=param.requires_grad)
                    return param.to(target_device)
            if "_buffers" in self.__dict__:
                buffers = self.__dict__["_buffers"]
                if name in buffers:
                    return buffers[name].to(target_device)
            return super(original_class, self).__getattr__(name)

        module.__class__ = type(
            "DynamicSwap_" + original_class.__name__,
            (original_class,),
            {"__getattr__": hacked_get_attr},
        )

    @staticmethod
    def _uninstall_module(module: torch.nn.Module):
        if "codex_backup_original_class" in module.__dict__:
            module.__class__ = module.__dict__.pop("codex_backup_original_class")

    @staticmethod
    def install_model(model: torch.nn.Module, target_device: torch.device):
        for module in model.modules():
            DynamicSwapInstaller._install_module(module, target_device)

    @staticmethod
    def uninstall_model(model: torch.nn.Module):
        for module in model.modules():
            DynamicSwapInstaller._uninstall_module(module)
