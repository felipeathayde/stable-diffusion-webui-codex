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
from .operations_gguf import dequantize_tensor

logger = logging.getLogger("backend.runtime.ops.operations")


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
    bnb_dtype: Optional[str] = None

    def describe(self) -> str:
        return (
            f"device={self.device}, dtype={self.dtype}, manual_cast={self.manual_cast_enabled}, "
            f"bnb_dtype={self.bnb_dtype}"
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
    if patches is None:
        patches = getattr(layer, "forge_online_loras", None)

    weight_patches = patches.get("weight") if patches is not None else None
    bias_patches = patches.get("bias") if patches is not None else None

    weight = None
    if layer.weight is not None:
        weight = layer.weight
        if weight_fn is not None:
            if weight_args is not None and (device := weight_args.get("device")) is not None:
                weight = weight.to(device=device)
            weight = weight_fn(weight)
        if weight_args is not None:
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
        if bias_fn is not None:
            if bias_args is not None and (device := bias_args.get("device")) is not None:
                bias = bias.to(device=device)
            bias = bias_fn(bias)
        if bias_args is not None:
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
):
    weight, bias, signal = None, None, None
    non_blocking = getattr(x.device, "type", None) != "mps"

    target_dtype = x.dtype
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
    if context.bnb_dtype in {"gguf"}:
        return CodexOperationsGGUF
    if _BNB_AVAILABLE and context.bnb_dtype in {"nf4", "fp4"}:
        return CodexOperationsBNB4bits
    return CodexOperations


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
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.group_norm(x, self.num_groups, weight, bias, self.eps)
            return super().forward(x)

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
            if self.parameters_manual_cast:
                weight, bias, signal = weights_manual_cast(self, x)
                with main_stream_worker(weight, bias, signal):
                    return torch.nn.functional.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
            return super().forward(x)

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


try:
    from .operations_bnb import (
        BnbQuantConfig,
        CodexLoader4Bit,
        CodexParams4bit,
        functional_dequantize_4bit,
        functional_linear_4bits,
    )

    _BNB_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    _BNB_AVAILABLE = False


if _BNB_AVAILABLE:

    class CodexOperationsBNB4bits(CodexOperations):
        class Linear(CodexLoader4Bit):
            def __init__(self, *args, **kwargs):
                ctx = get_operation_context()
                quant_type = ctx.bnb_dtype or "4bit"
                config = BnbQuantConfig(blocksize=64, quant_type=quant_type)
                super().__init__(device=ctx.device, dtype=ctx.dtype, quant_config=config)
                self.parameters_manual_cast = ctx.manual_cast_enabled

            def forward(self, x):
                if self.bias is not None and self.bias.dtype != x.dtype:
                    self.bias = utils.tensor2parameter(self.bias.to(x.dtype))

                if hasattr(self, "codex_online_loras"):
                    weight, bias, signal = weights_manual_cast(
                        self,
                        x,
                        weight_fn=functional_dequantize_4bit,
                        bias_fn=None,
                        skip_bias_dtype=True,
                    )
                    with main_stream_worker(weight, bias, signal):
                        return torch.nn.functional.linear(x, weight, bias)

                if not self.parameters_manual_cast:
                    return functional_linear_4bits(x, self.weight, self.bias)

                if not self.weight.bnb_quantized:
                    if x.device.type != "cuda":
                        raise AssertionError("BNB 4-bit quantization requires CUDA device.")
                    layer_original_device = self.weight.device
                    self.weight = self.weight._quantize(x.device)
                    bias = self.bias.to(x.device) if self.bias is not None else None
                    out = functional_linear_4bits(x, self.weight, bias)
                    self.weight = self.weight.to(layer_original_device)
                    return out

                weight, bias, signal = weights_manual_cast(
                    self,
                    x,
                    skip_weight_dtype=True,
                    skip_bias_dtype=True,
                )
                with main_stream_worker(weight, bias, signal):
                    return functional_linear_4bits(x, weight, bias)

else:

    class CodexOperationsBNB4bits(CodexOperations):  # pragma: no cover - fallback
        pass


class CodexOperationsGGUF(CodexOperations):
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
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy.device)
                    self.weight.computation_dtype = computation_dtype
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"].to(device=self.dummy.device)
                    self.bias.computation_dtype = computation_dtype
                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]
                if prefix + "bias" in state_dict:
                    self.bias = state_dict[prefix + "bias"]

        def _apply(self, fn, recurse=True):
            for name, param in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, name, utils.tensor2parameter(fn(param)))
            return self

        def forward(self, x):
            if self.bias is not None and self.bias.dtype != x.dtype:
                self.bias = utils.tensor2parameter(dequantize_tensor(self.bias).to(x.dtype))
            if self.weight is not None and self.weight.dtype != x.dtype and getattr(self.weight, "gguf_cls", None) is None:
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
            if ctx.device is not None:
                kwargs["device"] = ctx.device
            super().__init__(*args, **kwargs)
            self.parameters_manual_cast = ctx.manual_cast_enabled
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
                computation_dtype = self.dummy.dtype
                if computation_dtype not in (torch.float16, torch.bfloat16):
                    computation_dtype = torch.float16
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"].to(device=self.dummy.device)
                    self.weight.computation_dtype = computation_dtype
                del self.dummy
            else:
                if prefix + "weight" in state_dict:
                    self.weight = state_dict[prefix + "weight"]

        def _apply(self, fn, recurse=True):
            for name, param in self.named_parameters(recurse=False, remove_duplicate=True):
                setattr(self, name, utils.tensor2parameter(fn(param)))
            return self

        def forward(self, x):
            weight, bias, signal = weights_manual_cast(
                self,
                x,
                weight_fn=dequantize_tensor,
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


@contextlib.contextmanager
def using_codex_operations(operations=None, device=None, dtype=None, manual_cast_enabled=False, bnb_dtype=None):
    global _operation_context
    previous_context = _operation_context
    _operation_context = OperationContext(
        device=device,
        dtype=dtype,
        manual_cast_enabled=manual_cast_enabled,
        bnb_dtype=bnb_dtype,
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
    memory_management.free_memory(
        memory_required=3 * 1024 * 1024 * 1024,
        device=memory_management.get_torch_device(),
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

    memory_management.soft_empty_cache()
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
