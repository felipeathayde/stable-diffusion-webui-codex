# Copyright Codex 2024

"""bitsandbytes integration helpers.

Legacy Forge implementation pulled in 1:1; Codex rebuild introduces typed configs,
explicit validation, and a registry so downstream loaders can request q/bnb helpers
without touching raw bitsandbytes state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import bitsandbytes as bnb
import torch
from bitsandbytes.functional import dequantize_4bit
from bitsandbytes.nn.modules import Params4bit, QuantState

from apps.backend.runtime import utils
from apps.backend.runtime.memory import memory_management

logger = logging.getLogger("backend.runtime.bnb")


@dataclass(frozen=True, slots=True)
class BnbQuantConfig:
    """Typed configuration for bitsandbytes quantized weights."""

    blocksize: int
    quant_type: str
    compress_statistics: bool = False
    quant_storage: torch.dtype = torch.uint8

    def __post_init__(self) -> None:
        if self.blocksize <= 0:
            raise ValueError("blocksize must be > 0")
        if not self.quant_type:
            raise ValueError("quant_type must be provided")
        if self.quant_storage not in (torch.uint8, torch.int8):
            raise ValueError("unsupported quant_storage dtype")


def _copy_quant_state(state: Optional[QuantState], device: Optional[torch.device] = None) -> Optional[QuantState]:
    if state is None:
        return None
    device = device or state.absmax.device
    nested = (
        QuantState(
            absmax=state.state2.absmax.to(device),
            shape=state.state2.shape,
            code=state.state2.code.to(device),
            blocksize=state.state2.blocksize,
            quant_type=state.state2.quant_type,
            dtype=state.state2.dtype,
        )
        if state.nested
        else None
    )
    return QuantState(
        absmax=state.absmax.to(device),
        shape=state.shape,
        code=state.code.to(device),
        blocksize=state.blocksize,
        quant_type=state.quant_type,
        dtype=state.dtype,
        offset=state.offset.to(device) if state.nested else None,
        state2=nested,
    )


class CodexParams4bit(Params4bit):
    """Typed wrapper over bitsandbytes Params4bit with Codex memory hooks."""

    def __init__(self, *args, quant_config: Optional[BnbQuantConfig] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._quant_config = quant_config

    def _quantize(self, device):
        memory_management.signal_empty_cache = True
        return super()._quantize(device)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None and device.type == "cuda" and not self.bnb_quantized:
            logger.debug("Quantizing 4bit tensor on device %s", device)
            return self._quantize(device)
        return CodexParams4bit(
            torch.nn.Parameter.to(self, device=device, dtype=dtype, non_blocking=non_blocking),
            requires_grad=self.requires_grad,
            quant_state=_copy_quant_state(self.quant_state, device),
            blocksize=self.blocksize,
            compress_statistics=self.compress_statistics,
            quant_type=self.quant_type,
            quant_storage=self.quant_storage,
            bnb_quantized=self.bnb_quantized,
            quant_config=self._quant_config,
        )


class CodexLoader4Bit(torch.nn.Module):
    """Loader that materializes 4bit weights using registry configuration."""

    def __init__(self, *, device: torch.device, dtype: torch.dtype, quant_config: BnbQuantConfig):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.empty(1, device=device, dtype=dtype))
        self.weight: Optional[CodexParams4bit] = None
        self.bias: Optional[torch.nn.Parameter] = None
        self.quant_config = quant_config

    def _apply(self, fn, recurse: bool = True):
        for name, param in self.named_parameters(recurse=False, remove_duplicate=True):
            setattr(self, name, utils.tensor2parameter(fn(param)))
        return self

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        super()._save_to_state_dict(destination, prefix, keep_vars)
        quant_state = getattr(self.weight, "quant_state", None)
        if quant_state is not None:
            for key, value in quant_state.as_dict(packed=True).items():
                destination[prefix + "weight." + key] = value if keep_vars else value.detach()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        quant_keys = {k[len(prefix + "weight."):] for k in state_dict.keys() if k.startswith(prefix + "weight.")}

        if any("bitsandbytes" in k for k in quant_keys):
            quant_state_dict = {k: state_dict[prefix + "weight." + k] for k in quant_keys}
            self.weight = CodexParams4bit.from_prequantized(
                data=state_dict[prefix + "weight"],
                quantized_stats=quant_state_dict,
                requires_grad=False,
                device=self.dummy.device,
            )
            if prefix + "bias" in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + "bias"].to(self.dummy))
            del self.dummy
        elif hasattr(self, "dummy"):
            if prefix + "weight" in state_dict:
                self.weight = CodexParams4bit(
                    state_dict[prefix + "weight"].to(self.dummy),
                    requires_grad=False,
                    compress_statistics=self.quant_config.compress_statistics,
                    blocksize=self.quant_config.blocksize,
                    quant_type=self.quant_config.quant_type,
                    quant_storage=self.quant_config.quant_storage,
                    quant_config=self.quant_config,
                )
            if prefix + "bias" in state_dict:
                self.bias = torch.nn.Parameter(state_dict[prefix + "bias"].to(self.dummy))
            del self.dummy
        else:
            super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def reload_weight(self, weight: torch.Tensor) -> "CodexLoader4Bit":
        original_device = weight.device
        refreshed = CodexParams4bit(
            weight,
            requires_grad=False,
            compress_statistics=self.weight.compress_statistics,
            blocksize=self.weight.blocksize,
            quant_type=self.weight.quant_type,
            quant_storage=self.weight.quant_storage,
            bnb_quantized=False,
            quant_config=self.quant_config,
        )
        if original_device.type == "cuda":
            refreshed = refreshed.to(original_device)
        else:
            refreshed = refreshed.cuda().to(original_device)
        self.weight = refreshed
        return self


class BnbOperationRegistry:
    """Registry mapping quant types to functional helpers and loader factories."""

    def __init__(self) -> None:
        self._linear: Dict[str, Callable[[torch.Tensor, CodexParams4bit, Optional[torch.Tensor]], torch.Tensor]] = {}
        self._dequantize: Dict[str, Callable[[CodexParams4bit], torch.Tensor]] = {}
        self._loaders: Dict[str, Callable[[BnbQuantConfig, torch.device, torch.dtype], CodexLoader4Bit]] = {}

    def register_linear(self, quant_type: str, func: Callable[[torch.Tensor, CodexParams4bit, Optional[torch.Tensor]], torch.Tensor]) -> None:
        logger.debug("Registering linear op for quant_type=%s", quant_type)
        self._linear[quant_type] = func

    def register_dequantize(self, quant_type: str, func: Callable[[CodexParams4bit], torch.Tensor]) -> None:
        logger.debug("Registering dequantize op for quant_type=%s", quant_type)
        self._dequantize[quant_type] = func

    def register_loader(self, quant_type: str, factory: Callable[[BnbQuantConfig, torch.device, torch.dtype], CodexLoader4Bit]) -> None:
        logger.debug("Registering loader for quant_type=%s", quant_type)
        self._loaders[quant_type] = factory

    def linear(self, quant_type: str, x: torch.Tensor, weight: CodexParams4bit, bias: Optional[torch.Tensor]) -> torch.Tensor:
        if quant_type not in self._linear:
            raise KeyError(f"No linear implementation registered for quant_type={quant_type}")
        return self._linear[quant_type](x, weight, bias)

    def dequantize(self, quant_type: str, weight: CodexParams4bit) -> torch.Tensor:
        if quant_type not in self._dequantize:
            raise KeyError(f"No dequantize implementation registered for quant_type={quant_type}")
        return self._dequantize[quant_type](weight)

    def loader(self, quant_config: BnbQuantConfig, *, device: torch.device, dtype: torch.dtype) -> CodexLoader4Bit:
        factory = self._loaders.get(quant_config.quant_type)
        if factory is None:
            raise KeyError(f"No loader registered for quant_type={quant_config.quant_type}")
        return factory(quant_config, device, dtype)


REGISTRY = BnbOperationRegistry()


def _register_defaults() -> None:
    def _functional_linear(x: torch.Tensor, weight: CodexParams4bit, bias: Optional[torch.Tensor]) -> torch.Tensor:
        out = bnb.matmul_4bit(x, weight.t(), bias=bias, quant_state=weight.quant_state)
        return out.to(x)

    def _functional_dequantize(weight: CodexParams4bit) -> torch.Tensor:
        if not weight.bnb_quantized:
            return weight
        original_device = weight.device
        tmp = weight.cuda() if original_device.type != "cuda" else weight
        out = dequantize_4bit(tmp, quant_state=tmp.quant_state, blocksize=tmp.blocksize, quant_type=tmp.quant_type)
        return out.to(original_device)

    def _loader_factory(config: BnbQuantConfig, device: torch.device, dtype: torch.dtype) -> CodexLoader4Bit:
        return CodexLoader4Bit(device=device, dtype=dtype, quant_config=config)

    for quant_type in ("nf4", "fp4", "4bit"):
        REGISTRY.register_linear(quant_type, _functional_linear)
        REGISTRY.register_dequantize(quant_type, _functional_dequantize)
        REGISTRY.register_loader(quant_type, _loader_factory)


_register_defaults()


def functional_linear_4bits(x: torch.Tensor, weight: CodexParams4bit, bias: Optional[torch.Tensor]) -> torch.Tensor:
    return REGISTRY.linear(weight.quant_type, x, weight, bias)


def functional_dequantize_4bit(weight: CodexParams4bit) -> torch.Tensor:
    return REGISTRY.dequantize(weight.quant_type, weight)


__all__ = [
    "BnbQuantConfig",
    "CodexLoader4Bit",
    "CodexParams4bit",
    "functional_linear_4bits",
    "functional_dequantize_4bit",
    "REGISTRY",
]
