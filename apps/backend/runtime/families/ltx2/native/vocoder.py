"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native raw-key LTX2 vocoder for the backend runtime family seam.
Implements the LTX2 vocoder under `apps/**` with strict config parsing and strict raw-layout state-dict loading.
Supported state dicts are the parser-owned raw vocoder layout only: `conv_pre.*`, `ups.*`, `resblocks.*`, and
`conv_post.*`. Diffusers-remapped vocoder keys are rejected on purpose.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2VocoderConfig` (dataclass): Strict parsed config contract for the native LTX2 vocoder.
- `Ltx2Vocoder` (class): Native raw-key LTX2 vocoder module.
- `load_ltx2_vocoder` (function): Strict loader for parser-owned LTX2 vocoder state dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import math

import torch
import torch.nn.functional as F
from torch import nn

from apps.backend.runtime.models.state_dict import safe_load_state_dict

_ALLOWED_CONFIG_METADATA_KEYS = frozenset({"_class_name", "_diffusers_version"})
_RAW_VOCODER_PREFIXES = ("conv_pre.", "ups.", "resblocks.", "conv_post.")
_REJECTED_VOCODER_PREFIXES = ("conv_in.", "upsamplers.", "resnets.", "conv_out.")


@dataclass(frozen=True)
class Ltx2VocoderConfig:
    in_channels: int
    hidden_channels: int
    out_channels: int
    upsample_kernel_sizes: tuple[int, ...]
    upsample_factors: tuple[int, ...]
    resnet_kernel_sizes: tuple[int, ...]
    resnet_dilations: tuple[tuple[int, ...], ...]
    leaky_relu_negative_slope: float
    output_sampling_rate: int


class _ResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        *,
        kernel_size: int = 3,
        stride: int = 1,
        dilations: tuple[int, ...] = (1, 3, 5),
        leaky_relu_negative_slope: float = 0.1,
        padding_mode: str = "same",
    ) -> None:
        super().__init__()
        self.dilations = tuple(int(value) for value in dilations)
        self.negative_slope = float(leaky_relu_negative_slope)
        self.convs1 = nn.ModuleList(
            [
                nn.Conv1d(
                    int(channels),
                    int(channels),
                    int(kernel_size),
                    stride=int(stride),
                    dilation=int(dilation),
                    padding=padding_mode,
                )
                for dilation in self.dilations
            ]
        )
        self.convs2 = nn.ModuleList(
            [
                nn.Conv1d(
                    int(channels),
                    int(channels),
                    int(kernel_size),
                    stride=int(stride),
                    dilation=1,
                    padding=padding_mode,
                )
                for _ in self.dilations
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for conv1, conv2 in zip(self.convs1, self.convs2):
            residual = F.leaky_relu(hidden_states, negative_slope=self.negative_slope)
            residual = conv1(residual)
            residual = F.leaky_relu(residual, negative_slope=self.negative_slope)
            residual = conv2(residual)
            hidden_states = hidden_states + residual
        return hidden_states


class Ltx2Vocoder(nn.Module):
    def __init__(self, config: Ltx2VocoderConfig) -> None:
        super().__init__()
        self.config = config
        self.num_upsample_layers = len(config.upsample_kernel_sizes)
        self.resblocks_per_upsample = len(config.resnet_kernel_sizes)
        self.out_channels = int(config.out_channels)
        self.total_upsample_factor = math.prod(config.upsample_factors)
        self.negative_slope = float(config.leaky_relu_negative_slope)

        if self.num_upsample_layers != len(config.upsample_factors):
            raise RuntimeError(
                "Unsupported LTX2 vocoder config: `upsample_kernel_sizes` and `upsample_factors` must be the same length."
            )
        if self.resblocks_per_upsample != len(config.resnet_dilations):
            raise RuntimeError(
                "Unsupported LTX2 vocoder config: `resnet_kernel_sizes` and `resnet_dilations` must be the same length."
            )

        self.conv_pre = nn.Conv1d(int(config.in_channels), int(config.hidden_channels), kernel_size=7, stride=1, padding=3)
        self.ups = nn.ModuleList()
        self.resblocks = nn.ModuleList()

        input_channels = int(config.hidden_channels)
        output_channels = input_channels
        for stride, kernel_size in zip(config.upsample_factors, config.upsample_kernel_sizes):
            output_channels = input_channels // 2
            self.ups.append(
                nn.ConvTranspose1d(
                    input_channels,
                    output_channels,
                    int(kernel_size),
                    stride=int(stride),
                    padding=(int(kernel_size) - int(stride)) // 2,
                )
            )
            for resnet_kernel_size, dilations in zip(config.resnet_kernel_sizes, config.resnet_dilations):
                self.resblocks.append(
                    _ResBlock(
                        output_channels,
                        kernel_size=int(resnet_kernel_size),
                        dilations=tuple(int(value) for value in dilations),
                        leaky_relu_negative_slope=self.negative_slope,
                    )
                )
            input_channels = output_channels

        self.conv_post = nn.Conv1d(output_channels, int(config.out_channels), kernel_size=7, stride=1, padding=3)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2Vocoder":
        return cls(_parse_vocoder_config(config))

    def forward(self, hidden_states: torch.Tensor, time_last: bool = False) -> torch.Tensor:
        if hidden_states.ndim != 4:
            raise RuntimeError(
                "LTX2 vocoder expects a rank-4 mel tensor. "
                f"Got shape={tuple(int(dim) for dim in hidden_states.shape)!r}."
            )
        if not bool(time_last):
            hidden_states = hidden_states.transpose(2, 3)
        hidden_states = hidden_states.flatten(1, 2)
        hidden_states = self.conv_pre(hidden_states)

        for upsample_index in range(self.num_upsample_layers):
            hidden_states = F.leaky_relu(hidden_states, negative_slope=self.negative_slope)
            hidden_states = self.ups[upsample_index](hidden_states)
            start = upsample_index * self.resblocks_per_upsample
            end = (upsample_index + 1) * self.resblocks_per_upsample
            resblock_outputs = torch.stack(
                [self.resblocks[index](hidden_states) for index in range(start, end)],
                dim=0,
            )
            hidden_states = torch.mean(resblock_outputs, dim=0)

        hidden_states = F.leaky_relu(hidden_states, negative_slope=0.01)
        hidden_states = self.conv_post(hidden_states)
        return torch.tanh(hidden_states)



def load_ltx2_vocoder(
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Ltx2Vocoder:
    if not isinstance(state_dict, Mapping):
        raise TypeError(f"LTX2 vocoder state_dict must be a mapping; got {type(state_dict).__name__}.")

    _validate_raw_vocoder_state_dict(state_dict)
    try:
        module = Ltx2Vocoder.from_config(config)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 vocoder config instantiation failed: {exc}") from exc

    missing, unexpected = safe_load_state_dict(module, state_dict, log_name="ltx2.vocoder")
    if missing or unexpected:
        raise RuntimeError(
            "LTX2 vocoder strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )

    try:
        module = module.to(device=device, dtype=torch_dtype)
    except Exception:
        module = module.to(device=device)
    module.eval()
    return module



def _parse_vocoder_config(config: Mapping[str, Any]) -> Ltx2VocoderConfig:
    if not isinstance(config, Mapping):
        raise TypeError(f"LTX2 vocoder config must be a mapping; got {type(config).__name__}.")

    required_keys = {
        "in_channels",
        "hidden_channels",
        "out_channels",
        "upsample_kernel_sizes",
        "upsample_factors",
        "resnet_kernel_sizes",
        "resnet_dilations",
        "leaky_relu_negative_slope",
        "output_sampling_rate",
    }
    allowed_keys = required_keys | set(_ALLOWED_CONFIG_METADATA_KEYS)
    raw_keys = {str(key) for key in config.keys()}
    missing = sorted(required_keys - raw_keys)
    unexpected = sorted(raw_keys - allowed_keys)
    if missing or unexpected:
        raise RuntimeError(
            "Unsupported LTX2 vocoder config keys. "
            f"missing={missing!r} unexpected={unexpected!r}"
        )

    upsample_kernel_sizes = _require_positive_int_sequence(config, "upsample_kernel_sizes")
    upsample_factors = _require_positive_int_sequence(config, "upsample_factors")
    resnet_kernel_sizes = _require_positive_int_sequence(config, "resnet_kernel_sizes")
    raw_resnet_dilations = config.get("resnet_dilations")
    if not isinstance(raw_resnet_dilations, Sequence) or isinstance(raw_resnet_dilations, (str, bytes)):
        raise RuntimeError("LTX2 vocoder config key 'resnet_dilations' must be a sequence of int sequences.")
    resnet_dilations = tuple(_require_positive_int_tuple(entry, key="resnet_dilations") for entry in raw_resnet_dilations)

    parsed = Ltx2VocoderConfig(
        in_channels=_require_positive_int(config, "in_channels"),
        hidden_channels=_require_positive_int(config, "hidden_channels"),
        out_channels=_require_positive_int(config, "out_channels"),
        upsample_kernel_sizes=upsample_kernel_sizes,
        upsample_factors=upsample_factors,
        resnet_kernel_sizes=resnet_kernel_sizes,
        resnet_dilations=resnet_dilations,
        leaky_relu_negative_slope=_require_nonnegative_float(config, "leaky_relu_negative_slope"),
        output_sampling_rate=_require_positive_int(config, "output_sampling_rate"),
    )

    if len(parsed.upsample_kernel_sizes) != len(parsed.upsample_factors):
        raise RuntimeError(
            "Unsupported LTX2 vocoder config: `upsample_kernel_sizes` and `upsample_factors` must have equal length."
        )
    if len(parsed.resnet_kernel_sizes) != len(parsed.resnet_dilations):
        raise RuntimeError(
            "Unsupported LTX2 vocoder config: `resnet_kernel_sizes` and `resnet_dilations` must have equal length."
        )
    return parsed



def _validate_raw_vocoder_state_dict(state_dict: Mapping[str, Any]) -> None:
    raw_keys = tuple(str(key) for key in state_dict.keys())
    if not raw_keys:
        raise RuntimeError("LTX2 vocoder state_dict is empty.")
    if any(key.startswith("vocoder.") for key in raw_keys):
        raise RuntimeError(
            "LTX2 vocoder loader received unstripped `vocoder.*` keys. "
            "Expected parser-owned vocoder keys with that prefix removed."
        )
    if any(key.startswith(("model.diffusion_model.", "vae.", "audio_vae.")) for key in raw_keys):
        raise RuntimeError("LTX2 vocoder loader received non-vocoder component keys.")
    if any(key.startswith(_REJECTED_VOCODER_PREFIXES) for key in raw_keys):
        raise RuntimeError(
            "Unsupported Diffusers-remapped LTX2 vocoder keyspace. "
            "The native loader expects raw `conv_pre`/`ups`/`resblocks`/`conv_post` keys only."
        )

    required = ("conv_pre.weight", "conv_post.weight")
    missing = [key for key in required if key not in state_dict]
    if missing:
        raise RuntimeError(f"LTX2 vocoder state_dict is missing required raw keys: {missing!r}.")

    unexpected_prefixes = sorted({key.split(".", 1)[0] for key in raw_keys if not key.startswith(_RAW_VOCODER_PREFIXES)})
    if unexpected_prefixes:
        raise RuntimeError(
            "Unsupported LTX2 vocoder layout. "
            f"Unexpected top-level prefixes: {unexpected_prefixes!r}."
        )



def _require_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be an int; got {type(value).__name__}.")
    if value <= 0:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be > 0; got {value!r}.")
    return int(value)



def _require_nonnegative_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeError(
            f"LTX2 vocoder config key {key!r} must be a float-compatible number; got {type(value).__name__}."
        )
    value = float(value)
    if value < 0.0:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be >= 0; got {value!r}.")
    return value



def _require_positive_int_sequence(config: Mapping[str, Any], key: str) -> tuple[int, ...]:
    value = config.get(key)
    return _require_positive_int_tuple(value, key=key)



def _require_positive_int_tuple(value: Any, *, key: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must be a sequence of ints.")
    parsed: list[int] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int):
            raise RuntimeError(
                f"LTX2 vocoder config key {key!r}[{index}] must be an int; got {type(item).__name__}."
            )
        if item <= 0:
            raise RuntimeError(f"LTX2 vocoder config key {key!r}[{index}] must be > 0; got {item!r}.")
        parsed.append(int(item))
    if not parsed:
        raise RuntimeError(f"LTX2 vocoder config key {key!r} must not be empty.")
    return tuple(parsed)


__all__ = ["Ltx2VocoderConfig", "Ltx2Vocoder", "load_ltx2_vocoder"]
