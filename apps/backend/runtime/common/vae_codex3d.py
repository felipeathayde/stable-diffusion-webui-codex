"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared native 3D VAE runtime lane for temporal video autoencoders.
Defines `AutoencoderCodex3D` with causal 3D convolutions, explicit temporal
down/upsampling, and strict diffusers->codex key remap helpers for WAN-like
checkpoints without importing diffusers model classes.

Symbols (top-level; keep in sync; no ghosts):
- `AutoencoderCodex3D` (class): Native causal-3D KL VAE with codex keyspace (`encoder.downsamples.*`, `decoder.upsamples.*`, `conv1/conv2`).
- `sanitize_codex3d_vae_config` (function): Normalizes alias config fields into `AutoencoderCodex3D` constructor arguments.
- `remap_codex3d_vae_state_dict` (function): Normalizes wrapper prefixes and remaps diffusers WAN3D VAE keyspace to codex keyspace (strict/fail-loud).
- `is_codex3d_vae_instance` (function): Returns True when a model instance belongs to the native codex 3D VAE lane.
- `__all__` (constant): Explicit export list.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch import nn

from apps.backend.runtime.families.wan22.vae import DiagonalGaussianDistribution
from apps.backend.runtime.state_dict.keymap_anima import remap_anima_wan_vae_state_dict

_WRAPPER_PREFIXES = (
    "module.",
    "vae.",
    "first_stage_model.",
)

_FIXED_DIFFUSERS_TO_CODEX_KEYS: dict[str, str] = {
    "encoder.conv_in.weight": "encoder.conv1.weight",
    "encoder.conv_in.bias": "encoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_in.bias": "decoder.conv1.bias",
    "encoder.mid_block.resnets.0.norm1.gamma": "encoder.middle.0.residual.0.gamma",
    "encoder.mid_block.resnets.0.conv1.weight": "encoder.middle.0.residual.2.weight",
    "encoder.mid_block.resnets.0.conv1.bias": "encoder.middle.0.residual.2.bias",
    "encoder.mid_block.resnets.0.norm2.gamma": "encoder.middle.0.residual.3.gamma",
    "encoder.mid_block.resnets.0.conv2.weight": "encoder.middle.0.residual.6.weight",
    "encoder.mid_block.resnets.0.conv2.bias": "encoder.middle.0.residual.6.bias",
    "encoder.mid_block.resnets.1.norm1.gamma": "encoder.middle.2.residual.0.gamma",
    "encoder.mid_block.resnets.1.conv1.weight": "encoder.middle.2.residual.2.weight",
    "encoder.mid_block.resnets.1.conv1.bias": "encoder.middle.2.residual.2.bias",
    "encoder.mid_block.resnets.1.norm2.gamma": "encoder.middle.2.residual.3.gamma",
    "encoder.mid_block.resnets.1.conv2.weight": "encoder.middle.2.residual.6.weight",
    "encoder.mid_block.resnets.1.conv2.bias": "encoder.middle.2.residual.6.bias",
    "decoder.mid_block.resnets.0.norm1.gamma": "decoder.middle.0.residual.0.gamma",
    "decoder.mid_block.resnets.0.conv1.weight": "decoder.middle.0.residual.2.weight",
    "decoder.mid_block.resnets.0.conv1.bias": "decoder.middle.0.residual.2.bias",
    "decoder.mid_block.resnets.0.norm2.gamma": "decoder.middle.0.residual.3.gamma",
    "decoder.mid_block.resnets.0.conv2.weight": "decoder.middle.0.residual.6.weight",
    "decoder.mid_block.resnets.0.conv2.bias": "decoder.middle.0.residual.6.bias",
    "decoder.mid_block.resnets.1.norm1.gamma": "decoder.middle.2.residual.0.gamma",
    "decoder.mid_block.resnets.1.conv1.weight": "decoder.middle.2.residual.2.weight",
    "decoder.mid_block.resnets.1.conv1.bias": "decoder.middle.2.residual.2.bias",
    "decoder.mid_block.resnets.1.norm2.gamma": "decoder.middle.2.residual.3.gamma",
    "decoder.mid_block.resnets.1.conv2.weight": "decoder.middle.2.residual.6.weight",
    "decoder.mid_block.resnets.1.conv2.bias": "decoder.middle.2.residual.6.bias",
    "encoder.mid_block.attentions.0.norm.gamma": "encoder.middle.1.norm.gamma",
    "encoder.mid_block.attentions.0.to_qkv.weight": "encoder.middle.1.to_qkv.weight",
    "encoder.mid_block.attentions.0.to_qkv.bias": "encoder.middle.1.to_qkv.bias",
    "encoder.mid_block.attentions.0.proj.weight": "encoder.middle.1.proj.weight",
    "encoder.mid_block.attentions.0.proj.bias": "encoder.middle.1.proj.bias",
    "decoder.mid_block.attentions.0.norm.gamma": "decoder.middle.1.norm.gamma",
    "decoder.mid_block.attentions.0.to_qkv.weight": "decoder.middle.1.to_qkv.weight",
    "decoder.mid_block.attentions.0.to_qkv.bias": "decoder.middle.1.to_qkv.bias",
    "decoder.mid_block.attentions.0.proj.weight": "decoder.middle.1.proj.weight",
    "decoder.mid_block.attentions.0.proj.bias": "decoder.middle.1.proj.bias",
    "encoder.norm_out.gamma": "encoder.head.0.gamma",
    "encoder.conv_out.weight": "encoder.head.2.weight",
    "encoder.conv_out.bias": "encoder.head.2.bias",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
    "quant_conv.weight": "conv1.weight",
    "quant_conv.bias": "conv1.bias",
    "post_quant_conv.weight": "conv2.weight",
    "post_quant_conv.bias": "conv2.bias",
}

_DECODER_RESNET_TO_UPSAMPLE_INDEX: dict[tuple[int, int], int] = {
    (0, 0): 0,
    (0, 1): 1,
    (0, 2): 2,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 8,
    (2, 1): 9,
    (2, 2): 10,
    (3, 0): 12,
    (3, 1): 13,
    (3, 2): 14,
}

_DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX: dict[int, int] = {
    0: 3,
    1: 7,
    2: 11,
}


def _strip_wrapper_prefixes(key: str) -> str:
    normalized = str(key)
    while True:
        matched = False
        for prefix in _WRAPPER_PREFIXES:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                matched = True
        if not matched:
            return normalized


def _map_encoder_down_block_key(key: str) -> str:
    if ".resnets." in key:
        raise RuntimeError(
            "AutoencoderCodex3D key remap: unsupported encoder down_block layout "
            f"for key={key!r} (expected WAN diffusers canonical encoder.down_blocks.<idx>.<field>)."
        )
    mapped = key.replace("encoder.down_blocks.", "encoder.downsamples.", 1)
    mapped = mapped.replace(".norm1.gamma", ".residual.0.gamma")
    mapped = mapped.replace(".conv1.weight", ".residual.2.weight")
    mapped = mapped.replace(".conv1.bias", ".residual.2.bias")
    mapped = mapped.replace(".norm2.gamma", ".residual.3.gamma")
    mapped = mapped.replace(".conv2.weight", ".residual.6.weight")
    mapped = mapped.replace(".conv2.bias", ".residual.6.bias")
    mapped = mapped.replace(".conv_shortcut.weight", ".shortcut.weight")
    mapped = mapped.replace(".conv_shortcut.bias", ".shortcut.bias")
    return mapped


def _map_decoder_up_block_key(key: str) -> str:
    resnet_match = re.match(r"^decoder\.up_blocks\.(\d+)\.resnets\.(\d+)\.(.+)$", key)
    if resnet_match is not None:
        block_idx = int(resnet_match.group(1))
        resnet_idx = int(resnet_match.group(2))
        tail = str(resnet_match.group(3))
        if (block_idx, resnet_idx) not in _DECODER_RESNET_TO_UPSAMPLE_INDEX:
            raise RuntimeError(
                "AutoencoderCodex3D key remap: unsupported decoder up_block residual index "
                f"(block={block_idx} resnet={resnet_idx}) for key={key!r}."
            )
        upsample_index = _DECODER_RESNET_TO_UPSAMPLE_INDEX[(block_idx, resnet_idx)]
        if tail == "norm1.gamma":
            mapped_tail = "residual.0.gamma"
        elif tail == "conv1.weight":
            mapped_tail = "residual.2.weight"
        elif tail == "conv1.bias":
            mapped_tail = "residual.2.bias"
        elif tail == "norm2.gamma":
            mapped_tail = "residual.3.gamma"
        elif tail == "conv2.weight":
            mapped_tail = "residual.6.weight"
        elif tail == "conv2.bias":
            mapped_tail = "residual.6.bias"
        elif tail.startswith("conv_shortcut."):
            mapped_tail = "shortcut." + tail[len("conv_shortcut.") :]
        else:
            raise RuntimeError(
                "AutoencoderCodex3D key remap: unsupported decoder up_block residual field "
                f"tail={tail!r} key={key!r}."
            )
        return f"decoder.upsamples.{upsample_index}.{mapped_tail}"

    upsample_match = re.match(r"^decoder\.up_blocks\.(\d+)\.upsamplers\.0\.(.+)$", key)
    if upsample_match is not None:
        block_idx = int(upsample_match.group(1))
        tail = str(upsample_match.group(2))
        if block_idx not in _DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX:
            raise RuntimeError(
                "AutoencoderCodex3D key remap: unsupported decoder up_block upsampler index "
                f"(block={block_idx}) for key={key!r}."
            )
        upsample_index = _DECODER_UPSAMPLER_TO_UPSAMPLE_INDEX[block_idx]
        return f"decoder.upsamples.{upsample_index}.{tail}"

    raise RuntimeError(
        "AutoencoderCodex3D key remap: unsupported decoder up_block key layout "
        f"for key={key!r}."
    )


def remap_codex3d_vae_state_dict(
    state_dict: MutableMapping[str, Any],
) -> tuple[str, MutableMapping[str, Any]]:
    normalized: dict[str, Any] = {}
    for key, value in state_dict.items():
        normalized_key = _strip_wrapper_prefixes(str(key))
        if normalized_key in normalized:
            raise RuntimeError(
                "AutoencoderCodex3D key remap: normalized key collision "
                f"for key={normalized_key!r}."
            )
        normalized[normalized_key] = value

    has_codex = any(
        key.startswith("encoder.downsamples.")
        or key.startswith("decoder.upsamples.")
        or key in {"conv1.weight", "conv2.weight"}
        for key in normalized.keys()
    )
    has_diffusers = any(
        key.startswith("encoder.down_blocks.")
        or key.startswith("decoder.up_blocks.")
        or key.startswith("quant_conv.")
        or key.startswith("post_quant_conv.")
        for key in normalized.keys()
    )

    if has_codex and has_diffusers:
        raise RuntimeError(
            "AutoencoderCodex3D key remap: mixed codex/diffusers VAE keyspace detected "
            "(cannot resolve a single canonical lane)."
        )

    if has_diffusers:
        mapped: dict[str, Any] = {}
        for key, value in normalized.items():
            if key in _FIXED_DIFFUSERS_TO_CODEX_KEYS:
                mapped_key = _FIXED_DIFFUSERS_TO_CODEX_KEYS[key]
            elif key.startswith("encoder.down_blocks."):
                mapped_key = _map_encoder_down_block_key(key)
            elif key.startswith("decoder.up_blocks."):
                mapped_key = _map_decoder_up_block_key(key)
            else:
                mapped_key = key
            if mapped_key in mapped:
                raise RuntimeError(
                    "AutoencoderCodex3D key remap: output collision "
                    f"for mapped key={mapped_key!r} (source key={key!r})."
                )
            mapped[mapped_key] = value
        _, validated = remap_anima_wan_vae_state_dict(mapped)
        return "diffusers", validated

    _, validated = remap_anima_wan_vae_state_dict(normalized)
    return "codex", validated


class Codex3DCausalConv(nn.Conv3d):
    """Causal Conv3d with left-only temporal padding."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        padding: int | tuple[int, int, int] = 0,
    ) -> None:
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self._padding = (
            int(self.padding[2]),
            int(self.padding[2]),
            int(self.padding[1]),
            int(self.padding[1]),
            int(2 * self.padding[0]),
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D causal conv expects 5D input [B,C,T,H,W], "
                f"got shape={tuple(x.shape)}."
            )
        x = F.pad(x, self._padding)
        return super().forward(x)


class Codex3DRMSNorm(nn.Module):
    def __init__(self, dim: int, *, images: bool = True, channel_first: bool = True, bias: bool = False) -> None:
        super().__init__()
        broadcast_dims = (1, 1) if images else (1, 1, 1)
        shape = (int(dim), *broadcast_dims) if channel_first else (int(dim),)
        self._channel_first = bool(channel_first)
        self._scale = float(dim) ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dim = 1 if self._channel_first else -1
        out = F.normalize(x, dim=dim) * self._scale
        gamma = self.gamma.to(dtype=x.dtype, device=x.device)
        if gamma.ndim < x.ndim:
            gamma = gamma.view((1,) * (x.ndim - gamma.ndim) + tuple(gamma.shape))
        out = out * gamma
        if self.bias is not None:
            bias = self.bias.to(dtype=x.dtype, device=x.device)
            if bias.ndim < x.ndim:
                bias = bias.view((1,) * (x.ndim - bias.ndim) + tuple(bias.shape))
            out = out + bias
        return out


class Codex3DResidualBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.residual = nn.Sequential(
            Codex3DRMSNorm(self.in_dim, images=False),
            nn.SiLU(),
            Codex3DCausalConv(self.in_dim, self.out_dim, 3, padding=1),
            Codex3DRMSNorm(self.out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            Codex3DCausalConv(self.out_dim, self.out_dim, 3, padding=1),
        )
        self.shortcut = Codex3DCausalConv(self.in_dim, self.out_dim, 1) if self.in_dim != self.out_dim else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.residual(x) + self.shortcut(x)


class Codex3DAttentionBlock(nn.Module):
    """Single-head spatial attention applied frame-wise."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.norm = Codex3DRMSNorm(self.dim, images=True)
        self.to_qkv = nn.Conv2d(self.dim, self.dim * 3, 1)
        self.proj = nn.Conv2d(self.dim, self.dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D attention expects 5D input [B,C,T,H,W], "
                f"got shape={tuple(x.shape)}."
            )
        batch, channels, frames, height, width = x.shape
        residual = x
        normed = x.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
        normed = self.norm(normed)
        qkv = self.to_qkv(normed)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = q.reshape(batch * frames, channels, height * width).transpose(1, 2)
        k = k.reshape(batch * frames, channels, height * width)
        v = v.reshape(batch * frames, channels, height * width).transpose(1, 2)
        attn = torch.matmul(q, k) * (channels ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(batch * frames, channels, height, width)
        out = self.proj(out).view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
        return residual + out


class Codex3DResample(nn.Module):
    def __init__(self, dim: int, *, mode: str) -> None:
        super().__init__()
        self.dim = int(dim)
        self.mode = str(mode)

        if self.mode == "upsample2d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"), nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif self.mode == "downsample2d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif self.mode == "upsample3d":
            self.resample = nn.Sequential(nn.Upsample(scale_factor=(2.0, 2.0), mode="nearest"), nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = Codex3DCausalConv(dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif self.mode == "downsample3d":
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = Codex3DCausalConv(dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        elif self.mode == "none":
            self.resample = nn.Identity()
        else:
            raise RuntimeError(f"AutoencoderCodex3D unsupported resample mode={self.mode!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D resample expects 5D input [B,C,T,H,W], "
                f"got shape={tuple(x.shape)}."
            )
        batch, channels, frames, height, width = x.shape
        work = x
        if self.mode == "upsample3d":
            mixed = self.time_conv(work).reshape(batch, 2, channels, frames, height, width)
            work = torch.stack((mixed[:, 0], mixed[:, 1]), dim=3).reshape(batch, channels, frames * 2, height, width)
            if int(work.shape[2]) > 1:
                work = work[:, :, 1:, :, :]

        flat = work.permute(0, 2, 1, 3, 4).reshape(batch * int(work.shape[2]), channels, int(work.shape[3]), int(work.shape[4]))
        flat = self.resample(flat)
        out_channels = int(flat.shape[1])
        out_height = int(flat.shape[2])
        out_width = int(flat.shape[3])
        work = flat.reshape(batch, int(work.shape[2]), out_channels, out_height, out_width).permute(0, 2, 1, 3, 4).contiguous()

        if self.mode == "downsample3d":
            work = self.time_conv(work)
        return work


class Codex3DEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        input_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_downsample: Sequence[bool],
        dropout: float,
    ) -> None:
        super().__init__()
        dims = [int(dim) * m for m in (1, *tuple(int(x) for x in dim_mult))]
        scale = 1.0
        self.conv1 = Codex3DCausalConv(int(input_channels), int(dims[0]), 3, padding=1)
        downsamples: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            in_dim = int(in_dim)
            out_dim = int(out_dim)
            for _ in range(int(num_res_blocks)):
                downsamples.append(Codex3DResidualBlock(in_dim, out_dim, dropout=float(dropout)))
                if float(scale) in tuple(float(v) for v in attn_scales):
                    downsamples.append(Codex3DAttentionBlock(out_dim))
                in_dim = out_dim
            if index != len(tuple(dim_mult)) - 1:
                mode = "downsample3d" if bool(tuple(temperal_downsample)[index]) else "downsample2d"
                downsamples.append(Codex3DResample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)
        self.middle = nn.Sequential(
            Codex3DResidualBlock(out_dim, out_dim, dropout=float(dropout)),
            Codex3DAttentionBlock(out_dim),
            Codex3DResidualBlock(out_dim, out_dim, dropout=float(dropout)),
        )
        self.head = nn.Sequential(
            Codex3DRMSNorm(out_dim, images=False),
            nn.SiLU(),
            Codex3DCausalConv(out_dim, int(z_dim), 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.middle(self.downsamples(self.conv1(x))))


class Codex3DDecoder(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        z_dim: int,
        output_channels: int,
        dim_mult: Sequence[int],
        num_res_blocks: int,
        attn_scales: Sequence[float],
        temperal_upsample: Sequence[bool],
        dropout: float,
    ) -> None:
        super().__init__()
        dim_mult_tuple = tuple(int(x) for x in dim_mult)
        dims = [int(dim) * int(v) for v in (dim_mult_tuple[-1], *reversed(dim_mult_tuple))]
        scale = 1.0 / (2 ** max(len(tuple(dim_mult)) - 2, 0))
        self.conv1 = Codex3DCausalConv(int(z_dim), int(dims[0]), 3, padding=1)
        self.middle = nn.Sequential(
            Codex3DResidualBlock(int(dims[0]), int(dims[0]), dropout=float(dropout)),
            Codex3DAttentionBlock(int(dims[0])),
            Codex3DResidualBlock(int(dims[0]), int(dims[0]), dropout=float(dropout)),
        )
        upsamples: list[nn.Module] = []
        for index, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:], strict=True)):
            in_dim_i = int(in_dim)
            if index in (1, 2, 3):
                in_dim_i = in_dim_i // 2
            for _ in range(int(num_res_blocks) + 1):
                upsamples.append(Codex3DResidualBlock(in_dim_i, int(out_dim), dropout=float(dropout)))
                if float(scale) in tuple(float(v) for v in attn_scales):
                    upsamples.append(Codex3DAttentionBlock(int(out_dim)))
                in_dim_i = int(out_dim)
            if index != len(tuple(dim_mult)) - 1:
                mode = "upsample3d" if bool(tuple(temperal_upsample)[index]) else "upsample2d"
                upsamples.append(Codex3DResample(int(out_dim), mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)
        self.head = nn.Sequential(
            Codex3DRMSNorm(int(out_dim), images=False),
            nn.SiLU(),
            Codex3DCausalConv(int(out_dim), int(output_channels), 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.upsamples(self.middle(self.conv1(x))))


@dataclass(frozen=True, slots=True)
class _Codex3DConfigView:
    z_dim: int
    scaling_factor: float
    shift_factor: float | None


class AutoencoderCodex3D(nn.Module, ConfigMixin):
    config_name = "config.json"

    @register_to_config
    def __init__(
        self,
        *,
        base_dim: int = 96,
        decoder_base_dim: int | None = None,
        z_dim: int = 16,
        dim_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attn_scales: Sequence[float] = (),
        temperal_downsample: Sequence[bool] = (False, True, True),
        dropout: float = 0.0,
        in_channels: int = 3,
        out_channels: int = 3,
        scaling_factor: float = 1.0,
        shift_factor: float | None = None,
        latents_mean: Sequence[float] | None = None,
        latents_std: Sequence[float] | None = None,
        scale_factor_temporal: int | None = 4,
        scale_factor_spatial: int | None = 8,
    ) -> None:
        super().__init__()
        if decoder_base_dim is not None and int(decoder_base_dim) != int(base_dim):
            raise RuntimeError(
                "AutoencoderCodex3D requires decoder_base_dim to match base_dim "
                f"(got base_dim={int(base_dim)} decoder_base_dim={int(decoder_base_dim)})."
            )
        self.z_dim = int(z_dim)
        self.temperal_downsample = tuple(bool(x) for x in temperal_downsample)
        self.temperal_upsample = tuple(reversed(self.temperal_downsample))
        self.encoder = Codex3DEncoder(
            dim=int(base_dim),
            z_dim=int(self.z_dim) * 2,
            input_channels=int(in_channels),
            dim_mult=tuple(int(v) for v in dim_mult),
            num_res_blocks=int(num_res_blocks),
            attn_scales=tuple(float(v) for v in attn_scales),
            temperal_downsample=self.temperal_downsample,
            dropout=float(dropout),
        )
        self.conv1 = Codex3DCausalConv(int(self.z_dim) * 2, int(self.z_dim) * 2, 1)
        self.conv2 = Codex3DCausalConv(int(self.z_dim), int(self.z_dim), 1)
        self.decoder = Codex3DDecoder(
            dim=int(base_dim),
            z_dim=int(self.z_dim),
            output_channels=int(out_channels),
            dim_mult=tuple(int(v) for v in dim_mult),
            num_res_blocks=int(num_res_blocks),
            attn_scales=tuple(float(v) for v in attn_scales),
            temperal_upsample=self.temperal_upsample,
            dropout=float(dropout),
        )
        self.scaling_factor = float(scaling_factor)
        self.shift_factor = None if shift_factor is None else float(shift_factor)
        self._shift_factor_value = 0.0 if self.shift_factor is None else float(self.shift_factor)
        self.scale_factor_temporal = int(scale_factor_temporal) if scale_factor_temporal is not None else 4
        self.scale_factor_spatial = int(scale_factor_spatial) if scale_factor_spatial is not None else 8
        self.latents_mean = None if latents_mean is None else tuple(float(v) for v in latents_mean)
        self.latents_std = None if latents_std is None else tuple(float(v) for v in latents_std)
        self.use_tiling = False

    def enable_tiling(self, *_args: Any, **_kwargs: Any) -> None:
        self.use_tiling = True

    def encode(self, x: torch.Tensor, return_dict: bool = True, regulation: Any = None) -> Any:
        squeeze_t = False
        if x.ndim == 4:
            squeeze_t = True
            x = x.unsqueeze(2)
        if x.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D encode expects 4D or 5D input ([B,C,H,W] or [B,C,T,H,W]), "
                f"got shape={tuple(x.shape)}."
            )
        moments = self.conv1(self.encoder(x))
        posterior = DiagonalGaussianDistribution(moments)
        latents = regulation(posterior) if regulation is not None else posterior.sample()
        if squeeze_t and latents.ndim == 5 and int(latents.shape[2]) == 1:
            latents = latents.squeeze(2)
        if not return_dict:
            return (posterior,)
        return SimpleNamespace(latent_dist=posterior, latents=latents)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Any:
        squeeze_t = False
        if z.ndim == 4:
            squeeze_t = True
            z = z.unsqueeze(2)
        if z.ndim != 5:
            raise RuntimeError(
                "AutoencoderCodex3D decode expects 4D or 5D latents ([B,C,H,W] or [B,C,T,H,W]), "
                f"got shape={tuple(z.shape)}."
            )
        out = self.decoder(self.conv2(z))
        out = torch.clamp(out, min=-1.0, max=1.0)
        if squeeze_t and out.ndim == 5 and int(out.shape[2]) == 1:
            out = out.squeeze(2)
        if not return_dict:
            return (out,)
        return SimpleNamespace(sample=out)

    def process_in(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent - self._shift_factor_value) * self.scaling_factor

    def process_out(self, latent: torch.Tensor) -> torch.Tensor:
        return (latent / self.scaling_factor) + self._shift_factor_value


def sanitize_codex3d_vae_config(config: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(config)
    cleaned: dict[str, Any] = {}

    base_dim = source.get("base_dim")
    if base_dim is None:
        block_channels = source.get("block_out_channels")
        if isinstance(block_channels, (list, tuple)) and block_channels:
            base_dim = int(block_channels[0])
    if base_dim is None:
        base_dim = 96
    cleaned["base_dim"] = int(base_dim)

    decoder_base_dim = source.get("decoder_base_dim")
    if decoder_base_dim is not None:
        cleaned["decoder_base_dim"] = int(decoder_base_dim)

    z_dim = source.get("z_dim")
    if z_dim is None:
        z_dim = source.get("latent_channels")
    if z_dim is None:
        z_dim = 16
    cleaned["z_dim"] = int(z_dim)

    dim_mult = source.get("dim_mult")
    if dim_mult is None:
        block_channels = source.get("block_out_channels")
        if isinstance(block_channels, (list, tuple)) and block_channels:
            try:
                dim_mult = tuple(int(int(ch) // int(cleaned["base_dim"])) for ch in block_channels)
            except Exception:
                dim_mult = None
    cleaned["dim_mult"] = tuple(int(v) for v in (dim_mult if dim_mult is not None else (1, 2, 4, 4)))

    num_res_blocks = source.get("num_res_blocks")
    if num_res_blocks is None:
        num_res_blocks = source.get("layers_per_block")
    cleaned["num_res_blocks"] = int(num_res_blocks) if num_res_blocks is not None else 2

    attn_scales = source.get("attn_scales", ())
    cleaned["attn_scales"] = tuple(float(v) for v in attn_scales)

    temporal_flags = source.get("temperal_downsample")
    if temporal_flags is None:
        temporal_flags = source.get("temporal_downsample")
    if temporal_flags is None:
        temporal_flags = (False, True, True)
    cleaned["temperal_downsample"] = tuple(bool(v) for v in temporal_flags)

    cleaned["dropout"] = float(source.get("dropout", 0.0))
    cleaned["in_channels"] = int(source.get("in_channels", 3))
    cleaned["out_channels"] = int(source.get("out_channels", 3))
    cleaned["scaling_factor"] = float(source.get("scaling_factor", 1.0))
    cleaned["shift_factor"] = source.get("shift_factor")
    if source.get("latents_mean") is not None:
        cleaned["latents_mean"] = tuple(float(v) for v in source["latents_mean"])
    if source.get("latents_std") is not None:
        cleaned["latents_std"] = tuple(float(v) for v in source["latents_std"])
    if source.get("scale_factor_temporal") is not None:
        cleaned["scale_factor_temporal"] = int(source["scale_factor_temporal"])
    if source.get("scale_factor_spatial") is not None:
        cleaned["scale_factor_spatial"] = int(source["scale_factor_spatial"])
    return cleaned


def is_codex3d_vae_instance(model: object) -> bool:
    return isinstance(model, AutoencoderCodex3D)


__all__ = [
    "AutoencoderCodex3D",
    "is_codex3d_vae_instance",
    "remap_codex3d_vae_state_dict",
    "sanitize_codex3d_vae_config",
]
