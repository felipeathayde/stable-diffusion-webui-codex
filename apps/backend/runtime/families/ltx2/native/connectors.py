"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Native LTX2 text-connector stack for packed video/audio prompt embeddings.
Implements the parser-owned `connectors` component without relying on LTX2-specific Diffusers classes, keeps the
stored state-dict layout intact, and exposes strict config/state-driven construction for runtime assembly.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2TextConnectors` (class): Native connector stack that projects packed text embeddings into video/audio streams.
- `load_ltx2_connectors` (function): Strict config/state-driven connector loader used by the LTX2 runtime.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_CONNECTORS_WRAPPER_PREFIX = "connectors."


def _require_int(config: Mapping[str, Any], key: str) -> int:
    value = config.get(key)
    if not isinstance(value, int):
        raise RuntimeError(f"LTX2 connectors config requires integer `{key}`, got {value!r}.")
    return int(value)


def _require_float(config: Mapping[str, Any], key: str) -> float:
    value = config.get(key)
    if not isinstance(value, (int, float)):
        raise RuntimeError(f"LTX2 connectors config requires float `{key}`, got {value!r}.")
    return float(value)


def _require_bool(config: Mapping[str, Any], key: str) -> bool:
    value = config.get(key)
    if not isinstance(value, bool):
        raise RuntimeError(f"LTX2 connectors config requires bool `{key}`, got {value!r}.")
    return bool(value)


def _require_str(config: Mapping[str, Any], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"LTX2 connectors config requires non-empty string `{key}`, got {value!r}.")
    return value


def apply_interleaved_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    x_real, x_imag = x.unflatten(2, (-1, 2)).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(2)
    return (x.float() * cos + x_rotated.float() * sin).to(dtype=x.dtype)


def apply_split_rotary_emb(x: torch.Tensor, freqs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    cos, sin = freqs
    original_dtype = x.dtype
    needs_reshape = False
    if x.ndim != 4 and cos.ndim == 4:
        batch_size = x.shape[0]
        _, heads, tokens, _ = cos.shape
        x = x.reshape(batch_size, tokens, heads, -1).swapaxes(1, 2)
        needs_reshape = True

    if x.shape[-1] % 2 != 0:
        raise RuntimeError(f"LTX2 split rotary expects even last-dim width, got {x.shape[-1]}.")

    half = x.shape[-1] // 2
    split_x = x.reshape(*x.shape[:-1], 2, half).float()
    first = split_x[..., :1, :]
    second = split_x[..., 1:, :]
    cos_u = cos.unsqueeze(-2)
    sin_u = sin.unsqueeze(-2)

    out = split_x * cos_u
    out[..., :1, :].addcmul_(-sin_u, second)
    out[..., 1:, :].addcmul_(sin_u, first)
    out = out.reshape(*out.shape[:-2], x.shape[-1])

    if needs_reshape:
        out = out.swapaxes(1, 2).reshape(batch_size, tokens, -1)
    return out.to(dtype=original_dtype)


class _Ltx2RotaryPosEmbed1d(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        base_seq_len: int,
        theta: float,
        double_precision: bool,
        rope_type: str,
        num_attention_heads: int,
    ) -> None:
        super().__init__()
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 connectors rope_type must be 'interleaved' or 'split', got {rope_type!r}.")
        self.dim = int(dim)
        self.base_seq_len = int(base_seq_len)
        self.theta = float(theta)
        self.double_precision = bool(double_precision)
        self.rope_type = rope_type
        self.num_attention_heads = int(num_attention_heads)

    def forward(
        self,
        batch_size: int,
        positions: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid = torch.arange(positions, dtype=torch.float32, device=device) / float(self.base_seq_len)
        grid = grid.unsqueeze(0).repeat(batch_size, 1)
        freqs_dtype = torch.float64 if self.double_precision else torch.float32
        base = torch.pow(
            torch.tensor(self.theta, dtype=freqs_dtype, device=device),
            torch.linspace(0.0, 1.0, steps=self.dim // 2, dtype=freqs_dtype, device=device),
        )
        freqs = ((grid.unsqueeze(-1) * 2.0) - 1.0) * (base.to(dtype=torch.float32) * (torch.pi / 2.0))

        if self.rope_type == "interleaved":
            cos = freqs.cos().repeat_interleave(2, dim=-1)
            sin = freqs.sin().repeat_interleave(2, dim=-1)
            return cos, sin

        expected_freqs = self.dim // 2
        current_freqs = freqs.shape[-1]
        if current_freqs > expected_freqs:
            raise RuntimeError(
                f"LTX2 connectors split-rope expected <= {expected_freqs} frequencies, got {current_freqs}."
            )
        if current_freqs < expected_freqs:
            pad_size = expected_freqs - current_freqs
            cos = torch.cat([torch.ones_like(freqs[:, :, :pad_size]), freqs.cos()], dim=-1)
            sin = torch.cat([torch.zeros_like(freqs[:, :, :pad_size]), freqs.sin()], dim=-1)
        else:
            cos = freqs.cos()
            sin = freqs.sin()
        cos = cos.reshape(batch_size, positions, self.num_attention_heads, -1).swapaxes(1, 2)
        sin = sin.reshape(batch_size, positions, self.num_attention_heads, -1).swapaxes(1, 2)
        return cos, sin


class _GELUProjector(nn.Module):
    def __init__(self, dim_in: int, dim_out: int, *, approximate: str = "none", bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out, bias=bias)
        self.approximate = approximate

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        return F.gelu(hidden_states, approximate=self.approximate)


class _FeedForward(nn.Module):
    def __init__(self, dim: int, *, activation_fn: str = "gelu-approximate", bias: bool = True) -> None:
        super().__init__()
        if activation_fn != "gelu-approximate":
            raise RuntimeError(
                "LTX2 connectors feed-forward only supports activation_fn='gelu-approximate' in the native slice."
            )
        inner_dim = int(dim * 4)
        self.net = nn.ModuleList(
            [
                _GELUProjector(dim, inner_dim, approximate="tanh", bias=bias),
                nn.Dropout(0.0),
                nn.Linear(inner_dim, dim, bias=bias),
            ]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class _Ltx2Attention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        heads: int,
        kv_heads: int,
        dim_head: int,
        bias: bool = True,
        out_bias: bool = True,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.head_dim = int(dim_head)
        self.heads = int(heads)
        self.kv_heads = int(kv_heads)
        self.inner_dim = self.head_dim * self.heads
        self.inner_kv_dim = self.head_dim * self.kv_heads
        self.query_dim = int(query_dim)
        self.rope_type = rope_type
        self.norm_q = torch.nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(self.inner_kv_dim, eps=1e-6, elementwise_affine=True)
        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=bias)
        self.to_k = nn.Linear(self.query_dim, self.inner_kv_dim, bias=bias)
        self.to_v = nn.Linear(self.query_dim, self.inner_kv_dim, bias=bias)
        self.to_out = nn.ModuleList([nn.Linear(self.inner_dim, self.query_dim, bias=out_bias), nn.Dropout(0.0)])

    def _prepare_attention_mask(
        self,
        attention_mask: torch.Tensor | None,
        *,
        batch_size: int,
        query_len: int,
        key_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if attention_mask is None:
            return None
        mask = attention_mask
        if mask.ndim == 2:
            mask = mask[:, None, None, :]
        elif mask.ndim == 3:
            mask = mask[:, None, :, :]
        elif mask.ndim != 4:
            raise RuntimeError(f"LTX2 connectors attention mask must be 2D/3D/4D, got {mask.ndim}D.")
        if mask.shape[0] != batch_size:
            raise RuntimeError(
                f"LTX2 connectors attention mask batch mismatch: expected {batch_size}, got {mask.shape[0]}."
            )
        if mask.shape[-1] != key_len:
            raise RuntimeError(
                f"LTX2 connectors attention mask key length mismatch: expected {key_len}, got {mask.shape[-1]}."
            )
        if mask.shape[-2] == 1:
            mask = mask.expand(batch_size, 1, query_len, key_len)
        elif mask.shape[-2] != query_len:
            raise RuntimeError(
                f"LTX2 connectors attention mask query length mismatch: expected 1 or {query_len}, got {mask.shape[-2]}."
            )
        return mask.to(device=device, dtype=dtype).expand(batch_size, self.heads, query_len, key_len)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        query_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        key_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"LTX2 connectors attention expects [batch,tokens,channels], got shape={tuple(hidden_states.shape)!r}."
            )
        context = hidden_states if encoder_hidden_states is None else encoder_hidden_states
        batch_size, query_len, _ = hidden_states.shape
        key_len = int(context.shape[1])

        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(context))
        value = self.to_v(context)

        if query_rotary_emb is not None:
            if self.rope_type == "interleaved":
                query = apply_interleaved_rotary_emb(query, query_rotary_emb)
                key = apply_interleaved_rotary_emb(key, key_rotary_emb or query_rotary_emb)
            else:
                query = apply_split_rotary_emb(query, query_rotary_emb)
                key = apply_split_rotary_emb(key, key_rotary_emb or query_rotary_emb)

        query = query.unflatten(2, (self.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (self.kv_heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (self.kv_heads, -1)).transpose(1, 2)

        attention_bias = self._prepare_attention_mask(
            attention_mask,
            batch_size=batch_size,
            query_len=query_len,
            key_len=key_len,
            device=query.device,
            dtype=query.dtype,
        )
        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_bias,
            dropout_p=0.0,
            is_causal=False,
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class _Ltx2TransformerBlock1d(nn.Module):
    def __init__(self, *, dim: int, num_attention_heads: int, attention_head_dim: int, rope_type: str) -> None:
        super().__init__()
        self.norm1 = torch.nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.attn1 = _Ltx2Attention(
            query_dim=dim,
            heads=num_attention_heads,
            kv_heads=num_attention_heads,
            dim_head=attention_head_dim,
            rope_type=rope_type,
        )
        self.norm2 = torch.nn.RMSNorm(dim, eps=1e-6, elementwise_affine=False)
        self.ff = _FeedForward(dim, activation_fn="gelu-approximate")

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            query_rotary_emb=rotary_emb,
        )
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return hidden_states


class _Ltx2ConnectorTransformer1d(nn.Module):
    def __init__(
        self,
        *,
        num_attention_heads: int,
        attention_head_dim: int,
        num_layers: int,
        num_learnable_registers: int | None,
        rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str,
    ) -> None:
        super().__init__()
        self.num_attention_heads = int(num_attention_heads)
        self.inner_dim = self.num_attention_heads * int(attention_head_dim)
        self.causal_temporal_positioning = bool(causal_temporal_positioning)
        self.num_learnable_registers = num_learnable_registers
        self.learnable_registers = None
        if num_learnable_registers is not None:
            init_registers = (torch.rand(num_learnable_registers, self.inner_dim) * 2.0) - 1.0
            self.learnable_registers = nn.Parameter(init_registers)
        self.rope = _Ltx2RotaryPosEmbed1d(
            self.inner_dim,
            base_seq_len=rope_base_seq_len,
            theta=rope_theta,
            double_precision=rope_double_precision,
            rope_type=rope_type,
            num_attention_heads=self.num_attention_heads,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                _Ltx2TransformerBlock1d(
                    dim=self.inner_dim,
                    num_attention_heads=self.num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    rope_type=rope_type,
                )
                for _ in range(int(num_layers))
            ]
        )
        self.norm_out = torch.nn.RMSNorm(self.inner_dim, eps=1e-6, elementwise_affine=False)
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        attn_mask_binarize_threshold: float = -9000.0,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if hidden_states.ndim != 3:
            raise RuntimeError(
                f"LTX2 connector transformer expects [batch,tokens,channels], got shape={tuple(hidden_states.shape)!r}."
            )
        batch_size, seq_len, _ = hidden_states.shape

        if self.learnable_registers is not None:
            if attention_mask is None:
                raise RuntimeError("LTX2 connector transformer with learnable registers requires an attention_mask.")
            if seq_len % int(self.num_learnable_registers) != 0:
                raise RuntimeError(
                    "LTX2 connector transformer sequence length must be divisible by the number of learnable registers; "
                    f"got seq_len={seq_len} registers={self.num_learnable_registers}."
                )
            binary_mask = (attention_mask >= float(attn_mask_binarize_threshold)).int()
            if binary_mask.ndim == 4:
                binary_mask = binary_mask.squeeze(1).squeeze(1)
            if binary_mask.ndim != 2:
                raise RuntimeError(
                    f"LTX2 connector transformer binary mask must reduce to [batch,tokens], got {binary_mask.ndim}D."
                )
            if binary_mask.shape != (batch_size, seq_len):
                raise RuntimeError(
                    "LTX2 connector transformer binary mask shape mismatch: "
                    f"expected {(batch_size, seq_len)!r}, got {tuple(binary_mask.shape)!r}."
                )
            register_repeats = seq_len // int(self.num_learnable_registers)
            registers = torch.tile(self.learnable_registers, (register_repeats, 1))
            non_padded = [hidden_states[index, binary_mask[index].bool(), :] for index in range(batch_size)]
            pad_lengths = [seq_len - item.shape[0] for item in non_padded]
            padded = [F.pad(item, pad=(0, 0, 0, pad), value=0.0) for item, pad in zip(non_padded, pad_lengths)]
            hidden_states = torch.cat([item.unsqueeze(0) for item in padded], dim=0)
            flipped = torch.flip(binary_mask, dims=[1]).unsqueeze(-1)
            hidden_states = flipped * hidden_states + (1 - flipped) * registers
            attention_mask = torch.zeros_like(attention_mask)

        rotary_emb = self.rope(batch_size, seq_len, device=hidden_states.device)
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask, rotary_emb=rotary_emb)
        hidden_states = self.norm_out(hidden_states)
        return hidden_states, attention_mask


class Ltx2TextConnectors(nn.Module):
    def __init__(
        self,
        *,
        caption_channels: int,
        text_proj_in_factor: int,
        video_connector_num_attention_heads: int,
        video_connector_attention_head_dim: int,
        video_connector_num_layers: int,
        video_connector_num_learnable_registers: int | None,
        audio_connector_num_attention_heads: int,
        audio_connector_attention_head_dim: int,
        audio_connector_num_layers: int,
        audio_connector_num_learnable_registers: int | None,
        connector_rope_base_seq_len: int,
        rope_theta: float,
        rope_double_precision: bool,
        causal_temporal_positioning: bool,
        rope_type: str = "interleaved",
    ) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            caption_channels=int(caption_channels),
            text_proj_in_factor=int(text_proj_in_factor),
            video_connector_num_attention_heads=int(video_connector_num_attention_heads),
            video_connector_attention_head_dim=int(video_connector_attention_head_dim),
            video_connector_num_layers=int(video_connector_num_layers),
            video_connector_num_learnable_registers=video_connector_num_learnable_registers,
            audio_connector_num_attention_heads=int(audio_connector_num_attention_heads),
            audio_connector_attention_head_dim=int(audio_connector_attention_head_dim),
            audio_connector_num_layers=int(audio_connector_num_layers),
            audio_connector_num_learnable_registers=audio_connector_num_learnable_registers,
            connector_rope_base_seq_len=int(connector_rope_base_seq_len),
            rope_theta=float(rope_theta),
            rope_double_precision=bool(rope_double_precision),
            causal_temporal_positioning=bool(causal_temporal_positioning),
            rope_type=rope_type,
        )
        self.text_proj_in = nn.Linear(
            int(caption_channels) * int(text_proj_in_factor),
            int(caption_channels),
            bias=False,
        )
        self.video_connector = _Ltx2ConnectorTransformer1d(
            num_attention_heads=video_connector_num_attention_heads,
            attention_head_dim=video_connector_attention_head_dim,
            num_layers=video_connector_num_layers,
            num_learnable_registers=video_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )
        self.audio_connector = _Ltx2ConnectorTransformer1d(
            num_attention_heads=audio_connector_num_attention_heads,
            attention_head_dim=audio_connector_attention_head_dim,
            num_layers=audio_connector_num_layers,
            num_learnable_registers=audio_connector_num_learnable_registers,
            rope_base_seq_len=connector_rope_base_seq_len,
            rope_theta=rope_theta,
            rope_double_precision=rope_double_precision,
            causal_temporal_positioning=causal_temporal_positioning,
            rope_type=rope_type,
        )

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Ltx2TextConnectors":
        if not isinstance(config, Mapping):
            raise RuntimeError(f"LTX2 connectors from_config requires a mapping, got {type(config).__name__}.")
        rope_type = _require_str(config, "rope_type")
        if rope_type not in {"interleaved", "split"}:
            raise RuntimeError(f"LTX2 connectors rope_type must be 'interleaved' or 'split', got {rope_type!r}.")
        return cls(
            caption_channels=_require_int(config, "caption_channels"),
            text_proj_in_factor=_require_int(config, "text_proj_in_factor"),
            video_connector_num_attention_heads=_require_int(config, "video_connector_num_attention_heads"),
            video_connector_attention_head_dim=_require_int(config, "video_connector_attention_head_dim"),
            video_connector_num_layers=_require_int(config, "video_connector_num_layers"),
            video_connector_num_learnable_registers=_require_int(config, "video_connector_num_learnable_registers"),
            audio_connector_num_attention_heads=_require_int(config, "audio_connector_num_attention_heads"),
            audio_connector_attention_head_dim=_require_int(config, "audio_connector_attention_head_dim"),
            audio_connector_num_layers=_require_int(config, "audio_connector_num_layers"),
            audio_connector_num_learnable_registers=_require_int(config, "audio_connector_num_learnable_registers"),
            connector_rope_base_seq_len=_require_int(config, "connector_rope_base_seq_len"),
            rope_theta=_require_float(config, "rope_theta"),
            rope_double_precision=_require_bool(config, "rope_double_precision"),
            causal_temporal_positioning=_require_bool(config, "causal_temporal_positioning"),
            rope_type=rope_type,
        )

    def forward(
        self,
        text_encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        additive_mask: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if text_encoder_hidden_states.ndim != 3:
            raise RuntimeError(
                "LTX2 connectors expect packed text encoder hidden states with shape [batch,tokens,channels]; "
                f"got {tuple(text_encoder_hidden_states.shape)!r}."
            )
        if attention_mask is None:
            raise RuntimeError("LTX2 connectors require an attention_mask.")
        if not additive_mask:
            text_dtype = text_encoder_hidden_states.dtype
            attention_mask = (attention_mask - 1).reshape(
                attention_mask.shape[0],
                1,
                -1,
                attention_mask.shape[-1],
            )
            attention_mask = attention_mask.to(text_dtype) * torch.finfo(text_dtype).max

        text_encoder_hidden_states = self.text_proj_in(text_encoder_hidden_states)
        video_text_embedding, new_attn_mask = self.video_connector(text_encoder_hidden_states, attention_mask)
        attn_mask = (new_attn_mask < 1e-6).to(torch.int64)
        attn_mask = attn_mask.reshape(video_text_embedding.shape[0], video_text_embedding.shape[1], 1)
        video_text_embedding = video_text_embedding * attn_mask
        new_attn_mask = attn_mask.squeeze(-1)
        audio_text_embedding, _ = self.audio_connector(text_encoder_hidden_states, attention_mask)
        return video_text_embedding, audio_text_embedding, new_attn_mask


def _normalize_connector_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    wrapped_keys = [key for key in state_dict if key.startswith(_CONNECTORS_WRAPPER_PREFIX)]
    direct_keys = [key for key in state_dict if not key.startswith(_CONNECTORS_WRAPPER_PREFIX)]

    if wrapped_keys and direct_keys:
        raise RuntimeError(
            "LTX2 connectors state mixes wrapped `connectors.*` keys with direct keys; "
            "supported layouts are all-direct or all-wrapped only."
        )

    if wrapped_keys:
        for key, value in state_dict.items():
            if not key.startswith(_CONNECTORS_WRAPPER_PREFIX):
                raise RuntimeError(
                    "LTX2 connectors state mixes wrapped `connectors.*` keys with direct keys; "
                    "supported layouts are all-direct or all-wrapped only."
                )
            stripped = key[len(_CONNECTORS_WRAPPER_PREFIX) :]
            if not stripped:
                raise RuntimeError("LTX2 connectors state contains an empty `connectors.` wrapper key.")
            if stripped in normalized:
                raise RuntimeError(
                    f"LTX2 connectors wrapped layout collides after removing `connectors.`: {stripped!r}."
                )
            normalized[stripped] = value
        return normalized

    normalized.update(state_dict)
    return normalized


def load_ltx2_connectors(
    *,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Ltx2TextConnectors:
    module = Ltx2TextConnectors.from_config(config)
    normalized_state_dict = _normalize_connector_state_dict(state_dict)
    try:
        missing, unexpected = module.load_state_dict(normalized_state_dict, strict=False)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 connectors state load failed: {exc}") from exc
    if missing or unexpected:
        raise RuntimeError(
            "LTX2 connectors strict load failed: "
            f"missing={len(missing)} unexpected={len(unexpected)} "
            f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]}"
        )
    try:
        module = module.to(device=device, dtype=torch_dtype)
    except Exception:
        module = module.to(device=device)
    module.eval()
    return module
