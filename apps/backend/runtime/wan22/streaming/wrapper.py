"""StreamedWanDiTGGUF wrapper for block-based GGUF tensor streaming."""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import torch

from apps.backend.runtime.ops.operations_gguf import dequantize_tensor

from .specs import WanBlockInfo, WanExecutionPlan, build_execution_plan
from .controller import WanCoreController

logger = logging.getLogger("backend.runtime.wan22.streaming.wrapper")


class StreamedWanDiTGGUF:
    """Wrapper around WanDiTGGUF enabling block-based GGUF tensor streaming.

    This wrapper intercepts the forward pass and loads/unloads GGUF tensors
    per-block, using the WanCoreController to manage GPU memory.

    The original WanDiTGGUF state dict is not modified; tensors are
    dequantized and cached on-demand.

    Example:
        plan = build_execution_plan(dit.state, dit.spec.n_blocks)
        controller = WanCoreController(storage="cpu", compute="cuda")
        streamed = StreamedWanDiTGGUF(dit, plan, controller)
        output = streamed.forward(tokens, timestep, cond, dtype="bf16")
    """

    def __init__(
        self,
        base_dit: Any,  # WanDiTGGUF
        execution_plan: WanExecutionPlan,
        controller: WanCoreController,
    ) -> None:
        self._base = base_dit
        self._plan = execution_plan
        self._controller = controller

        # Cache references from base model
        self.state = base_dit.state
        self.spec = base_dit.spec
        self.stage_dir = base_dit.stage_dir
        self._logger = getattr(base_dit, "_logger", None)

        logger.info(
            "StreamedWanDiTGGUF initialized: %d blocks, %.2f MB total",
            len(execution_plan),
            execution_plan.total_bytes / (1024 * 1024),
        )

    @property
    def base_dit(self) -> Any:
        """Access the underlying WanDiTGGUF."""
        return self._base

    @property
    def controller(self) -> WanCoreController:
        """Access the streaming controller."""
        return self._controller

    def _get_dtype(self, dtype_str: str) -> torch.dtype:
        """Convert dtype string to torch.dtype."""
        return {
            "fp16": torch.float16,
            "bf16": getattr(torch, "bfloat16", torch.float16),
            "fp32": torch.float32,
        }.get(dtype_str, torch.float16)

    def _linear(
        self,
        x: torch.Tensor,
        weight_key: str,
        bias_key: Optional[str],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Apply linear transformation with streamed weights."""
        weight = self._controller.get_tensor(self.state, weight_key, dtype)
        bias = None
        if bias_key and bias_key in self.state:
            bias = self._controller.get_tensor(self.state, bias_key, dtype)

        weight = weight.to(device=x.device, dtype=x.dtype)
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)

        return torch.nn.functional.linear(x, weight, bias)

    def _layer_norm(
        self,
        x: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply layer normalization (potentially affine)."""
        normed = torch.nn.functional.layer_norm(x, x.shape[-1:], eps=1e-6)
        if weight is not None:
            weight = weight.to(device=x.device, dtype=x.dtype)
            normed = normed * weight
        if bias is not None:
            bias = bias.to(device=x.device, dtype=x.dtype)
            normed = normed + bias
        return normed

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        causal: bool = False,
    ) -> torch.Tensor:
        """Scaled dot-product attention via PyTorch SDPA."""
        return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)

    def _split_heads(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        """Split last dimension into [num_heads, head_dim]."""
        B, T, C = x.shape
        head_dim = C // num_heads
        x = x.view(B, T, num_heads, head_dim)
        x = x.permute(0, 2, 1, 3)  # B, H, T, D
        return x

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge heads back to [B, T, C]."""
        B, H, T, D = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(B, T, H * D)
        return x

    def _sinusoidal_position_embedding(
        self,
        length: int,
        dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build sinusoidal position embeddings."""
        position = torch.arange(0, length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q/K."""
        # q, k: (B, H, T, D)
        # cos, sin: (T, D)
        # Expand cos/sin to match q/k
        cos = cos.unsqueeze(0).unsqueeze(0)  # 1, 1, T, D
        sin = sin.unsqueeze(0).unsqueeze(0)

        q1, q2 = q[..., ::2], q[..., 1::2]
        k1, k2 = k[..., ::2], k[..., 1::2]

        q_rot = torch.stack([q1 * cos[..., ::2] - q2 * sin[..., ::2], q1 * sin[..., ::2] + q2 * cos[..., ::2]], dim=-1)
        k_rot = torch.stack([k1 * cos[..., ::2] - k2 * sin[..., ::2], k1 * sin[..., ::2] + k2 * cos[..., ::2]], dim=-1)

        q_out = q_rot.flatten(-2)
        k_out = k_rot.flatten(-2)

        return q_out, k_out

    def _get_block_tensors(
        self,
        block: WanBlockInfo,
        state: Dict[str, Any],
        dtype: torch.dtype,
    ) -> Dict[str, torch.Tensor]:
        """Load all tensors for a block and return them as a dict."""
        tensors: Dict[str, torch.Tensor] = {}
        for key in block.tensor_keys:
            tensors[key] = self._controller.get_tensor(state, key, dtype)
        return tensors

    def forward(
        self,
        tokens: torch.Tensor,
        timestep: float,
        cond: torch.Tensor,
        *,
        dtype: str = "fp16",
        return_time_proj: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with block-based GGUF tensor streaming."""
        device = tokens.device
        tt = self._get_dtype(dtype)
        x = tokens.to(device=device, dtype=tt)
        ctx = cond.to(device=device, dtype=tt)

        B, T, C = x.shape
        spec = self.spec

        # Time embedding
        t_in = torch.tensor([timestep], device=device, dtype=tt)
        if t_in.dim() == 1 and x.shape[0] > 1:
            t_in = t_in.expand(x.shape[0])

        # Time projection (non-streamed, small)
        tproj = self._base._compute_time_proj(t_in, device=device, dtype=tt)

        # Text embedding projection (non-streamed, small)
        if spec.text_emb_0_w and spec.text_emb_2_w:
            te0_w = self._controller.get_tensor(self.state, spec.text_emb_0_w, tt)
            te0_b = self.state.get(spec.text_emb_0_b)
            if te0_b is not None:
                te0_b = self._controller.get_tensor(self.state, spec.text_emb_0_b, tt)
            te2_w = self._controller.get_tensor(self.state, spec.text_emb_2_w, tt)
            te2_b = self.state.get(spec.text_emb_2_b)
            if te2_b is not None:
                te2_b = self._controller.get_tensor(self.state, spec.text_emb_2_b, tt)

            ctx = torch.nn.functional.linear(ctx, te0_w.to(ctx), te0_b.to(ctx) if te0_b is not None else None)
            ctx = torch.nn.functional.gelu(ctx)
            ctx = torch.nn.functional.linear(ctx, te2_w.to(ctx), te2_b.to(ctx) if te2_b is not None else None)
        else:
            if ctx.shape[-1] != C:
                raise RuntimeError(
                    f"WAN22: text embedding dim {ctx.shape[-1]} != d_model {C}"
                )

        h = x

        # === Block-wise streaming ===
        for block_idx, block_info in enumerate(self._plan):
            bs = spec.blocks[block_info.index]

            # Load block tensors
            self._controller.ensure_block_on_device(block_info, self.state, tt)

            # Per-block modulation
            e = tproj
            if bs.modulation and bs.modulation in self.state:
                mod = self._controller.get_tensor(self.state, bs.modulation, tt)
                if mod.dim() == 2:
                    mod = mod.unsqueeze(0)
                e = tproj + mod

            # Unpack modulation: [sa_shift, sa_scale, sa_gate, ffn_shift, ffn_scale, ffn_gate]
            sa_shift = e[:, 0]
            sa_scale = e[:, 1]
            sa_gate = e[:, 2]
            ffn_shift = e[:, 3]
            ffn_scale = e[:, 4]
            ffn_gate = e[:, 5]

            # Self-attention
            if bs.self_attn.q_w and bs.self_attn.k_w and bs.self_attn.v_w and bs.self_attn.o_w:
                x_sa = self._layer_norm(h)
                x_sa = x_sa * (1 + sa_scale[:, None, :]) + sa_shift[:, None, :]

                # QKV projections
                q = self._linear(x_sa, bs.self_attn.q_w, bs.self_attn.q_b, tt)
                k = self._linear(x_sa, bs.self_attn.k_w, bs.self_attn.k_b, tt)
                v = self._linear(x_sa, bs.self_attn.v_w, bs.self_attn.v_b, tt)

                qh = self._split_heads(q, spec.num_heads)
                kh = self._split_heads(k, spec.num_heads)
                vh = self._split_heads(v, spec.num_heads)

                # Optional RoPE
                if spec.rope_dim and spec.rope_dim > 0:
                    rope_dim = spec.rope_dim
                    if rope_dim > qh.shape[-1]:
                        rope_dim = qh.shape[-1]
                    pos = self._sinusoidal_position_embedding(T, rope_dim, device=device, dtype=tt)
                    cos = pos[..., ::2]
                    sin = pos[..., 1::2]
                    qh, kh = self._apply_rope(qh, kh, cos, sin)

                ah = self._sdpa(qh, kh, vh, causal=False)
                a = self._merge_heads(ah)

                sa_out = self._linear(a, bs.self_attn.o_w, bs.self_attn.o_b, tt)
                h = h + sa_out * sa_gate[:, None, :]

            # Cross-attention
            x_ca = h
            if bs.norm3_w:
                norm3_w = self._controller.get_tensor(self.state, bs.norm3_w, tt)
                norm3_b = None
                if bs.norm3_b:
                    norm3_b = self._controller.get_tensor(self.state, bs.norm3_b, tt)
                x_ca = self._layer_norm(h, norm3_w, norm3_b)

            # Cross-attn QKV
            ca = bs.cross_attn
            q_ca = self._linear(x_ca, ca.q_w, ca.q_b, tt)
            k_ca = self._linear(ctx, ca.k_w, ca.k_b, tt)
            v_ca = self._linear(ctx, ca.v_w, ca.v_b, tt)

            qh_ca = self._split_heads(q_ca, spec.num_heads)
            kh_ca = self._split_heads(k_ca, spec.num_heads)
            vh_ca = self._split_heads(v_ca, spec.num_heads)
            ah_ca = self._sdpa(qh_ca, kh_ca, vh_ca, causal=False)
            a_ca = self._merge_heads(ah_ca)

            ca_out = self._linear(a_ca, ca.o_w, ca.o_b, tt)
            h = h + ca_out

            # FFN
            if bs.ffn_in_w and bs.ffn_out_w:
                x_ffn = self._layer_norm(h)
                x_ffn = x_ffn * (1 + ffn_scale[:, None, :]) + ffn_shift[:, None, :]
                u = self._linear(x_ffn, bs.ffn_in_w, bs.ffn_in_b, tt)
                u = u * torch.sigmoid(u)  # SiLU
                u = self._linear(u, bs.ffn_out_w, bs.ffn_out_b, tt)
                h = h + u * ffn_gate[:, None, :]

            # Maybe evict block tensors
            self._controller.maybe_evict(block_info)

        if return_time_proj:
            return h, tproj
        return h

    def tokens_to_latents(
        self,
        tokens: torch.Tensor,
        grid: Tuple[int, int, int],
        timestep: float,
        device: torch.device,
        dtype: torch.dtype,
        tproj: torch.Tensor,
    ) -> torch.Tensor:
        """Delegate to base model's tokens_to_latents."""
        return self._base.tokens_to_latents(
            tokens, grid, timestep=timestep, device=device, dtype=dtype, tproj=tproj
        )

    def reset_controller(self) -> None:
        """Reset controller state (call between generations)."""
        self._controller.reset()

    def get_transfer_stats(self) -> dict:
        """Get transfer statistics summary."""
        return self._controller.stats.summary()

    def clear_cache(self) -> None:
        """Clear dequantization cache."""
        self._controller.evict_all()


def wrap_wan_dit_for_streaming(
    dit: Any,  # WanDiTGGUF
    policy: str = "naive",
    window_size: int = 2,
    compute_device: Optional[str] = None,
) -> StreamedWanDiTGGUF:
    """Factory function to wrap a WanDiTGGUF for streaming.

    Args:
        dit: The WanDiTGGUF model to wrap.
        policy: Streaming policy ("naive", "window", "aggressive").
        window_size: Window size for "window" policy.
        compute_device: Compute device (default: auto-detect).

    Returns:
        StreamedWanDiTGGUF wrapper.
    """
    from .controller import create_wan_controller

    plan = build_execution_plan(dit.state, dit.spec.n_blocks)
    controller = create_wan_controller(
        policy=policy,
        window_size=window_size,
        compute_device=compute_device,
    )

    return StreamedWanDiTGGUF(dit, plan, controller)

