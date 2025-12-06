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
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=causal
        )

    def _split_heads(
        self, x: torch.Tensor, num_heads: int
    ) -> torch.Tensor:
        """Split tensor into attention heads: [B, L, C] -> [B, H, L, C/H]."""
        B, L, C = x.shape
        head_dim = C // num_heads
        return x.view(B, L, num_heads, head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """Merge attention heads: [B, H, L, D] -> [B, L, H*D]."""
        B, H, L, D = x.shape
        return x.transpose(1, 2).contiguous().view(B, L, H * D)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor | float | int,
        cond: torch.Tensor,
        *,
        dtype: str = "bf16",
        return_time_proj: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with block-based streaming.

        This replicates WanDiTGGUF.forward() but loads block tensors
        on-demand using the streaming controller.
        """
        spec = self.spec
        if spec.d_model is None or spec.n_heads is None or not spec.blocks:
            raise RuntimeError("WAN22 spec incomplete (d_model/heads/blocks)")

        C = spec.d_model
        H = spec.n_heads
        tt = self._get_dtype(dtype)
        device = x.device

        cond = cond.to(device=device, dtype=tt)
        x = x.to(device=device, dtype=tt)

        # Time embedding
        if isinstance(t, torch.Tensor):
            t_in = t.to(device=device, dtype=torch.float32).view(-1)
        elif isinstance(t, (int, float)):
            t_in = torch.tensor([float(t)], device=device, dtype=torch.float32)
        else:
            t_in = torch.as_tensor(t, device=device, dtype=torch.float32).view(-1)

        if t_in.numel() == 1 and x.shape[0] > 1:
            t_in = t_in.expand(x.shape[0])

        # Time projection (non-streamed, small)
        tproj = self._base._compute_time_proj(t_in, device=device, dtype=tt)

        # Text embedding projection (non-streamed, small)
        ctx = cond
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

                qh = self._split_heads(q, H)
                kh = self._split_heads(k, H)
                vh = self._split_heads(v, H)
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

            qh_ca = self._split_heads(q_ca, H)
            kh_ca = self._split_heads(k_ca, H)
            vh_ca = self._split_heads(v_ca, H)
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
        self._controller.clear_cache()


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
