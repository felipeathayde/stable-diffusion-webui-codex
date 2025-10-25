from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class UNetMeta:
    channels: Optional[int] = None
    context_dim: Optional[int] = None
    heads: Optional[int] = None
    head_dim: Optional[int] = None


class GGUFUNet:
    """Minimal UNet wrapper that holds GGUF state and exposes a forward.

    Phase 1: loading + summary; forward not yet wired (raises for now).
    """

    def __init__(self, stage_dir: str, *, logger=None) -> None:
        self._logger = logger
        self.stage_dir = stage_dir
        self.meta = UNetMeta()
        self.state = self._load_state(stage_dir)
        self._summarize()

    def _load_state(self, stage_dir: str) -> Dict[str, Any]:
        import os
        from apps.server.backend.engines.video.wan.gguf_incore import _pick_stage_gguf  # reuse
        from apps.server.backend.runtime.utils import _load_gguf_state_dict
        from apps.server.backend.runtime import memory_management

        gguf_path = _pick_stage_gguf(stage_dir, "high") or _pick_stage_gguf(stage_dir, "low")
        if not gguf_path or not os.path.isfile(gguf_path):
            raise RuntimeError(f".gguf not found in {stage_dir}")
        state = _load_gguf_state_dict(gguf_path)
        # one-time bake
        class _D:
            def parameters(self_inner):
                for v in state.values():
                    if hasattr(v, "gguf_cls"):
                        yield v
        try:
            memory_management.bake_gguf_model(_D())
        except Exception:
            pass
        return state

    def _summarize(self) -> None:
        if self._logger:
            keys = list(self.state.keys())
            self._logger.info("[wan-gguf-core] UNet GGUF tensors: %d (e.g. %s)", len(keys), keys[:3])
            try:
                from .mapping import derive_spec_from_state
                spec = derive_spec_from_state(self.state)
                self._logger.info(
                    "[wan-gguf-core] Spec: d_model=%s heads=%s blocks=%d",
                    spec.d_model, spec.n_heads, spec.n_blocks,
                )
            except Exception as ex:
                if self._logger:
                    self._logger.warning("[wan-gguf-core] mapping derive failed: %s", ex)

    def forward(self, x, t, cond, guidance_scale: Optional[float] = None, *, dtype: str = "bf16"):
        """Partial forward: cross-attention stack only (residual on x).

        Requirements:
        - `cond` must be precomputed context embeddings [B, Lc, C].
        - Only cross-attention path is applied per block; MLP/residuals TBD.
        """
        from .mapping import derive_spec_from_state
        from .attn import cross_attention, self_attention
        from .ops import linear, to_dtype

        spec = derive_spec_from_state(self.state)
        if spec.d_model is None or spec.n_heads is None or not spec.blocks:
            raise RuntimeError("Unable to derive model spec (d_model/heads/blocks)")
        C = spec.d_model
        H = spec.n_heads
        device = x.device

        if hasattr(cond, "to"):
            cond = cond.to(device)
            if dtype:
                cond = to_dtype(cond, dtype)

        # Prepare time embedding and per-block modulation (scale/shift for SA, CA, FFN)
        def _silu(u):
            import torch
            return u * torch.sigmoid(u)

        import torch
        import math

        # Resolve timestep tensor to shape [B] on the sample device
        if isinstance(t, torch.Tensor):
            t_in = t.to(device=device, dtype=torch.float32)
            if t_in.ndim == 0:
                t_in = t_in.unsqueeze(0)
        elif isinstance(t, (int, float)):
            t_in = torch.tensor([float(t)], device=device, dtype=torch.float32)
        else:
            t_in = torch.as_tensor(t, device=device, dtype=torch.float32)
            if t_in.ndim == 0:
                t_in = t_in.unsqueeze(0)
        t_in = t_in.view(-1)
        if t_in.numel() == 1 and x.shape[0] > 1:
            t_in = t_in.expand(x.shape[0])
        if t_in.shape[0] != x.shape[0]:
            raise RuntimeError(f"Timestep batch {t_in.shape[0]} does not match sample batch {x.shape[0]}")

        # Basic sinusoidal embedding (match model expectation, typically 256 dim)
        te0_w = self.state.get(spec.time_emb_0_w)
        te0_b = self.state.get(spec.time_emb_0_b)
        te2_w = self.state.get(spec.time_emb_2_w)
        te2_b = self.state.get(spec.time_emb_2_b)
        tp_w = self.state.get(spec.time_proj_w)
        tp_b = self.state.get(spec.time_proj_b)
        required = {
            "time_embedding.0.weight": te0_w,
            "time_embedding.0.bias": te0_b,
            "time_embedding.2.weight": te2_w,
            "time_embedding.2.bias": te2_b,
            "time_projection.1.weight": tp_w,
            "time_projection.1.bias": tp_b,
        }
        missing = [name for name, tensor in required.items() if tensor is None]
        if missing:
            raise RuntimeError(f"Missing time embedding weights: {', '.join(missing)}")

        base_dim = getattr(te0_w, "shape", None)
        if base_dim:
            try:
                base_dim = int(base_dim[-1])
            except Exception:
                base_dim = None
        if not base_dim:
            base_dim = 256
        half = max(base_dim // 2, 1)
        freq_exponent = torch.arange(half, device=device, dtype=torch.float32)
        div_term = torch.exp(-math.log(10000.0) * freq_exponent / max(half - 1, 1))
        angles = t_in[:, None] * div_term[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # [B, base_dim]
        if emb.shape[1] != base_dim:
            emb = torch.nn.functional.pad(emb, (0, base_dim - emb.shape[1]))
        emb = to_dtype(emb, dtype)

        t5120 = linear(emb, te0_w, te0_b, dtype=dtype)
        t5120 = _silu(t5120)
        t5120 = linear(t5120, te2_w, te2_b, dtype=dtype)
        # time_projection: 5120 -> 30720 → [B,6,C]
        tproj = linear(t5120, tp_w, tp_b, dtype=dtype)
        tproj = tproj.view(tproj.shape[0], 6, C)

        # Iterate blocks and apply SA → CA → FFN with modulation
        h = to_dtype(x, dtype)
        if h.device != device:
            h = h.to(device)
        for bs in spec.blocks:
            ca = bs.cross_attn
            if not (ca.q_w and ca.k_w and ca.v_w and ca.o_w and ca.norm_q_w and ca.norm_k_w):
                raise RuntimeError(f"Missing cross-attn weights in block {bs.index}")
            # Per-block modulation: [1,6,C] * [B,6,C] → [B,6,C]
            if bs.modulation and bs.modulation in self.state:
                mod = self.state[bs.modulation]  # [1,6,C]
                if hasattr(mod, "gguf_cls"):
                    from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor
                    mod = dequantize_tensor(mod)
                if not torch.is_tensor(mod):
                    mod = torch.as_tensor(mod)
                mod = mod.to(device=device)
                mod = to_dtype(mod, dtype)
                m = tproj * mod  # broadcast
            else:
                m = tproj
            s_sa, b_sa, s_ca, b_ca, s_ffn, b_ffn = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4], m[:, 5]

            # Self-attention
            sa = bs.self_attn
            if sa.q_w and sa.k_w and sa.v_w and sa.o_w and sa.norm_q_w and sa.norm_k_w:
                h = self_attention(
                    h,
                    wq=self.state[sa.q_w], bq=self.state.get(sa.q_b),
                    wk=self.state[sa.k_w], bk=self.state.get(sa.k_b),
                    wv=self.state[sa.v_w], bv=self.state.get(sa.v_b),
                    wo=self.state[sa.o_w], bo=self.state.get(sa.o_b),
                    num_heads=H,
                    norm_q_w=self.state[sa.norm_q_w],
                    norm_k_w=self.state[sa.norm_k_w],
                    dtype=dtype,
                    scale=s_sa, shift=b_sa,
                )

            # Cross-attention
            h = cross_attention(
                h,
                cond,
                wq=self.state[ca.q_w], bq=self.state.get(ca.q_b),
                wk=self.state[ca.k_w], bk=self.state.get(ca.k_b),
                wv=self.state[ca.v_w], bv=self.state.get(ca.v_b),
                wo=self.state[ca.o_w], bo=self.state.get(ca.o_b),
                num_heads=H,
                norm_q_w=self.state[ca.norm_q_w],
                norm_k_w=self.state[ca.norm_k_w],
                dtype=dtype,
                scale=s_ca, shift=b_ca,
            )

            # FFN
            if bs.ffn_in_w and bs.ffn_out_w and bs.norm3_w:
                from .ops import rms_norm
                u = rms_norm(h, self.state[bs.norm3_w], dtype=dtype)
                u = u * (1 + s_ffn) + b_ffn
                u = linear(_silu(linear(u, self.state[bs.ffn_in_w], self.state.get(bs.ffn_in_b), dtype=dtype)),
                           self.state[bs.ffn_out_w], self.state.get(bs.ffn_out_b), dtype=dtype)
                h = h + u
        return h
