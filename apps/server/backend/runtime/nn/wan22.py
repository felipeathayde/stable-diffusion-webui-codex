from __future__ import annotations

"""WAN 2.2 — GGUF path (generic, PyTorch‑first, no custom kernels).

This module ports useful pieces from the prior WAN GGUF code into the generic
runtime, using only:
- apps.server.backend.gguf (readers/quants)
- apps.server.backend.runtime.ops (dequantize, ops)
- PyTorch SDPA for attention

It provides:
- derive_spec_from_state(): parse GGUF state keys into a model spec
- WanUNetGGUF: minimal UNet wrapper with forward over SA/CA/FFN stacks
- run_txt2vid/run_img2vid: skeletons that validate stages and prepare flow

Notes
- VAE encode/decode and full sampler loop are wired later; we keep errors
  explicit instead of faking outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import os
import math

import torch
from apps.server.backend.runtime.utils import _load_gguf_state_dict, read_arbitrary_config
from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor
from apps.server.backend.runtime import memory_management


# ------------------------------ spec/mapping

@dataclass
class CrossAttnWeights:
    q_w: str | None = None
    q_b: str | None = None
    k_w: str | None = None
    k_b: str | None = None
    v_w: str | None = None
    v_b: str | None = None
    o_w: str | None = None
    o_b: str | None = None
    norm_q_w: str | None = None
    norm_q_b: str | None = None
    norm_k_w: str | None = None
    norm_k_b: str | None = None


@dataclass
class BlockSpec:
    index: int
    cross_attn: CrossAttnWeights = field(default_factory=CrossAttnWeights)
    self_attn: CrossAttnWeights = field(default_factory=CrossAttnWeights)
    ffn_in_w: Optional[str] = None
    ffn_in_b: Optional[str] = None
    ffn_out_w: Optional[str] = None
    ffn_out_b: Optional[str] = None
    norm3_w: Optional[str] = None
    norm3_b: Optional[str] = None
    modulation: Optional[str] = None  # [1,6,C]


@dataclass
class ModelSpec:
    d_model: Optional[int] = None
    n_heads: Optional[int] = None
    n_blocks: int = 0
    blocks: List[BlockSpec] = field(default_factory=list)
    time_emb_0_w: Optional[str] = None
    time_emb_0_b: Optional[str] = None
    time_emb_2_w: Optional[str] = None
    time_emb_2_b: Optional[str] = None
    time_proj_w: Optional[str] = None
    time_proj_b: Optional[str] = None
    head_modulation: Optional[str] = None  # [1,2,C]


def _shape_of(state: Mapping[str, object], key: str) -> Optional[Tuple[int, ...]]:
    v = state.get(key)
    if v is None:
        return None
    try:
        shp = tuple(int(s) for s in getattr(v, 'shape', tuple()))
        return shp if shp else None
    except Exception:
        return None


def derive_spec_from_state(state: Mapping[str, object]) -> ModelSpec:
    by_block: Dict[int, Dict[str, str]] = {}
    for k in state.keys():
        ks = str(k)
        if not ks.startswith("blocks."):
            continue
        try:
            rest = ks.split(".", 2)
            bi = int(rest[1])
            tail = rest[2]
        except Exception:
            continue
        by_block.setdefault(bi, {})[tail] = ks

    d_model: Optional[int] = None
    heads: Optional[int] = None
    if by_block:
        bk = by_block[min(by_block.keys())]
        for cname in ("cross_attn.q.weight", "cross_attn.k.weight", "cross_attn.o.weight"):
            key = bk.get(cname)
            if key:
                shp = _shape_of(state, key)
                if shp and len(shp) == 2:
                    d_model = int(shp[0] if cname.endswith("o.weight") else shp[1])
                    break
        if d_model and d_model % 128 == 0:
            h = d_model // 128
            if 8 <= h <= 64:
                heads = h

    blocks: List[BlockSpec] = []
    for bi in sorted(by_block.keys()):
        entries = by_block[bi]
        ca = CrossAttnWeights(
            q_w=entries.get("cross_attn.q.weight"), q_b=entries.get("cross_attn.q.bias"),
            k_w=entries.get("cross_attn.k.weight"), k_b=entries.get("cross_attn.k.bias"),
            v_w=entries.get("cross_attn.v.weight"), v_b=entries.get("cross_attn.v.bias"),
            o_w=entries.get("cross_attn.o.weight"), o_b=entries.get("cross_attn.o.bias"),
            norm_q_w=entries.get("cross_attn.norm_q.weight"), norm_q_b=entries.get("cross_attn.norm_q.bias"),
            norm_k_w=entries.get("cross_attn.norm_k.weight"), norm_k_b=entries.get("cross_attn.norm_k.bias"),
        )
        sa = CrossAttnWeights(
            q_w=entries.get("self_attn.q.weight"), q_b=entries.get("self_attn.q.bias"),
            k_w=entries.get("self_attn.k.weight"), k_b=entries.get("self_attn.k.bias"),
            v_w=entries.get("self_attn.v.weight"), v_b=entries.get("self_attn.v.bias"),
            o_w=entries.get("self_attn.o.weight"), o_b=entries.get("self_attn.o.bias"),
            norm_q_w=entries.get("self_attn.norm_q.weight"), norm_q_b=entries.get("self_attn.norm_q.bias"),
            norm_k_w=entries.get("self_attn.norm_k.weight"), norm_k_b=entries.get("self_attn.norm_k.bias"),
        )
        bspec = BlockSpec(index=bi, cross_attn=ca, self_attn=sa)
        bspec.ffn_in_w = entries.get("ffn.0.weight")
        bspec.ffn_in_b = entries.get("ffn.0.bias")
        bspec.ffn_out_w = entries.get("ffn.2.weight")
        bspec.ffn_out_b = entries.get("ffn.2.bias")
        bspec.norm3_w = entries.get("norm3.weight")
        bspec.norm3_b = entries.get("norm3.bias")
        bspec.modulation = entries.get("modulation")
        blocks.append(bspec)

    time_emb_0_w = "time_embedding.0.weight" if "time_embedding.0.weight" in state else None
    time_emb_0_b = "time_embedding.0.bias" if "time_embedding.0.bias" in state else None
    time_emb_2_w = "time_embedding.2.weight" if "time_embedding.2.weight" in state else None
    time_emb_2_b = "time_embedding.2.bias" if "time_embedding.2.bias" in state else None
    time_proj_w = "time_projection.1.weight" if "time_projection.1.weight" in state else None
    time_proj_b = "time_projection.1.bias" if "time_projection.1.bias" in state else None
    head_mod = "head.modulation" if "head.modulation" in state else None

    return ModelSpec(
        d_model=d_model, n_heads=heads, n_blocks=len(blocks), blocks=blocks,
        time_emb_0_w=time_emb_0_w, time_emb_0_b=time_emb_0_b,
        time_emb_2_w=time_emb_2_w, time_emb_2_b=time_emb_2_b,
        time_proj_w=time_proj_w, time_proj_b=time_proj_b,
        head_modulation=head_mod,
    )


# ------------------------------ ops

def _rms_norm(x: torch.Tensor, w: Any) -> torch.Tensor:
    w = dequantize_tensor(w)
    if not torch.is_tensor(w):
        w = torch.as_tensor(w, device=x.device, dtype=x.dtype)
    eps = 1e-6
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * w


def _linear(x: torch.Tensor, w: Any, b: Any | None) -> torch.Tensor:
    w = dequantize_tensor(w)
    if not torch.is_tensor(w):
        w = torch.as_tensor(w)
    w = w.to(device=x.device, dtype=x.dtype)
    if b is not None:
        b = dequantize_tensor(b)
        if not torch.is_tensor(b):
            b = torch.as_tensor(b)
        b = b.to(device=x.device, dtype=x.dtype)
    return torch.nn.functional.linear(x, w, b)


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


def _split_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    B, L, C = x.shape
    D = C // h
    return x.view(B, L, h, D).permute(0, 2, 1, 3).contiguous()


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)


def _ca(x: torch.Tensor, ctx: torch.Tensor, *, w: CrossAttnWeights, state: Mapping[str, Any], heads: int, scale=None, shift=None) -> torch.Tensor:
    q_in = _rms_norm(x, state[w.norm_q_w]) if w.norm_q_w else x
    if scale is not None:
        q_in = q_in * (1 + scale)
    if shift is not None:
        q_in = q_in + shift
    q = _linear(q_in, state[w.q_w], state.get(w.q_b))
    k = _linear(_rms_norm(ctx, state[w.norm_k_w]) if w.norm_k_w else ctx, state[w.k_w], state.get(w.k_b))
    v = _linear(ctx, state[w.v_w], state.get(w.v_b))
    qh = _split_heads(q, heads)
    kh = _split_heads(k, heads)
    vh = _split_heads(v, heads)
    ah = _sdpa(qh, kh, vh, causal=False)
    a = _merge_heads(ah)
    out = _linear(a, state[w.o_w], state.get(w.o_b))
    return x + out


def _sa(x: torch.Tensor, *, w: CrossAttnWeights, state: Mapping[str, Any], heads: int, scale=None, shift=None) -> torch.Tensor:
    q_in = _rms_norm(x, state[w.norm_q_w]) if w.norm_q_w else x
    if scale is not None:
        q_in = q_in * (1 + scale)
    if shift is not None:
        q_in = q_in + shift
    q = _linear(q_in, state[w.q_w], state.get(w.q_b))
    k = _linear(_rms_norm(x, state[w.norm_k_w]) if w.norm_k_w else x, state[w.k_w], state.get(w.k_b))
    v = _linear(x, state[w.v_w], state.get(w.v_b))
    qh = _split_heads(q, heads)
    kh = _split_heads(k, heads)
    vh = _split_heads(v, heads)
    ah = _sdpa(qh, kh, vh, causal=False)
    a = _merge_heads(ah)
    out = _linear(a, state[w.o_w], state.get(w.o_b))
    return x + out


class WanUNetGGUF:
    def __init__(self, stage_dir: str, *, logger=None) -> None:
        self._logger = logger
        self.stage_dir = stage_dir
        self.state: Dict[str, Any] = self._load_state(stage_dir)
        self.spec: ModelSpec = derive_spec_from_state(self.state)

    def _load_state(self, stage_dir: str) -> Dict[str, Any]:
        path = _pick_stage_gguf(stage_dir, 'high') or _pick_stage_gguf(stage_dir, 'low')
        if not path or not os.path.isfile(path):
            raise RuntimeError(f".gguf not found in {stage_dir}")
        state = _load_gguf_state_dict(path)
        # bake once for speed on first use
        class _D:
            def parameters(self_inner):
                for v in state.values():
                    if hasattr(v, 'gguf_cls'):
                        yield v
        try:
            memory_management.bake_gguf_model(_D())
        except Exception:
            pass
        if self._logger:
            keys = list(state.keys())
            self._logger.info("[wan22.gguf] tensors=%d sample=%s", len(keys), keys[:3])
        return state

    def forward(self, x: torch.Tensor, t: torch.Tensor | float | int, cond: torch.Tensor, *, dtype: str = "bf16") -> torch.Tensor:
        spec = self.spec
        if spec.d_model is None or spec.n_heads is None or not spec.blocks:
            raise RuntimeError("WAN22 spec incomplete (d_model/heads/blocks)")
        C = spec.d_model
        H = spec.n_heads

        # dtype cast
        tt = {
            'fp16': torch.float16,
            'bf16': getattr(torch, 'bfloat16', torch.float16),
            'fp32': torch.float32,
        }.get(dtype, torch.float16)

        device = x.device
        cond = cond.to(device=device, dtype=tt)
        x = x.to(device=device, dtype=tt)

        # Time embedding (sinusoidal -> 5120 -> proj -> [B,6,C])
        te0_w = self.state.get(spec.time_emb_0_w)
        te0_b = self.state.get(spec.time_emb_0_b)
        te2_w = self.state.get(spec.time_emb_2_w)
        te2_b = self.state.get(spec.time_emb_2_b)
        tp_w = self.state.get(spec.time_proj_w)
        tp_b = self.state.get(spec.time_proj_b)
        for name, tensor in {
            'time_embedding.0.weight': te0_w,
            'time_embedding.0.bias': te0_b,
            'time_embedding.2.weight': te2_w,
            'time_embedding.2.bias': te2_b,
            'time_projection.1.weight': tp_w,
            'time_projection.1.bias': tp_b,
        }.items():
            if tensor is None:
                raise RuntimeError(f"Missing weight: {name}")

        if isinstance(t, torch.Tensor):
            t_in = t.to(device=device, dtype=torch.float32).view(-1)
        elif isinstance(t, (int, float)):
            t_in = torch.tensor([float(t)], device=device, dtype=torch.float32)
        else:
            t_in = torch.as_tensor(t, device=device, dtype=torch.float32).view(-1)
        if t_in.numel() == 1 and x.shape[0] > 1:
            t_in = t_in.expand(x.shape[0])

        base_dim = int(_shape_of(self.state, spec.time_emb_0_w)[-1] if _shape_of(self.state, spec.time_emb_0_w) else 256)
        half = max(base_dim // 2, 1)
        freq = torch.arange(half, device=device, dtype=torch.float32)
        div_term = torch.exp(-math.log(10000.0) * freq / max(half - 1, 1))
        angles = t_in[:, None] * div_term[None, :]
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        if emb.shape[1] != base_dim:
            emb = torch.nn.functional.pad(emb, (0, base_dim - emb.shape[1]))
        emb = emb.to(dtype=tt)

        t5120 = _linear(emb, te0_w, te0_b)
        t5120 = t5120 * torch.sigmoid(t5120)  # SiLU
        t5120 = _linear(t5120, te2_w, te2_b)
        tproj = _linear(t5120, tp_w, tp_b).view(t5120.shape[0], 6, C)

        h = x
        for bs in spec.blocks:
            # per-block modulation slices
            s_sa, b_sa, s_ca, b_ca, s_ffn, b_ffn = tproj[:, 0], tproj[:, 1], tproj[:, 2], tproj[:, 3], tproj[:, 4], tproj[:, 5]
            if bs.modulation and bs.modulation in self.state:
                mod = self.state[bs.modulation]
                mod = dequantize_tensor(mod)
                if not torch.is_tensor(mod):
                    mod = torch.as_tensor(mod, device=device, dtype=tt)
                m = tproj * mod  # broadcast [B,6,C]
                s_sa, b_sa, s_ca, b_ca, s_ffn, b_ffn = m[:, 0], m[:, 1], m[:, 2], m[:, 3], m[:, 4], m[:, 5]

            # Self-attention
            if bs.self_attn.q_w and bs.self_attn.k_w and bs.self_attn.v_w and bs.self_attn.o_w:
                h = _sa(h, w=bs.self_attn, state=self.state, heads=H, scale=s_sa, shift=b_sa)
            # Cross-attention
            h = _ca(h, cond, w=bs.cross_attn, state=self.state, heads=H, scale=s_ca, shift=b_ca)
            # FFN
            if bs.ffn_in_w and bs.ffn_out_w and bs.norm3_w:
                u = _rms_norm(h, self.state[bs.norm3_w])
                u = u * (1 + s_ffn) + b_ffn
                u = _linear(u, self.state[bs.ffn_in_w], self.state.get(bs.ffn_in_b))
                u = u * torch.sigmoid(u)  # SiLU
                u = _linear(u, self.state[bs.ffn_out_w], self.state.get(bs.ffn_out_b))
                h = h + u
        return h


# ------------------------------ helpers for stage files

def _normalize_win_path(p: str) -> str:
    if os.name == 'nt':
        return p
    if len(p) >= 2 and p[1] == ':' and p[0].isalpha():
        drive = p[0].lower()
        rest = p[2:].lstrip('\\/')
        return f"/mnt/{drive}/" + rest.replace('\\\\', '/').replace('\\', '/')
    return p


def _first_gguf_in(dir_path: Optional[str]) -> Optional[str]:
    if not dir_path:
        return None
    _dir = _normalize_win_path(dir_path)
    abspath = _dir if os.path.isabs(_dir) else os.path.abspath(_dir)
    try:
        for fn in os.listdir(abspath):
            if fn.lower().endswith('.gguf'):
                return os.path.join(abspath, fn)
    except Exception:
        return None
    return None


def _pick_stage_gguf(dir_path: Optional[str], stage: str) -> Optional[str]:
    p = _first_gguf_in(dir_path)
    if not dir_path:
        return p
    _dir = _normalize_win_path(dir_path)
    abspath = _dir if os.path.isabs(_dir) else os.path.abspath(_dir)
    try:
        stage_lc = stage.lower().strip()
        cands = [fn for fn in os.listdir(abspath) if fn.lower().endswith('.gguf')]
        for fn in cands:
            name = fn.lower()
            if stage_lc == 'high' and ('high' in name or 'highnoise' in name):
                return os.path.join(abspath, fn)
            if stage_lc == 'low' and ('low' in name or 'lownoise' in name):
                return os.path.join(abspath, fn)
    except Exception:
        pass
    return p


# ------------------------------ task entrypoints (skeletons)

@dataclass(frozen=True)
class StageConfig:
    model_dir: str
    sampler: str
    scheduler: str
    steps: int
    cfg_scale: Optional[float]


@dataclass(frozen=True)
class RunConfig:
    width: int
    height: int
    fps: int
    num_frames: int
    guidance_scale: Optional[float]
    dtype: str
    device: str
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    init_image: Optional[object] = None
    vae_dir: Optional[str] = None
    text_encoder_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    high: Optional[StageConfig] = None
    low: Optional[StageConfig] = None


def run_txt2vid(cfg: RunConfig, *, logger=None) -> List[object]:
    hi = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi or not lo:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    if logger:
        logger.info("[wan22.gguf] high=%s low=%s", hi, lo)
    # Forward mapping of tokens→latents→frames is pending; keep error explicit
    raise RuntimeError("WAN22 GGUF txt2vid forward pending (tokenization/latent decode)")


def run_img2vid(cfg: RunConfig, *, logger=None) -> List[object]:
    hi = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi or not lo:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")
    if logger:
        logger.info("[wan22.gguf] high=%s low=%s", hi, lo)
    raise RuntimeError("WAN22 GGUF img2vid forward pending (VAE encode/UNet loop/decode)")


__all__ = [
    'ModelSpec', 'BlockSpec', 'CrossAttnWeights', 'derive_spec_from_state',
    'WanUNetGGUF', 'run_txt2vid', 'run_img2vid',
]

