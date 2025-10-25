from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Tuple


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


def derive_spec_from_state(state: Mapping[str, object]) -> ModelSpec:
    """Best-effort parse of GGUF state dict keys into a block/cross-attn spec.

    Does not load tensors; only inspects key names and shapes when available.
    """
    # Collect keys by block
    by_block: Dict[int, Dict[str, str]] = {}
    shape_of: Dict[str, Tuple[int, ...]] = {}

    # Try to fetch shapes from ParameterGGUF tensors if present
    def _shape(k: str) -> Optional[Tuple[int, ...]]:
        v = state.get(k)
        if v is None:
            return None
        try:
            shp = tuple(int(s) for s in getattr(v, 'shape', tuple()))
            return shp if shp else None
        except Exception:
            return None

    pat = re.compile(r"^blocks\.(\d+)\.(.+)$")
    for k in state.keys():
        m = pat.match(str(k))
        if not m:
            continue
        bi = int(m.group(1))
        rest = m.group(2)
        by_block.setdefault(bi, {})[rest] = k
        shp = _shape(k)
        if shp is not None:
            shape_of[k] = shp

    # Derive d_model from a typical projection weight if available
    d_model: Optional[int] = None
    heads: Optional[int] = None
    if by_block:
        # pick first block with cross_attn.k.weight or q.weight
        first = min(by_block.keys())
        bk = by_block[first]
        for cname in ("cross_attn.q.weight", "cross_attn.k.weight", "cross_attn.o.weight"):
            key = bk.get(cname)
            if key and key in shape_of:
                shp = shape_of[key]
                # weight shapes are [out x in]
                if cname.endswith("o.weight"):
                    d_model = int(shp[0]) if len(shp) == 2 else None
                else:
                    d_model = int(shp[1]) if len(shp) == 2 else None
                break
        # Heuristic heads from 5120 dim → 40 heads (128 per head)
        if d_model and d_model % 128 == 0:
            h = d_model // 128
            if 8 <= h <= 64:
                heads = h

    blocks: List[BlockSpec] = []
    for bi in sorted(by_block.keys()):
        entries = by_block[bi]
        ca = CrossAttnWeights(
            q_w=entries.get("cross_attn.q.weight"),
            q_b=entries.get("cross_attn.q.bias"),
            k_w=entries.get("cross_attn.k.weight"),
            k_b=entries.get("cross_attn.k.bias"),
            v_w=entries.get("cross_attn.v.weight"),
            v_b=entries.get("cross_attn.v.bias"),
            o_w=entries.get("cross_attn.o.weight"),
            o_b=entries.get("cross_attn.o.bias"),
            norm_q_w=entries.get("cross_attn.norm_q.weight"),
            norm_q_b=entries.get("cross_attn.norm_q.bias"),
            norm_k_w=entries.get("cross_attn.norm_k.weight"),
            norm_k_b=entries.get("cross_attn.norm_k.bias"),
        )
        sa = CrossAttnWeights(
            q_w=entries.get("self_attn.q.weight"),
            q_b=entries.get("self_attn.q.bias"),
            k_w=entries.get("self_attn.k.weight"),
            k_b=entries.get("self_attn.k.bias"),
            v_w=entries.get("self_attn.v.weight"),
            v_b=entries.get("self_attn.v.bias"),
            o_w=entries.get("self_attn.o.weight"),
            o_b=entries.get("self_attn.o.bias"),
            norm_q_w=entries.get("self_attn.norm_q.weight"),
            norm_q_b=entries.get("self_attn.norm_q.bias"),
            norm_k_w=entries.get("self_attn.norm_k.weight"),
            norm_k_b=entries.get("self_attn.norm_k.bias"),
        )
        bspec = BlockSpec(index=bi, cross_attn=ca, self_attn=sa)
        # FFN
        bspec.ffn_in_w = entries.get("ffn.0.weight")
        bspec.ffn_in_b = entries.get("ffn.0.bias")
        bspec.ffn_out_w = entries.get("ffn.2.weight")
        bspec.ffn_out_b = entries.get("ffn.2.bias")
        bspec.norm3_w = entries.get("norm3.weight")
        bspec.norm3_b = entries.get("norm3.bias")
        bspec.modulation = entries.get("modulation")
        blocks.append(bspec)

    # Global time/head modulation
    time_emb_0_w = "time_embedding.0.weight" if "time_embedding.0.weight" in state else None
    time_emb_0_b = "time_embedding.0.bias" if "time_embedding.0.bias" in state else None
    time_emb_2_w = "time_embedding.2.weight" if "time_embedding.2.weight" in state else None
    time_emb_2_b = "time_embedding.2.bias" if "time_embedding.2.bias" in state else None
    time_proj_w = "time_projection.1.weight" if "time_projection.1.weight" in state else None
    time_proj_b = "time_projection.1.bias" if "time_projection.1.bias" in state else None
    head_mod = "head.modulation" if "head.modulation" in state else None

    return ModelSpec(
        d_model=d_model,
        n_heads=heads,
        n_blocks=len(blocks),
        blocks=blocks,
        time_emb_0_w=time_emb_0_w,
        time_emb_0_b=time_emb_0_b,
        time_emb_2_w=time_emb_2_w,
        time_emb_2_b=time_emb_2_b,
        time_proj_w=time_proj_w,
        time_proj_b=time_proj_b,
        head_modulation=head_mod,
    )
