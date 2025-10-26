from __future__ import annotations

"""WAN 2.2 — GGUF path (generic, PyTorch‑first, no custom kernels).

This module ports useful pieces from the prior WAN GGUF code into the generic
runtime, using only:
- apps.server.backend.gguf (readers/quants)
- apps.server.backend.runtime.ops (dequantize, ops)
- PyTorch SDPA for attention

It provides:
- derive_spec_from_state(): parse GGUF state keys into a model spec
- WanDiTGGUF: minimal Diffusion Transformer (DiT) wrapper with forward over SA/CA/FFN stacks
- run_txt2vid/run_img2vid: skeletons that validate stages and prepare flow

Notes
- VAE encode/decode and full sampler loop are wired later; we keep errors
  explicit instead of faking outputs.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import os
import math
from pathlib import Path

import torch
from apps.server.backend.runtime.utils import _load_gguf_state_dict, read_arbitrary_config
from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor
from apps.server.backend.runtime import memory_management
import logging
from apps.server.backend.engines.diffusion.wan22_common import resolve_wan_repo_candidates
# WAN DiT helpers are inlined here to keep the one-file-per-model convention,
# matching flux/chroma. No dependence on legacy wan_gguf_core/*.

# Debug helpers (lightweight)
_LOG_ONCE = {
    'patch_embed': False,
    'patch_unembed': False,
    'sdpa': False,
}

def _get_logger_legacy(logger: Any):
    # Legacy duplicate; keep for compatibility if referenced elsewhere
    import logging
    if logger is not None:
        return logger
    lg = logging.getLogger("wan22.gguf")
    if not lg.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter('[wan22.gguf] %(levelname)s: %(message)s')
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg

def _li(logger, msg, *args):
    try:
        _get_logger(logger).info(msg, *args)
    except Exception:
        pass

def _dbg(logger, name: str, where: str) -> None:
    try:
        _get_logger(logger).info("[wan22.gguf] DEBUG: %s de função %s", where, name)
    except Exception:
        pass

from functools import wraps
def _io(fn):
    @wraps(fn)
    def _wrap(*args, **kwargs):
        try:
            _dbg(kwargs.get('logger', None), fn.__name__, 'antes')
        except Exception:
            _dbg(None, fn.__name__, 'antes')
        out = fn(*args, **kwargs)
        try:
            _dbg(kwargs.get('logger', None), fn.__name__, 'depois')
        except Exception:
            _dbg(None, fn.__name__, 'depois')
        return out
    return _wrap

@_io
def _patch_embed3d(video, w, b):
    import torch
    from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor

    device = video.device
    dtype = video.dtype
    W = w
    if hasattr(W, 'gguf_cls'):
        W = dequantize_tensor(W)
    W = W.to(device=device, dtype=dtype)
    bias = None
    if b is not None:
        bias = b.to(device=device, dtype=dtype)
    B, C, T, H, Wd = video.shape
    kCout, kCin, kT, kH, kW = W.shape
    if C != kCin:
        raise RuntimeError(f"patch_embed: C_in mismatch: video C={C} vs weight {kCin}")
    y = torch.nn.functional.conv3d(video, W, bias=bias, stride=(1, kH, kW), padding=(0, 0, 0))
    B2, Cout, T2, H2, W2 = y.shape
    tokens = y.permute(0, 2, 3, 4, 1).contiguous().view(B2, T2 * H2 * W2, Cout)
    # One-time shape log for debugging
    global _LOG_ONCE
    if not _LOG_ONCE.get('patch_embed', False):
        _LOG_ONCE['patch_embed'] = True
        try:
            from .nn import wan22 as _self  # self-module for _li
        except Exception:
            _self = None
        try:
            (_self or globals()).get('_li', lambda *a, **k: None)(None, "[wan22.gguf] patch_embed3d: video=%s W=%s tokens=%s grid=(%d,%d,%d)", tuple(video.shape), tuple(W.shape), tuple(tokens.shape), T2, H2, W2)
        except Exception:
            pass
    return tokens, (T2, H2, W2)


@_io
def _patch_unembed3d(tokens, w, out_shape):
    import torch
    from apps.server.backend.runtime.ops.operations_gguf import dequantize_tensor

    device = tokens.device
    dtype = tokens.dtype
    W = w
    if hasattr(W, 'gguf_cls'):
        W = dequantize_tensor(W)
    W = W.to(device=device, dtype=dtype)
    B, L, Cout = tokens.shape
    kCout, kCin, kT, kH, kW = W.shape
    if Cout != kCout:
        raise RuntimeError(f"patch_unembed: C_out mismatch: tokens C={Cout} vs weight {kCout}")
    T2, H2, W2 = out_shape
    y = tokens.view(B, T2, H2, W2, Cout).permute(0, 4, 1, 2, 3).contiguous().to(device=device, dtype=dtype)
    video = torch.nn.functional.conv_transpose3d(y, W, bias=None, stride=(1, kH, kW), padding=(0, 0, 0))
    global _LOG_ONCE
    if not _LOG_ONCE.get('patch_unembed', False):
        _LOG_ONCE['patch_unembed'] = True
        try:
            (_self or globals()).get('_li', lambda *a, **k: None)(None, "[wan22.gguf] patch_unembed3d: tokens=%s W=%s out=%s grid=%s", tuple(tokens.shape), tuple(W.shape), tuple(video.shape), out_shape)
        except Exception:
            pass
    return video


@_io
def _get_text_context(
    model_dir: str,
    prompt: str,
    negative: Optional[str],
    *,
    device: str,
    dtype: str,
    text_encoder_dir: Optional[str] = None,
    tokenizer_dir: Optional[str] = None,
    vae_dir: Optional[str] = None,
    model_key: Optional[str] = None,
    metadata_dir: Optional[str] = None,
    offload_after: bool = True,
    te_device: Optional[str] = None,
):
    """GGUF path: use Transformers tokenizer + encoder only; do NOT fall back to Diffusers.

    - Searches explicit extras first (tokenizer_dir/text_encoder_dir), then common subfolders under model_dir.
    - Never downloads; never calls Diffusers. If not found, raises an explicit, actionable error.
    """
    import torch
    from transformers import AutoTokenizer, AutoConfig
    try:
        from transformers import UMT5EncoderModel as _Enc
    except Exception:
        from transformers import T5EncoderModel as _Enc

    # Resolve tokenizer dir: prefer explicit tokenizer_dir; else infer from metadata_dir/tokenizer*
    tk_dir = tokenizer_dir
    if (not tk_dir) and metadata_dir:
        cand = os.path.join(metadata_dir, 'tokenizer')
        cand2 = os.path.join(metadata_dir, 'tokenizer_2')
        if os.path.isdir(cand):
            tk_dir = cand
        elif os.path.isdir(cand2):
            tk_dir = cand2
    te_dir = text_encoder_dir
    te_file: Optional[str] = None
    if te_dir and os.path.isfile(te_dir) and te_dir.lower().endswith('.safetensors'):
        te_file = te_dir
        te_dir = os.path.dirname(te_dir)
    if tk_dir and os.path.isfile(tk_dir):
        tk_dir = os.path.dirname(tk_dir)

    # Strict: require tokenizer source
    if not tk_dir or not os.path.isdir(tk_dir):
        raise RuntimeError("WAN22 GGUF: tokenizer metadata missing or invalid; provide 'wan_metadata_dir' or 'wan_tokenizer_dir'.")

    # Load tokenizer from the single provided directory
    try:
        tok = AutoTokenizer.from_pretrained(tk_dir, use_fast=True, local_files_only=True)
    except Exception as ex:
        raise RuntimeError(f"WAN22 GGUF: failed to load tokenizer from '{tk_dir}': {ex}") from ex

    # Strict: require text encoder weights (file) OR a directory with config; when a file is provided,
    # the config is resolved from metadata_dir/text_encoder (vendored repo), never from the weights folder.
    if te_file is not None:
        if not metadata_dir or not os.path.isdir(metadata_dir):
            raise RuntimeError("WAN22 GGUF: 'wan_metadata_dir' is required when providing 'wan_text_encoder_path'.")
        enc_dir = os.path.join(metadata_dir, 'text_encoder')
        if not os.path.isdir(enc_dir):
            raise RuntimeError(
                f"WAN22 GGUF: expected text encoder config under metadata repo: '{enc_dir}'"
            )
        try:
            cfg = AutoConfig.from_pretrained(enc_dir, local_files_only=True)
        except Exception as ex:
            raise RuntimeError(
                f"WAN22 GGUF: failed to read text encoder config from '{enc_dir}': {ex}"
            ) from ex
        enc = _Enc(cfg)
        from safetensors.torch import load_file as _load_st
        try:
            sd = _load_st(te_file)
            enc.load_state_dict(sd, strict=False)
        except Exception as ex:
            raise RuntimeError(f"WAN22 GGUF: failed to load text encoder weights '{te_file}': {ex}") from ex
    else:
        # Strict mode: require a TE weights file; directory-based TE loading is not supported in WAN22 GGUF.
        raise RuntimeError(
            "WAN22 GGUF: 'wan_text_encoder_path' (.safetensors file) is required. Directory-based text encoders are not supported."
        )

    # Device for TE: explicit te_device overrides cfg.device; no silent fallback
    use_dev_name = (te_device or device or 'cpu').lower().strip()
    dev = torch.device('cuda' if use_dev_name == 'cuda' and torch.cuda.is_available() else 'cpu')
    enc = enc.to(dev)
    try:
        enc = enc.to(dtype=_as_dtype(dtype))
    except Exception:
        pass

    def _do(txt: str):
        inputs = tok([txt], padding='max_length', truncation=True, max_length=225, return_tensors='pt')
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        with torch.no_grad():
            out = enc(**inputs).last_hidden_state
            return out.to(_as_dtype(dtype))

    p = _do(prompt or '')
    n = _do(negative or '') if negative is not None else _do('')
    # Aggressive offload: drop TE from VRAM immediately after use
    if offload_after:
        try:
            enc.to('cpu')
        except Exception:
            pass
        del enc
        _cuda_empty_cache(logger=None, label='after-te')
    return p, n


@_io
def _load_vae(vae_path: Optional[str], *, torch_dtype):
    import os
    from diffusers import AutoencoderKLWan  # type: ignore
    if not vae_path:
        raise RuntimeError('wan_vae_dir is required when running WAN GGUF (VAE path missing)')
    path = os.path.expanduser(str(vae_path))
    if os.path.isdir(path):
        return AutoencoderKLWan.from_pretrained(path, torch_dtype=torch_dtype, local_files_only=True)
    if os.path.isfile(path):
        loader = getattr(AutoencoderKLWan, 'from_single_file', None)
        if loader is None:
            raise RuntimeError(f'AutoencoderKLWan.from_single_file not available; provide a directory instead of file: {path}')
        return loader(path, torch_dtype=torch_dtype)
    raise RuntimeError(f'VAE path not found: {path}')


@_io
def _get_scale_shift(vae) -> tuple[float, float]:
    try:
        cfg = getattr(vae, 'config', None) or {}
        sf = float(getattr(cfg, 'scaling_factor', getattr(cfg, 'scaling_factor', 0.18215)))
        sh = float(getattr(cfg, 'shift_factor', getattr(cfg, 'shift_factor', 0.0)))
        if isinstance(cfg, dict):
            sf = float(cfg.get('scaling_factor', sf))
            sh = float(cfg.get('shift_factor', sh))
        return sf, sh
    except Exception:
        return 0.18215, 0.0


@_io
def _vae_encode_init(init_image: Any, *, device: str, dtype: str, vae_dir: str | None = None, logger=None, offload_after: bool = True):
    import torch
    torch_dtype = _as_dtype(dtype)
    vae = _load_vae(vae_dir, torch_dtype=torch_dtype)
    sf, sh = _get_scale_shift(vae)
    if logger:
        try:
            logger.info('[wan22.gguf] VAE encode scale=%.6f shift=%.6f', sf, sh)
        except Exception:
            pass
    target = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
    from apps.server.backend.runtime.memory import memory_management as _mm
    _old = getattr(_mm, 'VAE_ALWAYS_TILED', False)
    try:
        _mm.VAE_ALWAYS_TILED = True
        vae = vae.to(device=target, dtype=torch_dtype)
    finally:
        _mm.VAE_ALWAYS_TILED = _old
    if not hasattr(init_image, 'to'):
        from PIL import Image
        import numpy as np
        if isinstance(init_image, Image.Image):
            img = init_image.convert('RGB')
            arr = np.array(img).astype('float32') / 255.0
            t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            t = t.to(target).to(torch_dtype)
            init_image = t * 2.0 - 1.0
        else:
            arr = np.asarray(init_image).astype('float32')
            if arr.ndim == 3 and arr.shape[2] in (1, 3):
                arr = arr / 255.0 if arr.max() > 1.0 else arr
                t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            elif arr.ndim == 3 and arr.shape[0] in (1, 3):
                t = torch.from_numpy(arr).unsqueeze(0)
            else:
                raise RuntimeError('unsupported init_image array shape')
            t = t.to(target).to(torch_dtype)
            init_image = t * 2.0 - 1.0
    # VAE expects video tensor [B,C,T,H,W]; expand a single frame to T=1
    if hasattr(init_image, 'ndim'):
        if init_image.ndim == 4:
            init_image = init_image.unsqueeze(2)
        elif init_image.ndim != 5:
            raise RuntimeError('init_image must be 4D (B,C,H,W) or 5D (B,C,T,H,W) after preprocessing')
    with torch.no_grad():
        latents = vae.encode(init_image).latent_dist.sample()
        latents = (latents - sh) * sf
    if offload_after:
        try:
            vae.to('cpu')
        except Exception:
            pass
        del vae
        _cuda_empty_cache(logger, label='after-vae-encode')
    return latents


@_io
def _vae_decode_video(video_latents: Any, *, model_dir: str, device: str, dtype: str, vae_dir: str | None = None, logger=None, offload_after: bool = True):
    import torch
    from PIL import Image
    torch_dtype = _as_dtype(dtype)
    dev = torch.device('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    vae = _load_vae(vae_dir, torch_dtype=torch_dtype)
    sf, sh = _get_scale_shift(vae)
    if logger:
        try:
            logger.info('[wan22.gguf] VAE decode scale=%.6f shift=%.6f', sf, sh)
        except Exception:
            pass
    from apps.server.backend.runtime.memory import memory_management as _mm
    _old = getattr(_mm, 'VAE_ALWAYS_TILED', False)
    try:
        _mm.VAE_ALWAYS_TILED = True
        vae = vae.to(device=dev, dtype=torch_dtype)
    finally:
        _mm.VAE_ALWAYS_TILED = _old
    B, C, T, H, W = video_latents.shape
    frames: list[Image.Image] = []
    with torch.no_grad():
        for t in range(T):
            lat = video_latents[:, :, t]
            lat = (lat / sf) + sh
            img = vae.decode(lat).sample
            img0 = img[0].detach().clamp(0, 1)
            arr = (img0.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            frames.append(Image.fromarray(arr))
    if offload_after:
        try:
            vae.to('cpu')
        except Exception:
            pass
        del vae
        _cuda_empty_cache(logger, label='after-vae-decode')
    return frames

try:  # progress bar for long loops (non-fatal if unavailable)
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


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


@_io
def _shape_of(state: Mapping[str, object], key: str) -> Optional[Tuple[int, ...]]:
    v = state.get(key)
    if v is None:
        return None
    try:
        shp = tuple(int(s) for s in getattr(v, 'shape', tuple()))
        return shp if shp else None
    except Exception:
        return None


@_io
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

@_io
def _rms_norm(x: torch.Tensor, w: Any) -> torch.Tensor:
    w = dequantize_tensor(w)
    if not torch.is_tensor(w):
        w = torch.as_tensor(w, device=x.device, dtype=x.dtype)
    eps = 1e-6
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)) * w


def _try_set_cache_policy(policy: Optional[str], limit_mb: Optional[int]) -> None:
    pol = (policy or 'none').strip().lower()
    lim = int(limit_mb or 0)
    if pol in ('none', '', 'off') or lim <= 0:
        return
    try:
        from apps.server.backend.runtime.ops.operations_gguf import set_cache_policy as _scp
    except Exception as ex:  # pragma: no cover
        raise RuntimeError("GGUF dequant cache requested but not available in this build (set_cache_policy missing). Update backend.") from ex
    _scp(pol, lim)


def _try_clear_cache() -> None:
    try:
        from apps.server.backend.runtime.ops.operations_gguf import clear_cache as _cc
        _cc()
    except Exception:
        pass


@_io
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


try:
    from contextlib import nullcontext
except Exception:  # pragma: no cover
    class nullcontext:  # type: ignore
        def __init__(self, *a, **k):
            ...
        def __enter__(self):
            return None
        def __exit__(self, *a):
            return False

_SDPA_SETTINGS = {
    'policy': 'mem_efficient',
    'chunk': 0,
}


def _set_sdpa_settings(policy: Optional[str], chunk: Optional[int]) -> None:
    pol = (policy or _SDPA_SETTINGS['policy']).strip().lower()
    if pol not in ('mem_efficient', 'flash', 'math'):
        pol = _SDPA_SETTINGS['policy']
    ch = int(chunk) if (chunk is not None and int(chunk) > 0) else 0
    _SDPA_SETTINGS['policy'] = pol
    _SDPA_SETTINGS['chunk'] = ch


def _sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False) -> torch.Tensor:
    pol = _SDPA_SETTINGS['policy']
    ch = _SDPA_SETTINGS['chunk']
    ctx = torch.backends.cuda.sdp_kernel(
        enable_flash=(pol == 'flash'),
        enable_math=(pol == 'math'),
        enable_mem_efficient=(pol == 'mem_efficient'),
    ) if (q.is_cuda and hasattr(torch.backends, 'cuda')) else nullcontext()

    if ch and ch > 0:
        with ctx:
            B, H, L, D = q.shape
            out_chunks = []
            for s in range(0, L, ch):
                e = min(L, s + ch)
                out_chunks.append(torch.nn.functional.scaled_dot_product_attention(q[:, :, s:e], k, v, is_causal=causal))
            return torch.cat(out_chunks, dim=2)
    else:
        with ctx:
            return torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=causal)


@_io
def _split_heads(x: torch.Tensor, h: int) -> torch.Tensor:
    B, L, C = x.shape
    D = C // h
    return x.view(B, L, h, D).permute(0, 2, 1, 3).contiguous()


@_io
def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    B, H, L, D = x.shape
    return x.permute(0, 2, 1, 3).contiguous().view(B, L, H * D)


@_io
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


@_io
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


class WanDiTGGUF:
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
    seed: Optional[int] = None
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    init_image: Optional[object] = None
    vae_dir: Optional[str] = None
    text_encoder_dir: Optional[str] = None
    tokenizer_dir: Optional[str] = None
    metadata_dir: Optional[str] = None
    high: Optional[StageConfig] = None
    low: Optional[StageConfig] = None
    # Memory/attention controls (optional)
    sdpa_policy: Optional[str] = None            # 'mem_efficient' | 'flash' | 'math'
    attn_chunk_size: Optional[int] = None        # split attention along sequence if set (>0)
    gguf_cache_policy: Optional[str] = None      # 'none' | 'cpu_lru'
    gguf_cache_limit_mb: Optional[int] = None    # MB limit for cpu_lru cache
    log_mem_interval: Optional[int] = None       # log CUDA mem every N steps if >0
    # Aggressive offload controls
    aggressive_offload: bool = True              # move modules off GPU immediately after use
    te_device: Optional[str] = None              # 'cuda' | 'cpu' (None = follow cfg.device)

def _as_dtype(dtype: str):
    return {
        'fp16': torch.float16,
        'bf16': getattr(torch, 'bfloat16', torch.float16),
        'fp32': torch.float32,
    }.get(str(dtype).lower(), torch.float16)


def _get_logger(logger: Any):
    import logging
    if logger is not None:
        return logger
    lg = logging.getLogger("wan22.gguf")
    if not lg.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter('[wan22.gguf] %(levelname)s: %(message)s')
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg

def _cuda_empty_cache(logger=None, label: str = "gc") -> None:
    try:
        import torch
        if not (getattr(torch, 'cuda', None) and torch.cuda.is_available()):
            return
        torch.cuda.synchronize()
        alloc_before = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved_before = torch.cuda.memory_reserved() // (1024 * 1024)
        torch.cuda.empty_cache()
        alloc_after = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved_after = torch.cuda.memory_reserved() // (1024 * 1024)
        _li(logger, "[wan22.gguf] cuda.gc(%s): alloc %d→%d MB reserved %d→%d MB", label, alloc_before, alloc_after, reserved_before, reserved_after)
    except Exception:
        pass


def _resolve_patch_weights(state: Mapping[str, Any]) -> Tuple[Any, Any]:
    w = state.get('patch_embedding.weight')
    b = state.get('patch_embedding.bias')
    if w is None:
        raise RuntimeError("GGUF missing 'patch_embedding.weight'")
    return w, b


def _infer_latent_grid(dit: 'WanDiTGGUF', *, T: int, H_lat: int, W_lat: int, device: torch.device, dtype: torch.dtype) -> Tuple[Tuple[int, int, int], Tuple[int, int]]:
    w, b = _resolve_patch_weights(dit.state)
    # Probe with zeros to avoid guessing shapes; cheap and reliable
    Cin = int(getattr(w, 'shape', [None, None, None, None, None])[1])
    if Cin is None:
        raise RuntimeError("failed to read patch_embedding weight shape")
    vid = torch.zeros(1, Cin, T, H_lat, W_lat, device=device, dtype=dtype)
    tokens, grid = _patch_embed3d(vid, w, b)
    L, Cout = int(tokens.shape[1]), int(tokens.shape[2])
    return (grid[0], grid[1], grid[2]), (L, Cout)


@_io
def _make_scheduler(steps: int, *, sampler: Optional[str] = None, scheduler: Optional[str] = None):
    """Instantiate a Diffusers scheduler based on requested sampler/scheduler names.

    Defaults to Euler when unspecified. No hardcoded counts; `steps` is passed through.
    """
    from diffusers import (
        EulerDiscreteScheduler,
        EulerAncestralDiscreteScheduler,
        DDIMScheduler,
        DPMSolverMultistepScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
    )

    s = (sampler or "").strip().lower()
    sch = (scheduler or "").strip().lower()

    cls = EulerDiscreteScheduler  # default
    # Try sampler first (explicit user intent)
    if s in ("euler",):
        cls = EulerDiscreteScheduler
    elif s in ("euler a", "euler_a", "euler-ancestral", "ancestral"):  # tolerant forms
        cls = EulerAncestralDiscreteScheduler
    elif s in ("ddim",):
        cls = DDIMScheduler
    elif s in ("dpm++ 2m", "dpm++ 2m sde", "dpm2m", "dpmpp2m", "dpmpp2m sde"):
        cls = DPMSolverMultistepScheduler
    elif s in ("plms", "lms"):
        cls = LMSDiscreteScheduler
    elif s in ("pndm",):
        cls = PNDMScheduler
    else:
        # Fall back to scheduler hint, if provided
        if "euler a" in sch or "ancestral" in sch:
            cls = EulerAncestralDiscreteScheduler
        elif "euler" in sch:
            cls = EulerDiscreteScheduler
        elif "ddim" in sch:
            cls = DDIMScheduler
        elif "dpm" in sch:
            cls = DPMSolverMultistepScheduler
        elif "lms" in sch:
            cls = LMSDiscreteScheduler
        elif "pndm" in sch:
            cls = PNDMScheduler

    sched = cls()
    sched.set_timesteps(max(1, int(steps)))
    return sched


@_io
def _cfg_merge(uncond: torch.Tensor, cond: torch.Tensor, scale: float | None) -> torch.Tensor:
    if scale is None:
        return cond
    return uncond + (cond - uncond) * float(scale)


@_io
def _log_cuda_mem(logger: Any, label: str = "mem") -> None:
    try:
        import torch
        if torch.cuda.is_available():
            alloc = float(torch.cuda.memory_allocated()) / (1024**2)
            reserv = float(torch.cuda.memory_reserved()) / (1024**2)
            total = float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
            logger.info("[wan22.gguf] %s: cuda mem alloc=%.0fMB reserved=%.0fMB total=%.0fMB", label, alloc, reserv, total)
    except Exception:
        pass


@_io
def _log_t_mapping(scheduler, timesteps, label: str, logger: Any) -> None:
    try:
        log = _get_logger(logger)
        n = len(timesteps)
        idxs = [0, max(0, n // 2 - 1), n - 1]
        vals: list[float] = []
        sigmas = getattr(scheduler, 'sigmas', None)
        for i in idxs:
            if sigmas is not None and len(sigmas) == n:
                s = float(sigmas[i]); s_min = float(sigmas[-1]); s_max = float(sigmas[0])
                t = max(0.0, min(1.0, (s - s_min) / (s_max - s_min))) if (s_max - s_min) > 0 else 0.0
            else:
                t = 1.0 - (float(i) / float(max(1, n - 1)))
            vals.append(t)
        log.info("[wan22.gguf] t-map(%s): t0=%.4f tmid=%.4f tend=%.4f (sigmas=%s)", label, vals[0], vals[1], vals[2], bool(sigmas is not None and len(sigmas)==n))
    except Exception:
        pass


def _time_snr_shift(alpha: float, t: float) -> float:
    # Same form as ComfyUI comfy.model_sampling.time_snr_shift
    if alpha == 1.0:
        return t
    return alpha * t / (1 + (alpha - 1) * t)


def _sample_stage_tokens(
    *,
    dit: 'WanDiTGGUF',
    steps: int,
    cfg_scale: Optional[float],
    prompt_embeds: torch.Tensor,
    negative_embeds: torch.Tensor,
    grid: Tuple[int, int, int],
    token_shape: Tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    logger: Any,
    sampler_name: Optional[str] = None,
    scheduler_name: Optional[str] = None,
    seed: Optional[int] = None,
    on_progress: Optional[Any] = None,
    log_mem_interval: Optional[int] = None,
) -> torch.Tensor:
    log = _get_logger(logger)
    T2, H2, W2 = grid
    L, Ctok = token_shape
    # Initialize token space with Gaussian noise (seeded when provided)
    if seed is not None and int(seed) >= 0:
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        x = torch.randn(1, L, Ctok, device=device, dtype=dtype, generator=g)
    else:
        x = torch.randn(1, L, Ctok, device=device, dtype=dtype)
    # Scheduler em espaço de tokens (parametrizável)
    scheduler = _make_scheduler(steps, sampler=sampler_name, scheduler=scheduler_name)
    # Diffusers timesteps; map to DiT time in [0,1]
    timesteps = scheduler.timesteps

    def _t_from_idx(idx: int) -> float:
        # ComfyUI FLOW mapping: percent -> sigma via time_snr_shift -> timestep = sigma * 1000
        n = max(1, len(timesteps) - 1)
        percent = float(idx) / float(n)
        sigma = _time_snr_shift(1.0, 1.0 - percent)
        return sigma * 1000.0

    _log_t_mapping(scheduler, timesteps, 'stage', logger)
    iterator = _tqdm(timesteps, desc="WAN22(stage)") if _tqdm else timesteps
    total = len(timesteps)
    if on_progress:
        try:
            on_progress(step=0, total=total, percent=0.0)
        except Exception:
            pass
    import time
    t0 = time.perf_counter()
    last = t0
    for i, t in enumerate(iterator):
        # Model forward: conditional and unconditional
        tt = _t_from_idx(i)
        eps_cond = dit.forward(x, tt, prompt_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dtype])
        eps_uncond = dit.forward(x, tt, negative_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dtype])
        eps = _cfg_merge(eps_uncond, eps_cond, cfg_scale)
        # Euler step
        out = scheduler.step(model_output=eps, timestep=t, sample=x)
        x = out.prev_sample
        pct = float(i + 1) / float(max(1, total))
        # Optional CUDA memory snapshot every N steps
        if log_mem_interval is not None:
            try:
                n = int(log_mem_interval or 0)
                if n > 0 and ((i + 1) % n) == 0:
                    _log_cuda_mem(logger, label=f'stage-step-{i+1}')
            except Exception:
                pass
        if on_progress:
            try:
                now = time.perf_counter()
                dt_step = now - last
                elapsed = now - t0
                remain = max(0, total - (i + 1))
                eta = (elapsed / max(1, i + 1)) * remain
                on_progress(step=i + 1, total=total, percent=pct, eta_seconds=eta, step_seconds=dt_step)
                last = now
            except Exception:
                pass
        elif (i + 1) % 5 == 0:
            log.info("[wan22.gguf] step %d/%d (%.1f%%)", i + 1, total, pct * 100.0)
    return x


@_io
def _decode_tokens_to_frames(
    *,
    tokens: torch.Tensor,
    dit: 'WanDiTGGUF',
    grid: Tuple[int, int, int],
    device: torch.device,
    dtype: torch.dtype,
    model_dir: str,
    cfg: RunConfig,
) -> List[object]:
    # Unembed tokens back to video latents and decode via VAE
    w, _b = _resolve_patch_weights(dit.state)
    video_latents = _patch_unembed3d(tokens, w, grid)  # [B,C,T,H,W]
    frames = _vae_decode_video(video_latents, model_dir=model_dir, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir)
    return frames


@_io
def _resolve_device_name(name: str) -> str:
    s = (name or 'auto').lower().strip()
    if s == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if s == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'


@_io
def run_txt2vid(cfg: RunConfig, *, logger=None, on_progress=None) -> List[object]:
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)
    # Configure attention and GGUF cache according to cfg (defaults emphasize memory efficiency)
    _set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))
    # Early activity signal so the UI shows immediate progress
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.0)
        except Exception:
            pass

    # Device & dtype
    dev_name = _resolve_device_name(getattr(cfg, 'device', 'auto'))
    dev = torch.device(dev_name)
    dt = _as_dtype(cfg.dtype)

    # Prepare UNet (High stage) and text context
    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.05)
        except Exception:
            pass
    # Determine model key for tokenizer/encoder presets
    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_t2v_{_variant}"
    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
        offload_after=bool(getattr(cfg, 'aggressive_offload', True)),
        te_device=getattr(cfg, 'te_device', None),
    )
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.1)
        except Exception:
            pass
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    # Infer latent grid and token shape from patch embedding
    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))
    grid, token_shape = _infer_latent_grid(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    log.info(
        "[wan22.gguf] device=%s dtype=%s grid(T,H',W')=%s token(L,C)=%s",
        str(dev), str(dt), grid, (token_shape[0], token_shape[1])
    )
    _log_cuda_mem(log, label='after-high-setup')
    if getattr(cfg, 'aggressive_offload', True):
        _cuda_empty_cache(log, label='pre-high')
    if on_progress:
        try:
            on_progress(stage='prepare', step=1, total=1, percent=0.15)
        except Exception:
            pass

    # Sample tokens in High stage
    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    log.info(
        "[wan22.gguf] HIGH: steps=%s sampler=%s scheduler=%s cfg_scale=%s seed=%s",
        steps_hi, sampler_hi, sched_hi, (getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale), cfg.seed,
    )
    total_hi = int(len(getattr(_make_scheduler(steps_hi), 'timesteps', list(range(steps_hi)))))
    if on_progress:
        try:
            on_progress(stage='high', step=0, total=total_hi, percent=0.0)
        except Exception:
            pass
    toks_hi = _sample_stage_tokens(
        dit=hi_dit,
        steps=steps_hi,
        cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        grid=grid,
        token_shape=token_shape,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=sampler_hi,
        scheduler_name=sched_hi,
        seed=cfg.seed,
        on_progress=(lambda **p: on_progress(stage='high', **p)) if on_progress else None,
        log_mem_interval=getattr(cfg, 'log_mem_interval', None),
    )
    if on_progress:
        try:
            on_progress(stage='high', step=total_hi, total=total_hi, percent=1.0)
        except Exception:
            pass

    # Decode High to frames and seed Low with last frame (Low is mandatory)
    frames_hi = _decode_tokens_to_frames(tokens=toks_hi, dit=hi_dit, grid=grid, device=dev, dtype=dt, model_dir=os.path.dirname(hi_path), cfg=cfg)
    if getattr(cfg, 'aggressive_offload', True):
        _cuda_empty_cache(log, label='after-high')
    if not frames_hi:
        raise RuntimeError("WAN22 GGUF: High stage produced no frames")

    seed_image = frames_hi[-1]
    # Prepare Low UNet
    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    # Encode seed to latents → tokens for Low
    lat_lo0 = _vae_encode_init(seed_image, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat_lo0.ndim == 4:
        lat_lo0 = lat_lo0.unsqueeze(2).repeat(1, 1, T, 1, 1)
    w_lo, b_lo = _resolve_patch_weights(lo_dit.state)
    toks_lo0, grid_lo = _patch_embed3d(lat_lo0.to(device=dev, dtype=dt), w_lo, b_lo)

    # Low stage sampling (mandatory)
    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    log.info(
        "[wan22.gguf] LOW: steps=%s sampler=%s scheduler=%s cfg_scale=%s (seeded from last HIGH frame)",
        steps_lo, sampler_lo, sched_lo, (getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
    )
    scheduler_lo = _make_scheduler(steps_lo, sampler=sampler_lo, scheduler=sched_lo)
    x_lo = toks_lo0.clone()
    _log_t_mapping(scheduler_lo, scheduler_lo.timesteps, 'low', log)
    iterator_lo = _tqdm(scheduler_lo.timesteps, desc="WAN22(low)") if _tqdm else scheduler_lo.timesteps
    total_lo = len(scheduler_lo.timesteps)
    if on_progress:
        try:
            on_progress(stage='low', step=0, total=total_lo, percent=0.0)
        except Exception:
            pass
    _log_cuda_mem(log, label='before-low')
    def _t_from_idx_lo(idx: int) -> float:
        try:
            sigmas = getattr(scheduler_lo, 'sigmas', None)
            if sigmas is not None and len(sigmas) == len(scheduler_lo.timesteps):
                s = float(sigmas[idx])
                s_min = float(sigmas[-1])
                s_max = float(sigmas[0])
                if s_max - s_min > 0:
                    return max(0.0, min(1.0, (s - s_min) / (s_max - s_min)))
        except Exception:
            pass
        n = max(1, len(scheduler_lo.timesteps) - 1)
        return 1.0 - (float(idx) / float(n))

    for i, t in enumerate(iterator_lo):
        tt = _t_from_idx_lo(i)
        eps_c = lo_dit.forward(x_lo, tt, prompt_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps_u = lo_dit.forward(x_lo, tt, negative_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps = _cfg_merge(eps_u, eps_c, getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale)
        out = scheduler_lo.step(model_output=eps, timestep=t, sample=x_lo)
        x_lo = out.prev_sample
        pct = float(i + 1) / float(max(1, total_lo))
        if on_progress:
            try:
                on_progress(stage='low', step=i + 1, total=total_lo, percent=pct)
            except Exception:
                pass
        elif ((i + 1) % 5) == 0:
            log.info("[wan22.gguf] low step %d/%d (%.1f%%)", i + 1, total_lo, pct * 100.0)

    frames_lo = _vae_decode_video(_patch_unembed3d(x_lo, w_lo, grid_lo), model_dir=os.path.dirname(lo_path), device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if getattr(cfg, 'aggressive_offload', True):
        _cuda_empty_cache(log, label='after-decode')
    _try_clear_cache()
    if not frames_lo:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    return frames_lo


def stream_txt2vid(cfg: RunConfig, *, logger=None):
    """Generator that yields progress and final frames for txt2vid."""
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (txt2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)

    dev = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dt = _as_dtype(cfg.dtype)

    # Prepare UNet (High) and text embeddings
    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_t2v_{_variant}"
    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=cfg.device,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
    )
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))
    grid, token_shape = _infer_latent_grid(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    total_hi = int(len(getattr(_make_scheduler(int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)), 'timesteps', list(range(int(getattr(cfg.high, 'steps', 12) if cfg.high else 12))))))
    yield {"type": "progress", "stage": "high", "step": 0, "total": total_hi, "percent": 0.0}

    toks_hi = _sample_stage_tokens(
        dit=hi_dit,
        steps=int(getattr(cfg.high, 'steps', 12) if cfg.high else 12),
        cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds,
        negative_embeds=negative_embeds,
        grid=grid,
        token_shape=token_shape,
        device=dev,
        dtype=dt,
        logger=log,
        sampler_name=(getattr(cfg.high, 'sampler', None) if cfg.high else None),
        scheduler_name=(getattr(cfg.high, 'scheduler', None) if cfg.high else None),
        seed=cfg.seed,
    )  # type: ignore[arg-type]

    # Cannot pass a lambda with yield directly; emulate by re-running progress here
    yield {"type": "progress", "stage": "high", "step": total_hi, "total": total_hi, "percent": 1.0}

    frames_hi = _decode_tokens_to_frames(tokens=toks_hi, dit=hi_dit, grid=grid, device=dev, dtype=dt, model_dir=os.path.dirname(hi_path), cfg=cfg)
    if not frames_hi:
        raise RuntimeError("WAN22 GGUF: High stage produced no frames")

    # Low stage
    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    lat_lo0 = _vae_encode_init(frames_hi[-1], device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat_lo0.ndim == 4:
        lat_lo0 = lat_lo0.unsqueeze(2).repeat(1, 1, T, 1, 1)
    w_lo, b_lo = _resolve_patch_weights(lo_dit.state)
    toks_lo0, grid_lo = _patch_embed3d(lat_lo0.to(device=dev, dtype=dt), w_lo, b_lo)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    scheduler_lo = _make_scheduler(steps_lo, sampler=sampler_lo, scheduler=sched_lo)
    x_lo = toks_lo0.clone()
    total_lo = len(scheduler_lo.timesteps)
    yield {"type": "progress", "stage": "low", "step": 0, "total": total_lo, "percent": 0.0}
    for i, t in enumerate(scheduler_lo.timesteps):
        tt = (1.0 - (float(i) / float(max(1, total_lo - 1)))) * 1000.0
        eps_c = lo_dit.forward(x_lo, tt, prompt_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps_u = lo_dit.forward(x_lo, tt, negative_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps = _cfg_merge(eps_u, eps_c, getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale)
        out = scheduler_lo.step(model_output=eps, timestep=t, sample=x_lo)
        x_lo = out.prev_sample
        pct = float(i + 1) / float(max(1, total_lo))
        yield {"type": "progress", "stage": "low", "step": i + 1, "total": total_lo, "percent": pct}

    frames_lo = _vae_decode_video(_patch_unembed3d(x_lo, w_lo, grid_lo), model_dir=os.path.dirname(lo_path), device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if not frames_lo:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames_lo}


def stream_img2vid(cfg: RunConfig, *, logger=None):
    # Reuse txt2vid streaming after forcing seed image path
    log = _get_logger(logger)
    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")
    # The rest mirrors stream_txt2vid, but seeds low stage from cfg.init_image
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")
    dev = torch.device('cuda' if cfg.device == 'cuda' and torch.cuda.is_available() else 'cpu')
    dt = _as_dtype(cfg.dtype)
    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
    _p = os.path.basename(hi_path).lower(); _variant = '5b' if '5b' in _p else '14b'; _model_key = f"wan_t2v_{_variant}"
    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path), prompt=cfg.prompt or "", negative=cfg.negative_prompt,
        device=cfg.device, dtype=cfg.dtype, text_encoder_dir=cfg.text_encoder_dir, tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir, model_key=_model_key, metadata_dir=cfg.metadata_dir,
    )
    if isinstance(prompt_embeds, torch.Tensor): prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor): negative_embeds = negative_embeds.to(device=dev, dtype=dt)
    H_lat = max(8, cfg.height // 8); W_lat = max(8, cfg.width // 8); T = max(1, int(cfg.num_frames))
    grid, token_shape = _infer_latent_grid(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    scheduler_hi = _make_scheduler(steps_hi, sampler=(getattr(cfg.high, 'sampler', None) if cfg.high else None), scheduler=(getattr(cfg.high, 'scheduler', None) if cfg.high else None))
    total_hi = len(scheduler_hi.timesteps)
    yield {"type": "progress", "stage": "high", "step": 0, "total": total_hi, "percent": 0.0}
    # Start from encoded init image latents for first step to warm VAE and token grids consistent
    toks_hi = _sample_stage_tokens(
        dit=hi_dit, steps=steps_hi, cfg_scale=(getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
        prompt_embeds=prompt_embeds, negative_embeds=negative_embeds, grid=grid, token_shape=token_shape,
        device=dev, dtype=dt, logger=log, sampler_name=(getattr(cfg.high, 'sampler', None) if cfg.high else None),
        scheduler_name=(getattr(cfg.high, 'scheduler', None) if cfg.high else None), seed=cfg.seed,
        on_progress=lambda step,total,percent: None,
    )
    yield {"type": "progress", "stage": "high", "step": total_hi, "total": total_hi, "percent": 1.0}
    # Seed low from cfg.init_image instead of last HIGH frame
    lat_lo0 = _vae_encode_init(cfg.init_image, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if lat_lo0.ndim == 4: lat_lo0 = lat_lo0.unsqueeze(2).repeat(1, 1, T, 1, 1)
    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)
    w_lo, b_lo = _resolve_patch_weights(lo_dit.state); toks_lo0, grid_lo = _patch_embed3d(lat_lo0.to(device=dev, dtype=dt), w_lo, b_lo)
    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    scheduler_lo = _make_scheduler(steps_lo, sampler=(getattr(cfg.low, 'sampler', None) if cfg.low else None), scheduler=(getattr(cfg.low, 'scheduler', None) if cfg.low else None))
    x_lo = toks_lo0.clone(); total_lo = len(scheduler_lo.timesteps)
    yield {"type": "progress", "stage": "low", "step": 0, "total": total_lo, "percent": 0.0}
    for i, t in enumerate(scheduler_lo.timesteps):
        tt = (1.0 - (float(i) / float(max(1, total_lo - 1)))) * 1000.0
        eps_c = lo_dit.forward(x_lo, tt, prompt_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps_u = lo_dit.forward(x_lo, tt, negative_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps = _cfg_merge(eps_u, eps_c, getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale)
        out = scheduler_lo.step(model_output=eps, timestep=t, sample=x_lo)
        x_lo = out.prev_sample
        pct = float(i + 1) / float(max(1, total_lo))
        yield {"type": "progress", "stage": "low", "step": i + 1, "total": total_lo, "percent": pct}
    frames_lo = _vae_decode_video(_patch_unembed3d(x_lo, w_lo, grid_lo), model_dir=os.path.dirname(lo_path), device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    if not frames_lo:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    yield {"type": "result", "frames": frames_lo}


@_io
def run_img2vid(cfg: RunConfig, *, logger=None, on_progress=None) -> List[object]:
    log = _get_logger(logger)
    hi_path = _pick_stage_gguf(getattr(cfg.high, 'model_dir', None) if cfg.high else None, 'high')
    lo_path = _pick_stage_gguf(getattr(cfg.low, 'model_dir', None) if cfg.low else None, 'low')
    if not hi_path or not lo_path:
        raise RuntimeError("WAN22 GGUF (img2vid) requires .gguf for both stages")
    log.info("[wan22.gguf] high=%s low=%s", hi_path, lo_path)
    # Configure attention and GGUF cache
    _set_sdpa_settings(getattr(cfg, 'sdpa_policy', None), getattr(cfg, 'attn_chunk_size', None))
    _try_set_cache_policy(getattr(cfg, 'gguf_cache_policy', None), getattr(cfg, 'gguf_cache_limit_mb', 0))
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.0)
        except Exception:
            pass

    if cfg.init_image is None:
        raise RuntimeError("img2vid requires init_image for GGUF path")

    dev_name = _resolve_device_name(getattr(cfg, 'device', 'auto'))
    dev = torch.device(dev_name)
    dt = _as_dtype(cfg.dtype)

    # Prepare UNet (High)
    hi_dit = WanDiTGGUF(os.path.dirname(hi_path), logger=log)
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.05)
        except Exception:
            pass

    # Text embeds
    _p = os.path.basename(hi_path).lower()
    _variant = '5b' if '5b' in _p else '14b'
    _model_key = f"wan_i2v_{_variant}"
    prompt_embeds, negative_embeds = _get_text_context(
        model_dir=os.path.dirname(hi_path),
        prompt=cfg.prompt or "",
        negative=cfg.negative_prompt,
        device=dev_name,
        dtype=cfg.dtype,
        text_encoder_dir=cfg.text_encoder_dir,
        tokenizer_dir=cfg.tokenizer_dir,
        vae_dir=cfg.vae_dir,
        model_key=_model_key,
        metadata_dir=cfg.metadata_dir,
        offload_after=bool(getattr(cfg, 'aggressive_offload', True)),
        te_device=getattr(cfg, 'te_device', None),
    )
    if on_progress:
        try:
            on_progress(stage='prepare', step=0, total=1, percent=0.1)
        except Exception:
            pass
    if isinstance(prompt_embeds, torch.Tensor):
        prompt_embeds = prompt_embeds.to(device=dev, dtype=dt)
    if isinstance(negative_embeds, torch.Tensor):
        negative_embeds = negative_embeds.to(device=dev, dtype=dt)

    # Encode init image to latents → tokens
    H_lat = max(8, cfg.height // 8)
    W_lat = max(8, cfg.width // 8)
    T = max(1, int(cfg.num_frames))
    grid, token_shape = _infer_latent_grid(hi_dit, T=T, H_lat=H_lat, W_lat=W_lat, device=dev, dtype=dt)
    log.info(
        "[wan22.gguf] device=%s dtype=%s grid(T,H',W')=%s token(L,C)=%s",
        str(dev), str(dt), grid, (token_shape[0], token_shape[1])
    )
    if on_progress:
        try:
            on_progress(stage='prepare', step=1, total=1, percent=0.15)
        except Exception:
            pass
    # For img2vid, we seed token space from VAE latents of the init image (repeated across T)
    lat0 = _vae_encode_init(cfg.init_image, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    # Tile along time to match requested frames
    if lat0.ndim == 4:  # [B,C,H,W]
        lat0 = lat0.unsqueeze(2).repeat(1, 1, T, 1, 1)
    w, b = _resolve_patch_weights(hi_dit.state)
    seed_tokens, _ = _patch_embed3d(lat0.to(device=dev, dtype=dt), w, b)  # [1, L, Ctok]

    # Run sampler starting from seeded tokens (replace init noise)
    steps_hi = int(getattr(cfg.high, 'steps', 12) if cfg.high else 12)
    sampler_hi = getattr(cfg.high, 'sampler', None) if cfg.high else None
    sched_hi = getattr(cfg.high, 'scheduler', None) if cfg.high else None
    log.info(
        "[wan22.gguf] HIGH(img2vid): steps=%s sampler=%s scheduler=%s cfg_scale=%s (seeded from init image)",
        steps_hi, sampler_hi, sched_hi, (getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale),
    )
    scheduler = _make_scheduler(steps_hi, sampler=sampler_hi, scheduler=sched_hi)
    timesteps = scheduler.timesteps
    x = seed_tokens.clone()
    _log_t_mapping(scheduler, timesteps, 'high', log)
    iterator = _tqdm(timesteps, desc="WAN22(high)") if _tqdm else timesteps
    def _t_from_idx_high(idx: int) -> float:
        n = max(1, len(timesteps) - 1)
        percent = float(idx) / float(n)
        sigma = _time_snr_shift(1.0, 1.0 - percent)
        return sigma * 1000.0

    for i, t in enumerate(iterator):
        tt = _t_from_idx_high(i)
        eps_c = hi_dit.forward(x, tt, prompt_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps_u = hi_dit.forward(x, tt, negative_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps = _cfg_merge(eps_u, eps_c, getattr(cfg.high, 'cfg_scale', None) if cfg.high else cfg.guidance_scale)
        out = scheduler.step(model_output=eps, timestep=t, sample=x)
        x = out.prev_sample
        if (i + 1) % 5 == 0:
            log.info("[wan22.gguf] high step %d/%d", i + 1, len(timesteps))

    # Decode High frames and seed Low with last frame latent
    frames_hi = _decode_tokens_to_frames(tokens=x, dit=hi_dit, grid=grid, device=dev, dtype=dt, model_dir=os.path.dirname(hi_path), cfg=cfg)
    seed_image = frames_hi[-1] if frames_hi else None

    # Low stage (mandatory): seed from last High frame
    if seed_image is None:
        raise RuntimeError("WAN22 GGUF: missing seed image for Low stage")
    lo_dit = WanDiTGGUF(os.path.dirname(lo_path), logger=log)

    # Prepare seed tokens for Low from seed_image
    lat_lo0 = _vae_encode_init(seed_image, device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log, offload_after=bool(getattr(cfg, 'aggressive_offload', True)))
    if lat_lo0.ndim == 4:
        lat_lo0 = lat_lo0.unsqueeze(2).repeat(1, 1, T, 1, 1)
    w_lo, b_lo = _resolve_patch_weights(lo_dit.state)
    toks_lo0, grid_lo = _patch_embed3d(lat_lo0.to(device=dev, dtype=dt), w_lo, b_lo)

    steps_lo = int(getattr(cfg.low, 'steps', 12) if cfg.low else 12)
    sampler_lo = getattr(cfg.low, 'sampler', None) if cfg.low else None
    sched_lo = getattr(cfg.low, 'scheduler', None) if cfg.low else None
    log.info(
        "[wan22.gguf] LOW(img2vid): steps=%s sampler=%s scheduler=%s cfg_scale=%s (seeded from last HIGH frame)",
        steps_lo, sampler_lo, sched_lo, (getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale),
    )
    scheduler_lo = _make_scheduler(steps_lo, sampler=sampler_lo, scheduler=sched_lo)
    tlist_lo = scheduler_lo.timesteps
    x_lo = toks_lo0.clone()
    _log_t_mapping(scheduler_lo, tlist_lo, 'low(img2vid)', log)
    iterator_lo = _tqdm(tlist_lo, desc="WAN22(low)") if _tqdm else tlist_lo
    def _t_from_idx_lo(idx: int) -> float:
        n = max(1, len(scheduler_lo.timesteps) - 1)
        percent = float(idx) / float(n)
        sigma = _time_snr_shift(1.0, 1.0 - percent)
        return sigma * 1000.0

    for i, t in enumerate(iterator_lo):
        tt = _t_from_idx_lo(i)
        eps_c = lo_dit.forward(x_lo, tt, prompt_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps_u = lo_dit.forward(x_lo, tt, negative_embeds, dtype={torch.float16: 'fp16', getattr(torch, 'bfloat16', torch.float16): 'bf16', torch.float32: 'fp32'}[dt])
        eps = _cfg_merge(eps_u, eps_c, getattr(cfg.low, 'cfg_scale', None) if cfg.low else cfg.guidance_scale)
        out = scheduler_lo.step(model_output=eps, timestep=t, sample=x_lo)
        x_lo = out.prev_sample
        if (i + 1) % 5 == 0:
            log.info("[wan22.gguf] low step %d/%d", i + 1, len(tlist_lo))
    frames_lo = _vae_decode_video(_patch_unembed3d(x_lo, w_lo, grid_lo), model_dir=os.path.dirname(lo_path), device=cfg.device, dtype=cfg.dtype, vae_dir=cfg.vae_dir, logger=log)
    _try_clear_cache()
    if not frames_lo:
        raise RuntimeError("WAN22 GGUF: Low stage produced no frames")
    return frames_lo


__all__ = [
    'ModelSpec', 'BlockSpec', 'CrossAttnWeights', 'derive_spec_from_state',
    'WanDiTGGUF', 'run_txt2vid', 'run_img2vid',
]
def _get_logger(logger: Any):
    if logger is not None:
        return logger
    lg = logging.getLogger("wan22.gguf")
    if not lg.handlers:
        h = logging.StreamHandler()
        fmt = logging.Formatter('[wan22.gguf] %(levelname)s: %(message)s')
        h.setFormatter(fmt)
        lg.addHandler(h)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    return lg
