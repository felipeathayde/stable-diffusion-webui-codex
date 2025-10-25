from __future__ import annotations

from contextlib import contextmanager
from typing import Tuple
import math
import torch


def _downsample_seq_avg(x: torch.Tensor, keep_tokens: int) -> torch.Tensor:
    """Downsample the sequence length axis (dim=1) by average pooling.

    Shapes: x [B, S, C] -> [B, keep_tokens, C]. Keeps order; groups are contiguous.
    """
    b, s, c = x.shape
    keep = max(1, min(int(keep_tokens), s))
    if keep == s:
        return x
    stride = math.ceil(s / keep)
    chunks = []
    for start in range(0, s, stride):
        end = min(start + stride, s)
        chunks.append(x[:, start:end, :].mean(dim=1, keepdim=True))
    return torch.cat(chunks, dim=1)


def _token_merge_patch(n: torch.Tensor, ctx: torch.Tensor, val: torch.Tensor, extra: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Attention pre-patch that reduces K/V tokens while keeping Q length intact.

    Inputs are pre-projection activations [B, S, C]. We only downsample `ctx`/`val`.
    The merge ratio is provided via extra["token_merge_ratio"].
    """
    ratio = float(extra.get("token_merge_ratio", 0.0) or 0.0)
    if ratio <= 0.0:
        return n, ctx, val
    s = ctx.shape[1]
    keep = max(1, int(s * (1.0 - min(ratio, 0.95))))
    ctx_d = _downsample_seq_avg(ctx, keep)
    val_d = _downsample_seq_avg(val, keep)
    return n, ctx_d, val_d


def apply_token_merging(engine, ratio: float | int | None) -> None:
    """Install token-merging attention patches on the engine's UNet.

    - Reduces only K/V sequence length by average pooling per contiguous chunk.
    - Leaves Q length unchanged so output sequence matches input and residual paths.
    - Ratio in [0,1). 0 disables. Values >= 0.95 are capped.
    """
    r = float(ratio or 0.0)
    unet_patcher = getattr(getattr(engine, "forge_objects", engine), "unet", None)
    if unet_patcher is None:
        return
    # Clear previous setting if turning off
    if r <= 0.0:
        # remove token_merge_ratio flag while preserving other options
        opts = getattr(unet_patcher, "model_options", {}) or {}
        if "transformer_options" in opts and isinstance(opts["transformer_options"], dict):
            opts["transformer_options"].pop("token_merge_ratio", None)
            patches = opts["transformer_options"].get("patches", {}) or {}
            for k in ("attn1_patch", "attn2_patch"):
                lst = patches.get(k, [])
                # remove our function if present
                patches[k] = [p for p in lst if getattr(p, "__name__", "") != _token_merge_patch.__name__]
            opts["transformer_options"]["patches"] = patches
            unet_patcher.model_options = opts
        return

    # Install patches
    unet_patcher.set_transformer_option("token_merge_ratio", r)
    patches = unet_patcher.model_options.get("transformer_options", {}).get("patches", {}) or {}
    attn1_list = list(patches.get("attn1_patch", []))
    attn2_list = list(patches.get("attn2_patch", []))
    if all(getattr(p, "__name__", "") != _token_merge_patch.__name__ for p in attn1_list):
        attn1_list.append(_token_merge_patch)
    if all(getattr(p, "__name__", "") != _token_merge_patch.__name__ for p in attn2_list):
        attn2_list.append(_token_merge_patch)
    patches["attn1_patch"] = attn1_list
    patches["attn2_patch"] = attn2_list
    unet_patcher.set_transformer_option("patches", patches)


@contextmanager
def SkipWritingToConfig():
    """No-op context manager kept for call-site compatibility.

    We do not write legacy config files; keeping this as a placeholder.
    """
    yield


__all__ = ["apply_token_merging", "SkipWritingToConfig"]
