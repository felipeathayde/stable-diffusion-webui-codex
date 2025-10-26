from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass
from io import BytesIO
import os
from typing import Any, Optional, Tuple

import torch

_LOG = logging.getLogger("wan22.test_cache")
if not _LOG.handlers:
    _LOG.addHandler(logging.StreamHandler())
_LOG.setLevel(logging.INFO)
_LOG.propagate = False


def _enabled() -> bool:
    v = os.getenv("WAN_TEST_CACHE", "0").strip().lower()
    return v in ("1", "true", "yes", "on")


@dataclass
class _Entry:
    key: str
    tensor_a: torch.Tensor
    tensor_b: Optional[torch.Tensor] = None


class _LRU:
    def __init__(self, max_items: int = 8) -> None:
        self.max = int(max(1, max_items))
        self._d: OrderedDict[str, _Entry] = OrderedDict()

    def get(self, key: str) -> Optional[_Entry]:
        it = self._d.get(key)
        if it is not None:
            self._d.move_to_end(key)
        return it

    def put(self, entry: _Entry) -> None:
        self._d[entry.key] = entry
        self._d.move_to_end(entry.key)
        while len(self._d) > self.max:
            self._d.popitem(last=False)


_te_cache = _LRU(int(os.getenv("WAN_TEST_CACHE_TE_MAX", "8")))
_vae_cache = _LRU(int(os.getenv("WAN_TEST_CACHE_VAE_MAX", "4")))


def _to_cpu32(x: torch.Tensor) -> torch.Tensor:
    return x.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _to_cpu16(x: torch.Tensor) -> torch.Tensor:
    # Latents are fine in fp16 to save memory; VAE decode will cast later
    return x.detach().to(device="cpu", dtype=torch.float16).contiguous()


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _hash_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _image_fingerprint(init_image: Any) -> Optional[str]:
    try:
        from PIL import Image
        if isinstance(init_image, Image.Image):
            buf = BytesIO()
            init_image.save(buf, format="PNG")
            return _hash_bytes(buf.getvalue())
    except Exception:
        pass
    try:
        if torch.is_tensor(init_image):
            t = init_image.detach().contiguous().to(device="cpu", dtype=torch.float32)
            return _hash_bytes(t.numpy().tobytes())
    except Exception:
        pass
    return None


def try_get_te(prompt: str, negative: Optional[str], *, tk_dir: Optional[str], te_file: Optional[str]) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if not _enabled():
        return None
    nrm_p = prompt or ""
    nrm_n = negative or ""
    key = _hash_str("TE|" + tk_dir if tk_dir else "") + "|" + _hash_str(te_file or "") + "|" + _hash_str(nrm_p) + "|" + _hash_str(nrm_n)
    ent = _te_cache.get(key)
    if ent is not None:
        _LOG.info("[test_cache] TE hit len(p)=%d len(n)=%d", len(nrm_p), len(nrm_n))
        # Return clones to avoid accidental in-place mods
        a = ent.tensor_a.clone()
        b = ent.tensor_b.clone() if ent.tensor_b is not None else _to_cpu32(torch.zeros_like(ent.tensor_a))
        return a, b
    _LOG.info("[test_cache] TE miss len(p)=%d len(n)=%d", len(nrm_p), len(nrm_n))
    return None


def store_te(prompt: str, negative: Optional[str], *, tk_dir: Optional[str], te_file: Optional[str], p: torch.Tensor, n: torch.Tensor) -> None:
    if not _enabled():
        return
    try:
        nrm_p = prompt or ""
        nrm_n = negative or ""
        key = _hash_str("TE|" + (tk_dir or "")) + "|" + _hash_str(te_file or "") + "|" + _hash_str(nrm_p) + "|" + _hash_str(nrm_n)
        entry = _Entry(key=key, tensor_a=_to_cpu32(p), tensor_b=_to_cpu32(n))
        _te_cache.put(entry)
        _LOG.info("[test_cache] TE store")
    except Exception:
        pass


def try_get_vae(init_image: Any, *, vae_dir: Optional[str], out_hw: Tuple[int, int]) -> Optional[torch.Tensor]:
    if not _enabled():
        return None
    fp = _image_fingerprint(init_image)
    if fp is None:
        return None
    key = _hash_str("VAE|" + (vae_dir or "")) + "|" + fp + f"|{int(out_hw[0])}x{int(out_hw[1])}"
    ent = _vae_cache.get(key)
    if ent is not None:
        _LOG.info("[test_cache] VAE hit HxW=%dx%d", int(out_hw[0]), int(out_hw[1]))
        return ent.tensor_a.clone()
    _LOG.info("[test_cache] VAE miss HxW=%dx%d", int(out_hw[0]), int(out_hw[1]))
    return None


def store_vae(init_image: Any, *, vae_dir: Optional[str], out_hw: Tuple[int, int], latents: torch.Tensor) -> None:
    if not _enabled():
        return
    try:
        fp = _image_fingerprint(init_image)
        if fp is None:
            return
        key = _hash_str("VAE|" + (vae_dir or "")) + "|" + fp + f"|{int(out_hw[0])}x{int(out_hw[1])}"
        entry = _Entry(key=key, tensor_a=_to_cpu16(latents))
        _vae_cache.put(entry)
        _LOG.info("[test_cache] VAE store")
    except Exception:
        pass

