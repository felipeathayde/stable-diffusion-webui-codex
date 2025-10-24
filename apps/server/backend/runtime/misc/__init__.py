"""Miscellaneous backend helpers (lazy exports to avoid import cycles).

This package exposes symbols on demand to prevent importing heavy modules
at package import time. It keeps `utils` and `memory` free from cycles.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    # attention utils
    "efficient_dot_product_attention": (".sub_quadratic_attention", "efficient_dot_product_attention"),
    # image resize helpers
    "adaptive_resize": (".image_resize", "adaptive_resize"),
    "bislerp": (".image_resize", "bislerp"),
    "lanczos": (".image_resize", "lanczos"),
    # state dict helpers
    "unet_to_diffusers": (".diffusers_state_dict", "unet_to_diffusers"),
    # for torch.load(..., pickle_module=checkpoint_pickle)
    # expose the submodule itself, not a symbol
    "checkpoint_pickle": (".checkpoint_pickle", None),
}

__all__ = sorted(_EXPORTS.keys())


def __getattr__(name: str) -> Any:  # pragma: no cover - runtime behavior
    try:
        mod_path, sym = _EXPORTS[name]
    except KeyError as e:
        raise AttributeError(name) from e
    mod = import_module(mod_path, __name__)
    return mod if sym is None else getattr(mod, sym)
