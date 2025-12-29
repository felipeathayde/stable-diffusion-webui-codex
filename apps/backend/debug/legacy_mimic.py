"""Legacy pipeline mimic/tracer (optional).

Goal
----
Provide a throwaway hook layer that mirrors the Forge/A1111 pipeline snapshot
under `.refs/Forge-A1111` so we can compare Codex vs legacy behaviour
end-to-end. When enabled, it logs checkpoint loading, processing, decode, and
image save events without altering outputs.

Usage
-----
```python
from apps.backend.debug import legacy_mimic
legacy_mimic.enable()  # idempotent
```

After enabling, run the legacy/Forge UI as usual. Logs are printed to stdout
with prefix `[legacy-mimic]` for the key stages. Disable by restarting the
process; this module avoids global side effects otherwise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

from apps.backend.infra.config.repo_root import get_repo_root

_ENABLED = False


def _tensor_stats(t: Any) -> str:
    if not isinstance(t, torch.Tensor):
        return f"non-tensor type={type(t).__name__}"
    x = t.float()
    try:
        return (
            f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device} "
            f"min={x.min():.6f} max={x.max():.6f} "
            f"mean={x.mean():.6f} std={x.std(unbiased=False):.6f}"
        )
    except Exception:
        return f"shape={tuple(x.shape)} dtype={x.dtype} device={x.device}"


def _log(stage: str, **kw: Any) -> None:
    parts = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[legacy-mimic] {stage} {parts}")


def _wrap_sd_model(sd_model: Any, log_fn: Callable[..., None]) -> None:
    if sd_model is None or getattr(sd_model, "_codex_legacy_mimic_wrapped", False):
        return

    if hasattr(sd_model, "decode_first_stage"):
        orig_decode = sd_model.decode_first_stage

        def _decode_wrapper(x, *args, **kwargs):  # noqa: ANN001
            t0 = time.perf_counter()
            out = orig_decode(x, *args, **kwargs)
            t1 = time.perf_counter()
            log_fn("decode_first_stage", seconds=f"{t1 - t0:.3f}", latents=_tensor_stats(x), image=_tensor_stats(out))
            return out

        sd_model.decode_first_stage = _decode_wrapper

    sd_model._codex_legacy_mimic_wrapped = True


def enable(log_fn: Callable[..., None] | None = None) -> None:
    """Install tracing hooks over the legacy pipeline (idempotent)."""

    global _ENABLED
    if _ENABLED:
        return

    log = log_fn or _log

    legacy_root = get_repo_root() / ".refs" / "Forge-A1111"
    if legacy_root.exists():
        sys.path.insert(0, str(legacy_root))
    else:
        log("warning", message=f"legacy path not found: {legacy_root}")

    # Late imports so Codex runtime stays clean when this module is unused
    import importlib

    sd_models = importlib.import_module("modules.sd_models")
    processing = importlib.import_module("modules.processing")
    images = importlib.import_module("modules.images")
    sd_vae_approx = importlib.import_module("modules.sd_vae_approx")

    # Hook checkpoint load
    if not hasattr(sd_models, "_codex_legacy_mimic_load_model"):
        sd_models._codex_legacy_mimic_load_model = sd_models.load_model

        def _load_model_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
            t0 = time.perf_counter()
            out = sd_models._codex_legacy_mimic_load_model(*args, **kwargs)
            t1 = time.perf_counter()
            ckpt = args[0] if args else kwargs.get("checkpoint_info") or kwargs.get("ckpt_info")
            ckpt_name = getattr(ckpt, "filename", None) or getattr(ckpt, "name", None) or str(ckpt)
            log("load_model", checkpoint=ckpt_name, seconds=f"{t1 - t0:.3f}")
            return out

        sd_models.load_model = _load_model_wrapper

    # Hook process_images_inner
    if not hasattr(processing, "_codex_legacy_mimic_process_images_inner"):
        processing._codex_legacy_mimic_process_images_inner = processing.process_images_inner

        def _process_images_inner_wrapper(p, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            t0 = time.perf_counter()
            sd_model = getattr(p, "sd_model", None)
            log(
                "process.start",
                width=getattr(p, "width", None),
                height=getattr(p, "height", None),
                steps=getattr(p, "steps", None),
                sampler=getattr(p, "sampler_name", None),
            )
            _wrap_sd_model(sd_model, log)
            res = processing._codex_legacy_mimic_process_images_inner(p, *args, **kwargs)
            t1 = time.perf_counter()
            log(
                "process.done",
                seconds=f"{t1 - t0:.3f}",
                images=len(getattr(res, "images", []) or []),
            )
            return res

        processing.process_images_inner = _process_images_inner_wrapper

    # Hook VAE approx to catch missing forge_objects
    if not hasattr(sd_vae_approx, "_codex_legacy_mimic_model"):
        sd_vae_approx._codex_legacy_mimic_model = sd_vae_approx.model

        def _vae_model_wrapper(*args, **kwargs):  # noqa: ANN002, ANN003
            shared = importlib.import_module("modules.shared")
            sd_model = getattr(shared, "sd_model", None)
            fo = getattr(sd_model, "forge_objects", None)
            log(
                "sd_vae_approx.model",
                forge_objects=fo is not None,
                vae_present=fo is not None and getattr(fo, "vae", None) is not None,
            )
            return sd_vae_approx._codex_legacy_mimic_model(*args, **kwargs)

        sd_vae_approx.model = _vae_model_wrapper

    # Hook image save
    if not hasattr(images, "_codex_legacy_mimic_save_image"):
        images._codex_legacy_mimic_save_image = images.save_image

        def _save_image_wrapper(image, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003
            res = images._codex_legacy_mimic_save_image(image, *args, **kwargs)
            target = kwargs.get("path") or (args[4] if len(args) >= 5 else None)
            filename = res if isinstance(res, str) else getattr(res, "filename", None)
            log("save_image", target=target or "<auto>", filename=filename or "<unknown>")
            return res

        images.save_image = _save_image_wrapper

    _ENABLED = True
    log("enabled", legacy_root=str(legacy_root))


__all__ = ["enable"]
