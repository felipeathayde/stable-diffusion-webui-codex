from __future__ import annotations

"""Thin compatibility bridge to A1111/Forge legacy modules.

All imports of `modules.*` and `modules_forge.*` should happen here so that
engine/task code in apps/server/backend does not depend on legacy paths.

This preserves behavior while we progressively port functionality into
runtime/ and engines/ proper.
"""

from typing import Any, Sequence


def get_opts():
    from modules import shared as _shared  # type: ignore

    return _shared.opts


def get_device():
    from modules import shared as _shared  # type: ignore

    return _shared.device


def device_cpu():
    from modules import devices as _devices  # type: ignore

    return _devices.cpu


def torch_gc() -> None:
    from modules import devices as _devices  # type: ignore

    _devices.torch_gc()


def create_sampler(name: str, sd_model: Any):
    from modules import sd_samplers as _sd_samplers  # type: ignore

    return _sd_samplers.create_sampler(name, sd_model)


def new_image_rng(
    shape: Sequence[int],
    seeds: Sequence[int],
    *,
    subseeds: Sequence[int] | None = None,
    subseed_strength: float = 0.0,
    seed_resize_from_h: int = 0,
    seed_resize_from_w: int = 0,
):
    from modules import rng as _rng  # type: ignore

    return _rng.ImageRNG(
        shape,
        seeds,
        subseeds=subseeds or [],
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
    )


def apply_token_merging(model: Any, ratio: float) -> None:
    from modules.sd_models import apply_token_merging as _apply  # type: ignore

    _apply(model, ratio)


def SkipWritingToConfig():  # noqa: N802 - mirrors legacy symbol
    from modules.sd_models import SkipWritingToConfig as _Skip  # type: ignore

    return _Skip()


def decode_first_stage(model: Any, batch: Any) -> Any:
    from modules.sd_samplers_common import decode_first_stage as _dec  # type: ignore

    return _dec(model, batch)


def images_tensor_to_samples_auto(tensor: Any, model: Any) -> Any:
    """Encode image tensor to latent samples using the configured VAE method."""
    from modules.sd_samplers_common import (  # type: ignore
        images_tensor_to_samples as _to_samples,
        approximation_indexes as _approx,
    )
    from modules import shared as _shared  # type: ignore

    method = getattr(_shared.opts, "sd_vae_encode_method", "Full")
    return _to_samples(tensor, _approx.get(method), model)


def shared_state():
    from modules import shared as _shared  # type: ignore

    return getattr(_shared, "state", None)


def scripts_post_sample(processing: Any, samples: Any) -> Any:
    from modules import scripts as _scripts  # type: ignore

    sr = getattr(processing, "scripts", None)
    if sr is None or not hasattr(sr, "post_sample"):
        return samples
    args_cls = getattr(_scripts, "PostSampleArgs", None)
    if args_cls is None:
        return samples
    args = args_cls(samples)
    sr.post_sample(processing, args)
    return getattr(args, "samples", samples)


def scripts_process(processing: Any) -> None:
    sr = getattr(processing, "scripts", None)
    if sr is None or not hasattr(sr, "process"):
        return
    sr.process(processing)


def main_entry_modules_change(mods: Any, *, save: bool, refresh: bool) -> bool:
    from modules_forge import main_entry as _main  # type: ignore

    return _main.modules_change(mods, save=save, refresh=refresh)


def main_entry_checkpoint_change(name: str, *, save: bool, refresh: bool) -> bool:
    from modules_forge import main_entry as _main  # type: ignore

    return _main.checkpoint_change(name, save=save, refresh=refresh)


def main_entry_refresh_model_loading_parameters() -> None:
    from modules_forge import main_entry as _main  # type: ignore

    _main.refresh_model_loading_parameters()


def sd_models_forge_model_reload() -> None:
    from modules import sd_models as _sd_models  # type: ignore

    _sd_models.forge_model_reload()

