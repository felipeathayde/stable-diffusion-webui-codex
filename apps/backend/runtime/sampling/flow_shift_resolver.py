"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Resolve flow-match `flow_shift` for sampling from canonical scheduler_config.json sources.
Used by the sampling context builder for predictors with `prediction_type='const'` (flow-match).

Symbols (top-level; keep in sync; no ghosts):
- `resolve_flow_shift_for_sampling` (function): Resolve the effective flow shift (fixed/dynamic) for the current run.
- `__all__` (constant): Export list for the resolver helper.
"""

from __future__ import annotations


def resolve_flow_shift_for_sampling(
    sd_model,
    predictor,
    *,
    height: int | None,
    width: int | None,
) -> float:
    # Flow-match models must resolve flow_shift from the canonical scheduler_config.json.
    from pathlib import Path

    from apps.backend.infra.config.repo_root import get_repo_root
    from apps.backend.runtime.model_registry.family_runtime import get_family_spec
    from apps.backend.runtime.model_registry.flow_shift import (
        FlowShiftMode,
        FlowShiftSpec,
        flow_shift_spec_from_repo_dir,
    )
    from apps.backend.runtime.model_registry.specs import ModelFamily

    spec_obj: FlowShiftSpec | None = None
    raw_spec = getattr(predictor, "flow_shift_spec", None)
    if isinstance(raw_spec, FlowShiftSpec):
        spec_obj = raw_spec
    else:
        bundle = getattr(sd_model, "_current_bundle", None)
        repo_ref = getattr(bundle, "model_ref", None)
        if isinstance(repo_ref, str):
            repo_path = Path(repo_ref)
            if repo_path.is_dir():
                spec_obj = flow_shift_spec_from_repo_dir(repo_path)

    if spec_obj is None:
        # If the model isn't a diffusers directory, try resolving the canonical
        # scheduler config from the vendored Hugging Face mirror using the
        # detected repo_hint.
        bundle = getattr(sd_model, "_current_bundle", None)
        sig = getattr(bundle, "signature", None)
        repo_hint = getattr(sig, "repo_hint", None) if sig is not None else None
        if isinstance(repo_hint, str) and repo_hint.strip():
            repo_root = get_repo_root()
            vendor_root = repo_root / "apps" / "backend" / "huggingface"
            vendor = vendor_root / repo_hint
            if not vendor.is_dir():
                # Some detectors use a full HF repo id as repo_hint, but the vendored
                # mirror may be stored under a shorter directory name (e.g., "Chroma").
                vendor = vendor_root / Path(repo_hint).name
            if vendor.is_dir():
                spec_obj = flow_shift_spec_from_repo_dir(vendor)

    if spec_obj is None:
        # Z-Image GGUF checkpoints are core-only; shift is defined by the vendored diffusers scheduler config.
        bundle = getattr(sd_model, "_current_bundle", None)
        family = getattr(bundle, "family", None)
        if family is ModelFamily.ZIMAGE:
            repo_root = get_repo_root()
            zimage_repo = repo_root / "apps" / "backend" / "huggingface" / "Alibaba-TongYi" / "Z-Image-Turbo"
            spec_obj = flow_shift_spec_from_repo_dir(zimage_repo)

    if spec_obj is None:
        raise RuntimeError(
            "Flow-match sampling requires a scheduler_config.json to resolve flow_shift, but none was found. "
            "Load a diffusers repo with scheduler/ configs or ensure the engine provides vendored HF assets."
        )

    if spec_obj.mode is FlowShiftMode.DYNAMIC:
        if height is None or width is None:
            raise RuntimeError("Dynamic flow_shift requires explicit height/width for seq_len calculation.")
        bundle = getattr(sd_model, "_current_bundle", None)
        family = getattr(bundle, "family", None)
        if not isinstance(family, ModelFamily):
            raise RuntimeError("Dynamic flow_shift requires a known ModelFamily on the loaded bundle.")
        fam = get_family_spec(family)
        scale = int(fam.latent_scale_factor)
        patch = int(fam.patch_size)
        if scale <= 0 or patch <= 0:
            raise RuntimeError(f"Invalid latent_scale_factor/patch_size for family={family}: {scale}/{patch}")
        step = scale * patch
        if (int(height) % step) != 0 or (int(width) % step) != 0:
            raise RuntimeError(
                f"Invalid size for dynamic flow shift: {int(width)}x{int(height)} (expected multiples of {step})."
            )
        seq_len = (int(height) // scale // patch) * (int(width) // scale // patch)
        return spec_obj.resolve_effective_shift(seq_len=seq_len)

    return spec_obj.resolve_effective_shift()


__all__ = [
    "resolve_flow_shift_for_sampling",
]

