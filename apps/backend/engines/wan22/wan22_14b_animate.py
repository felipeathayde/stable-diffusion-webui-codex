"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 Animate 14B video engine (Diffusers).
Loads `WanAnimatePipeline`-style weights with vendored Hugging Face metadata and dispatches `vid2vid` jobs via the shared `use_cases.vid2vid` runner.

Symbols (top-level; keep in sync; no ghosts):
- `Wan22Animate14BEngine` (class): VID2VID engine for WAN2.2 Animate 14B using Diffusers pipelines.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator, Optional

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.exceptions import EngineLoadError
from apps.backend.core.requests import InferenceEvent, Vid2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.engines.wan22.diffusers_loader import load_wan_diffusers_pipeline
from apps.backend.engines.wan22.wan22_common import (
    EngineOpts,
    WanComponents,
    resolve_wan_repo_candidates,
    unload_wan_components,
)
from apps.backend.huggingface.assets import ensure_repo_minimal_files
from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.runtime.memory import memory_management
from apps.backend.use_cases.vid2vid import run_vid2vid as _run_v2v


REPO_ROOT = get_repo_root()
HF_ROOT = REPO_ROOT / "apps" / "backend" / "huggingface"


class Wan22Animate14BEngine(BaseVideoEngine):
    engine_id = "wan22_14b_animate"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None
        self._opts = EngineOpts()

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.VID2VID,),
            model_types=("wan-2.2-animate-14b",),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": "WAN2.2 Animate 14B via Diffusers (requires preprocessing inputs)"},
        )

    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        default_mount_device = str(memory_management.manager.mount_device())
        requested_device = options.get("device")
        if requested_device not in (None, "", "auto"):
            raise EngineLoadError(
                "WAN22 engine-local 'device' override is not supported; "
                "configure the runtime main device via launcher/API canonical selection."
            )
        dty = str(options.get("dtype", "fp16"))
        self._opts = EngineOpts(device=default_mount_device, dtype=dty)

        p = str(model_ref)
        try:
            if os.path.isfile(p):
                p = os.path.dirname(p)
        except Exception:
            pass
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if not os.path.isdir(p):
            raise EngineLoadError(f"WAN22 Animate model path not found: {model_ref}")

        comp = WanComponents()

        # Resolve vendored HF metadata (model_index/tokenizers/etc).
        from apps.backend.infra.config.args import args as backend_args

        offline = bool(getattr(backend_args, "disable_online_tokenizer", False))
        vendor_dir: Optional[Path] = None
        last_exc: Optional[Exception] = None
        for rid in resolve_wan_repo_candidates(self.engine_id):
            local_dir = HF_ROOT / Path(rid.replace("/", os.sep))
            try:
                ensure_repo_minimal_files(rid, str(local_dir), offline=offline)
                vendor_dir = local_dir
                comp.hf_repo_dir = str(local_dir)
                te_dir = local_dir / "text_encoder"
                tk_dir = local_dir / "tokenizer"
                vae_dir = local_dir / "vae"
                comp.hf_text_encoder_dir = str(te_dir) if te_dir.exists() else None
                comp.hf_tokenizer_dir = str(tk_dir) if tk_dir.exists() else None
                comp.hf_vae_dir = str(vae_dir) if vae_dir.exists() else None
                break
            except Exception as exc:
                last_exc = exc
                self._logger.error("WAN22 Animate: failed to prepare HF metadata from %s: %s", rid, exc)
        if vendor_dir is None:
            raise EngineLoadError(
                f"WAN22 Animate: unable to prepare required HF assets; last error: {last_exc}"
            )

        comp.dtype = dty
        comp.device = default_mount_device
        comp.model_dir = p

        try:
            pipe = load_wan_diffusers_pipeline(
                weights_dir=Path(p),
                vendor_dir=vendor_dir,
                engine_id=self.engine_id,
                device=comp.device,
                dtype=dty,
                logger=self._logger,
            )
        except Exception as exc:
            raise EngineLoadError(f"WAN22 Animate: failed to load Diffusers pipeline: {exc}") from exc

        comp.pipeline = pipe
        comp.vae = getattr(pipe, "vae", None)
        self._comp = comp
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        unload_wan_components(self._comp, engine_id=self.engine_id, logger=self._logger)
        self._comp = None
        self.mark_unloaded()

    def vid2vid(self, request: Vid2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        assert self._comp is not None
        # Force the WAN Animate strategy when using this engine.
        from dataclasses import replace

        extras = dict(getattr(request, "extras", {}) or {})
        vid_cfg = extras.get("vid2vid") if isinstance(extras.get("vid2vid"), dict) else {}
        vid_cfg = dict(vid_cfg) if isinstance(vid_cfg, dict) else {}
        vid_cfg["method"] = "wan_animate"
        extras["vid2vid"] = vid_cfg

        req2 = replace(request, extras=extras)
        yield from _run_v2v(engine=self, comp=self._comp, request=req2)
