"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 5B video engine implementation (Diffusers and GGUF paths).
Implements txt2vid/img2vid/vid2vid by loading WAN components (pipeline, VAE, text encoder, metadata) and dispatching to the respective
use-cases, with strict asset resolution and stage overrides via request extras.

Symbols (top-level; keep in sync; no ghosts):
- `_is_diffusers_dir` (function): Heuristic check for a Diffusers-style WAN weights directory (config/model_index presence).
- `_looks_like_wan_diffusers_weights_dir` (function): Heuristic check for WAN Diffusers weights (safetensors/bin shards under transformer/vae).
- `Wan225BEngine` (class): `BaseVideoEngine` implementation for WAN22 5B; loads vendor HF metadata, builds/loads components, and runs
  txt2vid/img2vid/vid2vid via progress-streamed use-cases (contains nested helper logic for diffusers/GGUF mode selection).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import Img2VidRequest, InferenceEvent, Txt2VidRequest, Vid2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.use_cases.txt2vid import run_txt2vid as _run_t2v
from apps.backend.use_cases.img2vid import run_img2vid as _run_i2v
from apps.backend.use_cases.vid2vid import run_vid2vid as _run_v2v
from apps.backend.core.exceptions import EngineLoadError

import os

from apps.backend.engines.wan22.wan22_common import (
    WanComponents,
    resolve_wan_repo_candidates,
)
from apps.backend.engines.wan22.diffusers_loader import load_wan_diffusers_pipeline
from apps.backend.infra.config.repo_root import get_repo_root
from apps.backend.huggingface.assets import ensure_repo_minimal_files

REPO_ROOT = get_repo_root()
HF_ROOT = REPO_ROOT / "apps" / "backend" / "huggingface"


def _is_diffusers_dir(path: str) -> bool:
    try:
        return (
            os.path.isfile(os.path.join(path, "model_index.json"))
            or os.path.isfile(os.path.join(path, "unet", "config.json"))
            or os.path.isfile(os.path.join(path, "transformer", "config.json"))
            or os.path.isfile(os.path.join(path, "vae", "config.json"))
        )
    except Exception:
        return False


def _looks_like_wan_diffusers_weights_dir(path: str) -> bool:
    try:
        tdir = os.path.join(path, "transformer")
        if os.path.isdir(tdir):
            for name in os.listdir(tdir):
                n = name.lower()
                if n.endswith(".safetensors") or n.endswith(".bin") or n.endswith(".safetensors.index.json"):
                    return True
        vdir = os.path.join(path, "vae")
        if os.path.isdir(vdir):
            for name in os.listdir(vdir):
                n = name.lower()
                if n.endswith(".safetensors") or n.endswith(".bin") or n.endswith(".safetensors.index.json"):
                    return True
    except Exception:
        return False
    return False


class Wan225BEngine(BaseVideoEngine):
    engine_id = "wan22_5b"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID, TaskType.VID2VID),
            model_types=("wan-2.2-5b",),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": "WAN 2.2 5B via Diffusers or GGUF"},
        )

    # ------------------------------ lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.debug("[wan22_5b] before load()")
        dev = str(options.get("device", "auto"))
        dty = str(options.get("dtype", "fp16"))
        comp = WanComponents()

        p = model_ref
        try:
            if os.path.isfile(p):
                p = os.path.dirname(p)
        except Exception:
            pass
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if not os.path.isdir(p):
            alt = os.path.abspath(os.path.join("models", "Wan", model_ref))
            if os.path.isdir(alt):
                p = alt
            else:
                raise EngineLoadError(f"WAN22 5B model path not found: {model_ref}")

        comp.model_dir = p
        comp.dtype = dty
        try:
            from apps.backend.runtime.families.wan22.config import resolve_device_name

            resolved = resolve_device_name(dev)
            comp.device = "cpu" if resolved == "cpu" else "cuda"
        except Exception as exc:
            raise EngineLoadError(str(exc)) from exc

        if _is_diffusers_dir(p) or _looks_like_wan_diffusers_weights_dir(p):
            # Diffusers path requires vendored HF metadata (model_index/tokenizers/etc).
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
                    self._logger.error("WAN22 5B: failed to prepare minimal HF assets from %s: %s", rid, exc)
            if vendor_dir is None:
                raise EngineLoadError(
                    f"WAN22 5B: unable to prepare required HF assets for Diffusers path; last error: {last_exc}"
                )

            try:
                vendor = Path(str(comp.hf_repo_dir or "")).resolve() if comp.hf_repo_dir else None
                if vendor is None or not vendor.is_dir():
                    raise EngineLoadError("WAN22 5B diffusers path requires vendored HF metadata (hf_repo_dir).")
                pipe = load_wan_diffusers_pipeline(
                    weights_dir=Path(p),
                    vendor_dir=vendor,
                    engine_id=self.engine_id,
                    device=comp.device,
                    dtype=dty,
                    logger=self._logger,
                )
                comp.pipeline = pipe
                comp.vae = getattr(pipe, "vae", None)
            except Exception as exc:
                raise EngineLoadError(f"WAN22 5B diffusers pipeline load failed: {exc}") from exc
        else:
            # GGUF path: assets are payload-driven (sha-only); avoid any local/online HF metadata probing here.
            comp.pipeline = None
            ref_base = os.path.basename(str(model_ref or "")).lower()
            weights_hint = "14b" if "14b" in ref_base else ("5b" if "5b" in ref_base else "unknown")
            self._logger.info(
                "WAN22 GGUF runtime selected (engine=%s weights_hint=%s) for %s (device=%s dtype=%s)",
                self.engine_id,
                weights_hint,
                p,
                comp.device,
                dty,
            )

        self._comp = comp
        self._logger.debug("[wan22_5b] after load()")
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._logger.debug("[wan22_5b] before unload()")
        self._comp = None
        self._logger.debug("[wan22_5b] after unload()")
        self.mark_unloaded()

    # ------------------------------ tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug("[wan22_5b] before txt2vid()")
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_t2v(engine=self, comp=self._comp, request=request)
        self._logger.debug("[wan22_5b] after txt2vid()")

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug("[wan22_5b] before img2vid()")
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_i2v(engine=self, comp=self._comp, request=request)
        self._logger.debug("[wan22_5b] after img2vid()")

    def vid2vid(self, request: Vid2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.debug("[wan22_5b] before vid2vid()")
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_v2v(engine=self, comp=self._comp, request=request)
        self._logger.debug("[wan22_5b] after vid2vid()")
        return
