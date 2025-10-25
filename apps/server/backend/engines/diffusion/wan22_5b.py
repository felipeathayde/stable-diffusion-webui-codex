from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from apps.server.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.server.backend.core.requests import InferenceEvent, Txt2VidRequest, Img2VidRequest, ProgressEvent, ResultEvent
from apps.server.backend.engines.diffusion.base_video import BaseVideoEngine
from apps.server.backend.engines.diffusion.txt2vid import run_txt2vid as _run_t2v
from apps.server.backend.engines.diffusion.img2vid import run_img2vid as _run_i2v
from apps.server.backend.core.exceptions import EngineLoadError

import os

from apps.server.backend.huggingface.assets import ensure_repo_minimal_files
from .wan22_common import EngineOpts, WanComponents, resolve_wan_repo_candidates, resolve_user_supplied_assets

PROJECT_ROOT = Path(__file__).resolve().parents[5]
HF_ROOT = PROJECT_ROOT / "apps" / "server" / "backend" / "huggingface"


class Wan225BEngine(BaseVideoEngine):
    engine_id = "wan22_5b"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None
        self._opts = EngineOpts()

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("wan-2.2-5b",),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": "WAN 2.2 5B via Diffusers or GGUF"},
        )

    # ------------------------------ lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        dev = str(options.get("device", "auto"))
        dty = str(options.get("dtype", "fp16"))
        self._opts = EngineOpts(device=dev, dtype=dty)
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

        vendor_dir: Optional[Path] = None
        last_exc: Optional[Exception] = None
        for rid in resolve_wan_repo_candidates(self.engine_id):
            local_dir = HF_ROOT / Path(rid.replace('/', os.sep))
            try:
                ensure_repo_minimal_files(rid, str(local_dir), offline=False)
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
                self._logger.error(
                    "WAN22 5B: failed to fetch minimal HF assets from %s: %s", rid, exc
                )
        if vendor_dir is None and last_exc is not None:
            raise EngineLoadError(
                f"WAN22 5B: unable to prepare required HF assets; last error: {last_exc}"
            )

        def _is_diffusers_dir(path: str) -> bool:
            try:
                return (
                    os.path.isfile(os.path.join(path, 'model_index.json'))
                    or os.path.isfile(os.path.join(path, 'unet', 'config.json'))
                    or os.path.isfile(os.path.join(path, 'transformer', 'config.json'))
                    or os.path.isfile(os.path.join(path, 'vae', 'config.json'))
                )
            except Exception:
                return False

        comp.model_dir = p
        comp.device = "cuda" if (dev in ("auto", "cuda")) and getattr(__import__('torch').cuda, 'is_available', lambda: False)() else "cpu"
        comp.dtype = dty

        if _is_diffusers_dir(p):
            from diffusers import AutoencoderKLWan  # type: ignore
            from diffusers import WanPipeline  # type: ignore
            import torch  # type: ignore
            torch_dtype = {"fp16": torch.float16, "bf16": getattr(torch, "bfloat16", torch.float16), "fp32": torch.float32}[
                dty.lower() if dty.lower() in ("fp16", "bf16", "fp32") else "fp16"
            ]
            vae = AutoencoderKLWan.from_pretrained(p, subfolder="vae", torch_dtype=torch_dtype, local_files_only=True)
            pipe = WanPipeline.from_pretrained(p, torch_dtype=torch_dtype, vae=vae, local_files_only=True)
            pipe = pipe.to(comp.device)
            from apps.server.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as apply_attn  # type: ignore
            from apps.server.backend.engines.util.accelerator import apply_to_diffusers_pipeline as apply_accel  # type: ignore
            apply_attn(pipe, logger=self._logger)
            apply_accel(pipe, logger=self._logger)
            comp.pipeline = pipe
            comp.vae = vae
            self._logger.info("WAN22 5B diffusers pipeline loaded: %s", p)
        else:
            # GGUF path: tokenizer/config under backend/huggingface; all weights supplied by user.
            comp.pipeline = None
            self._logger.info("WAN22 5B GGUF runtime selected for %s", p)

        self._comp = comp
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._comp = None
        self.mark_unloaded()

    # ------------------------------ tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_t2v(engine=self, comp=self._comp, request=request)
        else:
            from apps.server.backend.runtime.nn import wan22 as gguf
            yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid (GGUF)")
            ex = getattr(request, 'extras', {}) or {}
            vae_path, te_path, meta_dir = resolve_user_supplied_assets(ex, self._comp.hf_repo_dir)
            try:
                self._logger.info(
                    "[wan22.gguf] assets: metadata=%s te=%s vae=%s",
                    os.path.basename(str(meta_dir)) if meta_dir else None,
                    os.path.basename(str(te_path)) if te_path else None,
                    os.path.basename(str(vae_path)) if vae_path else None,
                )
            except Exception:
                pass
            cfg = gguf.RunConfig(
                width=int(getattr(request, 'width', 768) or 768),
                height=int(getattr(request, 'height', 432) or 432),
                fps=int(getattr(request, 'fps', 24) or 24),
                num_frames=int(getattr(request, 'num_frames', 16) or 16),
                guidance_scale=getattr(request, 'cfg_scale', None),
                dtype=self._opts.dtype,
                device=self._opts.device,
                seed=getattr(request, 'seed', None),
                prompt=request.prompt,
                negative_prompt=getattr(request, 'negative_prompt', None),
                vae_dir=vae_path,
                text_encoder_dir=te_path,
                metadata_dir=meta_dir,
                high=gguf.StageConfig(
                    model_dir=self._comp.model_dir or '',
                    sampler=str(getattr(request, 'sampler', 'Euler a')),
                    scheduler=str(getattr(request, 'scheduler', 'Automatic')),
                    steps=max(1, int(getattr(request, 'steps', 12) or 12)),
                    cfg_scale=getattr(request, 'cfg_scale', None),
                ),
                low=gguf.StageConfig(
                    model_dir=self._comp.model_dir or '',
                    sampler=str(getattr(request, 'sampler', 'Euler a')),
                    scheduler=str(getattr(request, 'scheduler', 'Automatic')),
                    steps=max(1, int(getattr(request, 'steps', 12) or 12)),
                    cfg_scale=getattr(request, 'cfg_scale', None),
                ),
            )
            def _cb(**p):
                try:
                    self._logger.info("[wan22.gguf] %s %d/%d (%.1f%%)", p.get('stage','-'), int(p.get('step',0)), int(p.get('total',0)), float(p.get('percent',0))*100.0)
                except Exception:
                    pass
            frames = gguf.run_txt2vid(cfg, logger=self._logger, on_progress=_cb)
            yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "txt2vid", "frames": len(frames)})})

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_i2v(engine=self, comp=self._comp, request=request)
        else:
            from apps.server.backend.runtime.nn import wan22 as gguf
            if getattr(request, 'init_image', None) is None:
                raise RuntimeError("img2vid requires 'init_image'")
            yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid (GGUF)")
            ex = getattr(request, 'extras', {}) or {}
            vae_path, te_path, meta_dir = resolve_user_supplied_assets(ex, self._comp.hf_repo_dir)
            try:
                self._logger.info(
                    "[wan22.gguf] assets: metadata=%s te=%s vae=%s",
                    os.path.basename(str(meta_dir)) if meta_dir else None,
                    os.path.basename(str(te_path)) if te_path else None,
                    os.path.basename(str(vae_path)) if vae_path else None,
                )
            except Exception:
                pass
            cfg = gguf.RunConfig(
                width=int(getattr(request, 'width', 768) or 768),
                height=int(getattr(request, 'height', 432) or 432),
                fps=int(getattr(request, 'fps', 24) or 24),
                num_frames=int(getattr(request, 'num_frames', 16) or 16),
                guidance_scale=getattr(request, 'cfg_scale', None),
                dtype=self._opts.dtype,
                device=self._opts.device,
                seed=getattr(request, 'seed', None),
                prompt=request.prompt,
                negative_prompt=getattr(request, 'negative_prompt', None),
                init_image=getattr(request, 'init_image', None),
                vae_dir=vae_path,
                text_encoder_dir=te_path,
                metadata_dir=meta_dir,
                high=gguf.StageConfig(
                    model_dir=self._comp.model_dir or '',
                    sampler=str(getattr(request, 'sampler', 'Euler a')),
                    scheduler=str(getattr(request, 'scheduler', 'Automatic')),
                    steps=max(1, int(getattr(request, 'steps', 12) or 12)),
                    cfg_scale=getattr(request, 'cfg_scale', None),
                ),
                low=gguf.StageConfig(
                    model_dir=self._comp.model_dir or '',
                    sampler=str(getattr(request, 'sampler', 'Euler a')),
                    scheduler=str(getattr(request, 'scheduler', 'Automatic')),
                    steps=max(1, int(getattr(request, 'steps', 12) or 12)),
                    cfg_scale=getattr(request, 'cfg_scale', None),
                ),
            )
            def _cb2(**p):
                try:
                    self._logger.info("[wan22.gguf] %s %d/%d (%.1f%%)", p.get('stage','-'), int(p.get('step',0)), int(p.get('total',0)), float(p.get('percent',0))*100.0)
                except Exception:
                    pass
            frames = gguf.run_img2vid(cfg, logger=self._logger, on_progress=_cb2)
            yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "img2vid", "frames": len(frames)})})

    # ------------------------------ helpers
    # No per-stage GGUF builder needed here; runtimes will read request.extras directly
