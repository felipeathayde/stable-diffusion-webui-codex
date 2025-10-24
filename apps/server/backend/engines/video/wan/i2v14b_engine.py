"""Wan 2.2 I2V 14B (Image → Video) engine.

MVP: Validates params, loads pipeline via WAN loader, supports dual-stage
refinement by seeding stage 2 with last frame from stage 1. Streams progress
and returns frames.
"""

from __future__ import annotations

import time
from typing import Iterator, List

from apps.server.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.server.backend.core.requests import InferenceEvent, Img2VidRequest, ProgressEvent, ResultEvent
from apps.server.backend.core.param_registry import parse_params
from apps.server.backend.core.exceptions import UnsupportedTaskError, EngineLoadError, EngineExecutionError
from apps.server.backend.core.enums import Mode
from apps.server.backend.core.presets import apply_preset_to_request

from ...base import DiffusionEngine
from .loader import WanLoader, WanComponents
from .schedulers import apply_sampler_scheduler
from .gguf_exec import WanGGUFRunConfig, WanStageConfig, run_img2vid, GGUFExecutorUnavailable
from apps.server.backend.core.params.video import VideoInterpolationOptions
from apps.server.backend.video.interpolation import maybe_interpolate


class WanI2V14BEngine(DiffusionEngine):
    engine_id = "wan_i2v_14b"
    _loader: WanLoader | None = None
    _comp: WanComponents | None = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.IMG2VID,),
            model_types=("wan-2.2-i2v-14b",),
            precision=("fp16", "bf16"),
            extras={"notes": "Stub engine; integration pending with WAN 2.2 I2V 14B"},
        )

    def img2vid(self, request: Img2VidRequest, **kwargs: object) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        mode = self._resolve_mode(request)
        params = parse_params(self.engine_id, TaskType.IMG2VID, request)
        params, _applied = apply_preset_to_request(self.engine_id, mode, TaskType.IMG2VID, params, logger=self._logger)

        if getattr(params, "init_image", None) is None:
            raise EngineExecutionError("wan_i2v_14b/img2vid requires 'init_image'")

        comp = self._ensure_components()
        # Debug context for diagnostics
        try:
            extras = getattr(request, 'extras', {}) or {}
            self._logger.info(
                "wan_i2v_14b: extras keys=%s model_ref=%s", list(extras.keys()), getattr(self, '_model_ref', None)
            )
            wh = extras.get('wan_high') if isinstance(extras, dict) else None
            wl = extras.get('wan_low') if isinstance(extras, dict) else None
            self._logger.info(
                "wan_i2v_14b: wan_high.model_dir=%s wan_low.model_dir=%s comp(model_dir=%s, high_dir=%s, low_dir=%s, pipe=%s)",
                getattr(wh, 'get', lambda *_: None)('model_dir') if isinstance(wh, dict) else None,
                getattr(wl, 'get', lambda *_: None)('model_dir') if isinstance(wl, dict) else None,
                getattr(comp, 'model_dir', None), getattr(comp, 'high_dir', None), getattr(comp, 'low_dir', None),
                'yes' if getattr(comp, 'pipeline', None) is not None else 'no'
            )
        except Exception:
            pass
        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing WAN I2V‑14B")

        # Choose pipeline: if per-stage dirs provided, ensure both are loaded
        high_model = None
        low_model = None
        try:
            hi = (getattr(request, 'extras', {}) or {}).get('wan_high')
            lo = (getattr(request, 'extras', {}) or {}).get('wan_low')
            if isinstance(hi, dict) and isinstance(lo, dict) and hi.get('model_dir') and lo.get('model_dir'):
                prefer_format = None
                try:
                    fmt = (getattr(request, 'extras', {}) or {}).get('wan_format')
                    if isinstance(fmt, str) and fmt.lower() in ('gguf', 'diffusers', 'auto'):
                        prefer_format = None if fmt == 'auto' else fmt.lower()
                except Exception:
                    prefer_format = None
                comp = self._loader.load_stages(str(hi.get('model_dir')), str(lo.get('model_dir')), device=self._comp.device, dtype=self._comp.dtype, prefer_format=prefer_format)  # type: ignore[union-attr]
                high_model = comp.pipeline_high
                low_model = comp.pipeline_low
        except Exception as exc:
            self._logger.warning("wan_i2v_14b: failed to load per-stage models: %s", exc)
        pipe = low_model or comp.pipeline
        gguf_only = (pipe is None) and (comp.high_dir or comp.low_dir)
        if (pipe is None) and not gguf_only:
            # No diffusers pipeline and no GGUF dirs provided — cannot proceed
            raise EngineExecutionError("WAN I2V‑14B requires Diffusers pipeline or GGUF stage directories (wan_high/wan_low)")
        frames: List[object] = []
        try:
            if gguf_only:
                try:
                    hi_cfg = getattr(params, "high_stage", None)
                    lo_cfg = getattr(params, "low_stage", None)
                    cfg = WanGGUFRunConfig(
                        width=int(getattr(params, "width", 768) or 768),
                        height=int(getattr(params, "height", 432) or 432),
                        fps=int(getattr(request, "fps", 24) or 24),
                        num_frames=int(getattr(params, "num_frames", 16) or 16),
                        guidance_scale=getattr(params, "guidance_scale", None),
                        dtype=self._comp.dtype if self._comp else "fp16",
                        device=self._comp.device if self._comp else "cpu",
                        init_image=getattr(params, "init_image", None),
                        prompt=request.prompt,
                        negative_prompt=request.negative_prompt or None,
                        vae_dir=str(((getattr(request, 'extras', {}) or {}).get('wan_vae_dir')) or '') or None,
                        text_encoder_dir=str(((getattr(request, 'extras', {}) or {}).get('wan_text_encoder_dir')) or '') or None,
                        tokenizer_dir=str(((getattr(request, 'extras', {}) or {}).get('wan_tokenizer_dir')) or '') or None,
                        high=WanStageConfig(
                            model_dir=str(comp.high_dir),
                            sampler=str(getattr(hi_cfg or params, "sampler", "Automatic")),
                            scheduler=str(getattr(hi_cfg or params, "scheduler", "Automatic")),
                            steps=int(getattr(hi_cfg or params, "steps", 12) or 12),
                            cfg_scale=getattr(hi_cfg or params, "cfg_scale", getattr(params, "guidance_scale", None)),
                            seed=getattr(hi_cfg or params, "seed", getattr(params, "seed", None)),
                            lora_path=getattr(hi_cfg or object(), "lora_path", None),
                            lora_weight=getattr(hi_cfg or object(), "lora_weight", None),
                            lightning=getattr(hi_cfg or object(), "lightning", None),
                        ) if comp.high_dir else None,
                        low=WanStageConfig(
                            model_dir=str(comp.low_dir),
                            sampler=str(getattr(lo_cfg or params, "sampler", "Automatic")),
                            scheduler=str(getattr(lo_cfg or params, "scheduler", "Automatic")),
                            steps=int(getattr(lo_cfg or params, "steps", 12) or 12),
                            cfg_scale=getattr(lo_cfg or params, "cfg_scale", getattr(params, "guidance_scale", None)),
                            seed=getattr(lo_cfg or params, "seed", getattr(params, "seed", None)),
                            lora_path=getattr(lo_cfg or object(), "lora_path", None),
                            lora_weight=getattr(lo_cfg or object(), "lora_weight", None),
                            lightning=getattr(lo_cfg or object(), "lightning", None),
                        ) if comp.low_dir else None,
                    )
                    frames = run_img2vid(cfg, logger=self._logger)
                except GGUFExecutorUnavailable as ex:
                    raise EngineExecutionError(str(ex))
            # Stage 1 (diffusers)
            hi = getattr(params, "high_stage", None)
            hsam = getattr(hi, "sampler", getattr(params, "sampler", "Automatic")) if hi else getattr(params, "sampler", "Automatic")
            hsch = getattr(hi, "scheduler", getattr(params, "scheduler", "Automatic")) if hi else getattr(params, "scheduler", "Automatic")
            hstp = getattr(hi, "steps", getattr(params, "steps", 12)) if hi else getattr(params, "steps", 12)
            if hi is not None and getattr(hi, 'lightning', False):
                # Lightning defaults only set sampler/scheduler if unspecified; do not force steps
                if not hsam or str(hsam).lower() in ("auto", "automatic"):
                    hsam = "Euler"
                if not hsch or str(hsch).lower() in ("auto", "automatic"):
                    hsch = "Simple"
            hcfg = getattr(hi, "cfg_scale", getattr(params, "guidance_scale", None)) if hi else getattr(params, "guidance_scale", None)
            active_pipe_hi = high_model or pipe
            # Apply per-stage LoRA if provided
            try:
                if hi is not None and getattr(hi, 'lora_path', None):
                    self._loader.apply_lora(active_pipe_hi, getattr(hi, 'lora_path'), getattr(hi, 'lora_weight', None))  # type: ignore[union-attr]
            except Exception as _ex:
                self._logger.warning('LoRA application failed (high stage): %s', _ex)
            outcome = apply_sampler_scheduler(active_pipe_hi, str(hsam or "Automatic"), str(hsch or "Automatic"))
            for w in outcome.warnings:
                self._logger.warning("wan_i2v_14b: %s", w)
            yield ProgressEvent(stage="run_high", percent=5.0, message="Stage 1 (High Noise)")
            out_hi = active_pipe_hi(
                image=getattr(params, "init_image", None),
                prompt=request.prompt,
                negative_prompt=request.negative_prompt or None,
                num_frames=int(getattr(params, "num_frames", 16) or 16),
                num_inference_steps=max(1, int(hstp or 12)),
                height=int(getattr(params, "height", 432) or 432),
                width=int(getattr(params, "width", 768) or 768),
                guidance_scale=hcfg,
            )
            frames_hi = list(out_hi.frames[0]) if hasattr(out_hi, "frames") else []

            # Stage 2 (seed with last frame of Stage 1)
            lo = getattr(params, "low_stage", None)
            if lo is not None and frames_hi:
                lsam = getattr(lo, "sampler", getattr(params, "sampler", "Automatic"))
                lsch = getattr(lo, "scheduler", getattr(params, "scheduler", "Automatic"))
                lstp = getattr(lo, "steps", getattr(params, "steps", 12))
                if getattr(lo, 'lightning', False):
                    if not lsam or str(lsam).lower() in ("auto", "automatic"):
                        lsam = "Euler"
                    if not lsch or str(lsch).lower() in ("auto", "automatic"):
                        lsch = "Simple"
                lcfg = getattr(lo, "cfg_scale", getattr(params, "guidance_scale", None))
                active_pipe_lo = low_model or pipe
                # Apply LoRA for low stage if provided
                try:
                    if lo is not None and getattr(lo, 'lora_path', None):
                        self._loader.apply_lora(active_pipe_lo, getattr(lo, 'lora_path'), getattr(lo, 'lora_weight', None))  # type: ignore[union-attr]
                except Exception as _ex:
                    self._logger.warning('LoRA application failed (low stage): %s', _ex)
                outcome = apply_sampler_scheduler(active_pipe_lo, str(lsam or "Automatic"), str(lsch or "Automatic"))
                for w in outcome.warnings:
                    self._logger.warning("wan_i2v_14b: %s", w)
                yield ProgressEvent(stage="run_low", percent=50.0, message="Stage 2 (Low Noise)")
                seed_image = frames_hi[-1]
                out_lo = active_pipe_lo(
                    image=seed_image,
                    prompt=request.prompt,
                    negative_prompt=request.negative_prompt or None,
                    num_frames=int(getattr(params, "num_frames", 16) or 16),
                    num_inference_steps=max(1, int(lstp or 12)),
                    height=int(getattr(params, "height", 432) or 432),
                    width=int(getattr(params, "width", 768) or 768),
                    guidance_scale=lcfg,
                )
                frames = list(out_lo.frames[0]) if hasattr(out_lo, "frames") else frames_hi
            else:
                frames = frames_hi

            # Optional interpolation (VFI)
            vfi_opts = None
            try:
                vi = (getattr(request, 'extras', {}) or {}).get('video_interpolation')
                if isinstance(vi, dict):
                    vfi = VideoInterpolationOptions(
                        enabled=bool(vi.get('enabled', False)),
                        model=str(vi.get('model')) if vi.get('model') is not None else None,
                        times=int(vi.get('times')) if vi.get('times') is not None else None,
                    )
                    vfi_opts = vfi.as_dict()
                    if vfi.enabled and (vfi.times or 0) > 1:
                        yield ProgressEvent(stage="interpolate", percent=2.0, message="Interpolating frames (VFI)")
                        frames, vfi_meta = maybe_interpolate(frames, enabled=vfi.enabled, model=vfi.model, times=vfi.times or 2, logger=self._logger)
                        vfi_opts = {**vfi_opts, **{"result": vfi_meta}}
            except Exception:
                vfi_opts = None

            # Export video (honor options if provided)
            vid_opts = None
            try:
                v = (getattr(request, 'extras', {}) or {}).get('video')
                if isinstance(v, dict):
                    from apps.server.backend.core.params.video import VideoExportOptions
                    vid_opts = VideoExportOptions(
                        filename_prefix=v.get('video_filename_prefix') or v.get('filename_prefix'),
                        format=v.get('video_format') or v.get('format'),
                        pix_fmt=v.get('video_pix_fmt') or v.get('pix_fmt'),
                        crf=int(v.get('video_crf') or v.get('crf') or 0) or None,
                        loop_count=int(v.get('video_loop_count') or v.get('loop_count') or 0) or None,
                        pingpong=bool(v.get('video_pingpong') if v.get('video_pingpong') is not None else v.get('pingpong')) if 'video_pingpong' in v or 'pingpong' in v else None,
                        save_metadata=bool(v.get('video_save_metadata')) if 'video_save_metadata' in v else None,
                        save_output=bool(v.get('video_save_output')) if 'video_save_output' in v else None,
                        trim_to_audio=bool(v.get('video_trim_to_audio')) if 'video_trim_to_audio' in v else None,
                    ).as_dict()
            except Exception:
                vid_opts = None
            fps = int(getattr(request, "fps", 24) or 24)
            video_meta = self._maybe_export_video(frames, fps=fps, options=vid_opts)
            info = {
                "engine": self.engine_id,
                "task": "img2vid",
                "frames": len(frames),
                "video": video_meta,
                "video_options": vid_opts,
                "video_interpolation": vfi_opts,
                "wan_high": getattr(params, "high_stage", None).as_dict() if getattr(params, "high_stage", None) else None,
                "wan_low": getattr(params, "low_stage", None).as_dict() if getattr(params, "low_stage", None) else None,
            }
            yield ResultEvent(payload={"images": frames, "info": self._to_json(info)})
        except Exception as exc:  # noqa: BLE001
            raise EngineExecutionError(str(exc)) from exc

    def load(self, model_ref: str, **options: object) -> None:  # type: ignore[override]
        if not model_ref:
            raise EngineLoadError("wan_i2v_14b load failed: empty model_ref")
        self._loader = WanLoader(self._logger)
        comp = self._loader.load(model_ref, device=str(options.get("device", "auto")), dtype=str(options.get("dtype", "fp16")))
        self._comp = comp
        self.mark_loaded()

    def _ensure_components(self) -> WanComponents:
        if self._comp is None or self._loader is None:
            raise EngineLoadError("WAN components not loaded; call load(model_ref) first")
        return self._comp

    @staticmethod
    def _resolve_mode(request: object) -> Mode:
        try:
            meta = getattr(request, "metadata", {})
            mode_value = meta.get("mode", "Normal") if isinstance(meta, dict) else "Normal"
            return Mode(mode_value)  # type: ignore[arg-type]
        except Exception:
            return Mode.NORMAL
