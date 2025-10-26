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
from apps.server.backend.core.progress_stream import stream_run
from .wan22_common import EngineOpts, WanComponents, resolve_wan_repo_candidates, resolve_user_supplied_assets

PROJECT_ROOT = Path(__file__).resolve().parents[5]
HF_ROOT = PROJECT_ROOT / "apps" / "server" / "backend" / "huggingface"


class Wan2214BEngine(BaseVideoEngine):
    engine_id = "wan22_14b"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None
        self._opts = EngineOpts()

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("wan-2.2-14b",),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": "WAN 2.2 14B via Diffusers or GGUF"},
        )

    # ------------------------------ lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.info('[wan22_14b] DEBUG: antes de função load')
        dev = str(options.get("device", "auto"))
        dty = str(options.get("dtype", "fp16"))
        self._opts = EngineOpts(device=dev, dtype=dty)
        comp = WanComponents()
        # Resolve path
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
                raise EngineLoadError(f"WAN22 14B model path not found: {model_ref}")

        # Ensure assets are available under backend/huggingface
        vendor_dir: Optional[Path] = None
        last_exc: Optional[Exception] = None
        for rid in resolve_wan_repo_candidates(self.engine_id):
            local_dir = HF_ROOT / Path(rid.replace('/', os.sep))
            try:
                # We don’t know yet if user supplied a Diffusers dir or GGUF dir; prepare minimal assets.
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
                    "WAN22 14B: failed to fetch minimal HF assets from %s: %s", rid, exc
                )
        if vendor_dir is None and last_exc is not None:
            raise EngineLoadError(
                f"WAN22 14B: unable to prepare required HF assets; last error: {last_exc}"
            )

        # Decide path: Diffusers directory vs GGUF directory
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
        # Strict device policy: CPU only if explicitly chosen. Otherwise require CUDA.
        try:
            import torch
            cuda_ok = bool(getattr(torch, 'cuda', None) and torch.cuda.is_available())
        except Exception:
            cuda_ok = False
        dev_lc = (dev or 'auto').lower().strip()
        if dev_lc == 'cpu':
            comp.device = 'cpu'
        else:
            if not cuda_ok:
                raise EngineLoadError("CUDA is not available; set device='cpu' explicitly to force CPU.")
            comp.device = 'cuda'
        comp.dtype = dty

        if _is_diffusers_dir(p):
            # Diffusers pipeline
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
            self._logger.info("WAN22 14B diffusers pipeline loaded: %s", p)
        else:
            # GGUF path: tokenizer/config live under backend/huggingface; weights are user-provided.
            # comp.hf_* were already set (config/tokenizer). Do not attempt to fetch any weights here.
            comp.pipeline = None  # explicit: non-diffusers runtime
            self._logger.info("WAN22 14B GGUF runtime selected for %s", p)

        self._comp = comp
        self._logger.info('[wan22_14b] DEBUG: depois de função load')
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._logger.info('[wan22_14b] DEBUG: antes de função unload')
        self._comp = None
        self._logger.info('[wan22_14b] DEBUG: depois de função unload')
        self.mark_unloaded()

    # ------------------------------ tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.info('[wan22_14b] DEBUG: antes de função txt2vid')
        self.ensure_loaded()
        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_t2v(engine=self, comp=self._comp, request=request)
        else:
            # GGUF runtime
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
            # Stage config helpers (prefer per-stage extras when provided)
            def _as_dir(p: Optional[str]) -> Optional[str]:
                try:
                    if p and os.path.isfile(p):
                        return os.path.dirname(p)
                except Exception:
                    pass
                return p
            ex_high = ex.get('wan_high', {}) if isinstance(ex, dict) else {}
            ex_low = ex.get('wan_low', {}) if isinstance(ex, dict) else {}
            def _stage_from(ex_stage: Dict[str, Any]) -> gguf.StageConfig:
                model_p = _as_dir(str(ex_stage.get('model_dir') or (self._comp.model_dir or '')))
                sampler = str(ex_stage.get('sampler') or getattr(request, 'sampler', 'Euler a'))
                scheduler = str(ex_stage.get('scheduler') or getattr(request, 'scheduler', 'Automatic'))
                steps = max(1, int(ex_stage.get('steps') or getattr(request, 'steps', 12) or 12))
                cfg_scale = ex_stage.get('cfg_scale', getattr(request, 'cfg_scale', None))
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale,
                )

            # Stage config helpers (prefer per-stage extras when provided)
            def _as_dir(p: Optional[str]) -> Optional[str]:
                try:
                    if p and os.path.isfile(p):
                        return os.path.dirname(p)
                except Exception:
                    pass
                return p
            ex_high = ex.get('wan_high', {}) if isinstance(ex, dict) else {}
            ex_low = ex.get('wan_low', {}) if isinstance(ex, dict) else {}
            def _stage_from(ex_stage: Dict[str, Any]) -> gguf.StageConfig:
                model_p = _as_dir(str(ex_stage.get('model_dir') or (self._comp.model_dir or '')))
                sampler = str(ex_stage.get('sampler') or getattr(request, 'sampler', 'Euler a'))
                scheduler = str(ex_stage.get('scheduler') or getattr(request, 'scheduler', 'Automatic'))
                steps = max(1, int(ex_stage.get('steps') or getattr(request, 'steps', 12) or 12))
                cfg_scale = ex_stage.get('cfg_scale', getattr(request, 'cfg_scale', None))
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale,
                )

            # Stage config helpers (prefer per-stage extras when provided)
            def _as_dir(p: Optional[str]) -> Optional[str]:
                try:
                    if p and os.path.isfile(p):
                        return os.path.dirname(p)
                except Exception:
                    pass
                return p
            ex_high = ex.get('wan_high', {}) if isinstance(ex, dict) else {}
            ex_low = ex.get('wan_low', {}) if isinstance(ex, dict) else {}
            def _stage_from(ex_stage: Dict[str, Any]) -> gguf.StageConfig:
                model_p = _as_dir(str(ex_stage.get('model_dir') or (self._comp.model_dir or '')))
                sampler = str(ex_stage.get('sampler') or getattr(request, 'sampler', 'Euler a'))
                scheduler = str(ex_stage.get('scheduler') or getattr(request, 'scheduler', 'Automatic'))
                steps = max(1, int(ex_stage.get('steps') or getattr(request, 'steps', 12) or 12))
                cfg_scale = ex_stage.get('cfg_scale', getattr(request, 'cfg_scale', None))
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale,
                )

            # Stage config helpers (prefer per-stage extras when provided)
            def _as_dir(p: Optional[str]) -> Optional[str]:
                try:
                    if p and os.path.isfile(p):
                        return os.path.dirname(p)
                except Exception:
                    pass
                return p
            ex_high = ex.get('wan_high', {}) if isinstance(ex, dict) else {}
            ex_low = ex.get('wan_low', {}) if isinstance(ex, dict) else {}
            def _stage_from(ex_stage: Dict[str, Any]) -> gguf.StageConfig:
                model_p = _as_dir(str(ex_stage.get('model_dir') or (self._comp.model_dir or '')))
                sampler = str(ex_stage.get('sampler') or getattr(request, 'sampler', 'Euler a'))
                scheduler = str(ex_stage.get('scheduler') or getattr(request, 'scheduler', 'Automatic'))
                steps = max(1, int(ex_stage.get('steps') or getattr(request, 'steps', 12) or 12))
                cfg_scale = ex_stage.get('cfg_scale', getattr(request, 'cfg_scale', None))
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale,
                )

            import os
            # Env-only controls: offload level and TE implementation
            env_offload_lvl = os.getenv('WAN_GGUF_OFFLOAD_LEVEL')
            try:
                offload_level_env = int(env_offload_lvl) if env_offload_lvl is not None else None
            except Exception:
                offload_level_env = None
            te_impl_env = (os.getenv('WAN_TE_IMPL', '').strip().lower() or None)
            te_device_val = 'cuda' if te_impl_env == 'cuda_fp8' else None
            te_kernel_required_val = True if te_impl_env == 'cuda_fp8' else None
            import os
            env_offload_lvl = os.getenv('WAN_GGUF_OFFLOAD_LEVEL')
            try:
                offload_level_env = int(env_offload_lvl) if env_offload_lvl is not None else None
            except Exception:
                offload_level_env = None
            te_kernel_required_env = None
            try:
                te_kernel_required_env = str(os.getenv('WAN_TE_KERNEL_REQUIRED', '0')).strip().lower() in ('1','true','yes','on')
            except Exception:
                te_kernel_required_env = None
            # Env-only controls for this path as well
            env_offload_lvl = os.getenv('WAN_GGUF_OFFLOAD_LEVEL')
            try:
                offload_level_env = int(env_offload_lvl) if env_offload_lvl is not None else None
            except Exception:
                offload_level_env = None
            te_impl_env = (os.getenv('WAN_TE_IMPL', '').strip().lower() or None)
            te_device_val = 'cuda' if te_impl_env == 'cuda_fp8' else None
            te_kernel_required_val = True if te_impl_env == 'cuda_fp8' else None
            cfg = gguf.RunConfig(
                width=int(getattr(request, 'width', 768) or 768),
                height=int(getattr(request, 'height', 432) or 432),
                fps=int(getattr(request, 'fps', 24) or 24),
                num_frames=int(getattr(request, 'num_frames', 16) or 16),
                guidance_scale=getattr(request, 'cfg_scale', None),
                dtype=self._opts.dtype,
                device=self._comp.device,
                seed=getattr(request, 'seed', None),
                prompt=request.prompt,
                negative_prompt=getattr(request, 'negative_prompt', None),
                vae_dir=vae_path,
                text_encoder_dir=te_path,
                metadata_dir=meta_dir,
                # Memory/attention controls (optional extras)
                sdpa_policy=(ex.get('gguf_sdpa_policy') if isinstance(ex, dict) else None),
                attn_chunk_size=(int(ex.get('gguf_attn_chunk', 0)) if isinstance(ex, dict) and ex.get('gguf_attn_chunk') else None),
                gguf_cache_policy=(ex.get('gguf_cache_policy') if isinstance(ex, dict) else None),
                gguf_cache_limit_mb=(int(ex.get('gguf_cache_limit_mb', 0)) if isinstance(ex, dict) and ex.get('gguf_cache_limit_mb') else None),
                log_mem_interval=(int(ex.get('gguf_log_mem_interval', 0)) if isinstance(ex, dict) and ex.get('gguf_log_mem_interval') else None),
                aggressive_offload=bool(ex.get('gguf_offload', True)) if isinstance(ex, dict) else True,
                offload_level=offload_level_env,
                te_device=te_device_val,
                te_impl=te_impl_env,
                te_kernel_required=te_kernel_required_val,
                high=_stage_from(ex_high),
                low=_stage_from(ex_low),
            )
            for ev in stream_run(gguf.run_txt2vid, cfg=cfg, logger=self._logger):
                if isinstance(ev, dict) and ev.get('type') == 'progress':
                    st = str(ev.get('stage', ''))
                    step = int(ev.get('step', 0)); total = int(ev.get('total', 0)); pct = float(ev.get('percent', 0.0))
                    try:
                        self._logger.info("[wan22.gguf] %s %d/%d (%.1f%%)", st, step, total, pct * 100.0)
                    except Exception:
                        pass
                    yield ProgressEvent(stage=st, percent=pct, step=step, total_steps=total)
                elif isinstance(ev, dict) and ev.get('type') == 'result':
                    frames = ev.get('frames', [])
                    yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "txt2vid", "frames": len(frames)})})
            self._logger.info('[wan22_14b] DEBUG: depois de função txt2vid')
            return
            yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "txt2vid", "frames": len(frames)})})

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.info('[wan22_14b] DEBUG: antes de função img2vid')
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
            # Stage config helpers (prefer per-stage extras when provided)
            def _as_dir(p: Optional[str]) -> Optional[str]:
                try:
                    if p and os.path.isfile(p):
                        return os.path.dirname(p)
                except Exception:
                    pass
                return p
            ex_high = ex.get('wan_high', {}) if isinstance(ex, dict) else {}
            ex_low = ex.get('wan_low', {}) if isinstance(ex, dict) else {}
            def _stage_from(ex_stage: Dict[str, Any]) -> gguf.StageConfig:
                model_p = _as_dir(str(ex_stage.get('model_dir') or (self._comp.model_dir or '')))
                sampler = str(ex_stage.get('sampler') or getattr(request, 'sampler', 'Euler a'))
                scheduler = str(ex_stage.get('scheduler') or getattr(request, 'scheduler', 'Automatic'))
                steps = max(1, int(ex_stage.get('steps') or getattr(request, 'steps', 12) or 12))
                cfg_scale = ex_stage.get('cfg_scale', getattr(request, 'cfg_scale', None))
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale,
                )
            cfg = gguf.RunConfig(
                width=int(getattr(request, 'width', 768) or 768),
                height=int(getattr(request, 'height', 432) or 432),
                fps=int(getattr(request, 'fps', 24) or 24),
                num_frames=int(getattr(request, 'num_frames', 16) or 16),
                guidance_scale=getattr(request, 'cfg_scale', None),
                dtype=self._opts.dtype,
                device=self._comp.device,
                seed=getattr(request, 'seed', None),
                prompt=request.prompt,
                negative_prompt=getattr(request, 'negative_prompt', None),
                init_image=getattr(request, 'init_image', None),
                vae_dir=vae_path,
                text_encoder_dir=te_path,
                metadata_dir=meta_dir,
                sdpa_policy=(ex.get('gguf_sdpa_policy') if isinstance(ex, dict) else None),
                attn_chunk_size=(int(ex.get('gguf_attn_chunk', 0)) if isinstance(ex, dict) and ex.get('gguf_attn_chunk') else None),
                gguf_cache_policy=(ex.get('gguf_cache_policy') if isinstance(ex, dict) else None),
                gguf_cache_limit_mb=(int(ex.get('gguf_cache_limit_mb', 0)) if isinstance(ex, dict) and ex.get('gguf_cache_limit_mb') else None),
                log_mem_interval=(int(ex.get('gguf_log_mem_interval', 0)) if isinstance(ex, dict) and ex.get('gguf_log_mem_interval') else None),
                offload_level=offload_level_env,
                te_device=te_device_val,
                te_impl=te_impl_env,
                te_kernel_required=te_kernel_required_val,
                high=_stage_from(ex_high),
                low=_stage_from(ex_low),
            )
            for ev in stream_run(gguf.run_img2vid, cfg=cfg, logger=self._logger):
                if isinstance(ev, dict) and ev.get('type') == 'progress':
                    st = str(ev.get('stage', ''))
                    step = int(ev.get('step', 0)); total = int(ev.get('total', 0)); pct = float(ev.get('percent', 0.0))
                    try:
                        self._logger.info("[wan22.gguf] %s %d/%d (%.1f%%)", st, step, total, pct * 100.0)
                    except Exception:
                        pass
                    yield ProgressEvent(stage=st, percent=pct, step=step, total_steps=total)
                elif isinstance(ev, dict) and ev.get('type') == 'result':
                    frames = ev.get('frames', [])
                    yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "img2vid", "frames": len(frames)})})
                elif isinstance(ev, dict) and ev.get('type') == 'error':
                    err = ev.get('error')
                    try:
                        self._logger.error("[wan22.gguf] runtime error during img2vid: %s", err)
                    except Exception:
                        pass
                    # Strict: surface the error to orchestrator/UI (no silent fallback)
                    raise RuntimeError(f"WAN22 GGUF runtime error: {err}")
            self._logger.info('[wan22_14b] DEBUG: depois de função img2vid')
            return

    # ------------------------------ helpers
    # No per-stage GGUF builder needed here; runtimes will read request.extras directly when needed
