from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, Img2VidRequest, ProgressEvent, ResultEvent, Txt2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.use_cases.img2vid import run_img2vid as _run_i2v
from apps.backend.use_cases.txt2vid import run_txt2vid as _run_t2v
from apps.backend.core.exceptions import EngineLoadError
from apps.backend.core.progress_stream import stream_run
from apps.backend.engines.wan22.spec import WAN_14B_SPEC, WanEngineRuntime, assemble_wan_runtime
from apps.backend.engines.wan22.wan22_common import (
    EngineOpts,
    WanComponents,
    resolve_wan_repo_candidates,
    resolve_user_supplied_assets,
    _first_existing_path_for,
)
from apps.backend.huggingface.assets import ensure_repo_minimal_files
from apps.backend.runtime.wan22.sampler import sample_txt2vid as wan_sample_txt2vid
from apps.backend.runtime.workflows import build_video_plan, assemble_video_metadata, pil_to_tensor
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.codex import options as codex_options

REPO_ROOT = Path(__file__).resolve().parents[4]
HF_ROOT = REPO_ROOT / "apps" / "backend" / "huggingface"


class Wan2214BEngine(BaseVideoEngine):
    engine_id = "wan22_14b"

    def __init__(self) -> None:
        super().__init__()
        self._runtime_spec: Optional[WanEngineRuntime] = None
        self._comp: Optional[WanComponents] = None
        self._opts = EngineOpts()

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("wan-2.2-14b",),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": "WAN 2.2 14B via Diffusers/GGUF (default) or experimental Codex runtime/spec"},
        )

    # ------------------------------ lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.info('[wan22_14b] DEBUG: antes de função load')
        bundle = options.pop("_bundle", None)
        dev = str(options.get("device", "auto"))
        dty = str(options.get("dtype", "fp16"))
        self._opts = EngineOpts(device=dev, dtype=dty)
        self._runtime_spec = None

        # Runtime/spec is opt-in via options/UI; env toggles are deprecated.
        use_spec_runtime_opt = options.pop("use_codex_runtime", None)
        if use_spec_runtime_opt is None:
            snap = codex_options.get_snapshot()
            use_spec_runtime_opt = getattr(snap, "codex_wan22_use_spec_runtime", False)
        use_spec_runtime = bool(use_spec_runtime_opt)
        if bundle is not None and use_spec_runtime:
            try:
                if not isinstance(bundle, DiffusionModelBundle):
                    raise TypeError(
                        f"WAN22 14B: _bundle must be DiffusionModelBundle, got {type(bundle).__name__}"
                    )
                self._logger.info("WAN22 14B: assembling WanEngineRuntime from DiffusionModelBundle (experimental)")
                runtime = assemble_wan_runtime(
                    spec=WAN_14B_SPEC,
                    codex_components=bundle.components,
                    estimated_config=bundle.estimated_config,
                    device=(dev if dev != "auto" else "cuda"),
                    dtype=dty,
                )
                self._runtime_spec = runtime
                self._comp = None
                self.mark_loaded()
                # NOTE: tasks will explicitly raise NotImplementedError when this experimental
                # runtime path is active, until txt2vid/img2vid wiring is completed.
                return
            except Exception as exc:  # noqa: BLE001
                self._logger.error(
                    "WAN22 14B: failed to assemble Codex runtime from bundle; falling back to legacy path: %s",
                    exc,
                )
                self._runtime_spec = None

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
            from apps.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as apply_attn  # type: ignore
            from apps.backend.engines.util.accelerator import apply_to_diffusers_pipeline as apply_accel  # type: ignore
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
        self._runtime_spec = None
        self._comp = None
        self._logger.info('[wan22_14b] DEBUG: depois de função unload')
        self.mark_unloaded()

    # ------------------------------ runtime/spec helpers
    def _txt2vid_runtime_spec(self, request: Txt2VidRequest) -> Iterator[InferenceEvent]:
        import torch
        from PIL import Image

        runtime = self._runtime_spec
        if runtime is None:
            raise RuntimeError("WAN22 runtime/spec is not initialised; load() with bundle+options first.")

        start = time.perf_counter()

        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid (WAN22 spec runtime)")

        plan = build_video_plan(request)

        width = int(getattr(request, "width", plan.width))
        height = int(getattr(request, "height", plan.height))
        num_frames = int(getattr(request, "num_frames", plan.frames))
        steps = int(plan.steps or WAN_14B_SPEC.default_steps)

        raw_guidance = getattr(request, "guidance_scale", None)
        cfg_scale = float(raw_guidance) if raw_guidance is not None else float(WAN_14B_SPEC.default_cfg_scale)
        flow_shift = float(WAN_14B_SPEC.flow_shift)
        seed = getattr(request, "seed", None)

        device_name = (runtime.device or "cuda").strip().lower()
        if device_name == "cuda" and not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
            device_name = "cpu"
        device = torch.device(device_name)

        dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": getattr(torch, "bfloat16", torch.float16),
            "bfloat16": getattr(torch, "bfloat16", torch.float16),
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype_key = str(getattr(runtime, "dtype", "bf16")).strip().lower()
        torch_dtype = dtype_map.get(dtype_key, getattr(torch, "bfloat16", torch.float16))

        t5 = runtime.text.t5_text
        prompt = getattr(request, "prompt", "") or ""
        negative_prompt = getattr(request, "negative_prompt", "") or ""

        cond_all = t5([prompt])
        uncond_all = t5([negative_prompt]) if negative_prompt else None

        if cond_all.ndim != 3:
            raise RuntimeError(
                f"WAN22 T5 conditioning must be 3D [chunks, tokens, dim]; got shape={tuple(cond_all.shape)}",
            )
        cond = cond_all[:1].to(device=device, dtype=torch_dtype)
        uncond = None
        if uncond_all is not None:
            if uncond_all.ndim != 3:
                raise RuntimeError(
                    f"WAN22 T5 negative conditioning must be 3D [chunks, tokens, dim]; got shape={tuple(uncond_all.shape)}",
                )
            uncond = uncond_all[:1].to(device=device, dtype=torch_dtype)

        core_model = runtime.unet.model.diffusion_model
        core_model = core_model.to(device=device, dtype=torch_dtype)
        vae = runtime.vae

        yield ProgressEvent(stage="sample", percent=0.1, message="Sampling video latents (WAN22 spec runtime)")

        video = wan_sample_txt2vid(
            transformer=core_model,
            vae=vae,
            cond=cond,
            uncond=uncond,
            width=width,
            height=height,
            num_frames=num_frames,
            num_steps=steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            seed=seed,
            device=device,
            dtype=torch_dtype,
        )

        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)

        if video.ndim != 5:
            raise RuntimeError(
                f"WAN22 sampler expected tensor [B,C,T,H,W], got shape={tuple(video.shape)}",
            )

        yield ProgressEvent(stage="decode", percent=0.8, message="Decoding frames (WAN22 spec runtime)")

        video_cpu = video[0].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
        if video_cpu.ndim != 4:
            raise RuntimeError(
                f"WAN22 sampler output rank mismatch; expected [C,T,H,W], got shape={tuple(video_cpu.shape)}",
            )
        c_dim, t_dim, h_dim, w_dim = video_cpu.shape
        if c_dim not in (1, 3, 4):
            raise RuntimeError(
                f"WAN22 sampler produced unsupported channel count C={c_dim}; expected 1, 3, or 4.",
            )

        frames = []
        for i in range(t_dim):
            frame = video_cpu[:, i, :, :]
            if frame.ndim != 3:
                raise RuntimeError(
                    f"WAN22 frame tensor rank mismatch; expected [C,H,W], got shape={tuple(frame.shape)}",
                )
            arr = (frame.movedim(0, -1).numpy() * 255.0).round().clip(0, 255).astype("uint8")
            frames.append(Image.fromarray(arr))

        extras = getattr(request, "extras", {}) or {}
        video_opts = extras.get("video") if isinstance(extras, dict) else None
        video_meta = self._maybe_export_video(frames, fps=int(getattr(request, "fps", plan.fps)), options=video_opts)

        elapsed = time.perf_counter() - start
        metadata = assemble_video_metadata(
            self,
            plan,
            None,
            elapsed=elapsed,
            frame_count=len(frames),
            task="txt2vid",
            extra={
                "runtime": "wan22_spec",
                "steps": steps,
                "guidance_scale": cfg_scale,
            },
            video_meta=video_meta,
        )

        payload = {
            "images": frames,
            "info": self._to_json(metadata),
        }
        self._logger.info(
            "[wan22_14b] txt2vid runtime/spec completed: frames=%d, steps=%d, cfg=%.2f, device=%s, dtype=%s",
            len(frames),
            steps,
            cfg_scale,
            str(device),
            str(torch_dtype),
        )
        yield ResultEvent(payload=payload)
        self._logger.info('[wan22_14b] DEBUG: depois de função txt2vid (runtime/spec)')

    def _img2vid_runtime_spec(self, request: Img2VidRequest) -> Iterator[InferenceEvent]:
        import torch
        from PIL import Image

        runtime = self._runtime_spec
        if runtime is None:
            raise RuntimeError("WAN22 runtime/spec is not initialised; load() with bundle+options first.")
        if getattr(request, "init_image", None) is None:
            raise RuntimeError("img2vid requires 'init_image' when using WAN22 runtime/spec.")

        start = time.perf_counter()

        yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid (WAN22 spec runtime)")

        plan = build_video_plan(request)

        width = int(getattr(request, "width", plan.width))
        height = int(getattr(request, "height", plan.height))
        num_frames = int(getattr(request, "num_frames", plan.frames))
        steps = int(plan.steps or WAN_14B_SPEC.default_steps)

        raw_guidance = getattr(request, "guidance_scale", None)
        cfg_scale = float(raw_guidance) if raw_guidance is not None else float(WAN_14B_SPEC.default_cfg_scale)
        flow_shift = float(WAN_14B_SPEC.flow_shift)
        seed = getattr(request, "seed", None)

        device_name = (runtime.device or "cuda").strip().lower()
        if device_name == "cuda" and not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
            device_name = "cpu"
        device = torch.device(device_name)

        dtype_map = {
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": getattr(torch, "bfloat16", torch.float16),
            "bfloat16": getattr(torch, "bfloat16", torch.float16),
            "fp32": torch.float32,
            "float32": torch.float32,
        }
        dtype_key = str(getattr(runtime, "dtype", "bf16")).strip().lower()
        torch_dtype = dtype_map.get(dtype_key, getattr(torch, "bfloat16", torch.float16))

        t5 = runtime.text.t5_text
        prompt = getattr(request, "prompt", "") or ""
        negative_prompt = getattr(request, "negative_prompt", "") or ""

        cond_all = t5([prompt])
        uncond_all = t5([negative_prompt]) if negative_prompt else None

        if cond_all.ndim != 3:
            raise RuntimeError(
                f"WAN22 T5 conditioning must be 3D [chunks, tokens, dim]; got shape={tuple(cond_all.shape)}",
            )
        cond = cond_all[:1].to(device=device, dtype=torch_dtype)
        uncond = None
        if uncond_all is not None:
            if uncond_all.ndim != 3:
                raise RuntimeError(
                    f"WAN22 T5 negative conditioning must be 3D [chunks, tokens, dim]; got shape={tuple(uncond_all.shape)}",
                )
            uncond = uncond_all[:1].to(device=device, dtype=torch_dtype)

        core_model = runtime.unet.model.diffusion_model
        core_model = core_model.to(device=device, dtype=torch_dtype)
        vae = runtime.vae

        init_image = getattr(request, "init_image")
        if isinstance(init_image, Image.Image):
            init_pil = init_image
        else:
            raise RuntimeError(f"img2vid runtime/spec expects init_image as PIL.Image; got {type(init_image)!r}")

        init_tensor = pil_to_tensor([init_pil]).to(device=device, dtype=torch_dtype)
        with torch.inference_mode():
            # Encode once and tile across time dimension.
            latents_4d = vae.encode_inner(init_tensor.movedim(1, 3))
            if not isinstance(latents_4d, torch.Tensor):
                latents_4d = torch.as_tensor(latents_4d)
            b, c, h_lat, w_lat = latents_4d.shape
            latents_5d = latents_4d.view(b, c, 1, h_lat, w_lat).repeat(1, 1, num_frames, 1, 1)

        yield ProgressEvent(stage="sample", percent=0.1, message="Sampling video latents (WAN22 spec runtime, img2vid)")

        video = wan_sample_txt2vid(
            transformer=core_model,
            vae=vae,
            cond=cond,
            uncond=uncond,
            width=width,
            height=height,
            num_frames=num_frames,
            num_steps=steps,
            cfg_scale=cfg_scale,
            flow_shift=flow_shift,
            seed=seed,
            device=device,
            dtype=torch_dtype,
        )

        if not isinstance(video, torch.Tensor):
            video = torch.as_tensor(video)

        if video.ndim != 5:
            raise RuntimeError(
                f"WAN22 sampler expected tensor [B,C,T,H,W], got shape={tuple(video.shape)}",
            )

        yield ProgressEvent(stage="decode", percent=0.8, message="Decoding frames (WAN22 spec runtime, img2vid)")

        video_cpu = video[0].detach().to(device="cpu", dtype=torch.float32).clamp(0.0, 1.0)
        if video_cpu.ndim != 4:
            raise RuntimeError(
                f"WAN22 sampler output rank mismatch; expected [C,T,H,W], got shape={tuple(video_cpu.shape)}",
            )
        c_dim, t_dim, h_dim, w_dim = video_cpu.shape
        if c_dim not in (1, 3, 4):
            raise RuntimeError(
                f"WAN22 sampler produced unsupported channel count C={c_dim}; expected 1, 3, or 4.",
            )

        frames = []
        for i in range(t_dim):
            frame = video_cpu[:, i, :, :]
            if frame.ndim != 3:
                raise RuntimeError(
                    f"WAN22 frame tensor rank mismatch; expected [C,H,W], got shape={tuple(frame.shape)}",
                )
            arr = (frame.movedim(0, -1).numpy() * 255.0).round().clip(0, 255).astype("uint8")
            frames.append(Image.fromarray(arr))

        extras = getattr(request, "extras", {}) or {}
        video_opts = extras.get("video") if isinstance(extras, dict) else None
        video_meta = self._maybe_export_video(frames, fps=int(getattr(request, "fps", plan.fps)), options=video_opts)

        elapsed = time.perf_counter() - start
        metadata = assemble_video_metadata(
            self,
            plan,
            None,
            elapsed=elapsed,
            frame_count=len(frames),
            task="img2vid",
            extra={
                "runtime": "wan22_spec",
                "steps": steps,
                "guidance_scale": cfg_scale,
            },
            video_meta=video_meta,
        )

        payload = {
            "images": frames,
            "info": self._to_json(metadata),
        }
        self._logger.info(
            "[wan22_14b] img2vid runtime/spec completed: frames=%d, steps=%d, cfg=%.2f, device=%s, dtype=%s",
            len(frames),
            steps,
            cfg_scale,
            str(device),
            str(torch_dtype),
        )
        yield ResultEvent(payload=payload)
        self._logger.info('[wan22_14b] DEBUG: depois de função img2vid (runtime/spec)')

    # ------------------------------ tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.info('[wan22_14b] DEBUG: antes de função txt2vid')
        self.ensure_loaded()
        if self._runtime_spec is not None:
            yield from self._txt2vid_runtime_spec(request)
            return

        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_t2v(engine=self, comp=self._comp, request=request)
        else:
            # GGUF runtime
            from apps.backend.runtime.nn import wan22 as gguf
            yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid (GGUF)")
            ex = getattr(request, 'extras', {}) or {}
            vae_path, te_path, meta_dir = resolve_user_supplied_assets(ex, self._comp.hf_repo_dir)
            if not vae_path:
                vae_path = _first_existing_path_for("wan22_vae")
            if not te_path:
                te_path = _first_existing_path_for("wan22_tenc")
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
                flow_shift = ex_stage.get('flow_shift')
                flow_shift = float(flow_shift) if flow_shift is not None else None
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale, flow_shift=flow_shift,
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
                flow_shift = ex_stage.get('flow_shift')
                flow_shift = float(flow_shift) if flow_shift is not None else None
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale, flow_shift=flow_shift,
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
                flow_shift = ex_stage.get('flow_shift')
                flow_shift = float(flow_shift) if flow_shift is not None else None
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale, flow_shift=flow_shift,
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
                flow_shift = ex_stage.get('flow_shift')
                flow_shift = float(flow_shift) if flow_shift is not None else None
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale, flow_shift=flow_shift,
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
        if self._runtime_spec is not None:
            yield from self._img2vid_runtime_spec(request)
            return
        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_i2v(engine=self, comp=self._comp, request=request)
        else:
            from apps.backend.runtime.nn import wan22 as gguf
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
                flow_shift = ex_stage.get('flow_shift')
                flow_shift = float(flow_shift) if flow_shift is not None else None
                return gguf.StageConfig(
                    model_dir=model_p or '', sampler=sampler, scheduler=scheduler,
                    steps=steps, cfg_scale=cfg_scale, flow_shift=flow_shift,
                )
            # Env-only controls for offload/TE implementation (img2vid path)
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
