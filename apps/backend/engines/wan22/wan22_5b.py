from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, Txt2VidRequest, Img2VidRequest, Vid2VidRequest, ProgressEvent, ResultEvent
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.use_cases.txt2vid import run_txt2vid as _run_t2v
from apps.backend.use_cases.img2vid import run_img2vid as _run_i2v
from apps.backend.use_cases.vid2vid import run_vid2vid as _run_v2v
from apps.backend.core.exceptions import EngineLoadError

import os

from apps.backend.huggingface.assets import ensure_repo_minimal_files
from apps.backend.core.progress_stream import stream_run
from apps.backend.engines.wan22.wan22_common import (
    EngineOpts,
    WanComponents,
    WanStageOptions,
    resolve_wan_repo_candidates,
    resolve_user_supplied_assets,
    _first_existing_path_for,
)
from apps.backend.engines.wan22.diffusers_loader import load_wan_diffusers_pipeline
from apps.backend.infra.config.repo_root import get_repo_root

REPO_ROOT = get_repo_root()
HF_ROOT = REPO_ROOT / "apps" / "backend" / "huggingface"

def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except Exception:
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _build_wan22_gguf_run_config(
    *,
    request: Txt2VidRequest | Img2VidRequest,
    comp: WanComponents,
    dtype: str,
    device: str,
    logger: Any,
) -> Any:
    """Build a WAN22 GGUF RunConfig from the request + extras.

    This is intentionally a pure "mapping" helper so we can test stage overrides
    without running the GGUF runtime.
    """
    from apps.backend.runtime.wan22 import wan22 as gguf

    ex_raw = getattr(request, "extras", {}) or {}
    ex: Dict[str, Any] = dict(ex_raw) if isinstance(ex_raw, dict) else {}

    # Resolve assets (strict when provided; fallback roots are applied only when empty).
    vae_path, te_path, meta_dir = resolve_user_supplied_assets(ex, comp.hf_repo_dir)
    if not vae_path:
        vae_path = _first_existing_path_for("wan22_vae")
    if not te_path:
        te_path = _first_existing_path_for("wan22_tenc")
    if not te_path:
        raise RuntimeError(
            "WAN22 GGUF requires a text encoder weights file; provide 'wan_text_encoder_path' "
            "or configure 'wan22_tenc' with a directory containing exactly one '.safetensors' file."
        )

    te_path = os.path.expanduser(str(te_path))
    if os.path.isdir(te_path):
        try:
            candidates = sorted(
                fn for fn in os.listdir(te_path) if fn.lower().endswith(".safetensors")
            )
        except Exception as exc:
            raise RuntimeError(f"WAN22 GGUF: failed to list text encoder dir '{te_path}': {exc}") from exc
        if len(candidates) == 1:
            te_path = os.path.join(te_path, candidates[0])
        elif len(candidates) == 0:
            raise RuntimeError(
                f"WAN22 GGUF: no '.safetensors' file found under text encoder dir '{te_path}'. "
                "Provide 'wan_text_encoder_path' explicitly."
            )
        else:
            raise RuntimeError(
                f"WAN22 GGUF: multiple '.safetensors' files found under text encoder dir '{te_path}': {candidates}. "
                "Provide 'wan_text_encoder_path' explicitly."
            )

    if not str(te_path).lower().endswith(".safetensors"):
        raise RuntimeError(
            f"WAN22 GGUF: 'wan_text_encoder_path' must be a '.safetensors' file, got: {te_path}"
        )
    if not os.path.isfile(te_path):
        raise RuntimeError(f"WAN22 GGUF: text encoder weights not found: {te_path}")

    # WAN stage overrides
    wh_raw = ex.get("wan_high") if isinstance(ex.get("wan_high"), dict) else None
    wl_raw = ex.get("wan_low") if isinstance(ex.get("wan_low"), dict) else None

    default_steps = int(getattr(request, "steps", 12) or 12)
    default_cfg = getattr(request, "guidance_scale", None)

    hi_opts = WanStageOptions.from_mapping(wh_raw, default_steps=default_steps, default_cfg=default_cfg)
    lo_opts = WanStageOptions.from_mapping(wl_raw, default_steps=default_steps, default_cfg=default_cfg)

    base_dir = str(getattr(comp, "model_dir", "") or "")
    hi_dir = str(hi_opts.model_dir or base_dir)
    lo_dir = str(lo_opts.model_dir or base_dir)

    # Seed is applied to the High stage only in the current GGUF runtime.
    seed = getattr(request, "seed", None)
    if isinstance(wh_raw, dict) and wh_raw.get("seed") is not None:
        seed_override = _coerce_int(wh_raw.get("seed"))
        if seed_override is not None:
            seed = seed_override

    hi_flow_shift = _coerce_float(wh_raw.get("flow_shift")) if isinstance(wh_raw, dict) else None
    lo_flow_shift = _coerce_float(wl_raw.get("flow_shift")) if isinstance(wl_raw, dict) else None

    sampler_fallback = str(getattr(request, "sampler", "Automatic") or "Automatic")
    scheduler_fallback = str(getattr(request, "scheduler", "Automatic") or "Automatic")

    tokenizer_dir = str(ex.get("wan_tokenizer_dir")).strip() if ex.get("wan_tokenizer_dir") else None

    offload_level = _coerce_int(ex.get("gguf_offload_level"))
    if offload_level is not None and offload_level < 0:
        offload_level = None

    if logger is not None:
        try:
            logger.info(
                "[wan22.gguf] assets: metadata=%s te=%s vae=%s",
                os.path.basename(str(meta_dir)) if meta_dir else None,
                os.path.basename(str(te_path)) if te_path else None,
                os.path.basename(str(vae_path)) if vae_path else None,
            )
        except Exception:
            pass
        try:
            logger.info(
                "[wan22.gguf] stage overrides: high=(dir=%s steps=%s cfg=%s sampler=%s scheduler=%s seed=%s) low=(dir=%s steps=%s cfg=%s sampler=%s scheduler=%s)",
                os.path.basename(hi_dir) if hi_dir else None,
                int(hi_opts.steps),
                hi_opts.cfg_scale,
                hi_opts.sampler or sampler_fallback,
                hi_opts.scheduler or scheduler_fallback,
                seed,
                os.path.basename(lo_dir) if lo_dir else None,
                int(lo_opts.steps),
                lo_opts.cfg_scale,
                lo_opts.sampler or sampler_fallback,
                lo_opts.scheduler or scheduler_fallback,
            )
        except Exception:
            pass

    return gguf.RunConfig(
        width=int(getattr(request, "width", 768) or 768),
        height=int(getattr(request, "height", 432) or 432),
        fps=int(getattr(request, "fps", 24) or 24),
        num_frames=int(getattr(request, "num_frames", 16) or 16),
        guidance_scale=getattr(request, "guidance_scale", None),
        dtype=dtype,
        device=device,
        seed=seed,
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        init_image=getattr(request, "init_image", None),
        vae_dir=vae_path,
        text_encoder_dir=te_path,
        tokenizer_dir=tokenizer_dir,
        metadata_dir=meta_dir,
        sdpa_policy=(ex.get("gguf_sdpa_policy") if ex.get("gguf_sdpa_policy") is not None else None),
        attn_chunk_size=(
            int(ex.get("gguf_attn_chunk", 0)) if ex.get("gguf_attn_chunk") not in (None, "", 0) else None
        ),
        gguf_cache_policy=(ex.get("gguf_cache_policy") if ex.get("gguf_cache_policy") is not None else None),
        gguf_cache_limit_mb=(
            int(ex.get("gguf_cache_limit_mb", 0)) if ex.get("gguf_cache_limit_mb") not in (None, "", 0) else None
        ),
        log_mem_interval=(
            int(ex.get("gguf_log_mem_interval", 0)) if ex.get("gguf_log_mem_interval") not in (None, "", 0) else None
        ),
        aggressive_offload=bool(ex.get("gguf_offload", True)),
        offload_level=offload_level,
        te_device=(str(ex.get("gguf_te_device")).lower() if ex.get("gguf_te_device") is not None else None),
        te_impl=(str(ex.get("gguf_te_impl")).lower() if ex.get("gguf_te_impl") is not None else None),
        te_kernel_required=bool(ex.get("gguf_te_kernel_required", False)),
        high=gguf.StageConfig(
            model_dir=hi_dir,
            sampler=str(hi_opts.sampler or sampler_fallback),
            scheduler=str(hi_opts.scheduler or scheduler_fallback),
            steps=max(1, int(hi_opts.steps)),
            cfg_scale=hi_opts.cfg_scale,
            flow_shift=hi_flow_shift,
        ),
        low=gguf.StageConfig(
            model_dir=lo_dir,
            sampler=str(lo_opts.sampler or sampler_fallback),
            scheduler=str(lo_opts.scheduler or scheduler_fallback),
            steps=max(1, int(lo_opts.steps)),
            cfg_scale=lo_opts.cfg_scale,
            flow_shift=lo_flow_shift,
        ),
    )


class Wan225BEngine(BaseVideoEngine):
    engine_id = "wan22_5b"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None
        self._opts = EngineOpts()

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
        self._logger.info('[wan22_5b] DEBUG: antes de função load')
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

        if _is_diffusers_dir(p) or _looks_like_wan_diffusers_weights_dir(p):
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
            # GGUF path: tokenizer/config under backend/huggingface; all weights supplied by user.
            comp.pipeline = None
            self._logger.info("WAN22 5B GGUF runtime selected for %s (device=%s dtype=%s)", p, comp.device, dty)

        self._comp = comp
        self._logger.info('[wan22_5b] DEBUG: depois de função load')
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._logger.info('[wan22_5b] DEBUG: antes de função unload')
        self._comp = None
        self._logger.info('[wan22_5b] DEBUG: depois de função unload')
        self.mark_unloaded()

    # ------------------------------ tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.info('[wan22_5b] DEBUG: antes de função txt2vid')
        self.ensure_loaded()
        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_t2v(engine=self, comp=self._comp, request=request)
        else:
            from apps.backend.runtime.wan22 import wan22 as gguf
            yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing txt2vid (GGUF)")
            cfg = _build_wan22_gguf_run_config(
                request=request,
                comp=self._comp,
                dtype=self._opts.dtype,
                device=self._comp.device,
                logger=self._logger,
            )
            for ev in stream_run(gguf.run_txt2vid, cfg=cfg, logger=self._logger):
                if isinstance(ev, dict) and ev.get('type') == 'progress':
                    st = str(ev.get('stage', ''))
                    step = int(ev.get('step', 0)); total = int(ev.get('total', 0)); pct = float(ev.get('percent', 0.0))
                    pct_out = (pct * 100.0) if (0.0 <= pct <= 1.0) else pct
                    try:
                        self._logger.info("[wan22.gguf] %s %d/%d (%.1f%%)", st, step, total, pct_out)
                    except Exception:
                        pass
                    yield ProgressEvent(stage=st, percent=pct_out, step=step, total_steps=total)
                elif isinstance(ev, dict) and ev.get('type') == 'result':
                    frames = ev.get('frames', [])
                    yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "txt2vid", "frames": len(frames)})})
                elif isinstance(ev, dict) and ev.get('type') == 'error':
                    err = ev.get('error')
                    try:
                        self._logger.error("[wan22.gguf] runtime error during txt2vid: %s", err)
                    except Exception:
                        pass
                    raise RuntimeError(f"WAN22 GGUF runtime error: {err}")
            self._logger.info('[wan22_5b] DEBUG: depois de função txt2vid')
            return

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.info('[wan22_5b] DEBUG: antes de função img2vid')
        self.ensure_loaded()
        assert self._comp is not None
        if getattr(self._comp, 'pipeline', None) is not None:
            yield from _run_i2v(engine=self, comp=self._comp, request=request)
        else:
            from apps.backend.runtime.wan22 import wan22 as gguf
            if getattr(request, 'init_image', None) is None:
                raise RuntimeError("img2vid requires 'init_image'")
            yield ProgressEvent(stage="prepare", percent=0.0, message="Preparing img2vid (GGUF)")
            cfg = _build_wan22_gguf_run_config(
                request=request,
                comp=self._comp,
                dtype=self._opts.dtype,
                device=self._comp.device,
                logger=self._logger,
            )
            for ev in stream_run(gguf.run_img2vid, cfg=cfg, logger=self._logger):
                if isinstance(ev, dict) and ev.get('type') == 'progress':
                    st = str(ev.get('stage', ''))
                    step = int(ev.get('step', 0)); total = int(ev.get('total', 0)); pct = float(ev.get('percent', 0.0))
                    pct_out = (pct * 100.0) if (0.0 <= pct <= 1.0) else pct
                    try:
                        self._logger.info("[wan22.gguf] %s %d/%d (%.1f%%)", st, step, total, pct_out)
                    except Exception:
                        pass
                    yield ProgressEvent(stage=st, percent=pct_out, step=step, total_steps=total)
                elif isinstance(ev, dict) and ev.get('type') == 'result':
                    frames = ev.get('frames', [])
                    yield ResultEvent(payload={"images": frames, "info": self._to_json({"engine": self.engine_id, "task": "img2vid", "frames": len(frames)})})
                elif isinstance(ev, dict) and ev.get('type') == 'error':
                    err = ev.get('error')
                    try:
                        self._logger.error("[wan22.gguf] runtime error during img2vid: %s", err)
                    except Exception:
                        pass
                    raise RuntimeError(f"WAN22 GGUF runtime error: {err}")
            self._logger.info('[wan22_5b] DEBUG: depois de função img2vid')
            return

    def vid2vid(self, request: Vid2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self._logger.info('[wan22_5b] DEBUG: antes de função vid2vid')
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_v2v(engine=self, comp=self._comp, request=request)
        self._logger.info('[wan22_5b] DEBUG: depois de função vid2vid')
        return

    # ------------------------------ helpers
    # GGUF config assembly lives in `_build_wan22_gguf_run_config` so UI stage overrides
    # can be unit-tested without running the full runtime.
