from __future__ import annotations

import time
from pathlib import Path
import shutil
from uuid import uuid4
from typing import Any, Dict, Iterator, List, Optional, Sequence

from apps.backend.core.engine_interface import TaskType
from apps.backend.core.params.video import VideoInterpolationOptions
from apps.backend.core.requests import Img2VidRequest, InferenceEvent, ProgressEvent, ResultEvent, Vid2VidRequest
from apps.backend.video.export import export_video
from apps.backend.video.flow import FlowGuidanceError, RaftFlowEstimator, warp_frame
from apps.backend.video.interpolation import maybe_interpolate
from apps.backend.video.io import extract_frames, probe_video


def _blend(a: Any, b: Any, *, alpha: float) -> Any:
    from PIL import Image  # type: ignore

    if not isinstance(a, Image.Image) or not isinstance(b, Image.Image):
        raise RuntimeError("blend expects PIL images")
    aa = a.convert("RGB")
    bb = b.convert("RGB")
    if aa.size != bb.size:
        bb = bb.resize(aa.size)
    alpha_clamped = max(0.0, min(1.0, float(alpha)))
    return Image.blend(aa, bb, alpha_clamped)


def _load_pil_images(paths: Sequence[str]) -> list[Any]:
    from PIL import Image  # type: ignore

    out: list[Any] = []
    for p in paths:
        img = Image.open(p)
        out.append(img.copy())
        img.close()
    return out


def _extract_vid2vid_options(extras: Dict[str, Any]) -> dict[str, Any]:
    cfg = extras.get("vid2vid") if isinstance(extras.get("vid2vid"), dict) else {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def _extract_flow_options(extras: Dict[str, Any]) -> dict[str, Any]:
    cfg = extras.get("vid2vid_flow") if isinstance(extras.get("vid2vid_flow"), dict) else {}
    return dict(cfg) if isinstance(cfg, dict) else {}


def _as_wan_animate_mode(raw: str) -> str:
    v = str(raw or "").strip().lower()
    if v in {"animate", "animation"}:
        return "animate"
    if v in {"replace", "replacement"}:
        return "replace"
    raise RuntimeError("wan_animate mode must be 'animate' or 'replace'")


def _validate_4n_plus_1(value: int, *, name: str) -> int:
    v = int(value)
    if v <= 0 or (v - 1) % 4 != 0:
        raise RuntimeError(f"{name} must be 4N+1 (got {v})")
    return v


def _run_native_pipeline(
    *,
    engine: Any,
    comp: Any,
    request: Vid2VidRequest,
    frames_in: Sequence[Any],
) -> list[Any]:
    pipe = getattr(comp, "pipeline", None)
    if pipe is None:
        raise RuntimeError("vid2vid method 'native' requires a Diffusers WAN pipeline (comp.pipeline)")

    strength = request.strength
    if strength is None:
        strength = 0.8
    strength_val = max(0.0, min(1.0, float(strength)))

    output = pipe(
        video=list(frames_in),
        prompt=request.prompt,
        negative_prompt=getattr(request, "negative_prompt", None),
        num_frames=int(getattr(request, "num_frames", len(frames_in)) or len(frames_in)),
        num_inference_steps=int(getattr(request, "steps", 30) or 30),
        height=int(getattr(request, "height", 432) or 432),
        width=int(getattr(request, "width", 768) or 768),
        guidance_scale=getattr(request, "guidance_scale", None),
        strength=strength_val,
    )
    if hasattr(output, "frames"):
        frames = list(output.frames[0])
        if not frames:
            raise RuntimeError("vid2vid pipeline returned 0 frames")
        return frames
    raise RuntimeError("vid2vid pipeline returned no frames")


def _run_wan_animate(
    *,
    engine: Any,
    comp: Any,
    request: Vid2VidRequest,
    cfg: dict[str, Any],
) -> tuple[list[Any], int, bool, str | None]:
    logger = getattr(engine, "_logger", None)
    pipe = getattr(comp, "pipeline", None)
    if pipe is None:
        raise RuntimeError("vid2vid method 'wan_animate' requires a Diffusers WanAnimatePipeline (comp.pipeline)")

    ref = getattr(request, "reference_image", None)
    if ref is None:
        raise RuntimeError("vid2vid wan_animate requires 'reference_image'")

    pose_path = str(getattr(request, "pose_video_path", "") or "").strip()
    face_path = str(getattr(request, "face_video_path", "") or "").strip()
    if not pose_path or not face_path:
        raise RuntimeError("vid2vid wan_animate requires pose_video_path and face_video_path")

    mode = _as_wan_animate_mode(getattr(request, "animate_mode", "animate"))
    bg_path = str(getattr(request, "background_video_path", "") or "").strip()
    mask_path = str(getattr(request, "mask_video_path", "") or "").strip()
    if mode == "replace" and (not bg_path or not mask_path):
        raise RuntimeError("vid2vid wan_animate mode 'replace' requires background_video_path and mask_video_path")

    yield_fps = bool(cfg.get("use_source_fps", True))
    fps_val = int(getattr(request, "fps", 24) or 24)
    pose_probe = probe_video(pose_path)
    if yield_fps:
        fps_val = max(1, int(round(float(pose_probe.fps))))

    max_frames = cfg.get("max_frames")
    try:
        max_frames = int(max_frames) if max_frames is not None else None
    except Exception:
        max_frames = None

    tag = uuid4().hex
    work = Path.cwd() / "tmp" / "vid2vid" / f"task_{tag}"
    pose_dir = work / "pose_frames"
    face_dir = work / "face_frames"
    bg_dir = work / "bg_frames"
    mask_dir = work / "mask_frames"

    try:
        if logger:
            logger.info("[vid2vid.wan_animate] extracting frames (fps=%s max=%s)", fps_val, max_frames)
        pose_paths = extract_frames(
            pose_path,
            out_dir=str(pose_dir),
            fps=float(fps_val),
            max_frames=max_frames,
            width=int(getattr(request, "width", 1280) or 1280),
            height=int(getattr(request, "height", 720) or 720),
        )
        face_paths = extract_frames(
            face_path,
            out_dir=str(face_dir),
            fps=float(fps_val),
            max_frames=max_frames,
            width=int(getattr(request, "width", 1280) or 1280),
            height=int(getattr(request, "height", 720) or 720),
        )
        if len(pose_paths) != len(face_paths):
            raise RuntimeError(f"pose/face produced different frame counts (pose={len(pose_paths)} face={len(face_paths)})")

        bg_paths: list[str] | None = None
        mask_paths: list[str] | None = None
        if mode == "replace":
            bg_paths = extract_frames(
                bg_path,
                out_dir=str(bg_dir),
                fps=float(fps_val),
                max_frames=max_frames,
                width=int(getattr(request, "width", 1280) or 1280),
                height=int(getattr(request, "height", 720) or 720),
            )
            mask_paths = extract_frames(
                mask_path,
                out_dir=str(mask_dir),
                fps=float(fps_val),
                max_frames=max_frames,
                width=int(getattr(request, "width", 1280) or 1280),
                height=int(getattr(request, "height", 720) or 720),
            )
            if len(bg_paths) != len(pose_paths) or len(mask_paths) != len(pose_paths):
                raise RuntimeError(
                    "bg/mask frame count must match pose/face frame count "
                    f"(pose={len(pose_paths)} bg={len(bg_paths)} mask={len(mask_paths)})"
                )

        pose_frames = _load_pil_images(pose_paths)
        face_frames = _load_pil_images(face_paths)
        bg_frames = _load_pil_images(bg_paths) if bg_paths else None
        mask_frames = _load_pil_images(mask_paths) if mask_paths else None

        segment_len = _validate_4n_plus_1(int(getattr(request, "segment_frame_length", 77) or 77), name="segment_frame_length")
        prev_cond = _validate_4n_plus_1(
            int(getattr(request, "prev_segment_conditioning_frames", 1) or 1),
            name="prev_segment_conditioning_frames",
        )

        seed = getattr(request, "seed", None)
        generator = None
        if isinstance(seed, int) and seed >= 0:
            try:
                import torch  # type: ignore

                device_raw = getattr(pipe, "_execution_device", None) or getattr(pipe, "device", None) or "cpu"
                device = str(device_raw) if not hasattr(device_raw, "type") else str(device_raw.type)
                generator = torch.Generator(device=device).manual_seed(int(seed))
            except Exception:
                generator = None

        width = int(getattr(request, "width", 1280) or 1280)
        height = int(getattr(request, "height", 720) or 720)
        steps = int(getattr(request, "steps", 20) or 20)
        guidance = getattr(request, "guidance_scale", None)
        if guidance is None:
            guidance = 1.0

        output = pipe(
            image=ref,
            pose_video=pose_frames,
            face_video=face_frames,
            background_video=bg_frames,
            mask_video=mask_frames,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", None) or None,
            height=height,
            width=width,
            segment_frame_length=segment_len,
            num_inference_steps=steps,
            mode=mode,
            prev_segment_conditioning_frames=prev_cond,
            motion_encode_batch_size=(getattr(request, "motion_encode_batch_size", None) or None),
            guidance_scale=float(guidance),
            generator=generator,
            output_type="pil",
        )

        frames_out: list[Any] = []
        if hasattr(output, "frames"):
            frames_out = list(output.frames[0])
        elif hasattr(output, "images"):
            frames_out = list(output.images)  # type: ignore[attr-defined]
        if not frames_out:
            raise RuntimeError("WanAnimate pipeline returned 0 frames")

        # Optional: copy audio from the driving/original video if provided and has audio.
        audio_source = str(getattr(request, "video_path", "") or "").strip() or None
        has_audio = False
        if audio_source:
            try:
                has_audio = bool(probe_video(audio_source).has_audio)
            except Exception:
                has_audio = False
        return frames_out, fps_val, has_audio, audio_source
    finally:
        try:
            shutil.rmtree(work)
        except Exception:
            pass


def _run_flow_chunks(
    *,
    engine: Any,
    request: Vid2VidRequest,
    frames_in: Sequence[Any],
) -> list[Any]:
    logger = getattr(engine, "_logger", None)
    extras = dict(getattr(request, "extras", {}) or {})

    cfg = _extract_vid2vid_options(extras)
    flow_cfg = _extract_flow_options(extras)

    method = str(cfg.get("method") or "flow_chunks").strip().lower()
    if method not in {"flow_chunks", "chunks"}:
        raise RuntimeError(f"Unsupported vid2vid chunk method: {method}")

    strength = request.strength
    if strength is None:
        strength = float(cfg.get("strength", 0.8) or 0.8)
    strength_val = max(0.0, min(1.0, float(strength)))
    anchor_alpha = max(0.0, min(1.0, float(cfg.get("anchor_alpha", 1.0 - strength_val))))

    chunk_frames = int(cfg.get("chunk_frames") or getattr(request, "num_frames", 16) or 16)
    chunk_frames = max(2, min(128, int(chunk_frames)))
    overlap = int(cfg.get("overlap_frames") or max(2, chunk_frames // 4))
    overlap = max(0, min(chunk_frames - 1, overlap))
    stride = max(1, chunk_frames - overlap)

    flow_enabled = bool(flow_cfg.get("enabled", True))
    estimator: Optional[RaftFlowEstimator] = None
    if flow_enabled:
        estimator = RaftFlowEstimator(
            device=str(flow_cfg.get("device") or "cuda"),
            use_large=bool(flow_cfg.get("use_large", False)),
            downscale=int(flow_cfg.get("downscale") or 2),
        )

    out: list[Any] = []
    seed_base = getattr(request, "seed", None)
    seed_is_valid = isinstance(seed_base, int) and seed_base >= 0

    for start in range(0, len(frames_in), stride):
        needed = min(chunk_frames, len(frames_in) - start)
        if needed <= 0:
            break

        if start == 0 or not out:
            init = frames_in[0]
        else:
            prev_src = frames_in[start - 1]
            cur_src = frames_in[start]
            prev_out = out[start - 1] if (start - 1) < len(out) else out[-1]
            warped = prev_out
            if estimator is not None:
                try:
                    flow = estimator.estimate_backward_flow(target_frame=cur_src, source_frame=prev_src)
                    warped = warp_frame(prev_out, backward_flow=flow, device=estimator.device)
                except FlowGuidanceError as exc:
                    raise RuntimeError(f"Optical flow guidance failed: {exc}") from exc
            init = _blend(warped, cur_src, alpha=anchor_alpha)

        # Vary seed per chunk for seam robustness while keeping determinism.
        chunk_seed = None
        if seed_is_valid:
            chunk_seed = int(seed_base) + int(start)

        chunk_req = Img2VidRequest(
            task=TaskType.IMG2VID,
            prompt=request.prompt,
            negative_prompt=getattr(request, "negative_prompt", ""),
            init_image=init,
            width=int(getattr(request, "width", 768) or 768),
            height=int(getattr(request, "height", 432) or 432),
            steps=int(getattr(request, "steps", 30) or 30),
            num_frames=int(needed),
            fps=int(getattr(request, "fps", 24) or 24),
            seed=chunk_seed if chunk_seed is not None else getattr(request, "seed", None),
            guidance_scale=getattr(request, "guidance_scale", None),
            sampler=getattr(request, "sampler", None),
            scheduler=getattr(request, "scheduler", None),
            extras=extras,
        )

        # Engines may stream progress; we only need the final frames for stitching.
        frames_chunk: list[Any] = []
        for ev in engine.img2vid(chunk_req):
            if isinstance(ev, ResultEvent):
                payload = ev.payload or {}
                frames_chunk = list(payload.get("images", []) or [])
        if not frames_chunk:
            raise RuntimeError(f"vid2vid chunk produced 0 frames at start={start}")

        # Stitch with overlap crossfade when we already have frames at this index.
        overlap_count = min(overlap, len(frames_chunk), max(0, len(out) - start))
        for i in range(overlap_count):
            alpha = float(i + 1) / float(overlap_count)
            out[start + i] = _blend(out[start + i], frames_chunk[i], alpha=alpha)

        # Append new frames beyond current output length.
        for j in range(overlap_count, min(len(frames_chunk), len(frames_in) - start)):
            idx = start + j
            if idx < len(out):
                out[idx] = frames_chunk[j]
            else:
                out.append(frames_chunk[j])

    return out[: len(frames_in)]


def run_vid2vid(
    *,
    engine: Any,
    comp: Any,
    request: Vid2VidRequest,
) -> Iterator[InferenceEvent]:
    logger = getattr(engine, "_logger", None)
    extras = dict(getattr(request, "extras", {}) or {})
    cfg = _extract_vid2vid_options(extras)
    method = str(cfg.get("method") or "flow_chunks").strip().lower()
    if method not in {"native", "flow_chunks", "chunks", "wan_animate"}:
        raise RuntimeError(f"Unsupported vid2vid method: {method}")

    try:
        if method == "wan_animate":
            yield ProgressEvent(stage="probe", percent=0.0, message="Probing WAN Animate inputs")
            start = time.perf_counter()
            frames_out, fps_val, has_audio, audio_source = _run_wan_animate(engine=engine, comp=comp, request=request, cfg=cfg)
            probe = None
        else:
            video_path = str(getattr(request, "video_path", "") or "").strip()
            if not video_path:
                raise RuntimeError("vid2vid requires 'video_path'")

            src = Path(video_path)
            if not src.is_file():
                raise RuntimeError(f"vid2vid video not found: {video_path}")

            yield ProgressEvent(stage="probe", percent=0.0, message="Probing input video")
            probe = probe_video(video_path)

            use_source_fps = bool(cfg.get("use_source_fps", True))
            use_source_frames = bool(cfg.get("use_source_frames", True))

            fps_val = int(getattr(request, "fps", 24) or 24)
            if use_source_fps:
                fps_val = max(1, int(round(float(probe.fps))))

            frames_target = int(getattr(request, "num_frames", 0) or 0)
            if use_source_frames and probe.frame_count:
                frames_target = int(probe.frame_count)
            if frames_target <= 0:
                frames_target = 16

            start_s = cfg.get("start_seconds")
            end_s = cfg.get("end_seconds")
            max_frames = cfg.get("max_frames")
            try:
                start_s = float(start_s) if start_s is not None else None
            except Exception:
                start_s = None
            try:
                end_s = float(end_s) if end_s is not None else None
            except Exception:
                end_s = None
            try:
                max_frames = int(max_frames) if max_frames is not None else None
            except Exception:
                max_frames = None

            # Honor explicit max_frames first, otherwise cap by target.
            cap = frames_target
            if max_frames is not None and max_frames > 0:
                cap = min(cap, max_frames)

            work = Path.cwd() / "tmp" / "vid2vid" / f"task_{uuid4().hex}"
            frames_dir = work / "src_frames"
            yield ProgressEvent(stage="decode", percent=0.05, message="Decoding video frames")
            paths = extract_frames(
                video_path,
                out_dir=str(frames_dir),
                start_seconds=start_s,
                end_seconds=end_s,
                fps=float(fps_val),
                max_frames=int(cap) if cap > 0 else None,
                width=int(getattr(request, "width", 768) or 768),
                height=int(getattr(request, "height", 432) or 432),
            )
            frames_in = _load_pil_images(paths)
            try:
                shutil.rmtree(work)
            except Exception:
                pass

            yield ProgressEvent(stage="run", percent=0.1, message=f"Running vid2vid ({method})")
            start = time.perf_counter()

            if method == "native":
                frames_out = _run_native_pipeline(engine=engine, comp=comp, request=request, frames_in=frames_in)
            else:
                frames_out = _run_flow_chunks(engine=engine, request=request, frames_in=frames_in)
            has_audio = bool(probe.has_audio)
            audio_source = video_path if probe.has_audio else None

        vfi_opts = None
        vfi_cfg = extras.get("video_interpolation") if isinstance(extras, dict) else None
        if isinstance(vfi_cfg, dict):
            vio = VideoInterpolationOptions(
                enabled=bool(vfi_cfg.get("enabled", False)),
                model=str(vfi_cfg.get("model")) if vfi_cfg.get("model") is not None else None,
                times=int(vfi_cfg.get("times")) if vfi_cfg.get("times") is not None else None,
            )
            vfi_opts = vio.as_dict()
            if vio.enabled and (vio.times or 0) > 1:
                yield ProgressEvent(stage="interpolate", percent=0.95, message="Interpolating frames (VFI)")
                frames_out, vfi_meta = maybe_interpolate(
                    frames_out,
                    enabled=vio.enabled,
                    model=vio.model,
                    times=vio.times or 2,
                    logger=logger,
                )
                vfi_opts = {**vfi_opts, "result": vfi_meta}

        elapsed = time.perf_counter() - start
        info: dict[str, Any] = {
            "engine": getattr(engine, "engine_id", "unknown"),
            "task": "vid2vid",
            "method": method,
            "elapsed": round(elapsed, 3),
            "frames_in": (len(frames_in) if method != "wan_animate" else None),
            "frames_out": len(frames_out),
            "fps": fps_val,
            "width": int(getattr(request, "width", 768) or 768),
            "height": int(getattr(request, "height", 432) or 432),
            "steps": int(getattr(request, "steps", 30) or 30),
            "strength": request.strength,
            "audio_in": bool(has_audio),
        }
        if method == "wan_animate":
            info["animate_mode"] = getattr(request, "animate_mode", None)
            info["segment_frame_length"] = int(getattr(request, "segment_frame_length", 77) or 77)
            info["prev_segment_conditioning_frames"] = int(getattr(request, "prev_segment_conditioning_frames", 1) or 1)
        if vfi_opts is not None:
            info["video_interpolation"] = vfi_opts

        video_meta = None
        try:
            video_meta = export_video(
                frames_out,
                fps=fps_val,
                options=getattr(request, "video_options", None),
                task="vid2vid",
                audio_source_path=audio_source if has_audio else None,
                extra_metadata=info if bool(getattr(request, "video_options", None)) else None,
            )
        except Exception as exc:
            # Surface exporter failure explicitly; users can disable save_output to keep frames-only.
            raise RuntimeError(f"vid2vid export failed: {exc}") from exc

        if video_meta and getattr(video_meta, "saved", False):
            info["video_export"] = {
                "rel_path": getattr(video_meta, "rel_path", None),
                "mime": getattr(video_meta, "mime", None),
                "fps": getattr(video_meta, "fps", None),
                "frames": getattr(video_meta, "frame_count", None),
            }

        preview_n = int(cfg.get("preview_frames") or 48)
        preview = list(frames_out[: max(1, min(preview_n, len(frames_out)))])

        payload: dict[str, Any] = {
            "images": preview,
            "info": engine._to_json(info),  # type: ignore[attr-defined]
        }
        if video_meta and getattr(video_meta, "saved", False):
            payload["video"] = {
                "rel_path": getattr(video_meta, "rel_path", None),
                "mime": getattr(video_meta, "mime", None),
            }

        yield ResultEvent(payload=payload)
    finally:
        # Uploaded-file cleanup is handled at the API layer; keep the use-case side-effect-free.
        pass
