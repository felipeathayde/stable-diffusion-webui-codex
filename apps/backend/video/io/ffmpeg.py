from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


class FFmpegUnavailableError(RuntimeError):
    pass


def _which(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise FFmpegUnavailableError(
            f"{name} not found on PATH. Install ffmpeg/ffprobe and try again."
        )
    return path


def _parse_ratio(raw: str) -> Optional[float]:
    value = str(raw or "").strip()
    if not value or value in {"0/0", "N/A"}:
        return None
    if "/" in value:
        num_s, den_s = value.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except Exception:
            return None
        if den == 0:
            return None
        return num / den
    try:
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class VideoProbe:
    path: str
    width: int
    height: int
    fps: float
    duration_seconds: float | None
    frame_count: int | None
    has_audio: bool
    format_name: str | None = None
    video_codec: str | None = None
    audio_codec: str | None = None


def probe_video(path: str) -> VideoProbe:
    ffprobe = _which("ffprobe")
    p = str(path)
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        p,
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise RuntimeError(f"ffprobe failed for '{p}': {msg}") from exc
    try:
        data = json.loads(out.decode("utf-8", errors="replace"))
    except Exception as exc:
        raise RuntimeError("ffprobe returned invalid JSON") from exc

    streams = data.get("streams") if isinstance(data, dict) else None
    if not isinstance(streams, list):
        streams = []
    fmt = data.get("format") if isinstance(data, dict) else None
    fmt_name = str(fmt.get("format_name")).strip() if isinstance(fmt, dict) and fmt.get("format_name") else None
    duration = None
    if isinstance(fmt, dict) and fmt.get("duration") not in (None, "", "N/A"):
        try:
            duration = float(fmt.get("duration"))
        except Exception:
            duration = None

    vstream: dict[str, Any] | None = None
    astream: dict[str, Any] | None = None
    for s in streams:
        if not isinstance(s, dict):
            continue
        if s.get("codec_type") == "video" and vstream is None:
            vstream = s
        if s.get("codec_type") == "audio" and astream is None:
            astream = s

    if vstream is None:
        raise RuntimeError(f"ffprobe: no video stream found in '{p}'")

    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    if width <= 0 or height <= 0:
        raise RuntimeError(f"ffprobe: invalid dimensions for '{p}': {width}x{height}")

    fps = _parse_ratio(str(vstream.get("avg_frame_rate") or "")) or _parse_ratio(
        str(vstream.get("r_frame_rate") or "")
    )
    if fps is None or fps <= 0:
        raise RuntimeError(f"ffprobe: unable to determine fps for '{p}'")

    frame_count = None
    nb = vstream.get("nb_frames")
    if nb not in (None, "", "N/A"):
        try:
            frame_count = int(nb)
        except Exception:
            frame_count = None
    if frame_count is None and duration is not None:
        try:
            frame_count = max(1, int(round(duration * fps)))
        except Exception:
            frame_count = None

    vcodec = str(vstream.get("codec_name")).strip() if vstream.get("codec_name") else None
    acodec = str(astream.get("codec_name")).strip() if isinstance(astream, dict) and astream.get("codec_name") else None

    return VideoProbe(
        path=p,
        width=width,
        height=height,
        fps=float(fps),
        duration_seconds=duration,
        frame_count=frame_count,
        has_audio=astream is not None,
        format_name=fmt_name,
        video_codec=vcodec,
        audio_codec=acodec,
    )


def extract_frames(
    video_path: str,
    *,
    out_dir: str,
    start_seconds: float | None = None,
    end_seconds: float | None = None,
    fps: float | None = None,
    max_frames: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> list[str]:
    ffmpeg = _which("ffmpeg")
    src = str(video_path)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    vf_parts: list[str] = []
    if width and height and width > 0 and height > 0:
        vf_parts.append(f"scale={int(width)}:{int(height)}:flags=lanczos")
    if fps and fps > 0:
        vf_parts.append(f"fps={float(fps):.6f}")
    vf = ",".join(vf_parts) if vf_parts else None

    cmd: list[str] = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if start_seconds is not None and start_seconds >= 0:
        cmd += ["-ss", str(float(start_seconds))]
    if end_seconds is not None and end_seconds > 0:
        cmd += ["-to", str(float(end_seconds))]
    cmd += ["-i", src, "-an"]
    if vf:
        cmd += ["-vf", vf]
    if max_frames is not None and int(max_frames) > 0:
        cmd += ["-frames:v", str(int(max_frames))]

    cmd += ["-vsync", "0", str(out_path / "frame_%06d.png")]

    try:
        subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise RuntimeError(f"ffmpeg failed to extract frames: {msg}") from exc

    frames = sorted(str(p) for p in out_path.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("ffmpeg extracted 0 frames (check start/end/fps settings)")
    return frames
