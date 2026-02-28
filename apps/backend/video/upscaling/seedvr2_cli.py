"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Fail-loud SeedVR2 CLI runner for WAN video frame upscaling.
Validates frame inputs, provisions deterministic repo-local SeedVR2 runtime assets (repo checkout + isolated venv), encodes a
lossless ffmpeg intermediate video, invokes the external SeedVR2 CLI, and loads PNG outputs back into PIL frames.

Symbols (top-level; keep in sync; no ghosts):
- `run_seedvr2_upscaling` (function): Executes SeedVR2 upscaling from in-memory frames and returns `(frames_out, metadata)`.
- `__all__` (constant): Explicit export list for this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from apps.backend.core.params.video import VideoUpscalingOptions
from apps.backend.infra.config.repo_root import get_repo_root, repo_scratch_path
from apps.backend.video.runtime_dependencies import VideoDependencyResolutionError, resolve_ffmpeg_binary

_SEEDVR2_REPO_ENV = "CODEX_SEEDVR2_REPO_DIR"
_SEEDVR2_REPO_URL_ENV = "CODEX_SEEDVR2_REPO_URL"
_SEEDVR2_REPO_REF_ENV = "CODEX_SEEDVR2_REPO_REF"
_SEEDVR2_MODEL_DIR_ENV = "CODEX_SEEDVR2_MODEL_DIR"
_SEEDVR2_CUDA_DEVICE_ENV = "CODEX_SEEDVR2_CUDA_DEVICE"
_DEFAULT_SEEDVR2_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
_DEFAULT_SEEDVR2_REPO_REF = "4490bd1f482e026674543386bb2a4d176da245b9"
_DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE = Path(".uv/xdg-data/seedvr2")
_DEFAULT_SEEDVR2_REPO_RELATIVE = _DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE / "repo"
_DEFAULT_SEEDVR2_VENV_RELATIVE = _DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE / "venv"
_DEFAULT_SEEDVR2_MODEL_DIR_RELATIVE = Path(".uv/xdg-data/seedvr2")
_SEEDVR2_RUNTIME_STAMP_FILE = ".seedvr2-runtime-stamp.json"
_SEEDVR2_REPO_LOCK_FILE = ".seedvr2-repo.lock"
_SEEDVR2_VENV_LOCK_FILE = ".seedvr2-venv.lock"
_INTERMEDIATE_VIDEO_FPS = 24
_STDERR_PREVIEW_LIMIT = 4000

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover - non-posix fallback
    fcntl = None

try:
    import msvcrt  # type: ignore
except Exception:  # pragma: no cover - non-windows fallback
    msvcrt = None


@dataclass(frozen=True)
class _SeedVR2RepoResolution:
    repo_dir: Path
    uses_default_repo_path: bool
    pinned_ref: str


def _truncate_text(text: str, *, max_chars: int) -> str:
    normalized = str(text or "")
    if len(normalized) <= max_chars:
        return normalized
    keep = max(256, max_chars // 2)
    return normalized[:keep] + "\n...<truncated>...\n" + normalized[-keep:]


def _run_checked_subprocess(
    cmd: Sequence[str],
    *,
    purpose: str,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            list(cmd),
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd is not None else None,
        )
    except FileNotFoundError as exc:
        missing = cmd[0] if cmd else "<unknown>"
        raise RuntimeError(
            f"{purpose} failed because required executable '{missing}' was not found in PATH."
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"{purpose} failed to start: {exc}") from exc

    if proc.returncode == 0:
        return proc

    stderr_preview = _truncate_text(proc.stderr or proc.stdout or "", max_chars=_STDERR_PREVIEW_LIMIT)
    raise RuntimeError(
        f"{purpose} failed (exit {proc.returncode}; command={list(cmd)!r}).\n"
        f"stderr:\n{stderr_preview}"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_seedvr2_repo_url() -> str:
    raw = str(os.environ.get(_SEEDVR2_REPO_URL_ENV) or "").strip()
    return raw or _DEFAULT_SEEDVR2_REPO_URL


def _resolve_seedvr2_repo_ref() -> str:
    raw = str(os.environ.get(_SEEDVR2_REPO_REF_ENV) or "").strip()
    return raw or _DEFAULT_SEEDVR2_REPO_REF


def _resolve_seedvr2_runtime_root_dir() -> Path:
    return (get_repo_root() / _DEFAULT_SEEDVR2_RUNTIME_ROOT_RELATIVE).resolve()


@contextlib.contextmanager
def _exclusive_seedvr2_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        if os.name == "nt":
            if msvcrt is None:
                raise RuntimeError(
                    "SeedVR2 runtime locking requires the Windows 'msvcrt' module, but it is unavailable."
                )
            os.lseek(lock_fd, 0, os.SEEK_SET)
            os.write(lock_fd, b"0")
            os.lseek(lock_fd, 0, os.SEEK_SET)
            try:
                msvcrt.locking(lock_fd, msvcrt.LK_LOCK, 1)
            except OSError as exc:
                raise RuntimeError(f"Failed to acquire SeedVR2 lock at '{lock_path}': {exc}") from exc
            try:
                yield
            finally:
                msvcrt.locking(lock_fd, msvcrt.LK_UNLCK, 1)
        else:
            if fcntl is None:
                raise RuntimeError(
                    "SeedVR2 runtime locking requires the POSIX 'fcntl' module, but it is unavailable."
                )
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_EX)
            except OSError as exc:
                raise RuntimeError(f"Failed to acquire SeedVR2 lock at '{lock_path}': {exc}") from exc
            try:
                yield
            finally:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
    finally:
        os.close(lock_fd)


def _ensure_default_repo_is_git_checkout(*, git_bin: str, repo_dir: Path) -> None:
    try:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "rev-parse", "--git-dir"],
            purpose="SeedVR2 default repo bootstrap git checkout validation",
        )
    except RuntimeError as exc:
        raise RuntimeError(
            "SeedVR2 default repo path exists but is not a valid git checkout. "
            f"Expected '{repo_dir}'. Remove it or set {_SEEDVR2_REPO_ENV} to a valid SeedVR2 checkout."
        ) from exc


def _bootstrap_default_seedvr2_repo(*, repo_dir: Path, repo_url: str, repo_ref: str) -> None:
    git_bin = shutil.which("git")
    if not git_bin:
        raise RuntimeError(
            "SeedVR2 default repo bootstrap requires 'git', but it was not found in PATH. "
            f"Install git or set {_SEEDVR2_REPO_ENV} to an existing checkout."
        )

    if repo_dir.exists():
        if not repo_dir.is_dir():
            raise RuntimeError(
                "SeedVR2 repo path exists but is not a directory. "
                f"Expected '{repo_dir}'. Remove it or set {_SEEDVR2_REPO_ENV}."
            )
    else:
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        _run_checked_subprocess(
            [git_bin, "clone", "--filter=blob:none", "--no-checkout", repo_url, str(repo_dir)],
            purpose="SeedVR2 default repo bootstrap clone",
        )

    _ensure_default_repo_is_git_checkout(git_bin=git_bin, repo_dir=repo_dir)

    try:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "checkout", "--detach", repo_ref],
            purpose="SeedVR2 default repo bootstrap checkout",
        )
    except RuntimeError:
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "fetch", "--depth", "1", "origin", repo_ref],
            purpose="SeedVR2 default repo bootstrap fetch ref",
        )
        _run_checked_subprocess(
            [git_bin, "-C", str(repo_dir), "checkout", "--detach", repo_ref],
            purpose="SeedVR2 default repo bootstrap checkout",
        )


def _resolve_seedvr2_repo_dir() -> _SeedVR2RepoResolution:
    raw = str(os.environ.get(_SEEDVR2_REPO_ENV) or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
        uses_default_repo_path = False
        pinned_ref = ""
    else:
        candidate = get_repo_root() / _DEFAULT_SEEDVR2_REPO_RELATIVE
        uses_default_repo_path = True
        pinned_ref = _resolve_seedvr2_repo_ref()
        with _exclusive_seedvr2_lock(_resolve_seedvr2_runtime_root_dir() / _SEEDVR2_REPO_LOCK_FILE):
            _bootstrap_default_seedvr2_repo(
                repo_dir=candidate.resolve(),
                repo_url=_resolve_seedvr2_repo_url(),
                repo_ref=pinned_ref,
            )

    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            "SeedVR2 repo directory is missing. "
            f"Expected '{resolved}'. Set {_SEEDVR2_REPO_ENV} to a valid checkout path."
        )

    cli_path = resolved / "inference_cli.py"
    if not cli_path.is_file():
        raise RuntimeError(
            f"SeedVR2 CLI entrypoint not found at '{cli_path}'. "
            "Verify the repository checkout contains inference_cli.py."
        )
    return _SeedVR2RepoResolution(
        repo_dir=resolved,
        uses_default_repo_path=uses_default_repo_path,
        pinned_ref=pinned_ref,
    )


def _resolve_seedvr2_venv_dir() -> Path:
    return (get_repo_root() / _DEFAULT_SEEDVR2_VENV_RELATIVE).resolve()


def _seedvr2_venv_python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _create_seedvr2_venv(venv_dir: Path) -> None:
    try:
        import venv
    except Exception as exc:
        raise RuntimeError(
            "SeedVR2 isolated runtime requires Python's 'venv' module, but it is unavailable."
        ) from exc

    venv_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=os.name != "nt")
        builder.create(str(venv_dir))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to create SeedVR2 isolated runtime venv at '{venv_dir}': {exc}"
        ) from exc


def _read_seedvr2_runtime_stamp(stamp_path: Path) -> dict[str, str] | None:
    if not stamp_path.is_file():
        return None
    try:
        payload = json.loads(stamp_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(payload, dict):
        return None
    requirements_sha256 = payload.get("requirements_sha256")
    repo_ref = payload.get("repo_ref")
    if not isinstance(requirements_sha256, str) or not isinstance(repo_ref, str):
        return None
    return {
        "requirements_sha256": requirements_sha256,
        "repo_ref": repo_ref,
    }


def _write_seedvr2_runtime_stamp(stamp_path: Path, payload: dict[str, str]) -> None:
    stamp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = stamp_path.with_suffix(stamp_path.suffix + ".tmp")
    try:
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(stamp_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to persist SeedVR2 runtime stamp at '{stamp_path}': {exc}") from exc


def _ensure_seedvr2_runtime_venv(*, repo_resolution: _SeedVR2RepoResolution) -> Path:
    with _exclusive_seedvr2_lock(_resolve_seedvr2_runtime_root_dir() / _SEEDVR2_VENV_LOCK_FILE):
        requirements_path = repo_resolution.repo_dir / "requirements.txt"
        if not requirements_path.is_file():
            raise RuntimeError(
                f"SeedVR2 runtime requirements file is missing: '{requirements_path}'. "
                "The SeedVR2 checkout must include requirements.txt."
            )

        venv_dir = _resolve_seedvr2_venv_dir()
        venv_python = _seedvr2_venv_python_path(venv_dir)
        if not venv_python.is_file():
            _create_seedvr2_venv(venv_dir)
        if not venv_python.is_file():
            raise RuntimeError(
                f"SeedVR2 isolated runtime venv is missing Python executable at '{venv_python}'."
            )

        _run_checked_subprocess(
            [str(venv_python), "-m", "pip", "--version"],
            purpose="SeedVR2 isolated runtime pip validation",
        )

        repo_ref_token = (
            repo_resolution.pinned_ref if repo_resolution.uses_default_repo_path else "external-repo-path"
        )
        desired_stamp = {
            "requirements_sha256": _sha256_file(requirements_path),
            "repo_ref": repo_ref_token,
        }
        stamp_path = venv_dir / _SEEDVR2_RUNTIME_STAMP_FILE
        current_stamp = _read_seedvr2_runtime_stamp(stamp_path)
        if current_stamp == desired_stamp:
            return venv_python

        _run_checked_subprocess(
            [
                str(venv_python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                "-r",
                str(requirements_path),
            ],
            purpose="SeedVR2 isolated runtime dependency install",
        )
        _write_seedvr2_runtime_stamp(stamp_path, desired_stamp)
        return venv_python


def _resolve_seedvr2_model_dir() -> Path:
    raw = str(os.environ.get(_SEEDVR2_MODEL_DIR_ENV) or "").strip()
    if raw:
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = get_repo_root() / candidate
    else:
        candidate = get_repo_root() / _DEFAULT_SEEDVR2_MODEL_DIR_RELATIVE
    resolved = candidate.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    if not resolved.is_dir():
        raise RuntimeError(
            "SeedVR2 model directory is invalid. "
            f"Expected a directory at '{resolved}'."
        )
    return resolved


def _normalize_cuda_device_index(component_device: str | None) -> int | None:
    override_raw = str(os.environ.get(_SEEDVR2_CUDA_DEVICE_ENV) or "").strip()
    if override_raw:
        if not re.fullmatch(r"\d+", override_raw):
            raise RuntimeError(
                f"{_SEEDVR2_CUDA_DEVICE_ENV} must be a non-negative CUDA device index, got: {override_raw!r}"
            )
        return int(override_raw)

    raw_device = str(component_device or "").strip().lower()
    requested_component_cuda_index: int | None = None
    if raw_device:
        exact_match = re.fullmatch(r"cuda:(\d+)", raw_device)
        if exact_match:
            requested_component_cuda_index = int(exact_match.group(1))
        elif raw_device not in {"cuda", "cpu"}:
            return None

    raw_visible = str(os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not raw_visible:
        return requested_component_cuda_index

    entries = [entry.strip() for entry in raw_visible.split(",") if entry.strip()]
    if not entries:
        return requested_component_cuda_index

    if requested_component_cuda_index is not None:
        if requested_component_cuda_index < len(entries):
            return requested_component_cuda_index

        numeric_entries = [entry for entry in entries if re.fullmatch(r"\d+", entry)]
        if len(numeric_entries) == len(entries):
            if len(set(entries)) != len(entries):
                raise RuntimeError(
                    "CUDA_VISIBLE_DEVICES contains duplicate numeric entries, so SeedVR2 cannot map devices unambiguously. "
                    f"Got CUDA_VISIBLE_DEVICES={raw_visible!r}. Deduplicate it or set {_SEEDVR2_CUDA_DEVICE_ENV} explicitly."
                )
            requested_token = str(requested_component_cuda_index)
            if requested_token in entries:
                return entries.index(requested_token)

        raise RuntimeError(
            "SeedVR2 CUDA device mapping failed: component device "
            f"{component_device!r} is outside visible index range [0, {len(entries) - 1}] for "
            f"CUDA_VISIBLE_DEVICES={raw_visible!r}, and no physical-id fallback mapping is available. "
            f"Set {_SEEDVR2_CUDA_DEVICE_ENV} to an explicit visible index."
        )

    if raw_device == "cuda" or (not raw_device and len(entries) == 1):
        return 0
    return None


def _resolve_fallback_output_pngs(output_dir: Path) -> list[Path]:
    discovered_pngs = sorted(path for path in output_dir.rglob("*.png"))
    indexed_paths: list[tuple[int, Path]] = []
    index_to_path: dict[int, Path] = {}
    for path in discovered_pngs:
        index_matches = re.findall(r"\d+", path.stem)
        if not index_matches:
            raise RuntimeError(
                "SeedVR2 CLI fallback PNG discovery requires numeric frame indices in output filenames, "
                f"but '{path.name}' has no numeric token."
            )
        frame_index = int(index_matches[-1])
        previous = index_to_path.get(frame_index)
        if previous is not None:
            raise RuntimeError(
                "SeedVR2 CLI fallback PNG discovery found ambiguous frame ordering: "
                f"duplicate frame index {frame_index} in '{previous.name}' and '{path.name}'."
            )
        index_to_path[frame_index] = path
        indexed_paths.append((frame_index, path))
    indexed_paths.sort(key=lambda item: item[0])
    return [path for _, path in indexed_paths]


def _sanitize_metadata_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = get_repo_root().resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError:
        return f"<external>/{resolved.name}"
    return f"CODEX_ROOT/{relative.as_posix()}"


def _validate_input_frames(frames: Sequence[Any]) -> tuple[list[Any], tuple[int, int]]:
    from PIL import Image  # type: ignore

    frames_list = frames if isinstance(frames, list) else list(frames)
    if not frames_list:
        raise RuntimeError("SeedVR2 upscaling requires a non-empty frame sequence.")

    first = frames_list[0]
    if not isinstance(first, Image.Image):
        raise RuntimeError(
            "SeedVR2 upscaling requires PIL image frames; "
            f"frame[0] is {type(first).__name__}."
        )
    size = first.size
    if size[0] <= 0 or size[1] <= 0:
        raise RuntimeError(f"SeedVR2 upscaling received invalid frame size: {size!r}.")

    for index, frame in enumerate(frames_list, start=1):
        if not isinstance(frame, Image.Image):
            raise RuntimeError(
                "SeedVR2 upscaling requires PIL image frames; "
                f"frame[{index - 1}] is {type(frame).__name__}."
            )
        if frame.size != size:
            raise RuntimeError(
                "SeedVR2 upscaling requires same-size frames; "
                f"frame[0]={size!r} frame[{index - 1}]={frame.size!r}."
            )
    return frames_list, size


def _encode_lossless_input_video(frames_in_dir: Path, video_path: Path) -> None:
    try:
        ffmpeg_bin = resolve_ffmpeg_binary("ffmpeg")
    except VideoDependencyResolutionError as exc:
        raise RuntimeError(
            "SeedVR2 upscaling requires ffmpeg but it could not be resolved. "
            f"{exc}"
        ) from exc

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(_INTERMEDIATE_VIDEO_FPS),
        "-i",
        str(frames_in_dir / "frame_%06d.png"),
        "-an",
        "-c:v",
        "ffv1",
        "-pix_fmt",
        "rgb24",
        str(video_path),
    ]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode == 0:
        return
    stderr_preview = _truncate_text(proc.stderr or proc.stdout or "", max_chars=_STDERR_PREVIEW_LIMIT)
    raise RuntimeError(
        "SeedVR2 upscaling failed to encode lossless intermediate video "
        f"(exit {proc.returncode}). ffmpeg output:\n{stderr_preview}"
    )


def _run_seedvr2_cli(
    *,
    python_executable: Path,
    input_video_path: Path,
    output_dir: Path,
    options: VideoUpscalingOptions,
    model_dir: Path,
    repo_dir: Path,
    cuda_device_index: int | None,
) -> None:
    cli_path = repo_dir / "inference_cli.py"
    cmd: list[str] = [
        str(python_executable),
        str(cli_path),
        str(input_video_path),
        "--output",
        str(output_dir),
        "--output_format",
        "png",
        "--model_dir",
        str(model_dir),
    ]

    if options.dit_model:
        cmd.extend(["--dit_model", str(options.dit_model)])
    if options.resolution is not None:
        cmd.extend(["--resolution", str(int(options.resolution))])
    if options.max_resolution is not None:
        cmd.extend(["--max_resolution", str(int(options.max_resolution))])
    if options.batch_size is not None:
        cmd.extend(["--batch_size", str(int(options.batch_size))])
    if options.uniform_batch_size:
        cmd.append("--uniform_batch_size")
    if options.temporal_overlap is not None:
        cmd.extend(["--temporal_overlap", str(int(options.temporal_overlap))])
    if options.prepend_frames is not None:
        cmd.extend(["--prepend_frames", str(int(options.prepend_frames))])
    if options.color_correction:
        cmd.extend(["--color_correction", str(options.color_correction)])
    if options.input_noise_scale is not None:
        cmd.extend(["--input_noise_scale", str(float(options.input_noise_scale))])
    if options.latent_noise_scale is not None:
        cmd.extend(["--latent_noise_scale", str(float(options.latent_noise_scale))])
    if cuda_device_index is not None:
        cmd.extend(["--cuda_device", str(int(cuda_device_index))])

    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(repo_dir),
    )
    if proc.returncode == 0:
        return

    stderr_preview = _truncate_text(proc.stderr or proc.stdout or "", max_chars=_STDERR_PREVIEW_LIMIT)
    raise RuntimeError(
        "SeedVR2 CLI upscaling failed "
        f"(exit {proc.returncode}; command={cmd!r}).\n"
        f"stderr:\n{stderr_preview}"
    )


def _collect_output_frames(output_dir: Path, *, expected_count: int, input_stem: str) -> list[Any]:
    from PIL import Image  # type: ignore

    preferred = output_dir / input_stem
    candidate_paths: list[Path]
    if preferred.is_dir():
        candidate_paths = sorted(path for path in preferred.iterdir() if path.suffix.lower() == ".png")
    else:
        candidate_paths = _resolve_fallback_output_pngs(output_dir)
    if not candidate_paths:
        raise RuntimeError(f"SeedVR2 CLI produced no PNG output frames under '{output_dir}'.")

    out_frames: list[Any] = []
    for path in candidate_paths:
        with Image.open(path) as image:
            out_frames.append(image.convert("RGB").copy())

    if len(out_frames) != int(expected_count):
        raise RuntimeError(
            "SeedVR2 CLI output frame count mismatch: "
            f"expected {expected_count}, got {len(out_frames)}."
        )

    first_size = out_frames[0].size
    for index, frame in enumerate(out_frames, start=1):
        if frame.size != first_size:
            raise RuntimeError(
                "SeedVR2 CLI produced inconsistent output frame sizes: "
                f"frame[0]={first_size!r} frame[{index - 1}]={frame.size!r}."
            )

    return out_frames


def run_seedvr2_upscaling(
    frames: Sequence[Any],
    *,
    options: VideoUpscalingOptions,
    component_device: str | None,
    logger_: logging.Logger | None = None,
) -> tuple[list[Any], dict[str, Any]]:
    if not options.enabled:
        raise RuntimeError("SeedVR2 upscaling runner called with disabled options.")

    frames_list, _input_size = _validate_input_frames(frames)

    repo_resolution = _resolve_seedvr2_repo_dir()
    repo_dir = repo_resolution.repo_dir
    seedvr2_python = _ensure_seedvr2_runtime_venv(repo_resolution=repo_resolution)
    model_dir = _resolve_seedvr2_model_dir()
    cuda_device_index = _normalize_cuda_device_index(component_device)

    run_id = uuid4().hex
    work_dir = repo_scratch_path("seedvr2", run_id)
    frames_in_dir = work_dir / "frames_in"
    output_dir = work_dir / "out"
    input_video_path = work_dir / "input_lossless.mkv"
    frames_in_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    from PIL import Image  # type: ignore

    for index, frame in enumerate(frames_list, start=1):
        frame_rgb = frame.convert("RGB")
        frame_rgb.save(frames_in_dir / f"frame_{index:06d}.png", format="PNG")

    _encode_lossless_input_video(frames_in_dir, input_video_path)
    _run_seedvr2_cli(
        input_video_path=input_video_path,
        output_dir=output_dir,
        options=options,
        model_dir=model_dir,
        repo_dir=repo_dir,
        python_executable=seedvr2_python,
        cuda_device_index=cuda_device_index,
    )

    out_frames = _collect_output_frames(
        output_dir,
        expected_count=len(frames_list),
        input_stem=input_video_path.stem,
    )

    output_size = out_frames[0].size
    meta: dict[str, Any] = {
        "applied": True,
        "runner": "seedvr2_cli",
        "input_frames": len(frames_list),
        "output_frames": len(out_frames),
        "input_size": {"width": int(_input_size[0]), "height": int(_input_size[1])},
        "output_size": {"width": int(output_size[0]), "height": int(output_size[1])},
        "work_dir": _sanitize_metadata_path(work_dir),
        "repo_dir": _sanitize_metadata_path(repo_dir),
        "model_dir": _sanitize_metadata_path(model_dir),
    }
    if cuda_device_index is not None:
        meta["cuda_device"] = int(cuda_device_index)

    if logger_ is not None:
        logger_.info(
            "video upscaling (SeedVR2): %d frame(s) %dx%d -> %dx%d",
            len(frames_list),
            int(_input_size[0]),
            int(_input_size[1]),
            int(output_size[0]),
            int(output_size[1]),
        )

    return out_frames, meta


__all__ = ["run_seedvr2_upscaling"]
