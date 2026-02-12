"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Deterministic runtime dependency helpers for backend video features.
Resolves ffmpeg/ffprobe binaries and the default RIFE model path from repo-local runtime locations (with explicit overrides), and provisions
the default RIFE checkpoint with strict SHA-256 verification.

Symbols (top-level; keep in sync; no ghosts):
- `VIDEO_RUNTIME_FFMPEG_DIR_RELATIVE` (constant): Repo-relative ffmpeg runtime install directory.
- `VIDEO_RUNTIME_RIFE_MODEL_RELATIVE` (constant): Repo-relative default RIFE checkpoint path.
- `RIFE_IFNET_V426_URL` (constant): Canonical upstream URL for the default RIFE checkpoint artifact.
- `RIFE_IFNET_V426_SHA256` (constant): Expected SHA-256 for the default RIFE checkpoint artifact.
- `VideoDependencyResolutionError` (class): Raised when required runtime dependencies cannot be resolved or validated.
- `default_ffmpeg_bin_dir` (function): Returns the default repo-local ffmpeg runtime directory.
- `default_rife_model_path` (function): Returns the default repo-local RIFE checkpoint path.
- `resolve_ffmpeg_binary` (function): Resolves `ffmpeg`/`ffprobe` binary paths with deterministic precedence.
- `resolve_rife_model_path` (function): Resolves and validates a requested/default RIFE checkpoint path.
- `ensure_ffmpeg_binaries` (function): Installs ffmpeg runtime via ffmpeg-downloader and mirrors binaries into deterministic repo-local runtime paths.
- `ensure_rife_model_file` (function): Downloads/verifies the default RIFE checkpoint into repo-local runtime storage.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.request import urlopen

from apps.backend.infra.config.repo_root import get_repo_root

VIDEO_RUNTIME_FFMPEG_DIR_RELATIVE = Path(".uv/xdg-data/ffmpeg-downloader/ffmpeg")
VIDEO_RUNTIME_RIFE_MODEL_RELATIVE = Path(".uv/xdg-data/rife/rife47.pth")

RIFE_IFNET_V426_URL = "https://github.com/EutropicAI/ccvfi/releases/download/model_zoo/RIFE_IFNet_v426_heavy.pkl"
RIFE_IFNET_V426_SHA256 = "4cc518e172156ad6207b9c7a43364f518832d83a4325d484240493a9e2980537"


class VideoDependencyResolutionError(RuntimeError):
    pass


def _is_windows() -> bool:
    return os.name == "nt"


def default_ffmpeg_bin_dir() -> Path:
    root = get_repo_root() / VIDEO_RUNTIME_FFMPEG_DIR_RELATIVE
    return root / "bin" if _is_windows() else root


def default_rife_model_path() -> Path:
    return get_repo_root() / VIDEO_RUNTIME_RIFE_MODEL_RELATIVE


def _binary_filename(binary_name: str) -> str:
    suffix = ".exe" if _is_windows() else ""
    return f"{binary_name}{suffix}"


def _resolve_optional_path(raw: str | None) -> Path | None:
    value = str(raw or "").strip()
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = get_repo_root() / path
    return path


def resolve_ffmpeg_binary(binary_name: str) -> str:
    if binary_name not in {"ffmpeg", "ffprobe"}:
        raise VideoDependencyResolutionError(f"Unsupported ffmpeg binary name: {binary_name!r}")

    env_var = f"CODEX_{binary_name.upper()}_PATH"
    attempts: list[str] = []

    env_path = _resolve_optional_path(os.environ.get(env_var))
    if env_path is not None:
        attempts.append(f"{env_var}={env_path}")
        if env_path.is_file():
            return str(env_path)

    bin_dir_override = _resolve_optional_path(os.environ.get("CODEX_FFMPEG_BIN_DIR"))
    if bin_dir_override is not None:
        override_candidate = bin_dir_override / _binary_filename(binary_name)
        attempts.append(f"CODEX_FFMPEG_BIN_DIR/{_binary_filename(binary_name)}={override_candidate}")
        if override_candidate.is_file():
            return str(override_candidate)

    deterministic_candidates: list[Path] = [default_ffmpeg_bin_dir() / _binary_filename(binary_name)]
    if not _is_windows():
        deterministic_candidates.append(default_ffmpeg_bin_dir() / "bin" / _binary_filename(binary_name))

    for candidate in deterministic_candidates:
        attempts.append(f"default={candidate}")
        if candidate.is_file():
            return str(candidate)

    try:
        import ffmpeg_downloader

        downloader_path = ffmpeg_downloader.installed(binary_name, return_path=True)
        if isinstance(downloader_path, str) and downloader_path:
            from_downloader = Path(downloader_path)
            attempts.append(f"ffmpeg_downloader={from_downloader}")
            if from_downloader.is_file():
                return str(from_downloader)
    except Exception:
        attempts.append("ffmpeg_downloader=unavailable")

    path_candidate = shutil.which(binary_name)
    attempts.append(f"PATH={path_candidate or '<missing>'}")
    if path_candidate:
        return path_candidate

    raise VideoDependencyResolutionError(
        f"Unable to resolve required binary '{binary_name}'. Tried: {', '.join(attempts)}. "
        "Run install-webui.sh/install-webui.bat to provision ffmpeg runtime dependencies."
    )


def resolve_rife_model_path(model: str | None) -> Path:
    raw = str(model or "").strip()
    default_path = default_rife_model_path()

    if raw:
        requested = Path(raw).expanduser()
        if requested.is_absolute():
            candidate = requested
        elif requested.parent == Path("."):
            candidate = default_path.parent / requested.name
        else:
            candidate = get_repo_root() / requested
    else:
        env_override = _resolve_optional_path(os.environ.get("CODEX_RIFE_MODEL_PATH"))
        candidate = env_override if env_override is not None else default_path

    if not candidate.is_file():
        raise VideoDependencyResolutionError(
            f"RIFE model checkpoint not found: {candidate}. "
            "Run install-webui.sh/install-webui.bat to provision the default model, or set CODEX_RIFE_MODEL_PATH."
        )
    return candidate


def ensure_ffmpeg_binaries(version: str | None = None, *, no_symlinks: bool = False) -> dict[str, Path]:
    try:
        import ffmpeg_downloader
    except Exception as exc:
        raise VideoDependencyResolutionError(
            f"ffmpeg-downloader runtime is unavailable ({exc}). Re-run install-webui to provision it."
        ) from exc

    install_cmd: list[str] = [sys.executable, "-m", "ffmpeg_downloader", "install", "-y"]
    if version:
        install_cmd.append(str(version))
    if no_symlinks and not _is_windows():
        install_cmd.append("--no-simlinks")
    try:
        subprocess.check_output(install_cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        msg = exc.output.decode("utf-8", errors="replace") if exc.output else str(exc)
        raise VideoDependencyResolutionError(f"ffmpeg-downloader install failed: {msg}") from exc

    destination_dir = default_ffmpeg_bin_dir()
    destination_dir.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    for binary_name in ("ffmpeg", "ffprobe"):
        source_raw = ffmpeg_downloader.installed(binary_name, return_path=True)
        if not source_raw:
            raise VideoDependencyResolutionError(f"ffmpeg-downloader did not provision '{binary_name}'.")
        source = Path(str(source_raw)).resolve()
        if not source.is_file():
            raise VideoDependencyResolutionError(
                f"ffmpeg-downloader reported missing binary '{binary_name}' at '{source}'."
            )
        destination = (destination_dir / source.name).resolve()
        if source != destination:
            shutil.copy2(source, destination)
        resolved[binary_name] = destination
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_rife_model_file(
    target: Path | None = None,
    *,
    source_url: str = RIFE_IFNET_V426_URL,
    expected_sha256: str = RIFE_IFNET_V426_SHA256,
) -> Path:
    destination = target if target is not None else default_rife_model_path()
    destination = destination.expanduser()
    if not destination.is_absolute():
        destination = get_repo_root() / destination

    if destination.exists() and not destination.is_file():
        raise VideoDependencyResolutionError(f"RIFE model destination exists but is not a file: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.is_file():
        existing_hash = _sha256_file(destination)
        if existing_hash.lower() == expected_sha256.lower():
            return destination
        destination.unlink()

    tmp_path = destination.with_suffix(destination.suffix + ".download")
    if tmp_path.exists():
        tmp_path.unlink()

    digest = hashlib.sha256()
    try:
        with urlopen(source_url, timeout=120) as response, tmp_path.open("wb") as output:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                digest.update(block)
                output.write(block)
    except Exception as exc:
        if tmp_path.exists():
            tmp_path.unlink()
        raise VideoDependencyResolutionError(f"Failed to download RIFE model from {source_url}: {exc}") from exc

    downloaded_hash = digest.hexdigest().lower()
    if downloaded_hash != expected_sha256.lower():
        if tmp_path.exists():
            tmp_path.unlink()
        raise VideoDependencyResolutionError(
            f"Downloaded RIFE model hash mismatch: expected {expected_sha256}, got {downloaded_hash}."
        )

    tmp_path.replace(destination)
    return destination


__all__ = [
    "VIDEO_RUNTIME_FFMPEG_DIR_RELATIVE",
    "VIDEO_RUNTIME_RIFE_MODEL_RELATIVE",
    "RIFE_IFNET_V426_URL",
    "RIFE_IFNET_V426_SHA256",
    "VideoDependencyResolutionError",
    "default_ffmpeg_bin_dir",
    "default_rife_model_path",
    "resolve_ffmpeg_binary",
    "resolve_rife_model_path",
    "ensure_ffmpeg_binaries",
    "ensure_rife_model_file",
]
