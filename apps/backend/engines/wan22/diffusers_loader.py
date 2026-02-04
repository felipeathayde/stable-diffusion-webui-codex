"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN 2.2 diffusers pipeline loader (prepare vendored dir + instantiate pipeline).
Prepares a local diffusers directory (copy/link configs + weights) from an engine weights directory, resolves the pipeline class from
`model_index.json`, and instantiates the diffusers pipeline with selected dtype/device defaults.

Symbols (top-level; keep in sync; no ghosts):
- `_torch_dtype` (function): Maps dtype strings (`fp16`/`bf16`/`fp32`) to torch dtypes.
- `_read_model_index_class_name` (function): Reads `_class_name` from a diffusers `model_index.json`.
- `_resolve_pipeline_class` (function): Resolves a diffusers pipeline class by name (errors with actionable message when missing).
- `_copy_tree` (function): Copies a directory tree into a destination directory (preserving structure).
- `_link_or_copy` (function): Hard-links (or copies) a file into a destination path.
- `prepare_wan_diffusers_dir` (function): Builds a ready-to-load diffusers directory under the vendor cache for a WAN engine.
- `load_wan_diffusers_pipeline` (function): Loads a WAN diffusers pipeline from a weights directory (calls `prepare_wan_diffusers_dir`).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

from apps.backend.infra.config.repo_root import repo_scratch_path


def _torch_dtype(dtype: str):
    import torch  # type: ignore

    key = str(dtype or "fp16").strip().lower()
    if key == "bf16":
        return getattr(torch, "bfloat16", torch.float16)
    if key == "fp32":
        return torch.float32
    return torch.float16


def _read_model_index_class_name(model_dir: Path) -> str:
    model_index = model_dir / "model_index.json"
    if not model_index.is_file():
        raise RuntimeError(f"Missing model_index.json under: {model_dir}")
    try:
        data = json.loads(model_index.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Invalid model_index.json under {model_dir}: {exc}") from exc
    cls = data.get("_class_name")
    if not isinstance(cls, str) or not cls.strip():
        raise RuntimeError(f"model_index.json missing _class_name under: {model_dir}")
    return cls.strip()


def _resolve_pipeline_class(class_name: str):
    try:
        import diffusers  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"diffusers is required to load WAN pipelines: {exc}") from exc

    if hasattr(diffusers, class_name):
        return getattr(diffusers, class_name)
    raise RuntimeError(
        f"diffusers does not provide '{class_name}'. "
        "This model likely requires a newer diffusers version."
    )


def _copy_tree(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for fname in files:
            s = Path(root) / fname
            d = dst / rel / fname
            if d.exists():
                continue
            shutil.copy2(s, d)


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    if os.name == "nt":
        raise RuntimeError(
            "WAN diffusers overlay requires symlinks; on Windows, enable Developer Mode "
            "or copy the config/tokenizer files into your weights directory."
        )
    os.symlink(src, dst)


def prepare_wan_diffusers_dir(*, weights_dir: Path, vendor_dir: Path, engine_id: str) -> Path:
    """Create a lightweight overlay dir combining vendor configs and local weights.

    Some setups keep WAN weights in a directory missing tokenizer/config metadata.
    We vendor the minimal HuggingFace metadata under apps/backend/huggingface/Wan-AI/**,
    and stitch them together via a repo-local overlay so diffusers `from_pretrained()` can
    operate in local-files-only mode.
    """
    weights_dir = weights_dir.resolve()
    vendor_dir = vendor_dir.resolve()
    if not vendor_dir.is_dir():
        raise RuntimeError(f"vendor_dir does not exist: {vendor_dir}")

    overlay_root = repo_scratch_path("hf_overlays", engine_id)
    overlay_root.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(str(weights_dir).encode("utf-8")).hexdigest()[:12]
    out = overlay_root / key

    marker = out / ".codex_overlay.json"
    if out.is_dir() and marker.is_file():
        try:
            meta = json.loads(marker.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        if meta.get("weights_dir") == str(weights_dir) and meta.get("vendor_dir") == str(vendor_dir):
            return out

    out.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"weights_dir": str(weights_dir), "vendor_dir": str(vendor_dir)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Root-level metadata
    for fname in ("model_index.json", "configuration.json", "config.json"):
        src = vendor_dir / fname
        if src.is_file():
            dst = out / fname
            if not dst.exists():
                shutil.copy2(src, dst)

    # Small, metadata-only folders
    for dname in ("tokenizer", "tokenizer_2", "scheduler", "image_processor"):
        src = vendor_dir / dname
        if src.is_dir():
            _copy_tree(src, out / dname)

    # Component folders: copy vendor configs + link local weights.
    component_dirs = (
        "transformer",
        "transformer_2",
        "vae",
        "text_encoder",
        "text_encoder_2",
        "image_encoder",
    )
    for dname in component_dirs:
        dst_dir = out / dname
        dst_dir.mkdir(parents=True, exist_ok=True)

        vend = vendor_dir / dname
        if vend.is_dir():
            _copy_tree(vend, dst_dir)

        wdir = weights_dir / dname
        if wdir.is_dir():
            for entry in wdir.iterdir():
                if not entry.is_file():
                    continue
                dst = dst_dir / entry.name
                if dst.exists():
                    continue
                _link_or_copy(entry, dst)

    return out


def load_wan_diffusers_pipeline(
    *,
    weights_dir: Path,
    vendor_dir: Path,
    engine_id: str,
    device: str,
    dtype: str,
    logger: Any | None = None,
):
    """Load a WAN diffusers pipeline, using vendor_dir configs when needed."""

    base_dir = weights_dir
    if not (weights_dir / "model_index.json").is_file():
        base_dir = prepare_wan_diffusers_dir(weights_dir=weights_dir, vendor_dir=vendor_dir, engine_id=engine_id)

    class_name = _read_model_index_class_name(base_dir)
    pipeline_cls = _resolve_pipeline_class(class_name)
    torch_dtype = _torch_dtype(dtype)

    try:
        from diffusers import AutoencoderKLWan  # type: ignore
    except Exception as exc:
        raise RuntimeError(f"diffusers AutoencoderKLWan is required for WAN pipelines: {exc}") from exc

    vae = AutoencoderKLWan.from_pretrained(
        str(base_dir),
        subfolder="vae",
        torch_dtype=torch_dtype,
        local_files_only=True,
    )
    pipe = pipeline_cls.from_pretrained(
        str(base_dir),
        torch_dtype=torch_dtype,
        vae=vae,
        local_files_only=True,
    )

    pipe = pipe.to(device)

    try:
        from apps.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as apply_attn  # type: ignore
        from apps.backend.engines.util.accelerator import apply_to_diffusers_pipeline as apply_accel  # type: ignore

        apply_attn(pipe, logger=logger)
        apply_accel(pipe, logger=logger)
    except Exception:
        # Best effort; do not block execution on optional accelerator hooks.
        pass

    if logger is not None:
        try:
            logger.info(
                "WAN diffusers pipeline loaded: class=%s dir=%s device=%s dtype=%s",
                class_name,
                str(base_dir),
                device,
                str(torch_dtype).replace("torch.", ""),
            )
        except Exception:
            pass

    return pipe


__all__ = [
    "prepare_wan_diffusers_dir",
    "load_wan_diffusers_pipeline",
]
