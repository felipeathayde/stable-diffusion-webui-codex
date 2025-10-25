from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Dict, Any
import json


@dataclass
class CheckpointEntry:
    name: str
    title: str
    path: str
    model_name: str
    filename: str
    short_hash: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def _iter_files(root: str, exts: Iterable[str]) -> Iterable[str]:
    try:
        for name in os.listdir(root):
            full = os.path.join(root, name)
            if os.path.isfile(full) and any(name.lower().endswith(ext) for ext in exts):
                yield full
    except Exception:
        return []


def _collect_safetensors(models_root: str) -> List[CheckpointEntry]:
    out: List[CheckpointEntry] = []
    if not os.path.isdir(models_root):
        return out
    for full in _iter_files(models_root, (".safetensors", ".ckpt", ".pt")):
        name = os.path.splitext(os.path.basename(full))[0]
        out.append(
            CheckpointEntry(
                name=name,
                title=name,
                path=os.path.dirname(full),
                model_name=name,
                filename=full,
            )
        )
    # Also consider subfolders like models/Stable-diffusion/*
    for sub in ("Stable-diffusion", "stable-diffusion", "checkpoints", "sd", "sdxl"):
        d = os.path.join(models_root, sub)
        if not os.path.isdir(d):
            continue
        for full in _iter_files(d, (".safetensors", ".ckpt", ".pt")):
            name = os.path.splitext(os.path.basename(full))[0]
            out.append(
                CheckpointEntry(
                    name=name,
                    title=name,
                    path=os.path.dirname(full),
                    model_name=name,
                    filename=full,
                )
            )
    return out


def _collect_diffusers(vendored_hf_root: str) -> List[CheckpointEntry]:
    out: List[CheckpointEntry] = []
    if not os.path.isdir(vendored_hf_root):
        return out
    try:
        for org in os.listdir(vendored_hf_root):
            org_dir = os.path.join(vendored_hf_root, org)
            if not os.path.isdir(org_dir):
                continue
            for repo in os.listdir(org_dir):
                repo_dir = os.path.join(org_dir, repo)
                if not os.path.isdir(repo_dir):
                    continue
                mi = os.path.join(repo_dir, "model_index.json")
                unet = os.path.join(repo_dir, "unet")
                transformer = os.path.join(repo_dir, "transformer")
                if os.path.isfile(mi) or os.path.isdir(unet) or os.path.isdir(transformer):
                    name = f"{org}/{repo}"
                    out.append(
                        CheckpointEntry(
                            name=name,
                            title=name,
                            path=repo_dir,
                            model_name=repo,
                            filename=mi if os.path.isfile(mi) else repo_dir,
                            metadata={"format": "diffusers"},
                        )
                    )
    except Exception:
        return out
    return out


def list_checkpoints(models_root: str = "models", vendored_hf_root: str = "apps/server/backend/huggingface") -> List[CheckpointEntry]:
    """Discover checkpoints under models/ and vendored HuggingFace repos.

    Returns entries in a stable order (alphabetical by title).
    """
    entries: List[CheckpointEntry] = []
    entries.extend(_collect_safetensors(models_root))
    entries.extend(_collect_diffusers(vendored_hf_root))
    entries.sort(key=lambda e: e.title.lower())
    return entries


def _read_json(path: str) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def describe_checkpoints(models_root: str = "models", vendored_hf_root: str = "apps/server/backend/huggingface") -> List[Dict[str, Any]]:
    """Return metadata for discovered checkpoints (non-destructive, lightweight).

    Fields per entry (best-effort):
    - name, title, filename, path, format: 'diffusers'|'checkpoint'
    - components (diffusers): list of present subfolders (unet/transformer/vae/text_encoder[/_2]/tokenizer[/_2]/scheduler)
    - prediction_type (if scheduler/config present)
    - vae.latent_channels, vae.scaling_factor (if VAE config present)
    - file_ext, file_size (for checkpoint files)
    """
    out: List[Dict[str, Any]] = []
    for e in list_checkpoints(models_root, vendored_hf_root):
        meta: Dict[str, Any] = {
            "name": e.name,
            "title": e.title,
            "model_name": e.model_name,
            "filename": e.filename,
            "path": e.path,
            "format": e.metadata.get("format") or ("diffusers" if os.path.isfile(os.path.join(e.path, "model_index.json")) else "checkpoint"),
        }
        if meta["format"] == "diffusers":
            comps = []
            for sub in ("unet", "transformer", "vae", "text_encoder", "text_encoder_2", "tokenizer", "tokenizer_2", "scheduler"):
                if os.path.isdir(os.path.join(e.path, sub)):
                    comps.append(sub)
            meta["components"] = comps
            # scheduler prediction_type
            sch_cfg = os.path.join(e.path, "scheduler", "config.json")
            if os.path.isfile(sch_cfg):
                cfg = _read_json(sch_cfg)
                pt = cfg.get("prediction_type")
                if isinstance(pt, str):
                    meta["prediction_type"] = pt
            # vae config
            vae_cfg = os.path.join(e.path, "vae", "config.json")
            if os.path.isfile(vae_cfg):
                cfg = _read_json(vae_cfg)
                meta.setdefault("vae", {})
                meta["vae"]["latent_channels"] = cfg.get("latent_channels")
                meta["vae"]["scaling_factor"] = cfg.get("scaling_factor")
        else:
            # single-file checkpoint
            try:
                stat = os.stat(e.filename)
                meta["file_ext"] = os.path.splitext(e.filename)[1].lower()
                meta["file_size"] = stat.st_size
            except Exception:
                meta["file_ext"] = os.path.splitext(e.filename)[1].lower()
                meta["file_size"] = None
        out.append(meta)
    return out


__all__ = ["CheckpointEntry", "list_checkpoints", "describe_checkpoints"]
