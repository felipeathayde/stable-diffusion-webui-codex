from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class Inventory:
    vaes: List[Dict[str, str]]
    text_encoders: List[Dict[str, str]]
    loras: List[Dict[str, str]]
    wan22: List[Dict[str, str]]  # .gguf files under models/wan22
    metadata: List[Dict[str, str]]  # org/repo roots under backend/huggingface


_CACHE: Inventory | None = None


def _repo_root() -> Path:
    # apps/server/backend/inventory/cache.py -> repo_root = parents[4]
    return Path(__file__).resolve().parents[4]


def _models_root() -> str:
    return str(_repo_root() / "models")


def _hf_root() -> str:
    return str(_repo_root() / "apps" / "server" / "backend" / "huggingface")


def _list_files(dir_path: str, exts: tuple[str, ...]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not os.path.isdir(dir_path):
        return out
    try:
        for name in os.listdir(dir_path):
            full = os.path.join(dir_path, name)
            if os.path.isfile(full) and name.lower().endswith(exts):
                out.append({"name": name, "path": full})
    except Exception:
        return out
    return sorted(out, key=lambda d: d["name"].lower())


def _list_dirs(dir_path: str) -> List[str]:
    try:
        return [os.path.join(dir_path, n) for n in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, n))]
    except Exception:
        return []


def scan_all(models_root: str | None = None, hf_root: str | None = None) -> Inventory:
    mr = models_root or _models_root()
    hr = hf_root or _hf_root()

    # Exact subfolder policy (no guessing):
    vae_dir = os.path.join(mr, "VAE")
    te_dir = os.path.join(mr, "text-encoder")
    lora_dir = os.path.join(mr, "Lora")
    wan_dir = os.path.join(mr, "wan22")

    vaes = _list_files(vae_dir, (".safetensors", ".bin", ".pt"))
    text_encoders = _list_files(te_dir, (".safetensors", ".bin", ".pt"))
    loras = _list_files(lora_dir, (".safetensors", ".bin", ".pt", ".ckpt"))
    # WAN22 GGUF with stage detection (high/low/unknown)
    wan22: List[Dict[str, str]] = []
    if os.path.isdir(wan_dir):
        try:
            for name in os.listdir(wan_dir):
                full = os.path.join(wan_dir, name)
                if os.path.isfile(full) and name.lower().endswith('.gguf'):
                    lower = name.lower()
                    stage = 'unknown'
                    if any(k in lower for k in ('high', 'highnoise', 'high_noise')):
                        stage = 'high'
                    elif any(k in lower for k in ('low', 'lownoise', 'low_noise')):
                        stage = 'low'
                    wan22.append({"name": name, "path": full, "stage": stage})
            wan22.sort(key=lambda d: d["name"].lower())
        except Exception:
            wan22 = []

    # Metadata folders: org/repo roots under hf_root
    metadata: List[Dict[str, str]] = []
    for org in _list_dirs(hr):
        for repo in _list_dirs(org):
            metadata.append({
                "name": f"{os.path.basename(org)}/{os.path.basename(repo)}",
                "path": repo,
            })
    metadata.sort(key=lambda d: d["name"].lower())

    return Inventory(vaes=vaes, text_encoders=text_encoders, loras=loras, wan22=wan22, metadata=metadata)


def init(models_root: str | None = None, hf_root: str | None = None) -> None:
    global _CACHE
    _CACHE = scan_all(models_root=models_root, hf_root=hf_root)


def get() -> Dict[str, List[Dict[str, str]]]:
    global _CACHE
    if _CACHE is None:
        init()
    assert _CACHE is not None
    return asdict(_CACHE)


def refresh(models_root: str | None = None, hf_root: str | None = None) -> Dict[str, List[Dict[str, str]]]:
    """Re-scan models and HF metadata roots and replace the in-memory cache.

    Returns the refreshed inventory as a plain dict suitable for JSON responses.
    """
    global _CACHE
    _CACHE = scan_all(models_root=models_root, hf_root=hf_root)
    return asdict(_CACHE)
