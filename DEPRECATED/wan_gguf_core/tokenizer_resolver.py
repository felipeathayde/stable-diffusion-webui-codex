from __future__ import annotations

# DEPRECATED: moved out of active backend; reference only.

"""Auto-resolve and fetch tokenizers from Hugging Face when missing.

Design
- Try local candidates first (explicit dirs/files, text encoder dir, model_dir, model_dir/tokenizer).
- If nothing found and model_key is known, fetch from a predefined HF repo/subdir
  into models/tokenizers/<cache_name> and return that path.

No external deps beyond stdlib (urllib). Works offline unless fetch is needed.
"""

import os
import urllib.request
from typing import Iterable, Optional, Tuple
from pathlib import Path


KNOWN_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
    "added_tokens.json",
)

# Predefined map (can be extended later or loaded from a json)
PRESET = {
    "wan_i2v_14b": {
        "repo": "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "subdir": "tokenizer",
        "cache": "Wan2.2-I2V-A14B-Diffusers",
    },
    "wan_t2v_14b": {
        "repo": "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "subdir": "tokenizer",
        "cache": "Wan2.2-T2V-A14B-Diffusers",
    },
    # best-effort naming for 5B variants; adjust if your local layout differs
    "wan_i2v_5b": {
        "repo": "Wan-AI/Wan2.2-I2V-A5B-Diffusers",
        "subdir": "tokenizer",
        "cache": "Wan2.2-I2V-A5B-Diffusers",
    },
    "wan_t2v_5b": {
        "repo": "Wan-AI/Wan2.2-T2V-A5B-Diffusers",
        "subdir": "tokenizer",
        "cache": "Wan2.2-T2V-A5B-Diffusers",
    },
}


def _is_tokenizer_dir(path: str) -> bool:
    try:
        entries = set(os.listdir(path))
    except Exception:
        return False
    needed = {"tokenizer.json", "tokenizer_config.json", "spiece.model"}
    return bool(needed & entries)  # any presence is acceptable


def _fetch_hf(repo: str, subdir: str, dest_dir: str) -> bool:
    base = f"https://huggingface.co/{repo}/resolve/main/{subdir}"
    os.makedirs(dest_dir, exist_ok=True)
    fetched = 0
    for name in KNOWN_FILES:
        url = f"{base}/{name}?download=1"
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                if not (200 <= (resp.status or 0) < 400):
                    continue
                data = resp.read()
            out = os.path.join(dest_dir, name)
            with open(out, "wb") as f:
                f.write(data)
            fetched += 1
        except Exception:
            continue
    return fetched > 0


def ensure_tokenizer(
    *,
    model_key: Optional[str],
    candidates: Iterable[str],
) -> Tuple[Optional[str], Optional[str]]:
    """Return (tokenizer_dir, info) where info describes the source.

    - If a local candidate directory with tokenizer files is found → return it.
    - Else, if model_key is known, fetch from HF into models/tokenizers/<cache> and return that dir.
    - Else, return (None, None).
    """
    # First, local candidates
    for cand in candidates:
        if not cand:
            continue
        path = cand
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if _is_tokenizer_dir(path):
            return path, f"local:{path}"

    # Look under our vendored HF snapshot if present
    try:
        root = Path(__file__).resolve().parents[3] / "huggingface"
        # Search up to depth 3 for a 'tokenizer' folder with expected files
        for path in root.rglob("tokenizer"):
            if _is_tokenizer_dir(str(path)):
                return str(path), f"vendored:{path}"
    except Exception:
        pass

    # Then try preset
    if model_key and model_key in PRESET:
        meta = PRESET[model_key]
        cache_dir = os.path.join("models", "tokenizers", meta["cache"])  # not committed to VCS
        if _is_tokenizer_dir(cache_dir):
            return cache_dir, f"cache:{cache_dir}"
        ok = _fetch_hf(meta["repo"], meta["subdir"], cache_dir)
        if ok:
            return cache_dir, f"fetched:{meta['repo']}/{meta['subdir']}"

    return None, None


# ---------------- Text Encoder resolver ----------------

def _is_text_encoder_dir(path: str) -> bool:
    try:
        entries = set(os.listdir(path))
    except Exception:
        return False
    # Presence of any model weight/config suggests a usable encoder folder
    needed_any = {"config.json", "model.safetensors", "pytorch_model.bin"}
    if entries & needed_any:
        return True
    # Or nested under a subfolder
    for sub in ("text_encoder",):
        p = os.path.join(path, sub)
        try:
            subents = set(os.listdir(p))
            if subents & needed_any:
                return True
        except Exception:
            continue
    return False


def ensure_text_encoder(*, model_key: Optional[str], candidates: Iterable[str]) -> Tuple[Optional[str], Optional[str]]:
    """Return (text_encoder_dir, info). Fetches from HF if not found locally.

    Fetch target: models/text-encoders/<cache> from PRESET repo's 'text_encoder' subdir.
    Warning: downloads model weights (hundreds of MB). Use only when user allows network.
    """
    # Local candidates first
    for cand in candidates:
        if not cand:
            continue
        path = cand
        if os.path.isfile(path):
            path = os.path.dirname(path)
        if _is_text_encoder_dir(path):
            return path, f"local:{path}"

    # Vendored
    try:
        root = Path(__file__).resolve().parents[3] / "huggingface"
        for sub in ("text_encoder",):
            for path in root.rglob(sub):
                if _is_text_encoder_dir(str(path)):
                    return str(path), f"vendored:{path}"
    except Exception:
        pass

    # Preset fetch
    if model_key and model_key in PRESET:
        meta = PRESET[model_key]
        cache_dir = os.path.join("models", "text-encoders", meta["cache"])  # not committed
        if _is_text_encoder_dir(cache_dir):
            return cache_dir, f"cache:{cache_dir}"
        ok = _fetch_hf(meta["repo"], "text_encoder", cache_dir)
        if ok:
            return cache_dir, f"fetched:{meta['repo']}/text_encoder"
    return None, None
