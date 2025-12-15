from __future__ import annotations

import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional, List

from apps.backend.infra.config.paths import get_paths_for


@dataclass
class EngineOpts:
    device: str = "auto"
    dtype: str = "fp16"


@dataclass
class WanComponents:
    text_encoder: Any | None = None
    transformer: Any | None = None
    vae: Any | None = None
    pipeline: Any | None = None
    pipeline_high: Any | None = None
    pipeline_low: Any | None = None
    model_dir: str | None = None
    high_dir: str | None = None
    low_dir: str | None = None
    device: str = "cpu"
    dtype: str = "fp16"
    hf_repo_dir: Optional[str] = None
    hf_text_encoder_dir: Optional[str] = None
    hf_tokenizer_dir: Optional[str] = None
    hf_vae_dir: Optional[str] = None


@dataclass
class WanStageOptions:
    model_dir: Optional[str] = None
    sampler: Optional[str] = None
    scheduler: Optional[str] = None
    steps: int = 12
    cfg_scale: Optional[float] = None
    lora_path: Optional[str] = None
    lora_weight: Optional[float] = None
    lightning: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}

    @staticmethod
    def from_mapping(obj: Any, *, default_steps: int = 12, default_cfg: Optional[float] = None) -> "WanStageOptions":
        if not isinstance(obj, dict):
            return WanStageOptions(steps=default_steps, cfg_scale=default_cfg)
        return WanStageOptions(
            model_dir=str(obj.get("model_dir")) if obj.get("model_dir") else None,
            sampler=str(obj.get("sampler")) if obj.get("sampler") else None,
            scheduler=str(obj.get("scheduler")) if obj.get("scheduler") else None,
            steps=int(obj.get("steps") or default_steps),
            cfg_scale=(float(obj.get("cfg_scale")) if obj.get("cfg_scale") is not None else default_cfg),
            lora_path=str(obj.get("lora_path")) if obj.get("lora_path") else None,
            lora_weight=(float(obj.get("lora_weight")) if obj.get("lora_weight") is not None else None),
            lightning=bool(obj.get("lightning", False)),
        )


WAN_DIFFUSERS_REPO_CANDIDATES = {
    # Known, published Diffusers repos only (avoid ambiguous non-diffusers names)
    "wan22_14b": (
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
    ),
    "wan22_5b": (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ),
    "wan_t2v_14b": (
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
    ),
    "wan_t2v_5b": (
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    ),
}


def resolve_wan_repo_candidates(model_key: Optional[str] = None) -> List[str]:
    """Return an ordered list of WAN 2.2 Diffusers repo IDs to try.

    Strict mode: no implicit or generic fallbacks. Resolution order is:
    1) Explicit env override `CODEX_WAN_DIFFUSERS_REPO` (single repo id)
    2) Exact key match in `WAN_DIFFUSERS_REPO_CANDIDATES`

    If neither yields any candidate, raise with an actionable message.
    """
    key = (model_key or "").lower().strip()

    # 1) explicit env override (highest precedence)
    env_repo = (os.environ.get("CODEX_WAN_DIFFUSERS_REPO") or "").strip()
    if env_repo:
        return [env_repo]

    # 2) exact/contained key match from known map
    candidates: List[str] = []
    for mk, repos in WAN_DIFFUSERS_REPO_CANDIDATES.items():
        if mk in key:
            candidates.extend(repos)
            break

    if not candidates:
        valid = ", ".join(sorted(WAN_DIFFUSERS_REPO_CANDIDATES.keys()))
        raise ValueError(
            (
                "resolve_wan_repo_candidates: unable to resolve repo for key '{key}'. "
                "Provide a valid engine/model key (one of: {valid}) or set "
                "CODEX_WAN_DIFFUSERS_REPO to an explicit repo id."
            ).format(key=model_key, valid=valid)
        )

    # Deduplicate preserving order
    seen: set[str] = set()
    uniq: List[str] = []
    for rid in candidates:
        if rid and rid not in seen:
            uniq.append(rid)
            seen.add(rid)
    return uniq


__all__ = ["EngineOpts", "WanComponents", "WanStageOptions", "resolve_wan_repo_candidates", "WAN_DIFFUSERS_REPO_CANDIDATES"]


def _first_existing_path_for(key: str) -> Optional[str]:
    """Return the first existing path for a logical key.

    Semantics:
    - Values in apps/paths.json (via get_paths_for) are treated as user overrides.
    - If no override yields an existing path, fall back to built-in defaults
      under the repo's `models` tree (e.g., /models/wan22-vae).
    """
    candidates: List[str] = list(get_paths_for(key))

    # Built-in fallbacks quando não há override configurado.
    if not candidates:
        repo_root = Path(__file__).resolve().parents[4]
        if key == "wan22_vae":
            candidates.append(str(repo_root / "models" / "wan22-vae"))
        elif key == "wan22_tenc":
            # Prefer WAN22-specific encoder root.
            candidates.append(str(repo_root / "models" / "wan22-tenc"))

    for path in candidates:
        p = os.path.expanduser(path)
        if os.path.isdir(p) or os.path.isfile(p):
            return p
    return None


def resolve_user_supplied_assets(extras: dict | None, fallback_metadata_dir: Optional[str]) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Strict parsing of user-supplied asset paths.

    Accepted keys only:
    - VAE: 'wan_vae_path'
    - Text encoder: 'wan_text_encoder_path' (file) or 'wan_text_encoder_dir' (directory)
    - Tokenizer (optional): 'wan_tokenizer_dir'; if missing, use fallback_tokenizer_dir

    Returns (vae_path, text_encoder_path_or_dir, metadata_dir). No guessing for user inputs;
    WAN22 defaults are applied separately using apps/paths.json when fields stay empty.
    """
    import os
    ex = extras or {}
    vae = str(ex.get('wan_vae_path')).strip() if ex.get('wan_vae_path') else None
    te = None
    if ex.get('wan_text_encoder_path'):
        te = str(ex.get('wan_text_encoder_path')).strip()
    elif ex.get('wan_text_encoder_dir'):
        te = str(ex.get('wan_text_encoder_dir')).strip()
    meta = str(ex.get('wan_metadata_dir')).strip() if ex.get('wan_metadata_dir') else None
    if not meta and fallback_metadata_dir and os.path.isdir(fallback_metadata_dir):
        meta = fallback_metadata_dir
    return (vae if vae else None), (te if te else None), (meta if meta else None)
