"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend-owned engine dependency check contract for WebUI readiness surfaces.
Builds deterministic per-engine check rows from backend inventory/model-registry state so the frontend can render a strict
"Dependency Check" panel and disable generation when required assets are missing. Semantic-engine asset checks resolve through the
canonical contract owner seam (`contract_owner_for_semantic_engine`) to prevent drift between API surfaces.

Symbols (top-level; keep in sync; no ghosts):
- `DependencyCheckRow` (dataclass): One backend dependency row (`id/label/ok/message`) rendered by the frontend.
- `EngineDependencyStatus` (dataclass): Aggregated dependency status for one semantic engine (`ready + checks`).
- `build_engine_dependency_checks` (function): Build per-engine dependency status map for `/api/engines/capabilities`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from apps.backend.core.contracts.asset_requirements import (
    contract_for_engine,
    contract_owner_for_semantic_engine,
)
from apps.backend.infra.config.paths import get_paths_for
from apps.backend.inventory import cache as inventory_cache


@dataclass(frozen=True, slots=True)
class DependencyCheckRow:
    """One backend dependency check row shown in the frontend panel."""

    id: str
    label: str
    ok: bool
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "id": str(self.id),
            "label": str(self.label),
            "ok": bool(self.ok),
            "message": str(self.message),
        }


@dataclass(frozen=True, slots=True)
class EngineDependencyStatus:
    """Aggregated dependency checks for one semantic engine surface."""

    ready: bool
    checks: tuple[DependencyCheckRow, ...]

    @classmethod
    def from_checks(cls, checks: Iterable[DependencyCheckRow]) -> "EngineDependencyStatus":
        rows = tuple(checks)
        return cls(
            ready=all(bool(row.ok) for row in rows),
            checks=rows,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "ready": bool(self.ready),
            "checks": [row.as_dict() for row in self.checks],
        }


_CHECKPOINT_REQUIRED_ENGINES: frozenset[str] = frozenset(
    {
        "sd15",
        "sdxl",
        "flux1",
        "flux2",
        "chroma",
        "zimage",
        "anima",
        "svd",
        "hunyuan_video",
    }
)

_WAN_METADATA_PREFIX = "wan-ai/wan2.2-"

_CHECKPOINT_ROOT_KEYS_BY_ENGINE: dict[str, tuple[str, ...]] = {
    "sd15": ("sd15_ckpt",),
    "sdxl": ("sdxl_ckpt",),
    "flux1": ("flux1_ckpt",),
    "flux2": ("flux2_ckpt",),
    "chroma": ("flux1_ckpt",),
    "zimage": ("zimage_ckpt",),
    "anima": ("anima_ckpt",),
    "wan22": ("wan22_ckpt",),
}

_CHECKPOINT_FAMILY_HINTS_BY_ENGINE: dict[str, tuple[str, ...]] = {
    "sd15": ("sd15",),
    "sdxl": ("sdxl",),
    "flux1": ("flux1",),
    "flux2": ("flux2",),
    "chroma": ("chroma", "flux1"),
    "zimage": ("zimage",),
    "anima": ("anima",),
    "wan22": ("wan22",),
}

_VAE_ROOT_KEYS_BY_CONTRACT_OWNER: dict[str, tuple[str, ...]] = {
    "flux1": ("flux1_vae",),
    "flux2": ("flux2_vae",),
    "zimage": ("zimage_vae", "flux1_vae"),
    "anima": ("anima_vae",),
    "wan22_5b": ("wan22_vae",),
    "wan22_14b": ("wan22_vae",),
    "wan22_14b_animate": ("wan22_vae",),
}

_TEXT_ENCODER_ROOT_KEYS_BY_CONTRACT_OWNER: dict[str, tuple[str, ...]] = {
    "flux1": ("flux1_tenc",),
    "flux2": ("flux2_tenc",),
    "zimage": ("zimage_tenc",),
    "anima": ("anima_tenc",),
    "wan22_5b": ("wan22_tenc",),
    "wan22_14b": ("wan22_tenc",),
    "wan22_14b_animate": ("wan22_tenc",),
}


def _count_list_entries(value: object) -> int:
    if not isinstance(value, list):
        return 0
    return sum(1 for item in value if isinstance(item, dict))


def _count_wan_metadata_repos(value: object) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip().lower()
        if name.startswith(_WAN_METADATA_PREFIX):
            count += 1
    return count


def _normalized_path(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    if len(text) <= 1:
        return text
    return text.rstrip("/")


def _path_in_roots(path: str, roots: list[str]) -> bool:
    norm_path = _normalized_path(path)
    if not norm_path:
        return False

    for raw_root in roots:
        root = _normalized_path(raw_root)
        if not root:
            continue
        if norm_path == root or norm_path.startswith(root + "/"):
            return True
        if root.startswith("/"):
            rel = root.lstrip("/")
            if norm_path.endswith("/" + rel) or ("/" + rel + "/") in norm_path:
                return True
    return False


def _count_assets_in_roots(value: object, roots: list[str]) -> int:
    if not isinstance(value, list):
        return 0
    count = 0
    for item in value:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "").strip()
        if _path_in_roots(path, roots):
            count += 1
    return count


def _roots_for_keys(keys: tuple[str, ...]) -> list[str]:
    roots: list[str] = []
    for key in keys:
        for raw in get_paths_for(key):
            try:
                resolved = str(Path(raw).resolve())
            except Exception:
                resolved = str(raw or "").strip()
            if resolved and resolved not in roots:
                roots.append(resolved)
    return roots


def _count_checkpoints_for_engine(model_api: Any, semantic_engine: str) -> int:
    records = model_api.list_checkpoints(refresh=False)
    if not isinstance(records, list):
        return 0

    family_hints = tuple(str(value).strip().lower() for value in _CHECKPOINT_FAMILY_HINTS_BY_ENGINE.get(semantic_engine, ()))
    if family_hints:
        scoped = 0
        for record in records:
            hint = str(getattr(record, "family_hint", "") or "").strip().lower()
            if hint in family_hints:
                scoped += 1
        if scoped > 0:
            return scoped

    root_keys = _CHECKPOINT_ROOT_KEYS_BY_ENGINE.get(semantic_engine, ())
    roots = _roots_for_keys(root_keys)
    if not roots:
        return len(records)

    scoped = 0
    for record in records:
        path = str(getattr(record, "filename", "") or "").strip()
        if _path_in_roots(path, roots):
            scoped += 1
    return scoped


def _text_encoder_slots(value: object) -> set[str]:
    if not isinstance(value, list):
        return set()
    slots: set[str] = set()
    for item in value:
        if not isinstance(item, dict):
            continue
        slot = str(item.get("slot") or "").strip()
        if slot:
            slots.add(slot)
    return slots


def _checkpoint_count(model_api: Any) -> int:
    records = model_api.list_checkpoints(refresh=False)
    if isinstance(records, list):
        return len(records)
    return 0


def build_engine_dependency_checks(
    *,
    engine_capabilities: Mapping[str, Mapping[str, object]],
    model_api: Any,
) -> dict[str, dict[str, object]]:
    """Build backend-owned dependency checks for semantic engines.

    Args:
        engine_capabilities: Capability surfaces keyed by semantic engine.
        model_api: Runtime models API facade (must expose `list_checkpoints(refresh=...)`).

    Returns:
        Dict keyed by semantic engine where each value has:
        - `ready`: bool
        - `checks`: list of dependency rows (`id`, `label`, `ok`, `message`)
    """

    inventory = inventory_cache.get()
    vae_count = _count_list_entries(inventory.get("vaes"))
    text_encoder_count = _count_list_entries(inventory.get("text_encoders"))
    text_encoder_slots = _text_encoder_slots(inventory.get("text_encoders"))
    wan_model_count = _count_list_entries(inventory.get("wan22"))
    wan_metadata_count = _count_wan_metadata_repos(inventory.get("metadata"))
    wan_tenc_roots = [str(Path(p).resolve()) for p in get_paths_for("wan22_tenc")]
    wan_vae_roots = [str(Path(p).resolve()) for p in get_paths_for("wan22_vae")]
    wan_text_encoder_count = _count_assets_in_roots(inventory.get("text_encoders"), wan_tenc_roots)
    wan_vae_count = _count_assets_in_roots(inventory.get("vaes"), wan_vae_roots)

    result: dict[str, dict[str, object]] = {}
    for semantic_engine in sorted(engine_capabilities.keys()):
        checks: list[DependencyCheckRow] = []

        checks.append(
            DependencyCheckRow(
                id="capability_surface",
                label="Capability Surface",
                ok=True,
                message=f"Backend capability surface '{semantic_engine}' loaded.",
            )
        )

        if semantic_engine in _CHECKPOINT_REQUIRED_ENGINES:
            checkpoint_count = _count_checkpoints_for_engine(model_api, semantic_engine)
            has_checkpoint = checkpoint_count > 0
            checks.append(
                DependencyCheckRow(
                    id="checkpoint_inventory",
                    label="Model Checkpoints",
                    ok=has_checkpoint,
                    message=(
                        f"{checkpoint_count} checkpoint(s) discovered by backend registry."
                        if has_checkpoint
                        else (
                            "No checkpoints discovered by backend registry. "
                            "Add at least one checkpoint and refresh model inventory."
                        )
                    ),
                )
            )

        contract_engine = contract_owner_for_semantic_engine(semantic_engine)
        contract = contract_for_engine(contract_engine)
        scoped_vae_roots = _roots_for_keys(_VAE_ROOT_KEYS_BY_CONTRACT_OWNER.get(contract_engine, ()))
        scoped_tenc_roots = _roots_for_keys(_TEXT_ENCODER_ROOT_KEYS_BY_CONTRACT_OWNER.get(contract_engine, ()))
        scoped_vae_count = (
            _count_assets_in_roots(inventory.get("vaes"), scoped_vae_roots)
            if scoped_vae_roots
            else vae_count
        )
        scoped_text_encoder_count = (
            _count_assets_in_roots(inventory.get("text_encoders"), scoped_tenc_roots)
            if scoped_tenc_roots
            else text_encoder_count
        )
        scoped_text_encoder_slots = (
            {
                str(item.get("slot") or "").strip()
                for item in inventory.get("text_encoders", [])
                if isinstance(item, dict)
                and _path_in_roots(str(item.get("path") or "").strip(), scoped_tenc_roots)
                and str(item.get("slot") or "").strip()
            }
            if scoped_tenc_roots
            else text_encoder_slots
        )
        if contract.requires_vae:
            has_vae = scoped_vae_count > 0
            checks.append(
                DependencyCheckRow(
                    id="vae_inventory",
                    label="VAE Inventory",
                    ok=has_vae,
                    message=(
                        f"{scoped_vae_count} VAE file(s) discovered."
                        if has_vae
                        else "No VAE files discovered. Configure VAE roots and refresh inventory."
                    ),
                )
            )
        if contract.tenc_count > 0:
            required = int(contract.tenc_count)
            has_tenc = scoped_text_encoder_count >= required
            checks.append(
                DependencyCheckRow(
                    id="text_encoder_inventory",
                    label="Text Encoder Inventory",
                    ok=has_tenc,
                    message=(
                        f"{scoped_text_encoder_count} text encoder file(s) discovered (requires >= {required})."
                        if has_tenc
                        else (
                            f"Only {scoped_text_encoder_count} text encoder file(s) discovered "
                            f"(requires >= {required}). Configure text-encoder roots and refresh inventory."
                        )
                    ),
                )
            )
            required_slots = tuple(str(slot) for slot in contract.tenc_slots)
            if required_slots:
                missing_slots = [slot for slot in required_slots if slot not in scoped_text_encoder_slots]
                has_required_slots = len(missing_slots) == 0
                checks.append(
                    DependencyCheckRow(
                        id="text_encoder_slots",
                        label="Text Encoder Slots",
                        ok=has_required_slots,
                        message=(
                            f"Required slots available: {', '.join(required_slots)}."
                            if has_required_slots
                            else (
                                "Missing required text encoder slot(s): "
                                f"{', '.join(missing_slots)}."
                            )
                        ),
                    )
                )

        if semantic_engine == "wan22":
            has_wan_models = wan_model_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_models_inventory",
                    label="WAN Models",
                    ok=has_wan_models,
                    message=(
                        f"{wan_model_count} WAN GGUF model(s) discovered."
                        if has_wan_models
                        else "No WAN GGUF models discovered. Configure WAN roots and refresh inventory."
                    ),
                )
            )

            has_wan_text_encoder = wan_text_encoder_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_text_encoder_inventory",
                    label="WAN Text Encoder",
                    ok=has_wan_text_encoder,
                    message=(
                        f"{wan_text_encoder_count} WAN text encoder file(s) discovered."
                        if has_wan_text_encoder
                        else (
                            "No text encoders discovered under WAN roots. "
                            "Configure `wan22_tenc` roots and refresh inventory."
                        )
                    ),
                )
            )

            has_wan_vae = wan_vae_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_vae_inventory",
                    label="WAN VAE",
                    ok=has_wan_vae,
                    message=(
                        f"{wan_vae_count} WAN VAE file(s) discovered."
                        if has_wan_vae
                        else (
                            "No VAE files discovered under WAN roots. "
                            "Configure `wan22_vae` roots and refresh inventory."
                        )
                    ),
                )
            )

            has_wan_metadata = wan_metadata_count > 0
            checks.append(
                DependencyCheckRow(
                    id="wan_metadata_inventory",
                    label="WAN Metadata",
                    ok=has_wan_metadata,
                    message=(
                        f"{wan_metadata_count} WAN metadata repo(s) discovered under apps/backend/huggingface."
                        if has_wan_metadata
                        else (
                            "No WAN metadata repository discovered under apps/backend/huggingface. "
                            "Vendor a Wan2.2 metadata repo (e.g. Wan-AI/Wan2.2-I2V-A14B-Diffusers)."
                        )
                    ),
                )
            )

        status = EngineDependencyStatus.from_checks(checks)
        result[semantic_engine] = status.as_dict()

    return result


__all__ = [
    "DependencyCheckRow",
    "EngineDependencyStatus",
    "build_engine_dependency_checks",
]
