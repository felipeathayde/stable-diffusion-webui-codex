"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical per-engine asset requirements (VAE/text encoders) for generation requests.
Centralizes “what is required” so UI ↔ API ↔ loader can stay in sync and drift cannot reappear via duplicated `engine_id in (...)` logic.

Symbols (top-level; keep in sync; no ghosts):
- `TextEncoderKind` (enum): UI-friendly label for the expected text encoder selection kind.
- `EngineAssetContract` (dataclass): Required VAE/text encoder contract for a specific engine request context.
- `contract_for_engine` (function): Base contract for an engine when the selected checkpoint is not core-only.
- `contract_for_core_only` (function): Contract for an engine when the selected checkpoint is core-only.
- `contract_for_request` (function): Resolve the effective contract for an engine request (e.g. core-only checkpoints).
- `format_text_encoder_kind_label` (function): Human label used in error messages and UI copy.
- `known_engine_ids` (function): Returns the set of engine ids covered by this contract module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from apps.backend.infra.config.env_flags import env_flag


class TextEncoderKind(str, Enum):
    NONE = "none"
    CLIP = "clip"
    SDXL = "sdxl"
    CLIP_T5 = "clip+t5"
    T5 = "t5"
    QWEN = "qwen"
    SD3 = "sd3"


def format_text_encoder_kind_label(kind: TextEncoderKind) -> str:
    if kind is TextEncoderKind.NONE:
        return "None"
    if kind is TextEncoderKind.CLIP:
        return "CLIP"
    if kind is TextEncoderKind.SDXL:
        return "SDXL (CLIP-L + CLIP-G)"
    if kind is TextEncoderKind.CLIP_T5:
        return "CLIP + T5"
    if kind is TextEncoderKind.T5:
        return "T5"
    if kind is TextEncoderKind.QWEN:
        return "Qwen"
    if kind is TextEncoderKind.SD3:
        return "SD3 (CLIP-L + CLIP-G + T5)"
    return str(kind.value)


@dataclass(frozen=True, slots=True)
class EngineAssetContract:
    """Asset requirements for an engine request context."""

    requires_vae: bool
    tenc_count: int
    tenc_kind: TextEncoderKind
    sha_only: bool
    notes: str = ""

    def __post_init__(self) -> None:
        if int(self.tenc_count) < 0:
            raise ValueError("tenc_count must be >= 0")
        if int(self.tenc_count) == 0 and self.tenc_kind is not TextEncoderKind.NONE:
            raise ValueError("tenc_kind must be NONE when tenc_count is 0")
        if int(self.tenc_count) > 0 and self.tenc_kind is TextEncoderKind.NONE:
            raise ValueError("tenc_kind must not be NONE when tenc_count is > 0")

    @property
    def requires_text_encoders(self) -> bool:
        return int(self.tenc_count) > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "requires_vae": bool(self.requires_vae),
            "tenc_count": int(self.tenc_count),
            "tenc_kind": str(self.tenc_kind.value),
            "tenc_kind_label": format_text_encoder_kind_label(self.tenc_kind),
            "sha_only": bool(self.sha_only),
            "notes": str(self.notes or ""),
        }


_BASE_CONTRACTS: dict[str, EngineAssetContract] = {
    # Diffusion checkpoints embed VAE/text encoders; external assets are optional overrides.
    "sd15": EngineAssetContract(
        requires_vae=False,
        tenc_count=0,
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sd20": EngineAssetContract(
        requires_vae=False,
        tenc_count=0,
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sdxl": EngineAssetContract(
        requires_vae=False,
        tenc_count=0,
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sdxl_refiner": EngineAssetContract(
        requires_vae=False,
        tenc_count=0,
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Monolithic checkpoint; external VAE/text encoders are optional overrides.",
    ),
    "sd35": EngineAssetContract(
        requires_vae=False,
        tenc_count=0,
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Diffusers-style checkpoint; external VAE/text encoders are optional overrides.",
    ),
    # External-assets-first families.
    "flux1": EngineAssetContract(
        requires_vae=True,
        tenc_count=2,
        tenc_kind=TextEncoderKind.CLIP_T5,
        sha_only=True,
        notes="External-assets-first: requires VAE + 2 text encoders (CLIP + T5) via sha selection.",
    ),
    "flux1_kontext": EngineAssetContract(
        requires_vae=True,
        tenc_count=2,
        tenc_kind=TextEncoderKind.CLIP_T5,
        sha_only=True,
        notes="External-assets-first: requires VAE + 2 text encoders (CLIP + T5) via sha selection.",
    ),
    "zimage": EngineAssetContract(
        requires_vae=True,
        tenc_count=1,
        tenc_kind=TextEncoderKind.QWEN,
        sha_only=True,
        notes="External-assets-first: requires Flow16 VAE + 1 Qwen text encoder via sha selection.",
    ),
    # Chroma safetensors are treated as monolithic; GGUF selections remain core-only.
    "flux1_chroma": EngineAssetContract(
        requires_vae=False,
        tenc_count=0,
        tenc_kind=TextEncoderKind.NONE,
        sha_only=True,
        notes="Chroma safetensors are treated as monolithic; external assets are optional overrides.",
    ),
}


def contract_for_engine(engine_id: str) -> EngineAssetContract:
    """Return the base contract for an engine.

    This is the contract for non-core-only checkpoint selections.
    """

    key = str(engine_id or "").strip().lower()
    if not key:
        raise ValueError("engine_id required")
    contract = _BASE_CONTRACTS.get(key)
    if contract is None:
        raise KeyError(f"Engine asset contract missing for engine_id={key!r}")
    return contract


def contract_for_core_only(engine_id: str) -> EngineAssetContract:
    """Return the contract when the selected checkpoint is core-only."""

    key = str(engine_id or "").strip().lower()
    if not key:
        raise ValueError("engine_id required")

    if key in ("flux1", "flux1_kontext", "zimage"):
        return contract_for_engine(key)

    if key == "flux1_chroma":
        return EngineAssetContract(
            requires_vae=True,
            tenc_count=1,
            tenc_kind=TextEncoderKind.T5,
            sha_only=True,
            notes="Core-only checkpoint: requires external VAE + 1 T5 text encoder.",
        )

    if key in ("sd15", "sd20"):
        return EngineAssetContract(
            requires_vae=True,
            tenc_count=1,
            tenc_kind=TextEncoderKind.CLIP,
            sha_only=True,
            notes="Core-only checkpoint: requires external VAE + 1 CLIP text encoder.",
        )

    if key in ("sdxl", "sdxl_refiner"):
        return EngineAssetContract(
            requires_vae=True,
            tenc_count=2,
            tenc_kind=TextEncoderKind.SDXL,
            sha_only=True,
            notes="Core-only checkpoint: requires external VAE + 2 SDXL text encoders.",
        )

    if key == "sd35":
        enable_t5 = env_flag("CODEX_SD3_ENABLE_T5", default=True)
        count = 3 if enable_t5 else 2
        return EngineAssetContract(
            requires_vae=True,
            tenc_count=count,
            tenc_kind=TextEncoderKind.SD3,
            sha_only=True,
            notes=(
                "Core-only checkpoint: requires external VAE + SD3 text encoders "
                f"(tenc_count={count}; CODEX_SD3_ENABLE_T5={bool(enable_t5)})."
            ),
        )

    base = contract_for_engine(key)
    return EngineAssetContract(
        requires_vae=True,
        tenc_count=1,
        tenc_kind=TextEncoderKind.CLIP,
        sha_only=bool(base.sha_only),
        notes="Core-only checkpoint: requires external VAE + at least one text encoder (default contract).",
    )


def contract_for_request(*, engine_id: str, checkpoint_core_only: bool) -> EngineAssetContract:
    """Resolve the effective asset contract for an engine request."""

    if checkpoint_core_only:
        return contract_for_core_only(engine_id)
    return contract_for_engine(engine_id)


def known_engine_ids() -> tuple[str, ...]:
    """Return engine ids covered by the contract mapping."""

    return tuple(sorted(_BASE_CONTRACTS.keys()))
