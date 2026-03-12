"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 bundle planning helpers for loader/runtime handoff.
Converts the parser-normalized LTX2 checkpoint result into a typed bundle-planning contract with immutable parser-owned
component names, vendored metadata paths, and one resolved external Gemma3 text-encoder asset.

Symbols (top-level; keep in sync; no ghosts):
- `prepare_ltx2_bundle_inputs` (function): Build the typed LTX2 bundle-planning contract from loader-side parser output.
- `build_ltx2_bundle_metadata` (function): Serialize stable loader metadata for future runtime/engine assembly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.model_registry.specs import ModelFamily, ModelSignature

from .audio import validate_ltx2_audio_bundle_contract
from .config import LTX2_COMPONENT_NAMES, LTX2_REQUIRED_TEXT_ENCODER_SLOT, resolve_ltx2_vendor_paths
from .model import Ltx2BundleInputs, Ltx2ComponentStates
from .text_encoder import resolve_ltx2_text_encoder_asset
from .vae import validate_ltx2_video_vae_contract


def prepare_ltx2_bundle_inputs(
    *,
    model_ref: str,
    estimated_config: CodexEstimatedConfig,
    signature: ModelSignature,
    text_encoder_override_paths: Mapping[str, str],
    backend_root: Path,
) -> Ltx2BundleInputs:
    if signature.family is not ModelFamily.LTX2:
        raise RuntimeError(
            "LTX2 bundle planning requires `ModelFamily.LTX2`; "
            f"got {getattr(signature.family, 'value', signature.family)!r}."
        )

    components = Ltx2ComponentStates.from_component_map(
        {name: component.state_dict for name, component in estimated_config.components.items()}
    )
    validate_ltx2_video_vae_contract(components.vae)
    validate_ltx2_audio_bundle_contract(
        audio_vae_state=components.audio_vae,
        vocoder_state=components.vocoder,
    )

    vendor_paths = resolve_ltx2_vendor_paths(
        backend_root=backend_root,
        repo_id=str(estimated_config.repo_id or "").strip(),
    )
    text_encoder = resolve_ltx2_text_encoder_asset(
        override_paths=text_encoder_override_paths,
        vendor_paths=vendor_paths,
    )
    if text_encoder.alias != LTX2_REQUIRED_TEXT_ENCODER_SLOT:
        raise RuntimeError(
            "LTX2 bundle planning resolved an unexpected text-encoder alias; "
            f"expected {LTX2_REQUIRED_TEXT_ENCODER_SLOT!r}, got {text_encoder.alias!r}."
        )

    return Ltx2BundleInputs(
        model_ref=model_ref,
        signature=signature,
        estimated_config=estimated_config,
        components=components,
        text_encoder=text_encoder,
        vendor_paths=vendor_paths,
    )


def build_ltx2_bundle_metadata(inputs: Ltx2BundleInputs) -> dict[str, object]:
    return {
        "engine_key": "ltx2",
        "asset_repo_id": str(inputs.estimated_config.repo_id),
        "parser_split": str(inputs.estimated_config.extras.get("parser_split", "")),
        "component_names": LTX2_COMPONENT_NAMES,
        "tenc_path": inputs.text_encoder.path,
        "tenc_kind": inputs.text_encoder.kind,
        "tenc_alias": inputs.text_encoder.alias,
        "tokenizer_dir": inputs.text_encoder.tokenizer_dir,
        "model_index_path": inputs.vendor_paths.model_index_path,
        "connectors_config_path": inputs.vendor_paths.connectors_config_path,
        "vendor_repo_dir": inputs.vendor_paths.repo_dir,
    }
