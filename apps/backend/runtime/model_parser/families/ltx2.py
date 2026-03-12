"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Parser plan builder for backend-only monolithic LTX 2.x checkpoints.
Builds a strict parser plan for the combined LTX checkpoint slice where transformer/connectors, video VAE, audio VAE, and vocoder share
one state dict. Supports both direct connector aliases and wrapped `connectors.` surfaces without renaming the raw keys, keeps the family
unadvertised, and registers only the external Gemma 3 12B text-encoder alias needed for future overrides.

Symbols (top-level; keep in sync; no ghosts):
- `_LTX2_ASSET_REPO` (constant): Vendored HF repo used for parser-side asset/config resolution.
- `_DIT_REQUIRED_MARKER_GROUPS` (constant): Accepted transformer marker groups for supported LTX monoliths.
- `_CONNECTOR_WRAPPER_PREFIX` (constant): Optional wrapper prefix for monolithic connector weights embedded under `dit_root`.
- `_CONNECTOR_GROUPS` (constant): Required connector prefix groups accepted under direct or wrapped monolithic surfaces.
- `_OPTIONAL_CONNECTOR_PREFIXES` (constant): Supported connector-only prefixes used for routing but not required as standalone invariants.
- `_CONNECTOR_PREFIXES` (constant): Flattened accepted connector prefixes used for split/validation matching.
- `_COMPONENT_REQUIRED_KEYS` (constant): Required key markers per parsed component.
- `_register_ltx2_text_encoders` (function): Registers the `gemma3_12b` external text-encoder alias.
- `_strip_connector_wrapper` (function): Removes the optional `connectors.` wrapper from a `dit_root` key for validation-only matching.
- `_matches_connector_prefixes` (function): Returns True when a raw `dit_root` key matches accepted connector prefixes after wrapper normalization.
- `_has_connector_group` (function): Returns True when a key collection contains one accepted connector group.
- `_key_is_connector` (function): Returns True when a raw `dit_root` key belongs in the connector bucket.
- `_split_dit_root_state` (function): Splits stripped `dit_root` tensors into transformer vs connectors by strict prefixes.
- `_validate_dit_root_component` (function): Validates the combined LTX Dit/connector contract before build-config separation.
- `_validate_component_required_keys` (function): Validates required key markers for VAE/audio VAE/vocoder components.
- `_build_ltx2_estimated_config` (function): Builds `CodexEstimatedConfig` with explicit `transformer` and `connectors` components.
- `build_plan` (function): Builds and returns the LTX2 `ParserPlanBundle`.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from apps.backend.runtime.model_registry.specs import ModelSignature

from ..builders import build_estimated_config, register_text_encoder
from ..errors import ValidationError
from ..quantization import validate_component_dtypes
from ..specs import CodexComponent, CodexEstimatedConfig, ParserPlan, ParserPlanBundle, SplitSpec, ValidationSpec

_LTX2_ASSET_REPO = "Lightricks/LTX-2"

_DIT_REQUIRED_MARKER_GROUPS = (
    ("adaln_single.emb.timestep_embedder.linear_1.bias", "patchify_proj.weight"),
    ("av_ca_a2v_gate_adaln_single.emb.timestep_embedder.linear_1.weight", "patchify_proj.weight"),
)

_CONNECTOR_WRAPPER_PREFIX = "connectors."

_CONNECTOR_GROUPS = {
    "video_connector": ("video_embeddings_connector", "video_connector"),
    "audio_connector": ("audio_embeddings_connector", "audio_connector"),
    "text_projection": ("text_embedding_projection.aggregate_embed", "text_proj_in"),
}

_OPTIONAL_CONNECTOR_PREFIXES = ("transformer_1d_blocks",)
_CONNECTOR_PREFIXES = tuple(prefix for prefixes in _CONNECTOR_GROUPS.values() for prefix in prefixes) + _OPTIONAL_CONNECTOR_PREFIXES

_COMPONENT_REQUIRED_KEYS = {
    "vae": ("per_channel_statistics.mean-of-means", "per_channel_statistics.std-of-means"),
    "audio_vae": ("per_channel_statistics.mean-of-means",),
    "vocoder": ("conv_pre.weight", "conv_post.weight"),
}


def _register_ltx2_text_encoders(context) -> None:
    register_text_encoder(context, "gemma3_12b", "text_encoder")


def _strip_connector_wrapper(key: str) -> str:
    if key.startswith(_CONNECTOR_WRAPPER_PREFIX):
        return key[len(_CONNECTOR_WRAPPER_PREFIX) :]
    return key


def _matches_connector_prefixes(key: str, prefixes: tuple[str, ...]) -> bool:
    return _strip_connector_wrapper(key).startswith(prefixes)


def _has_connector_group(keys: Iterable[str], prefixes: tuple[str, ...]) -> bool:
    return any(_matches_connector_prefixes(key, prefixes) for key in keys)


def _key_is_connector(key: str) -> bool:
    return key.startswith(_CONNECTOR_WRAPPER_PREFIX) or _matches_connector_prefixes(key, _CONNECTOR_PREFIXES)


def _split_dit_root_state(tensors: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    transformer: dict[str, Any] = {}
    connectors: dict[str, Any] = {}
    for key, value in tensors.items():
        if _key_is_connector(key):
            connectors[key] = value
        else:
            transformer[key] = value
    return transformer, connectors


def _validate_dit_root_component(context) -> None:
    dit_root = context.require("dit_root").tensors
    if not any(all(key in dit_root for key in group) for group in _DIT_REQUIRED_MARKER_GROUPS):
        raise ValidationError(
            "LTX2 monolithic transformer markers are missing after stripping `model.diffusion_model.`. "
            "Expected one supported adaln marker set plus `patchify_proj.weight`.",
            component="dit_root",
        )

    missing_connector_groups = [
        label
        for label, prefixes in _CONNECTOR_GROUPS.items()
        if not _has_connector_group(dit_root, prefixes)
    ]
    if missing_connector_groups:
        raise ValidationError(
            "LTX2 monolithic checkpoint is missing required connector groups inside `model.diffusion_model.`. "
            "Accepted surfaces include direct connector aliases and wrapped `connectors.` keys. "
            f"missing_groups={missing_connector_groups!r}",
            component="dit_root",
        )

    transformer, connectors = _split_dit_root_state(dit_root)
    if not transformer:
        raise ValidationError(
            "LTX2 Dit root did not leave any transformer tensors after connector separation.",
            component="dit_root",
        )
    if not connectors:
        raise ValidationError(
            "LTX2 Dit root did not yield any connector tensors after connector separation.",
            component="dit_root",
        )
    if "transformer_blocks.0.attn2.to_k.weight" not in transformer:
        raise ValidationError(
            "LTX2 Dit root split removed a required core transformer key from the transformer bucket.",
            component="dit_root",
        )
    if not _has_connector_group(connectors, _CONNECTOR_PREFIXES):
        raise ValidationError(
            "LTX2 Dit root split failed to retain any connector-only keys in the connectors bucket.",
            component="dit_root",
        )


def _validate_component_required_keys(component: str, tensors: Mapping[str, Any]) -> None:
    required = _COMPONENT_REQUIRED_KEYS[component]
    missing = [key for key in required if key not in tensors]
    if missing:
        raise ValidationError(
            f"LTX2 component {component!r} is missing required keys: {missing!r}",
            component=component,
        )


def _validate_vae_component(context) -> None:
    _validate_component_required_keys("vae", context.require("vae").tensors)


def _validate_audio_vae_component(context) -> None:
    _validate_component_required_keys("audio_vae", context.require("audio_vae").tensors)


def _validate_vocoder_component(context) -> None:
    _validate_component_required_keys("vocoder", context.require("vocoder").tensors)


def _build_ltx2_estimated_config(context, signature: ModelSignature) -> CodexEstimatedConfig:
    transformer_state, connector_state = _split_dit_root_state(context.require("dit_root").tensors)
    if not transformer_state or not connector_state:
        raise ValidationError(
            "LTX2 parser build-config reached an invalid split state; transformer/connectors must both be non-empty.",
            component="config",
        )
    if "transformer_blocks.0.attn2.to_k.weight" not in transformer_state:
        raise ValidationError(
            "LTX2 parser build-config produced a transformer component without the required core transformer block key.",
            component="config",
        )
    if not _has_connector_group(connector_state, _CONNECTOR_PREFIXES):
        raise ValidationError(
            "LTX2 parser build-config produced a connectors component without any connector-only sentinel keys.",
            component="config",
        )

    base = build_estimated_config(
        context,
        signature,
        repo_override=_LTX2_ASSET_REPO,
        extra_metadata={
            "asset_repo_id": _LTX2_ASSET_REPO,
            "source_checkpoint_repo_id": (signature.extras or {}).get("source_checkpoint_repo_id", ""),
            "parser_split": "ltx2_monolith",
        },
    )

    components = {
        "transformer": CodexComponent(name="transformer", state_dict=transformer_state),
        "connectors": CodexComponent(name="connectors", state_dict=connector_state),
        "vae": CodexComponent(name="vae", state_dict=context.require("vae").tensors),
        "audio_vae": CodexComponent(name="audio_vae", state_dict=context.require("audio_vae").tensors),
        "vocoder": CodexComponent(name="vocoder", state_dict=context.require("vocoder").tensors),
    }

    return CodexEstimatedConfig(
        signature=base.signature,
        repo_id=base.repo_id,
        family=base.family,
        prediction=base.prediction,
        latent_format=base.latent_format,
        quantization=base.quantization,
        components=components,
        text_encoder_map=base.text_encoder_map,
        extras=base.extras,
        core_config=base.core_config,
    )


def build_plan(signature: ModelSignature) -> ParserPlanBundle:
    plan = ParserPlan(
        splits=(
            SplitSpec(name="dit_root", prefixes=("model.diffusion_model.",), strip_prefix=""),
            SplitSpec(name="vae", prefixes=("vae.",), strip_prefix=""),
            SplitSpec(name="audio_vae", prefixes=("audio_vae.",), strip_prefix=""),
            SplitSpec(name="vocoder", prefixes=("vocoder.",), strip_prefix=""),
        ),
        validations=(
            ValidationSpec(name="register_ltx2_text_encoders", function=_register_ltx2_text_encoders),
            ValidationSpec(name="ltx2_dit_root", function=_validate_dit_root_component),
            ValidationSpec(name="ltx2_vae", function=_validate_vae_component),
            ValidationSpec(name="ltx2_audio_vae", function=_validate_audio_vae_component),
            ValidationSpec(name="ltx2_vocoder", function=_validate_vocoder_component),
            ValidationSpec(name="dtype_sanity", function=validate_component_dtypes),
        ),
    )
    return ParserPlanBundle(plan=plan, build_config=lambda ctx: _build_ltx2_estimated_config(ctx, signature))
