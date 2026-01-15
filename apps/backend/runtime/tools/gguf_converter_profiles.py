"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Converter profile registry for GGUF conversion (selects layout/planner/key mapping + per-model dtype policies).

Symbols (top-level; keep in sync; no ghosts):
- `_is_flux` (function): Detect whether a config.json describes a Flux transformer.
- `_is_zimage` (function): Detect whether a config.json describes a ZImage transformer.
- `_build_llama_mapping` (function): Build a Llama HF→GGUF key mapping from the model config.
- `_COND_QUANTIZED` (constant): Condition helper matching any quantized preset (non-F16/F32).
- `_COND_FLUX_MIXED` (constant): Condition helper matching Flux mixed presets (`Q5_K_M`/`Q4_K_M`).
- `FLUX_QUANT_POLICY` (constant): Flux per-tensor dtype policy (mixed presets keep more IO weights in float32).
- `ZIMAGE_QUANT_POLICY` (constant): ZImage per-tensor dtype policy (pad tokens must remain float).
- `LLAMA_QUANT_POLICY` (constant): Llama per-tensor dtype policy (mixed presets bump key weights to higher precision).
- `PROFILE_REGISTRY` (constant): Registry of built-in converter profiles (table-driven dispatch).
- `resolve_profile` (function): Resolve the effective `ConverterProfileSpec` from a config.json and `comfy_layout`.
- `profile_by_id` (function): Resolve a profile by its stable id string (no heuristics).
"""

from __future__ import annotations

from typing import Any, Mapping

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.runtime.tools import gguf_converter_key_mapping as _key_mapping
from apps.backend.runtime.tools import gguf_converter_tensor_planner as _tensor_planner
from apps.backend.runtime.tools.gguf_converter_specs import (
    CompiledTensorTypeRule,
    ConverterProfileId,
    ConverterProfileSpec,
    GGUFArch,
    GGUFKeyLayout,
    KeyMappingSpec,
    PlannerSpec,
    QuantizationCondition,
    QuantizationPolicySpec,
    TensorNameTarget,
    TensorTypeRule,
)
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType


def _is_flux(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_flux_transformer_config(config)


def _is_zimage(config: Mapping[str, Any]) -> bool:
    return _tensor_planner.is_zimage_transformer_config(config)


def _build_llama_mapping(config: Mapping[str, Any]) -> dict[str, str]:
    num_layers = int(config.get("num_hidden_layers", 32))
    return _key_mapping.build_key_mapping(num_layers)


_COND_QUANTIZED = QuantizationCondition(exclude=frozenset({QuantizationType.F16, QuantizationType.F32}))
_COND_FLUX_MIXED = QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M, QuantizationType.Q4_K_M}))


FLUX_QUANT_POLICY = QuantizationPolicySpec(
    id="flux",
    # Required model policy: do not allow user overrides to violate these.
    required_rules=(
        TensorTypeRule(
            pattern=r"^img_in\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux input embedder is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^(?:guidance_in|time_in|vector_in)\.in_layer\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux in-projections are quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^final_layer\.linear\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux output projection is quality-sensitive; keep float",
        ),
        TensorTypeRule(
            pattern=r"^(?:guidance_in|time_in|vector_in)\.out_layer\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux out-projections can be F16 without visible regressions",
        ),
        TensorTypeRule(
            pattern=r"^txt_in\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux txt_in can be F16 while preserving prompt semantics",
        ),
        TensorTypeRule(
            pattern=r"^final_layer\.adaLN_modulation\.1\.weight$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux final_layer modulation can be F16",
        ),
        TensorTypeRule(
            pattern=r"^(?:double_blocks|single_blocks)\..*\.(?:bias|scale)$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux 1D tensors (biases/scales) stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^(?:img_in|txt_in)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux biases stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^(?:guidance_in|time_in|vector_in)\.(?:in_layer|out_layer)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux biases stay F32 for stability",
        ),
        TensorTypeRule(
            pattern=r"^final_layer\.(?:linear|adaLN_modulation\.1)\.bias$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_QUANTIZED,
            reason="Flux biases stay F32 for stability",
        ),
        # Mixed presets are explicitly allowed to trade size for quality.
        TensorTypeRule(
            pattern=r"^(?:guidance_in|time_in|vector_in)\.out_layer\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_FLUX_MIXED,
            reason="Flux mixed preset: keep out-projections float32 for higher quality",
        ),
        TensorTypeRule(
            pattern=r"^txt_in\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_FLUX_MIXED,
            reason="Flux mixed preset: keep txt_in float32 for higher quality",
        ),
        TensorTypeRule(
            pattern=r"^final_layer\.adaLN_modulation\.1\.weight$",
            ggml_type=GGMLQuantizationType.F32,
            apply_to=TensorNameTarget.DST,
            when=_COND_FLUX_MIXED,
            reason="Flux mixed preset: keep final modulation float32 for higher quality",
        ),
    ),
)


ZIMAGE_QUANT_POLICY = QuantizationPolicySpec(
    id="zimage",
    required_rules=(
        TensorTypeRule(
            pattern=r"^(?:x_pad_token|cap_pad_token)$",
            ggml_type=GGMLQuantizationType.F16,
            apply_to=TensorNameTarget.BOTH,
            when=_COND_QUANTIZED,
            reason="ZImage pad tokens must remain float (load_state_dict cannot load quantized tensors)",
        ),
    ),
)


LLAMA_QUANT_POLICY = QuantizationPolicySpec(
    id="llama",
    default_rules=(
        TensorTypeRule(
            pattern=r"(?:^|\.)token_embd\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM embeddings: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)output\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM output head: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"model\.embed_tokens\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM embeddings: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"lm_head\.weight$",
            ggml_type=GGMLQuantizationType.Q8_0,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM output head: keep higher precision to preserve semantics",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)attn_(?:q|k|v|output)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM attention projections: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q5_K_M})),
            reason="LLM attention projections: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)token_embd\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM embeddings: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)output\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM output head: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"model\.embed_tokens\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM embeddings: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"lm_head\.weight$",
            ggml_type=GGMLQuantizationType.Q6_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM output head: bump to 6-bit K",
        ),
        TensorTypeRule(
            pattern=r"(?:^|\.)attn_(?:q|k|v|output)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM attention projections: bump to 5-bit K",
        ),
        TensorTypeRule(
            pattern=r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj)\.weight$",
            ggml_type=GGMLQuantizationType.Q5_K,
            apply_to=TensorNameTarget.BOTH,
            when=QuantizationCondition(include=frozenset({QuantizationType.Q4_K_M})),
            reason="LLM attention projections: bump to 5-bit K",
        ),
    ),
)


GENERIC_QUANT_POLICY = QuantizationPolicySpec(id="generic")


_LLAMA_KEY_MAPPING = KeyMappingSpec(id="llama_hf_to_gguf", build=_build_llama_mapping)


def _plan_flux(
    tensor_names: list[str],
    safetensors_handle: Any,
    requested_type: GGMLQuantizationType,
    rules: list[CompiledTensorTypeRule],
) -> tuple[list[Any], dict[str, str]]:
    return _tensor_planner.plan_flux_transformer_tensors(tensor_names, safetensors_handle, requested_type, rules)


def _plan_zimage(
    tensor_names: list[str],
    safetensors_handle: Any,
    requested_type: GGMLQuantizationType,
    rules: list[CompiledTensorTypeRule],
) -> tuple[list[Any], dict[str, str]]:
    return _tensor_planner.plan_zimage_transformer_tensors(tensor_names, safetensors_handle, requested_type, rules)


_FLUX_PLANNER = PlannerSpec(
    id="flux_transformer",
    plan=_plan_flux,
    normalize_metadata=_tensor_planner.normalize_flux_transformer_metadata_config,
)

_ZIMAGE_PLANNER = PlannerSpec(
    id="zimage_transformer",
    plan=_plan_zimage,
    normalize_metadata=_tensor_planner.normalize_zimage_transformer_metadata_config,
)


PROFILE_REGISTRY: tuple[ConverterProfileSpec, ...] = (
    ConverterProfileSpec(
        id=ConverterProfileId.FLUX_TRANSFORMER_COMFY,
        arch=GGUFArch.FLUX,
        layout=GGUFKeyLayout.COMFY_CODEX,
        detect=_is_flux,
        quant_policy=FLUX_QUANT_POLICY,
        planner=_FLUX_PLANNER,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.FLUX_TRANSFORMER_NATIVE,
        arch=GGUFArch.FLUX,
        layout=GGUFKeyLayout.NATIVE_KEYS,
        detect=_is_flux,
        quant_policy=FLUX_QUANT_POLICY,
        planner=_FLUX_PLANNER,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.ZIMAGE_TRANSFORMER_COMFY,
        arch=GGUFArch.ZIMAGE,
        layout=GGUFKeyLayout.COMFY_CODEX,
        detect=_is_zimage,
        quant_policy=ZIMAGE_QUANT_POLICY,
        planner=_ZIMAGE_PLANNER,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.ZIMAGE_TRANSFORMER_NATIVE,
        arch=GGUFArch.ZIMAGE,
        layout=GGUFKeyLayout.NATIVE_KEYS,
        detect=_is_zimage,
        quant_policy=ZIMAGE_QUANT_POLICY,
        planner=_ZIMAGE_PLANNER,
    ),
    ConverterProfileSpec(
        id=ConverterProfileId.LLAMA_HF_TO_GGUF,
        arch=GGUFArch.LLAMA,
        layout=GGUFKeyLayout.LLAMA_GGUF,
        detect=lambda cfg: not _is_flux(cfg) and not _is_zimage(cfg),
        quant_policy=LLAMA_QUANT_POLICY,
        key_mapping=_LLAMA_KEY_MAPPING,
    ),
)


def resolve_profile(config_json: Mapping[str, Any], *, comfy_layout: bool) -> ConverterProfileSpec:
    if _is_flux(config_json):
        return PROFILE_REGISTRY[0] if comfy_layout else PROFILE_REGISTRY[1]
    if _is_zimage(config_json):
        return PROFILE_REGISTRY[2] if comfy_layout else PROFILE_REGISTRY[3]
    return PROFILE_REGISTRY[4]


def profile_by_id(profile_id: str) -> ConverterProfileSpec:
    try:
        pid = ConverterProfileId(str(profile_id))
    except ValueError as exc:
        raise ValueError(f"Unknown GGUF converter profile_id: {profile_id!r}") from exc

    for profile in PROFILE_REGISTRY:
        if profile.id is pid:
            return profile

    raise ValueError(f"GGUF converter profile_id not registered: {profile_id!r}")


__all__ = ["PROFILE_REGISTRY", "profile_by_id", "resolve_profile"]
