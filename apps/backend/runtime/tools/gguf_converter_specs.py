"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed “profile + policy” specs for the GGUF converter (layouts, planners, and per-model tensor dtype rules).

Symbols (top-level; keep in sync; no ghosts):
- `GGUFArch` (enum): High-level GGUF architecture buckets used by conversion profiles.
- `GGUFKeyLayout` (enum): Target key layout kind (native keys vs. Comfy/Codex runtime keys vs. Llama HF→GGUF mapping).
- `TensorNameTarget` (enum): Whether a tensor-type rule matches source names, destination names, or both.
- `ConverterProfileId` (enum): Stable identifiers for converter profiles (model-kind + layout).
- `QuantizationCondition` (dataclass): Declarative condition for when a rule applies (include/exclude quantization selectors).
- `TensorTypeRule` (dataclass): Declarative per-tensor dtype rule (regex + target + condition + reason).
- `CompiledTensorTypeRule` (dataclass): Compiled rule used during planning (compiled regex + target + dtype + reason).
- `QuantizationPolicySpec` (dataclass): Bundle of built-in dtype rules; compiles them with optional user overrides.
- `KeyMappingSpec` (dataclass): Typed wrapper around “key mapping builders” (e.g. Llama HF→GGUF mapping).
- `PlannerSpec` (dataclass): Typed wrapper around planner implementations (Flux/ZImage planners).
- `ConverterProfileSpec` (dataclass): Full conversion profile (detection + layout + planner + policies).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Literal, Mapping, Sequence

from apps.backend.quantization.gguf import GGMLQuantizationType
from apps.backend.runtime.tools.gguf_converter_quantization import requested_ggml_type
from apps.backend.runtime.tools.gguf_converter_types import QuantizationType

_TensorNameTargetLiteral = Literal["src", "dst", "both"]


class GGUFArch(str, Enum):
    LLAMA = "llama"
    FLUX = "flux"
    ZIMAGE = "zimage"


class GGUFKeyLayout(str, Enum):
    NATIVE_KEYS = "native_keys"
    LLAMA_GGUF = "llama_gguf"
    COMFY_CODEX = "comfy_codex"


class TensorNameTarget(str, Enum):
    SRC = "src"
    DST = "dst"
    BOTH = "both"

    @classmethod
    def from_literal(cls, value: _TensorNameTargetLiteral) -> TensorNameTarget:
        return cls(value)

    def matches_src(self) -> bool:
        return self in {TensorNameTarget.SRC, TensorNameTarget.BOTH}

    def matches_dst(self) -> bool:
        return self in {TensorNameTarget.DST, TensorNameTarget.BOTH}


class ConverterProfileId(str, Enum):
    FLUX_TRANSFORMER_COMFY = "flux_transformer_comfy"
    FLUX_TRANSFORMER_NATIVE = "flux_transformer_native"
    ZIMAGE_TRANSFORMER_COMFY = "zimage_transformer_comfy"
    ZIMAGE_TRANSFORMER_NATIVE = "zimage_transformer_native"
    LLAMA_HF_TO_GGUF = "llama_hf_to_gguf"
    GENERIC_NATIVE = "generic_native"


@dataclass(frozen=True, slots=True)
class QuantizationCondition:
    include: frozenset[QuantizationType] | None = None
    exclude: frozenset[QuantizationType] = frozenset()

    def matches(self, quant: QuantizationType) -> bool:
        if quant in self.exclude:
            return False
        if self.include is None:
            return True
        return quant in self.include


@dataclass(frozen=True, slots=True)
class TensorTypeRule:
    pattern: str
    ggml_type: GGMLQuantizationType
    apply_to: TensorNameTarget = TensorNameTarget.BOTH
    when: QuantizationCondition = QuantizationCondition()
    reason: str = ""


@dataclass(frozen=True, slots=True)
class CompiledTensorTypeRule:
    pattern: re.Pattern[str]
    ggml_type: GGMLQuantizationType
    apply_to: TensorNameTarget
    reason: str = ""


@dataclass(frozen=True, slots=True)
class QuantizationPolicySpec:
    id: str
    default_rules: tuple[TensorTypeRule, ...] = ()
    required_rules: tuple[TensorTypeRule, ...] = ()

    def compile(
        self,
        *,
        quant: QuantizationType,
        user_rules: Sequence[str],
    ) -> list[CompiledTensorTypeRule]:
        compiled: list[CompiledTensorTypeRule] = []

        for rule in self.default_rules:
            if not rule.when.matches(quant):
                continue
            compiled.append(
                CompiledTensorTypeRule(
                    pattern=re.compile(rule.pattern),
                    ggml_type=rule.ggml_type,
                    apply_to=rule.apply_to,
                    reason=rule.reason,
                )
            )

        for entry in user_rules:
            raw = str(entry or "").strip()
            if not raw:
                continue
            if "=" not in raw:
                raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")
            pattern, qname = raw.split("=", 1)
            pattern = pattern.strip()
            qname = qname.strip()
            if not pattern or not qname:
                raise ValueError(f"Invalid tensor override (expected '<regex>=<quant>'): {raw!r}")

            try:
                q_enum = QuantizationType(qname.upper())
            except ValueError as exc:
                raise ValueError(f"Invalid quant type in override {raw!r}: {qname!r}") from exc

            compiled.append(
                CompiledTensorTypeRule(
                    pattern=re.compile(pattern),
                    ggml_type=requested_ggml_type(q_enum),
                    apply_to=TensorNameTarget.BOTH,
                    reason="user override",
                )
            )

        for rule in self.required_rules:
            if not rule.when.matches(quant):
                continue
            compiled.append(
                CompiledTensorTypeRule(
                    pattern=re.compile(rule.pattern),
                    ggml_type=rule.ggml_type,
                    apply_to=rule.apply_to,
                    reason=rule.reason,
                )
            )

        return compiled


@dataclass(frozen=True, slots=True)
class KeyMappingSpec:
    id: str
    build: Callable[[Mapping[str, Any]], dict[str, str]]


@dataclass(frozen=True, slots=True)
class PlannerSpec:
    id: str
    plan: Callable[..., tuple[list[Any], dict[str, str]]]
    normalize_metadata: Callable[[Mapping[str, Any]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class ConverterProfileSpec:
    id: ConverterProfileId
    arch: GGUFArch
    layout: GGUFKeyLayout
    detect: Callable[[Mapping[str, Any]], bool]
    quant_policy: QuantizationPolicySpec
    key_mapping: KeyMappingSpec | None = None
    planner: PlannerSpec | None = None


__all__ = [
    "CompiledTensorTypeRule",
    "ConverterProfileId",
    "ConverterProfileSpec",
    "GGUFArch",
    "GGUFKeyLayout",
    "KeyMappingSpec",
    "PlannerSpec",
    "QuantizationCondition",
    "QuantizationPolicySpec",
    "TensorNameTarget",
    "TensorTypeRule",
]
