"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux-family runtime factory helpers (Flux/Chroma/Kontext) for consistent engine assembly.
Centralizes Flux-family runtime assembly and `CodexObjects` construction so Flux-family engines keep only model-specific logic.

Symbols (top-level; keep in sync; no ghosts):
- `CodexFluxFamilyAssembly` (dataclass): Assembled Flux-family runtime + `CodexObjects` bundle.
- `CodexFluxFamilyFactory` (class): Builder that assembles a `FluxEngineRuntime` from a model bundle and produces `CodexObjects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects, TextEncoderHandle
from apps.backend.engines.flux.spec import FluxEngineRuntime, FluxEngineSpec, assemble_flux_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexFluxFamilyAssembly:
    runtime: FluxEngineRuntime
    codex_objects: CodexObjects


class CodexFluxFamilyFactory:
    """Assemble Flux-family runtimes and the corresponding engine component bundle."""

    def __init__(self, *, spec: FluxEngineSpec) -> None:
        self._spec = spec

    def assemble(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexFluxFamilyAssembly:
        runtime = assemble_flux_runtime(
            spec=self._spec,
            estimated_config=bundle.estimated_config,
            codex_components=bundle.components,
            engine_options=options,
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={
                "clip": TextEncoderHandle(
                    patcher=runtime.clip.patcher,
                    runtime=runtime.clip,
                )
            },
            clipvision=None,
        )
        return CodexFluxFamilyAssembly(runtime=runtime, codex_objects=codex_objects)
