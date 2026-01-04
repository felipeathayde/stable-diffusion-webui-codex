"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Kontext runtime factory helpers for consistent engine assembly.
Centralizes Flux-derived Kontext runtime assembly and `CodexObjects` construction so the engine keeps only Kontext-specific behavior.

Symbols (top-level; keep in sync; no ghosts):
- `CodexKontextAssembly` (dataclass): Assembled Kontext runtime + `CodexObjects` bundle.
- `CodexKontextFactory` (class): Builder that assembles a `FluxEngineRuntime` from a model bundle and produces `CodexObjects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects
from apps.backend.engines.flux.spec import FluxEngineRuntime, FluxEngineSpec, assemble_flux_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexKontextAssembly:
    runtime: FluxEngineRuntime
    codex_objects: CodexObjects


class CodexKontextFactory:
    """Assemble Kontext runtimes and the corresponding engine component bundle."""

    def __init__(self, *, spec: FluxEngineSpec) -> None:
        self._spec = spec

    def assemble(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexKontextAssembly:
        runtime = assemble_flux_runtime(
            spec=self._spec,
            estimated_config=bundle.estimated_config,
            codex_components=bundle.components,
            engine_options=options,
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={"clip": runtime.clip},
            clipvision=None,
        )
        return CodexKontextAssembly(runtime=runtime, codex_objects=codex_objects)

