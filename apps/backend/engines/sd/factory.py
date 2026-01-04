"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: SD-family runtime factory helpers (SD15/SD20/SDXL) for consistent engine assembly.
Centralizes SD-family runtime assembly and `CodexObjects` construction so each SD engine keeps only model-specific logic.

Symbols (top-level; keep in sync; no ghosts):
- `CodexSDFamilyAssembly` (dataclass): Assembled SD-family runtime + `CodexObjects` bundle.
- `CodexSDFamilyFactory` (class): Builder that assembles an `SDEngineRuntime` from a model bundle and produces `CodexObjects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from apps.backend.engines.common.base import CodexObjects
from apps.backend.engines.sd.spec import SDEngineRuntime, SDEngineSpec, assemble_engine_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexSDFamilyAssembly:
    runtime: SDEngineRuntime
    codex_objects: CodexObjects


class CodexSDFamilyFactory:
    """Assemble SD-family runtimes and the corresponding engine component bundle."""

    def __init__(self, *, spec: SDEngineSpec) -> None:
        self._spec = spec

    def assemble(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: dict[str, Any] | None = None,
    ) -> CodexSDFamilyAssembly:
        _ = options  # reserved for future shared SD-family build knobs
        runtime = assemble_engine_runtime(self._spec, bundle.estimated_config, bundle.components)
        codex_objects = CodexObjects(
            denoiser=runtime.unet,
            vae=runtime.vae,
            text_encoders={"clip": runtime.clip},
            clipvision=None,
        )
        return CodexSDFamilyAssembly(runtime=runtime, codex_objects=codex_objects)
