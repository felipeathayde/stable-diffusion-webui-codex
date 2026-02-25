"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima runtime factory helpers for consistent engine assembly.
Centralizes Anima runtime assembly and `CodexObjects` construction so the engine facade stays thin and delegates
mode pipelines to canonical use-cases (Option A).

Symbols (top-level; keep in sync; no ghosts):
- `CodexAnimaAssembly` (dataclass): Assembled runtime + `CodexObjects` bundle.
- `CodexAnimaFactory` (class): Builder that assembles `AnimaEngineRuntime` from a model bundle and engine options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects, TextEncoderHandle
from apps.backend.engines.anima.spec import AnimaEngineRuntime, AnimaEngineSpec, assemble_anima_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexAnimaAssembly:
    runtime: AnimaEngineRuntime
    codex_objects: CodexObjects


class CodexAnimaFactory:
    def __init__(self, *, spec: AnimaEngineSpec) -> None:
        self._spec = spec

    def assemble(self, bundle: DiffusionModelBundle, *, options: Mapping[str, Any]) -> CodexAnimaAssembly:
        runtime = assemble_anima_runtime(
            spec=self._spec,
            estimated_config=bundle.estimated_config,
            codex_components=bundle.components,
            engine_options=options,
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={
                "qwen3": TextEncoderHandle(
                    patcher=runtime.qwen,
                    runtime=runtime.qwen,
                )
            },
            clipvision=None,
        )
        return CodexAnimaAssembly(runtime=runtime, codex_objects=codex_objects)


__all__ = ["CodexAnimaAssembly", "CodexAnimaFactory"]
