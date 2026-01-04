"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 runtime factory helpers for consistent engine assembly.
Centralizes WAN22 Codex runtime assembly and `CodexObjects` construction so WAN engines keep only model-specific logic.

Symbols (top-level; keep in sync; no ghosts):
- `CodexWan22Assembly` (dataclass): Assembled WAN runtime + `CodexObjects` bundle.
- `CodexWan22Factory` (class): Builder that assembles a `WanEngineRuntime` from a model bundle and produces `CodexObjects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects
from apps.backend.engines.wan22.spec import WanEngineRuntime, WanEngineSpec, assemble_wan_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexWan22Assembly:
    runtime: WanEngineRuntime
    codex_objects: CodexObjects


class CodexWan22Factory:
    """Assemble WAN22 Codex runtimes and the corresponding engine component bundle."""

    def __init__(self, *, spec: WanEngineSpec) -> None:
        self._spec = spec

    def assemble(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexWan22Assembly:
        device = str(options.get("device", "cuda"))
        dtype = str(options.get("dtype", "bf16"))
        runtime = assemble_wan_runtime(
            spec=self._spec,
            codex_components=bundle.components,
            estimated_config=bundle.estimated_config,
            device=device,
            dtype=dtype,
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={"t5": runtime.text.t5_text},
            clipvision=None,
        )
        return CodexWan22Assembly(runtime=runtime, codex_objects=codex_objects)

