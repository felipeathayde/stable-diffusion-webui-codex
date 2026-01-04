"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image runtime factory helpers for consistent engine assembly.
Centralizes Z-Image runtime assembly and `CodexObjects` construction so the engine keeps only Z-Image-specific behavior.

Symbols (top-level; keep in sync; no ghosts):
- `CodexZImageAssembly` (dataclass): Assembled Z-Image runtime + `CodexObjects` bundle.
- `CodexZImageFactory` (class): Builder that assembles a `ZImageEngineRuntime` from a model bundle and produces `CodexObjects`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from apps.backend.engines.common.base import CodexObjects
from apps.backend.engines.zimage.spec import ZImageEngineRuntime, ZImageEngineSpec, assemble_zimage_runtime
from apps.backend.runtime.models.loader import DiffusionModelBundle


@dataclass(frozen=True, slots=True)
class CodexZImageAssembly:
    runtime: ZImageEngineRuntime
    codex_objects: CodexObjects


class CodexZImageFactory:
    """Assemble Z-Image runtimes and the corresponding engine component bundle."""

    def __init__(self, *, spec: ZImageEngineSpec) -> None:
        self._spec = spec

    def assemble(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexZImageAssembly:
        device = str(options.get("device", "cuda"))
        dtype = str(options.get("dtype", "bf16"))
        runtime = assemble_zimage_runtime(
            spec=self._spec,
            codex_components=bundle.components,
            estimated_config=bundle.estimated_config,
            device=device,
            dtype=dtype,
            external_vae_path=options.get("vae_path"),
            external_tenc_path=options.get("tenc_path"),
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={"qwen3": runtime.text.qwen3_text},
            clipvision=None,
        )
        return CodexZImageAssembly(runtime=runtime, codex_objects=codex_objects)

