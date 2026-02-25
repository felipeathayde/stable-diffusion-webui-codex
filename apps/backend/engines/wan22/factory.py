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

import torch

from apps.backend.engines.common.base import CodexObjects, TextEncoderHandle
from apps.backend.patchers.base import ModelPatcher
from apps.backend.engines.wan22.spec import WanEngineRuntime, WanEngineSpec, assemble_wan_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
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
        default_mount_device = str(memory_management.manager.mount_device())
        device = str(options.get("device") or default_mount_device)
        dtype = str(options.get("dtype", "bf16"))
        runtime = assemble_wan_runtime(
            spec=self._spec,
            codex_components=bundle.components,
            estimated_config=bundle.estimated_config,
            device=device,
            dtype=dtype,
        )
        t5_module = getattr(runtime.text.t5_text, "text_encoder", None)
        if not isinstance(t5_module, torch.nn.Module):
            raise RuntimeError(
                "WAN22 runtime exposes invalid text encoder for CodexObjects.text_encoders['t5']; "
                f"expected torch.nn.Module, got {type(t5_module).__name__}."
            )
        t5_patcher = ModelPatcher(
            t5_module,
            load_device=memory_management.manager.get_device(DeviceRole.TEXT_ENCODER),
            offload_device=memory_management.manager.get_offload_device(DeviceRole.TEXT_ENCODER),
        )
        codex_objects = CodexObjects(
            denoiser=runtime.denoiser,
            vae=runtime.vae,
            text_encoders={
                "t5": TextEncoderHandle(
                    patcher=t5_patcher,
                    runtime=runtime.text.t5_text,
                )
            },
            clipvision=None,
        )
        return CodexWan22Assembly(runtime=runtime, codex_objects=codex_objects)
