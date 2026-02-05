"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Anima engine facade for txt2img/img2img.
Defines the `AnimaEngine` class and integrates with the common `CodexDiffusionEngine` lifecycle.
The full runtime (Cosmos Predict2 + Anima adapter + Qwen3-0.6B + VAE) is ported in phases; until then, loading fails loudly.

Symbols (top-level; keep in sync; no ghosts):
- `_ANIMA_FACTORY` (constant): Factory used to assemble the Anima runtime and `CodexObjects`.
- `_canonical_device_label` (function): Normalize device identity labels for strict runtime consistency checks.
- `AnimaEngine` (class): Engine facade registered under engine id `anima`.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.common.runtime_lifecycle import require_runtime
from apps.backend.runtime.model_registry.capabilities import ENGINE_SURFACES, SemanticEngine
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import DiffusionModelBundle

from .factory import CodexAnimaFactory
from .spec import ANIMA_SPEC, AnimaEngineRuntime


logger = logging.getLogger("backend.engines.anima")
_ANIMA_FACTORY = CodexAnimaFactory(spec=ANIMA_SPEC)


def _canonical_device_label(value: object, *, field_name: str) -> str:
    if isinstance(value, torch.device):
        device = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            raise RuntimeError(f"Anima runtime assembly returned empty `{field_name}` device identity.")
        try:
            device = torch.device(raw)
        except Exception as exc:  # noqa: BLE001 - fail-loud identity parsing
            raise RuntimeError(
                f"Anima runtime assembly returned invalid `{field_name}` device identity: {raw!r}"
            ) from exc
    else:
        raise RuntimeError(
            f"Anima runtime assembly returned invalid `{field_name}` device identity; expected torch.device or str."
        )

    if device.type == "cuda" and device.index is None:
        return "cuda:0"
    return str(device)


class AnimaEngine(CodexDiffusionEngine):
    """Anima engine (Cosmos Predict2 + Anima adapter)."""

    engine_id = "anima"
    expected_family = ModelFamily.ANIMA

    def __init__(self) -> None:
        super().__init__()
        self._runtime: AnimaEngineRuntime | None = None

    def capabilities(self) -> EngineCapabilities:
        surface = ENGINE_SURFACES[SemanticEngine.ANIMA]
        tasks: list[TaskType] = []
        if surface.supports_txt2img:
            tasks.append(TaskType.TXT2IMG)
        if surface.supports_img2img:
            tasks.append(TaskType.IMG2IMG)
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=tuple(tasks),
            model_types=("anima",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        return ("qwen3",)

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _ANIMA_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        runtime_device_label = _canonical_device_label(getattr(runtime, "device", None), field_name="runtime.device")
        denoiser_model = getattr(runtime.denoiser, "model", None)
        if denoiser_model is None:
            raise RuntimeError("Anima runtime assembly returned denoiser without `model` for device consistency checks.")
        denoiser_load_device = getattr(denoiser_model, "load_device", None)
        if denoiser_load_device is None:
            raise RuntimeError(
                "Anima runtime assembly returned denoiser.model without `load_device` for device consistency checks."
            )
        denoiser_device_label = _canonical_device_label(
            denoiser_load_device,
            field_name="denoiser.model.load_device",
        )
        if denoiser_device_label != runtime_device_label:
            raise RuntimeError(
                "Anima runtime device mismatch: runtime.device="
                f"{runtime_device_label} but denoiser.model.load_device={denoiser_device_label}"
            )
        runtime_compute_dtype = getattr(runtime, "core_compute_dtype", None)
        if not isinstance(runtime_compute_dtype, str) or not runtime_compute_dtype:
            raise RuntimeError(
                "Anima runtime assembly returned invalid `core_compute_dtype`; expected non-empty dtype label."
            )
        self._device = runtime_device_label
        self._dtype = runtime_compute_dtype
        logger.debug("Anima runtime assembled")
        return assembly.codex_objects

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> AnimaEngineRuntime:
        return require_runtime(self._runtime, label=self.engine_id)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        raise NotImplementedError("Anima text conditioning not yet ported.")

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        raise NotImplementedError("Anima prompt length calculation not yet ported.")
