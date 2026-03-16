"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LTX2 engine specification and truthful runtime assembly.
Rehydrates the loader-produced typed LTX2 bundle contract into a dedicated native runtime container, threads normalized
internal engine options into that assembly, and lets the registered `ltx2` engine execute canonical `txt2vid` /
`img2vid` without drifting into WAN paths.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2EngineRuntime` (dataclass): Loaded LTX2 engine runtime container holding bundle inputs plus the assembled native components.
- `Ltx2EngineSpec` (dataclass): Canonical LTX2 engine spec metadata.
- `assemble_ltx2_runtime` (function): Assemble the loaded LTX2 engine runtime from a loader-produced diffusion bundle.
- `LTX2_SPEC` (constant): Canonical LTX2 engine spec instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

from apps.backend.runtime.families.ltx2.model import Ltx2BundleInputs
from apps.backend.runtime.families.ltx2.runtime import (
    Ltx2NativeComponents,
    build_ltx2_native_components,
    require_ltx2_bundle_inputs,
    run_ltx2_img2vid,
    run_ltx2_txt2vid,
)
from apps.backend.runtime.model_registry.specs import ModelFamily

if TYPE_CHECKING:
    from apps.backend.core.requests import Img2VidRequest, Txt2VidRequest
    from apps.backend.runtime.families.ltx2.runtime import Ltx2RunResult
    from apps.backend.runtime.pipeline_stages.video import GeneratedAudioExportPolicy
    from apps.backend.runtime.processing.datatypes import VideoPlan


@dataclass(frozen=True, slots=True)
class Ltx2EngineRuntime:
    bundle_inputs: Ltx2BundleInputs
    native: Ltx2NativeComponents
    device: str
    dtype: str

    def run_txt2vid(
        self,
        *,
        request: "Txt2VidRequest",
        plan: "VideoPlan",
        generated_audio_export_policy: "GeneratedAudioExportPolicy",
    ) -> "Ltx2RunResult":
        return run_ltx2_txt2vid(
            native=self.native,
            request=request,
            plan=plan,
            generated_audio_export_policy=generated_audio_export_policy,
        )

    def run_img2vid(
        self,
        *,
        request: "Img2VidRequest",
        plan: "VideoPlan",
        generated_audio_export_policy: "GeneratedAudioExportPolicy",
    ) -> "Ltx2RunResult":
        return run_ltx2_img2vid(
            native=self.native,
            request=request,
            plan=plan,
            generated_audio_export_policy=generated_audio_export_policy,
        )


@dataclass(frozen=True, slots=True)
class Ltx2EngineSpec:
    name: str = "ltx2"
    family: ModelFamily = ModelFamily.LTX2


def assemble_ltx2_runtime(
    *,
    spec: Ltx2EngineSpec,
    bundle,
    engine_options: Mapping[str, Any] | None = None,
) -> Ltx2EngineRuntime:
    if getattr(bundle, "family", None) is not spec.family:
        raise RuntimeError(
            f"LTX2 engine assembly expected bundle family {spec.family.value!r}, "
            f"got {getattr(getattr(bundle, 'family', None), 'value', getattr(bundle, 'family', None))!r}."
        )

    inputs = require_ltx2_bundle_inputs(bundle)
    options = dict(engine_options or {})
    raw_dtype = str(options.get("dtype", "bf16") or "bf16").strip().lower()
    if raw_dtype not in {"fp16", "bf16", "fp32"}:
        raise RuntimeError(f"LTX2 engine dtype must be one of fp16|bf16|fp32, got {raw_dtype!r}.")
    raw_device = str(options.get("device", "auto") or "auto").strip().lower()
    if raw_device not in {"auto", "cpu", "cuda"}:
        raise RuntimeError(f"LTX2 engine device must be one of auto|cpu|cuda, got {raw_device!r}.")
    native = build_ltx2_native_components(
        bundle_inputs=inputs,
        device=raw_device,
        dtype=raw_dtype,
        engine_options=options,
    )

    return Ltx2EngineRuntime(
        bundle_inputs=inputs,
        native=native,
        device=native.device_label,
        dtype=native.dtype_label,
    )


LTX2_SPEC = Ltx2EngineSpec()
