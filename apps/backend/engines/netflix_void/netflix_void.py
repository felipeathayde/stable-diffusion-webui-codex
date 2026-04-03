"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend Netflix VOID video engine facade for the native vid2vid scaffold.
Registers the `netflix_void` engine as a thin `BaseVideoEngine` adapter, resolves the base-bundle + literal overlay-pair
contract through the dedicated family loader/runtime seam, and delegates canonical `vid2vid` ownership to the shared
use-case layer.

Symbols (top-level; keep in sync; no ghosts):
- `_NETFLIX_VOID_FACTORY` (constant): Factory used to assemble the Netflix VOID engine runtime.
- `NetflixVoidEngine` (class): Backend video engine registered under engine id `netflix_void`.
"""

from __future__ import annotations

from typing import Any, Iterator

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.core.requests import InferenceEvent, Vid2VidRequest
from apps.backend.engines.common.base_video import BaseVideoEngine
from apps.backend.engines.netflix_void.factory import CodexNetflixVoidFactory
from apps.backend.engines.netflix_void.spec import NETFLIX_VOID_SPEC, NetflixVoidEngineRuntime
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.use_cases.vid2vid import run_vid2vid as _run_v2v

_NETFLIX_VOID_FACTORY = CodexNetflixVoidFactory(spec=NETFLIX_VOID_SPEC)


class NetflixVoidEngine(BaseVideoEngine):
    engine_id = "netflix_void"
    expected_family = ModelFamily.NETFLIX_VOID
    model_types: tuple[str, ...] = ("netflix_void",)
    runtime_note: str = "Netflix VOID native vid2vid runtime"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: NetflixVoidEngineRuntime | None = None

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.VID2VID,),
            model_types=self.model_types,
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": self.runtime_note},
        )

    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        self._logger.debug("[%s] before load()", self.engine_id)
        assembly = _NETFLIX_VOID_FACTORY.assemble(model_ref, options=dict(options))
        self._runtime = assembly.runtime
        self.mark_loaded()
        self._logger.debug("[%s] after load()", self.engine_id)

    def unload(self) -> None:  # type: ignore[override]
        self._logger.debug("[%s] before unload()", self.engine_id)
        self._runtime = None
        self.mark_unloaded()
        self._logger.debug("[%s] after unload()", self.engine_id)

    def _require_runtime(self) -> NetflixVoidEngineRuntime:
        runtime = self._runtime
        if runtime is None:
            raise RuntimeError("netflix_void runtime is not initialised; call load() first.")
        return runtime

    def vid2vid(self, request: Vid2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        del kwargs
        self.ensure_loaded()
        runtime = self._require_runtime()
        yield from _run_v2v(engine=self, comp=runtime, request=request)
