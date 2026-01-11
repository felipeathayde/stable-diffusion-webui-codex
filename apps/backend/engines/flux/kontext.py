"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux.1 Kontext engine implementation (Flux-derived, image-conditioned flow model).
Overrides the img2img contract to encode the init image as conditioning tokens (`image_latents`) instead of starting from a noisy latent.

Symbols (top-level; keep in sync; no ghosts):
- `Kontext` (class): Flux-derived image-conditioned engine with a custom img2img execution path.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexObjects
from apps.backend.engines.flux.flux import Flux
from apps.backend.engines.flux.factory import CodexFluxFamilyFactory
from apps.backend.engines.flux.spec import FLUX_SPEC
from apps.backend.runtime.models.loader import DiffusionModelBundle

logger = logging.getLogger("backend.engines.flux.kontext")

_KONTEXT_FACTORY = CodexFluxFamilyFactory(spec=FLUX_SPEC)


class Kontext(Flux):
    """Flux Kontext engine (Flux-derived, image-conditioned)."""

    engine_id = "flux1_kontext"

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("flux1_kontext", "kontext"),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
            extras={
                "samplers": ("euler", "euler a", "ddim", "dpm++ 2m"),
                "schedulers": ("simple", "beta", "normal"),
            },
        )

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        assembly = _KONTEXT_FACTORY.assemble(bundle, options=options)
        runtime = assembly.runtime
        self._runtime = runtime
        self.use_distilled_cfg_scale = runtime.use_distilled_cfg
        logger.debug("Kontext runtime prepared (distilled cfg=%s)", runtime.use_distilled_cfg)

        from apps.backend.runtime.flux.streaming import StreamedFluxCore

        core_model = getattr(runtime.denoiser.model, "diffusion_model", runtime.denoiser.model)
        if isinstance(core_model, StreamedFluxCore):
            self._streaming_controller = core_model.controller
        else:
            self._streaming_controller = None

        return assembly.codex_objects

    def img2img(self, request: Any, **kwargs: Any) -> Iterable[Any]:  # type: ignore[override]
        import json
        import secrets
        import threading
        import time

        from apps.backend.core.requests import Img2ImgRequest, ProgressEvent, ResultEvent
        from apps.backend.core.state import state as backend_state
        from apps.backend.engines.util.adapters import build_img2img_processing
        from apps.backend.use_cases.kontext_img2img import generate_kontext_img2img as _generate_kontext_img2img
        from apps.backend.runtime.processing.conditioners import decode_latent_batch
        from apps.backend.runtime.workflows.image_io import latents_to_pil
        from apps.backend.runtime.text_processing import last_extra_generation_params

        self.ensure_loaded()

        if not isinstance(request, Img2ImgRequest):
            raise TypeError(f"{self.__class__.__name__}.img2img expects Img2ImgRequest")

        raw_seed = int(getattr(request, "seed", -1) or -1)
        if raw_seed < 0:
            raw_seed = secrets.randbits(32) & 0x7FFFFFFF

        proc = build_img2img_processing(request)
        proc.sd_model = self
        proc.seed = raw_seed
        proc.seeds = [raw_seed]
        proc.subseed = -1
        proc.subseeds = [-1]

        prompt_texts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
        prompts = prompt_texts

        result: dict[str, Any] = {"latents": None, "error": None}
        sampling_times: dict[str, float | None] = {"start": None, "end": None}
        done = threading.Event()

        def _worker() -> None:
            try:
                sampling_times["start"] = time.perf_counter()
                result["latents"] = _generate_kontext_img2img(
                    processing=proc,
                    conditioning=None,
                    unconditional_conditioning=None,
                    prompts=prompts,
                )
            except Exception as _exc:
                result["error"] = _exc
            finally:
                sampling_times["end"] = time.perf_counter()
                done.set()

        threading.Thread(target=_worker, name=f"{self.engine_id}-img2img-worker", daemon=True).start()

        t0 = time.perf_counter()
        last_step = -1
        while not done.is_set():
            try:
                step = int(getattr(backend_state, "sampling_step", 0) or 0)
                total = int(getattr(backend_state, "sampling_steps", 0) or 0)
            except Exception:
                step, total = 0, 0
            if total > 0 and step != last_step:
                elapsed = time.perf_counter() - t0
                eta = (elapsed * (total - step) / max(step, 1)) if step > 0 else None
                pct = max(5.0, min(99.0, (step / total) * 100.0))
                yield ProgressEvent(stage="sampling", percent=pct, step=step, total_steps=total, eta_seconds=eta)
                last_step = step
            time.sleep(0.12)

        if result["error"] is not None:
            raise result["error"]
        latents = result["latents"]

        if not isinstance(latents, torch.Tensor):
            raise RuntimeError(
                f"kontext img2img returned {type(latents).__name__}, expected torch.Tensor (latents)"
            )

        decode_start = time.perf_counter()
        decoded = decode_latent_batch(self, latents)
        images = latents_to_pil(decoded)
        decode_end = time.perf_counter()

        extra_params: dict[str, object] = {}
        try:
            extra_params.update(last_extra_generation_params)
            extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
        except Exception:
            extra_params = getattr(proc, "extra_generation_params", {}) or {}

        info: dict[str, object] = {
            "engine": self.engine_id,
            "task": "img2img",
            "width": int(proc.width),
            "height": int(proc.height),
            "steps": int(proc.steps),
            "guidance_scale": float(proc.guidance_scale),
            "sampler": (str(getattr(proc, "sampler_name", "")).strip() or None),
            "scheduler": (str(getattr(proc, "scheduler", "")).strip() or None),
        }
        if getattr(proc, "prompt", None):
            info["prompt"] = str(getattr(proc, "prompt", ""))
        if getattr(proc, "negative_prompt", None):
            info["negative_prompt"] = str(getattr(proc, "negative_prompt", ""))
        info["seed"] = int(raw_seed)
        if extra_params:
            info["extra"] = extra_params

        timings: dict[str, float] = {}
        try:
            if sampling_times["start"] is not None and sampling_times["end"] is not None:
                timings["sampling_ms"] = max(0.0, (sampling_times["end"] - sampling_times["start"]) * 1000.0)
            timings["decode_ms"] = max(0.0, (decode_end - decode_start) * 1000.0)
            info["timings_ms"] = timings
        except Exception:
            pass

        self._post_txt2img_cleanup()

        yield ResultEvent(payload={"images": images, "info": json.dumps(info)})
