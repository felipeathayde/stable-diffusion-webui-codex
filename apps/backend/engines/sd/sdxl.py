from __future__ import annotations

import logging
from types import SimpleNamespace
import threading
import time
from typing import Any, Iterable, List, Mapping, Optional, Tuple

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.spec import SDXL_REFINER_SPEC, SDXL_SPEC, SDEngineRuntime, assemble_engine_runtime
from apps.backend.runtime.memory import memory_management
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.common.nn.unet import Timestep
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.use_cases.txt2img import generate_txt2img as _generate_txt2img
import json
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent
from apps.backend.runtime.processing.conditioners import decode_latent_batch
from apps.backend.runtime.workflows.common import latents_to_pil


# note: no extra device assertions here; diagnostics should be captured upstream

logger = logging.getLogger("backend.engines.sd.sdxl")


def _opts() -> SimpleNamespace:
    return SimpleNamespace(
        sdxl_crop_left=0,
        sdxl_crop_top=0,
        sdxl_refiner_low_aesthetic_score=2.5,
        sdxl_refiner_high_aesthetic_score=6.0,
    )


def _prompt_meta(prompt: Iterable[str]) -> Tuple[int, int, bool]:
    obj = prompt  # type: ignore
    width = getattr(obj, "width", 1024) or 1024
    height = getattr(obj, "height", 1024) or 1024
    is_negative = getattr(obj, "is_negative_prompt", False)
    return width, height, is_negative


class StableDiffusionXL(CodexDiffusionEngine):
    """Codex-native SDXL base engine."""

    engine_id = "sdxl"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None
        self.embedder = Timestep(256)

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("sdxl",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    # load() behavior inherited from CodexDiffusionEngine

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        runtime = assemble_engine_runtime(SDXL_SPEC, bundle.estimated_config, bundle.components)
        self._runtime = runtime
        self.register_model_family("sdxl")

        logger.debug(
            "StableDiffusionXL runtime prepared with branches=%s clip_skip=%d",
            runtime.classic_order,
            runtime.classic_engine("clip_l").clip_skip,
        )

        return CodexObjects(
            unet=runtime.unet,
            clip=runtime.clip,
            vae=runtime.vae,
            clipvision=None,
        )

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> SDEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("StableDiffusionXL runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        runtime.set_clip_skip(clip_skip)
        logger.debug("Clip skip set to %d for SDXL.", clip_skip)

    # ------------------------------------------------------------------ Tasks
    def txt2img(self, request, **kwargs: Any):  # type: ignore[override]
        """Run txt2img using the staged pipeline runner.

        Avoids external builder imports; constructs a minimal CodexProcessingTxt2Img
        from the request and prepares conditioning via SDXL runtime.
        """
        from apps.backend.runtime.processing.models import CodexProcessingTxt2Img

        self.ensure_loaded()
        _ = self._require_runtime()

        # Build processing descriptor from request
        proc = CodexProcessingTxt2Img(
            prompt=str(getattr(request, "prompt", "")),
            negative_prompt=str(getattr(request, "negative_prompt", "")),
            width=int(getattr(request, "width", 1024) or 1024),
            height=int(getattr(request, "height", 1024) or 1024),
            steps=int(getattr(request, "steps", 30) or 30),
            guidance_scale=float(getattr(request, "guidance_scale", 7.0) or 7.0),
            distilled_guidance_scale=float(getattr(getattr(request, "metadata", {}), "get", lambda _k, _d=None: None)(
                "distilled_cfg_scale", 3.5
            ) or 3.5),
            batch_size=int(getattr(request, "batch_size", 1) or 1),
            iterations=1,
            seed=int(getattr(request, "seed", -1) or -1),
            sampler_name=getattr(request, "sampler", None),
            scheduler=getattr(request, "scheduler", None),
            metadata=getattr(request, "metadata", {}),
        )
        # Bind current model
        proc.sd_model = self

        # Prepare conditioning (SDXL: CLIP-L/CLIP-G)
        prompts = [proc.prompt]
        seeds = [proc.seed]
        subseeds = [-1]
        subseed_strength = 0.0
        cond = self.get_learned_conditioning(prompts)
        uncond = self.get_learned_conditioning([""])

        yield ProgressEvent(stage="prepare", percent=5.0, message="Preparing conditioning")

        # Run pipeline on a worker thread while streaming progress from backend_state
        result: dict[str, Any] = {"latents": None, "error": None}
        done = threading.Event()

        def _worker() -> None:
            try:
                result["latents"] = _generate_txt2img(
                    processing=proc,
                    conditioning=cond,
                    unconditional_conditioning=uncond,
                    seeds=seeds,
                    subseeds=subseeds,
                    subseed_strength=subseed_strength,
                    prompts=prompts,
                )
            except Exception as _exc:  # noqa: BLE001
                result["error"] = _exc
            finally:
                done.set()

        threading.Thread(target=_worker, name="sdxl-txt2img-worker", daemon=True).start()

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
                f"txt2img pipeline returned {type(latents).__name__}, expected torch.Tensor (latents)"
            )

        # Decode to RGB and package result
        decoded = decode_latent_batch(self, latents)
        images = latents_to_pil(decoded)
        info = {
            "engine": self.engine_id,
            "task": "txt2img",
            "width": int(proc.width),
            "height": int(proc.height),
            "steps": int(proc.steps),
            "guidance_scale": float(proc.guidance_scale),
            "sampler": str(getattr(proc, "sampler_name", "Automatic") or "Automatic"),
            "scheduler": str(getattr(proc, "scheduler", "Automatic") or "Automatic"),
        }
        yield ResultEvent(payload={"images": images, "info": json.dumps(info)})

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)
        out_l = runtime.classic_engine("clip_l")(prompt)
        if isinstance(out_l, tuple) and len(out_l) == 2:
            cond_l, _ = out_l
        else:
            cond_l = out_l

        out_g = runtime.classic_engine("clip_g")(prompt)
        if isinstance(out_g, tuple) and len(out_g) == 2:
            cond_g, pooled_g = out_g
        else:
            # Fallback: older engines attach pooled on the tensor
            pooled_g = getattr(out_g, "pooled", None)
            cond_g = out_g
            if pooled_g is None:
                raise RuntimeError("SDXL CLIP-G did not provide a pooled embedding; cannot build conditioning vector.")

        width, height, is_negative = _prompt_meta(prompt)
        opts = _opts()

        embed_values = [
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
            self.embedder(torch.tensor([opts.sdxl_crop_top])),
            self.embedder(torch.tensor([opts.sdxl_crop_left])),
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
        ]

        flat = torch.flatten(torch.cat(embed_values)).unsqueeze(dim=0).repeat(pooled_g.shape[0], 1).to(pooled_g)

        if is_negative and all(x == "" for x in prompt):
            pooled_l = torch.zeros_like(pooled_l)
            pooled_g = torch.zeros_like(pooled_g)
            cond_l = torch.zeros_like(cond_l)
            cond_g = torch.zeros_like(cond_g)

        cond = {
            "crossattn": torch.cat([cond_l, cond_g], dim=2),
            "vector": torch.cat([pooled_g, flat], dim=1),
        }

        logger.debug("Generated SDXL conditioning for %d prompts.", len(prompt))
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.classic_engine("clip_l")
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.codex_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.codex_objects.vae.first_stage_model.process_out(x)
        sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)



class StableDiffusionXLRefiner(CodexDiffusionEngine):
    """Codex-native SDXL refiner engine."""

    engine_id = "sdxl_refiner"

    def __init__(self) -> None:
        super().__init__()
        self._runtime: Optional[SDEngineRuntime] = None
        self.embedder = Timestep(256)

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2IMG, TaskType.IMG2IMG),
            model_types=("sdxl_refiner",),
            devices=("cpu", "cuda"),
            precision=("fp16", "bf16", "fp32"),
        )

    # load() behavior inherited from CodexDiffusionEngine

    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        runtime = assemble_engine_runtime(SDXL_REFINER_SPEC, bundle.estimated_config, bundle.components)
        self._runtime = runtime
        self.register_model_family("sdxl")

        logger.debug(
            "StableDiffusionXLRefiner runtime prepared with clip_skip=%d",
            runtime.classic_engine("clip_g").clip_skip,
        )

        return CodexObjects(
            unet=runtime.unet,
            clip=runtime.clip,
            vae=runtime.vae,
            clipvision=None,
        )

    def _on_unload(self) -> None:
        self._runtime = None

    def _require_runtime(self) -> SDEngineRuntime:
        if self._runtime is None:
            raise RuntimeError("StableDiffusionXLRefiner runtime is not initialised; call load() first.")
        return self._runtime

    def set_clip_skip(self, clip_skip: int) -> None:
        runtime = self._require_runtime()
        runtime.set_clip_skip(clip_skip)
        logger.debug("Clip skip set to %d for SDXL refiner.", clip_skip)

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)

        cond_g, pooled = runtime.classic_engine("clip_g")(prompt)

        width, height, is_negative = _prompt_meta(prompt)
        opts = _opts()

        embed_values = [
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
            self.embedder(torch.tensor([opts.sdxl_crop_top])),
            self.embedder(torch.tensor([opts.sdxl_crop_left])),
            self.embedder(torch.tensor([height])),
            self.embedder(torch.tensor([width])),
        ]
        flat = torch.flatten(torch.cat(embed_values)).unsqueeze(dim=0).repeat(pooled.shape[0], 1).to(pooled)

        if is_negative and all(x == "" for x in prompt):
            pooled = torch.zeros_like(pooled)
            cond_g = torch.zeros_like(cond_g)

        cond = {
            "crossattn": cond_g,
            "vector": torch.cat([pooled, flat], dim=1),
        }

        logger.debug("Generated SDXL refiner conditioning for %d prompts.", len(prompt))
        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.classic_engine("clip_g")
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.codex_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        sample = self.codex_objects.vae.first_stage_model.process_out(x)
        sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)
