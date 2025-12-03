from __future__ import annotations

import logging
from types import SimpleNamespace
import threading
import time
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from apps.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.engines.sd.spec import SDXL_REFINER_SPEC, SDXL_SPEC, SDEngineRuntime, assemble_engine_runtime
from apps.backend.engines.util.adapters import build_txt2img_processing
from apps.backend.infra.config import args as backend_args
from apps.backend.runtime.memory import memory_management
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.common.nn.unet import Timestep
from apps.backend.runtime.models.loader import DiffusionModelBundle
from apps.backend.runtime.wan22.vae import AutoencoderKLWan
from apps.backend.use_cases.txt2img import generate_txt2img as _generate_txt2img
import json
from apps.backend.core.requests import InferenceEvent, ProgressEvent, ResultEvent
import secrets
from apps.backend.runtime.processing.conditioners import decode_latent_batch
from apps.backend.runtime.workflows.common import latents_to_pil
from apps.backend.runtime.text_processing import last_extra_generation_params


# note: no extra device assertions here; diagnostics should be captured upstream

logger = logging.getLogger("backend.engines.sd.sdxl")


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    if tensor is None:
        return {"shape": None, "dtype": None, "device": None}
    with torch.no_grad():
        data = tensor.detach()
        stats_tensor = data.float()
        return {
            "shape": tuple(data.shape),
            "dtype": str(data.dtype),
            "device": str(data.device),
            "min": float(stats_tensor.min().item()),
            "max": float(stats_tensor.max().item()),
            "mean": float(stats_tensor.mean().item()),
            "std": float(stats_tensor.std(unbiased=False).item()),
        }


def _opts() -> SimpleNamespace:
    return SimpleNamespace(
        sdxl_crop_left=0,
        sdxl_crop_top=0,
        sdxl_refiner_low_aesthetic_score=2.5,
        sdxl_refiner_high_aesthetic_score=6.0,
    )


def _validate_conditioning_payload(runtime: SDEngineRuntime, payload: Mapping[str, Any], *, label: str) -> None:
    """Fail fast when CLIP/conditioning outputs are malformed.

    Ensures shapes match the UNet config and that no NaN/Inf values sneak into
    the sampling loop, which would otherwise produce noisy “golesma” results.
    """

    def _require_tensor(key: str) -> torch.Tensor:
        value = payload.get(key) if isinstance(payload, Mapping) else None
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"SDXL conditioning '{label}' missing tensor '{key}' (got {type(value).__name__}).")
        return value

    cross = _require_tensor("crossattn")
    vector = _require_tensor("vector")

    if cross.ndim != 3:
        raise RuntimeError(
            f"SDXL conditioning '{label}' crossattn must be 3D (B, S, C); got shape={tuple(cross.shape)}."
        )
    if vector.ndim != 2:
        raise RuntimeError(
            f"SDXL conditioning '{label}' vector must be 2D (B, F); got shape={tuple(vector.shape)}."
        )

    for name, tensor in (("crossattn", cross), ("vector", vector)):
        if not torch.isfinite(tensor).all():
            raise RuntimeError(
                f"SDXL conditioning '{label}' contains non-finite values in '{name}'. "
                "Check CLIP weights/conversion before sampling."
            )

    cfg = getattr(runtime.unet.model, "diffusion_model", None)
    if cfg is not None:
        cfg = getattr(cfg, "codex_config", None)

    expected_ctx = getattr(cfg, "context_dim", None) if cfg is not None else None
    if isinstance(expected_ctx, int) and int(cross.shape[-1]) != expected_ctx:
        raise RuntimeError(
            f"SDXL conditioning '{label}' context dim {int(cross.shape[-1])} does not match UNet context_dim={expected_ctx}."
        )

    expected_adm = getattr(cfg, "adm_in_channels", None) if cfg is not None else None
    if isinstance(expected_adm, int) and int(vector.shape[1]) != expected_adm:
        raise RuntimeError(
            f"SDXL conditioning '{label}' ADM vector dim {int(vector.shape[1])} does not match adm_in_channels={expected_adm}."
        )


class _SDXLPrompt(str):
    """String subclass that carries SDXL spatial metadata for conditioning."""

    __slots__ = ("width", "height", "target_width", "target_height", "crop_left", "crop_top", "is_negative_prompt")

    def __new__(
        cls,
        text: str,
        *,
        width: int,
        height: int,
        target_width: Optional[int] = None,
        target_height: Optional[int] = None,
        crop_left: int = 0,
        crop_top: int = 0,
        is_negative_prompt: bool = False,
    ) -> "_SDXLPrompt":
        obj = super().__new__(cls, text or "")
        obj.width = int(width or 1024)
        obj.height = int(height or 1024)
        obj.target_width = int(target_width or obj.width)
        obj.target_height = int(target_height or obj.height)
        obj.crop_left = int(crop_left or 0)
        obj.crop_top = int(crop_top or 0)
        obj.is_negative_prompt = bool(is_negative_prompt)
        return obj


def _prompt_meta(prompts: Sequence[str]) -> Tuple[int, int, int, int, int, int, bool]:
    reference: Any = prompts
    if isinstance(prompts, (list, tuple)) and prompts:
        reference = prompts[0]

    def _meta(attr: str, default: int) -> int:
        value = getattr(reference, attr, None)
        if value is None:
            value = getattr(prompts, attr, None)
        return int(value if value not in (None, "") else default)

    fallback = _opts()
    width = _meta("width", 1024)
    height = _meta("height", 1024)
    target_width = _meta("target_width", width)
    target_height = _meta("target_height", height)
    crop_left = _meta("crop_left", getattr(fallback, "sdxl_crop_left", 0))
    crop_top = _meta("crop_top", getattr(fallback, "sdxl_crop_top", 0))
    is_negative = bool(getattr(reference, "is_negative_prompt", getattr(prompts, "is_negative_prompt", False)))
    return width, height, target_width, target_height, crop_left, crop_top, is_negative


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

        base_vae = getattr(runtime.vae.first_stage_model, "_base", runtime.vae.first_stage_model)
        if isinstance(base_vae, AutoencoderKLWan):
            raise RuntimeError(
                "SDXL engine received a WAN22-style VAE (AutoencoderKLWan); "
                "this combination is not supported. Use a compatible SDXL VAE or remove the WAN VAE from the checkpoint."
            )

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

        Uses the staged txt2img pipeline runner backed by CodexProcessingTxt2Img,
        deriving hires/refiner overrides from the request payload.
        """
        from apps.backend.core.requests import Txt2ImgRequest

        self.ensure_loaded()
        _ = self._require_runtime()

        if not isinstance(request, Txt2ImgRequest):
            raise TypeError("StableDiffusionXL.txt2img expects Txt2ImgRequest")

        # Build processing descriptor from request
        raw_seed = int(getattr(request, "seed", -1) or -1)
        if raw_seed < 0:
            raw_seed = secrets.randbits(32) & 0x7FFFFFFF

        proc = build_txt2img_processing(request)
        proc.sd_model = self
        proc.seed = raw_seed
        proc.seeds = [raw_seed]
        proc.subseed = -1
        proc.subseeds = [-1]

        # Bind current model

        # Defer conditioning to the pipeline runner (after prompt parsing / hires overrides)
        prompt_texts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
        prompts = prompt_texts
        seeds = [raw_seed]
        subseeds = [-1]
        subseed_strength = 0.0
        cond = None
        uncond = None

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
        # Surface prompt/seed metadata so the frontend can show the real
        # generation inputs instead of guessing from request/store state.
        try:
            primary_prompt = getattr(proc, "primary_prompt", proc.prompt)
        except Exception:  # pragma: no cover - defensive only
            primary_prompt = str(getattr(proc, "prompt", ""))

        try:
            primary_negative = getattr(proc, "primary_negative_prompt", proc.negative_prompt)
        except Exception:  # pragma: no cover - defensive only
            primary_negative = str(getattr(proc, "negative_prompt", ""))

        all_seeds = list(getattr(proc, "all_seeds", []) or [])
        seed_value = None
        if all_seeds:
            try:
                seed_value = int(all_seeds[0])
            except Exception:  # pragma: no cover - defensive only
                seed_value = None
        else:
            raw_seed = getattr(proc, "seed", None)
            if raw_seed is not None:
                try:
                    seed_value = int(raw_seed)
                except Exception:  # pragma: no cover - defensive only
                    seed_value = None

        # Merge core runtime metadata with extra-generation parameters collected
        # during text processing (e.g., TI names, emphasis mode, LoRA hashes).
        extra_params: dict[str, object] = {}
        try:
            extra_params.update(last_extra_generation_params)
            extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
        except Exception:  # pragma: no cover - defensive only
            extra_params = getattr(proc, "extra_generation_params", {}) or {}

        info: dict[str, object] = {
            "engine": self.engine_id,
            "task": "txt2img",
            "width": int(proc.width),
            "height": int(proc.height),
            "steps": int(proc.steps),
            "guidance_scale": float(proc.guidance_scale),
            "sampler": str(getattr(proc, "sampler_name", "Automatic") or "Automatic"),
            "scheduler": str(getattr(proc, "scheduler", "Automatic") or "Automatic"),
        }
        if primary_prompt:
            info["prompt"] = str(primary_prompt)
        if primary_negative:
            info["negative_prompt"] = str(primary_negative)
        if seed_value is not None:
            info["seed"] = int(seed_value)
        if all_seeds:
            info["all_seeds"] = [int(s) for s in all_seeds]
        if getattr(proc, "enable_hr", False):
            info["hires"] = {
                "enabled": True,
                "width": int(getattr(proc, "hr_upscale_to_x", 0) or proc.width),
                "height": int(getattr(proc, "hr_upscale_to_y", 0) or proc.height),
                "steps": int(getattr(proc, "hr_second_pass_steps", 0) or proc.steps),
                "denoise": float(getattr(getattr(proc, "hires", None) or proc, "hr_distilled_cfg", 0.0) or 0.0),
                "sampler": str(getattr(proc, "hr_sampler_name", "Use same sampler") or "Use same sampler"),
                "scheduler": str(getattr(proc, "hr_scheduler", "Use same scheduler") or "Use same scheduler"),
                "checkpoint": str(getattr(proc, "hr_checkpoint_name", "Use same checkpoint") or "Use same checkpoint"),
                "modules": list(getattr(proc, "hr_additional_modules", []) or []),
            }
        if extra_params:
            info["extra"] = extra_params
        yield ResultEvent(payload={"images": images, "info": json.dumps(info)})

    def _prepare_prompt_wrappers(
        self,
        texts: Sequence[str],
        proc: Any,
        *,
        is_negative: bool,
    ) -> List[_SDXLPrompt]:
        width = int(getattr(proc, "width", 1024) or 1024)
        height = int(getattr(proc, "height", 1024) or 1024)
        target_width = int(getattr(proc, "hr_upscale_to_x", 0) or width)
        target_height = int(getattr(proc, "hr_upscale_to_y", 0) or height)
        opts = _opts()
        crop_left = int(getattr(proc, "sdxl_crop_left", getattr(opts, "sdxl_crop_left", 0)) or 0)
        crop_top = int(getattr(proc, "sdxl_crop_top", getattr(opts, "sdxl_crop_top", 0)) or 0)

        wrappers: List[_SDXLPrompt] = []
        for entry in texts:
            raw_text = str(entry or "")
            entry_width = int(getattr(entry, "width", width) or width)
            entry_height = int(getattr(entry, "height", height) or height)
            entry_target_width = int(getattr(entry, "target_width", target_width) or target_width)
            entry_target_height = int(getattr(entry, "target_height", target_height) or target_height)
            entry_crop_left = int(getattr(entry, "crop_left", crop_left) or crop_left)
            entry_crop_top = int(getattr(entry, "crop_top", crop_top) or crop_top)
            wrappers.append(
                _SDXLPrompt(
                    raw_text,
                    width=entry_width,
                    height=entry_height,
                    target_width=entry_target_width,
                    target_height=entry_target_height,
                    crop_left=entry_crop_left,
                    crop_top=entry_crop_top,
                    is_negative_prompt=is_negative,
                )
            )
        return wrappers

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: List[str]):
        runtime = self._require_runtime()
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)
        unload_clip = self.smart_offload_enabled
        try:
            out_l = runtime.classic_engine("clip_l")(prompt)
            pooled_l = None
            if isinstance(out_l, tuple) and len(out_l) == 2:
                cond_l, pooled_l = out_l
            else:
                cond_l = out_l
                pooled_l = getattr(cond_l, "pooled", None)

            out_g = runtime.classic_engine("clip_g")(prompt)
            if isinstance(out_g, tuple) and len(out_g) == 2:
                cond_g, pooled_g = out_g
            else:
                # Fallback: older engines attach pooled on the tensor
                pooled_g = getattr(out_g, "pooled", None)
                cond_g = out_g
                if pooled_g is None:
                    raise RuntimeError("SDXL CLIP-G did not provide a pooled embedding; cannot build conditioning vector.")

            width, height, target_width, target_height, crop_left, crop_top, is_negative = _prompt_meta(prompt)
            label = "uncond" if is_negative else "cond"

            embed_values = [
                self.embedder(torch.tensor([height])),
                self.embedder(torch.tensor([width])),
                self.embedder(torch.tensor([crop_top])),
                self.embedder(torch.tensor([crop_left])),
                self.embedder(torch.tensor([target_height])),
                self.embedder(torch.tensor([target_width])),
            ]

            flat = torch.flatten(torch.cat(embed_values)).unsqueeze(dim=0).repeat(pooled_g.shape[0], 1).to(pooled_g)

            # Only zero-out negative embeddings when all underlying texts are truly empty.
            raw_texts = [str(x or "") for x in prompt]
            force_zero_negative_prompt = is_negative and all(t.strip() == "" for t in raw_texts)

            if force_zero_negative_prompt:
                if pooled_l is not None:
                    pooled_l = torch.zeros_like(pooled_l)
                pooled_g = torch.zeros_like(pooled_g)
                cond_l = torch.zeros_like(cond_l)
                cond_g = torch.zeros_like(cond_g)

            cond = {
                "crossattn": torch.cat([cond_l, cond_g], dim=2),
                "vector": torch.cat([pooled_g, flat], dim=1),
            }

            _validate_conditioning_payload(runtime, cond, label=label)

            logger.debug("Generated SDXL conditioning for %d prompts.", len(prompt))
            return cond
        finally:
            if unload_clip:
                memory_management.unload_model(self.codex_objects.clip.patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.classic_engine("clip_l")
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
            sample = self.codex_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            logger.info("[decode] latents stats=%s", _tensor_stats(x))
            sample = self.codex_objects.vae.first_stage_model.process_out(x)
            logger.info("[decode] after_process_out=%s", _tensor_stats(sample))
            sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
            logger.info("[decode] decoded_tensor=%s", _tensor_stats(sample))
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)



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
        unload_clip = self.smart_offload_enabled
        try:
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
        finally:
            if unload_clip:
                memory_management.unload_model(self.codex_objects.clip.patcher)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str):
        runtime = self._require_runtime()
        engine = runtime.classic_engine("clip_g")
        _, token_count = engine.process_texts([prompt])
        target = engine.get_target_prompt_token_count(token_count)
        return token_count, target

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
            sample = self.codex_objects.vae.first_stage_model.process_in(sample)
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)

    @torch.inference_mode()
    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        memory_management.load_model_gpu(self.codex_objects.vae)
        unload_vae = self.smart_offload_enabled
        try:
            sample = self.codex_objects.vae.first_stage_model.process_out(x)
            sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
            return sample.to(x)
        finally:
            if unload_vae:
                memory_management.unload_model(self.codex_objects.vae)
