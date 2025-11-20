"""Stage-based txt2img pipeline orchestrator."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np
import torch
from PIL import Image

from apps.backend.codex import main as codex_main
from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG
from apps.backend.patchers.token_merging import SkipWritingToConfig
from apps.backend.infra.config import args as backend_args
from apps.backend.runtime.pipeline_debug import pipeline_trace
from apps.backend.runtime.processing.conditioners import (
    decode_latent_batch,
    img2img_conditioning,
)
from apps.backend.runtime.processing.datatypes import (
    ConditioningPayload,
    HiResPlan,
    PromptContext,
    SamplingPlan,
)
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.backend.runtime.workflows import (
    apply_dimension_overrides,
    apply_prompt_context,
    apply_sampling_overrides,
    apply_tiling_if_requested,
    build_prompt_context,
    build_sampling_plan,
    ensure_sampler_and_rng,
    execute_sampling,
    finalize_tiling,
    latents_to_pil,
    maybe_decode_for_hr,
    pil_to_tensor,
    run_process_scripts,
)
from apps.backend.codex.loader import EngineLoadOptions, load_engine as _load_engine

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


@dataclass(slots=True)
class PrepareState:
    """State captured after the preparation stage."""

    prompt_context: PromptContext
    sampling_plan: SamplingPlan
    rng: ImageRNG
    payload: ConditioningPayload
    hires_plan: HiResPlan | None
    init_latents: torch.Tensor | None
    init_decoded: torch.Tensor | None
    cond: object | None = None
    uncond: object | None = None


@dataclass(slots=True)
class SamplingOutput:
    """Result of executing a sampling stage."""

    samples: torch.Tensor
    decoded: torch.Tensor | None


class Txt2ImgPipelineRunner:
    """Orchestrates txt2img generation through well-defined stages."""

    def __init__(self) -> None:
        self._logger = logging.getLogger("backend.use_cases.txt2img.pipeline")

    @staticmethod
    def _log_tensor_stats(label: str, tensor: torch.Tensor | None) -> None:
        logger = logging.getLogger("backend.use_cases.txt2img.pipeline")
        if tensor is None:
            logger.info("[sampling] %s: <none>", label)
            return
        with torch.no_grad():
            data = tensor.detach()
            try:
                stats_tensor = data.float()
            except Exception:
                stats_tensor = data
            mean = float(stats_tensor.mean().item())
            std = float(stats_tensor.std(unbiased=False).item())
            min_value = float(stats_tensor.min().item())
            max_value = float(stats_tensor.max().item())
        logger.info(
            "[sampling] %s: shape=%s dtype=%s device=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            min_value,
            max_value,
            mean,
            std,
        )

    def _compute_conditioning(self, processing: CodexProcessingTxt2Img, context: PromptContext):
        """Build cond/uncond using the engine's SDXL-aware helpers after prompt parsing and dimension overrides."""
        sd_model = getattr(processing, "sd_model", None)
        if sd_model is None or not hasattr(sd_model, "get_learned_conditioning"):
            return None, None

        prompts = list(context.prompts or [getattr(processing, "prompt", "")])
        negative_prompts = list(context.negative_prompts or [getattr(processing, "negative_prompt", "")])

        # Preserve spatial metadata via engine helper when available
        if hasattr(sd_model, "_prepare_prompt_wrappers"):
            prompts_wrapped = sd_model._prepare_prompt_wrappers(prompts, processing, is_negative=False)
            negative_wrapped = sd_model._prepare_prompt_wrappers(negative_prompts, processing, is_negative=True)
            cond = sd_model.get_learned_conditioning(prompts_wrapped)
            uncond = sd_model.get_learned_conditioning(negative_wrapped)
        else:
            cond = sd_model.get_learned_conditioning(prompts)
            uncond = sd_model.get_learned_conditioning(negative_prompts)
        return cond, uncond

    def _log_conditioning(self, cond: object, uncond: object) -> None:
        """Optional conditioning diagnostics controlled by --debug-conditioning / CODEX_DEBUG_COND."""
        try:
            if not getattr(backend_args.args, "debug_conditioning", False):
                return

            def _shape(t):
                return tuple(t.shape) if isinstance(t, torch.Tensor) else None

            def _dtype(t):
                return str(t.dtype) if isinstance(t, torch.Tensor) else None

            def _device(t):
                return str(t.device) if isinstance(t, torch.Tensor) else None

            def _norm(t):
                return float(t.detach().abs().mean().item()) if isinstance(t, torch.Tensor) else None

            ca = cond.get("crossattn") if isinstance(cond, dict) else None
            va = cond.get("vector") if isinstance(cond, dict) else None
            ua = uncond.get("crossattn") if isinstance(uncond, dict) else None
            uv = uncond.get("vector") if isinstance(uncond, dict) else None

            self._logger.info(
                "[sdxl] cond: cross shape=%s dtype=%s dev=%s norm=%.4f | vector shape=%s dtype=%s dev=%s norm=%.4f",
                _shape(ca), _dtype(ca), _device(ca), (_norm(ca) or 0.0),
                _shape(va), _dtype(va), _device(va), (_norm(va) or 0.0),
            )
            self._logger.info(
                "[sdxl] uncond: cross shape=%s dtype=%s dev=%s norm=%.4f | vector shape=%s dtype=%s dev=%s norm=%.4f",
                _shape(ua), _dtype(ua), _device(ua), (_norm(ua) or 0.0),
                _shape(uv), _dtype(uv), _device(uv), (_norm(uv) or 0.0),
            )
        except Exception as exc:  # noqa: BLE001
            self._logger.debug("[sdxl] conditioning diagnostics skipped: %s", exc)

    # ------------------------------------------------------------------ public API
    @pipeline_trace
    def run(
        self,
        processing: CodexProcessingTxt2Img,
        conditioning_data,
        unconditional_data,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> torch.Tensor:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "StableDiffusionXL exige CUDA ativo (torch.cuda.is_available() retornou False). "
                "Instale os drivers/CUDA corretos ou execute o modo somente CPU conscientemente."
            )

        model_device, model_dtype = self._sd_model_device_info(processing)
        if model_device is not None:
            self._logger.info("SDXL sd_model device=%s dtype=%s", model_device, model_dtype)
            if model_device.type != "cuda":
                raise RuntimeError(
                    "StableDiffusionXL carregou o modelo em %s; configure uma GPU CUDA antes de continuar."
                    % model_device
                )

        state = self._prepare_state(
            processing,
            conditioning_data,
            unconditional_data,
            seeds,
            subseeds,
            subseed_strength,
            prompts,
        )
        base_result = self._execute_base_sampling(processing, state)

        if state.hires_plan is None:
            return base_result.samples

        self._reload_for_hires(processing, state)
        hires_samples = self._run_hires_pass(processing, state, base_result)
        return hires_samples

    # ------------------------------------------------------------------ stages
    @pipeline_trace
    def _prepare_state(
        self,
        processing: CodexProcessingTxt2Img,
        conditioning_data,
        unconditional_data,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> PrepareState:
        prompt_context = build_prompt_context(processing, prompts)
        apply_prompt_context(processing, prompt_context)
        apply_dimension_overrides(processing, prompt_context.controls)

        plan = build_sampling_plan(processing, seeds, subseeds, subseed_strength)
        plan = apply_sampling_overrides(processing, prompt_context.controls, plan)
        rng = ensure_sampler_and_rng(processing, plan)

        processing.seeds = list(plan.seeds)
        processing.subseeds = list(plan.subseeds)
        processing.guidance_scale = plan.guidance_scale
        processing.cfg_scale = plan.guidance_scale
        processing.steps = plan.steps
        processing.prepare_prompt_data()

        run_process_scripts(processing)

        payload = ConditioningPayload(conditioning=conditioning_data, unconditional=unconditional_data)

        init_latents, init_decoded = self._prepare_first_pass_from_image(processing)
        hires_plan = self._build_hires_plan(processing)

        # Compute conditioning if not provided (SDXL path); preserve metadata (width/height/targets) after overrides.
        cond = conditioning_data
        uncond = unconditional_data
        if cond is None or uncond is None:
            cond, uncond = self._compute_conditioning(processing, prompt_context)
            if cond is None or uncond is None:
                raise RuntimeError("Failed to build conditioning for txt2img; get_learned_conditioning returned None.")
            payload = ConditioningPayload(conditioning=cond, unconditional=uncond)
            self._log_conditioning(cond, uncond)

        return PrepareState(
            prompt_context=prompt_context,
            sampling_plan=plan,
            rng=rng,
            payload=payload,
            hires_plan=hires_plan,
            init_latents=init_latents,
            init_decoded=init_decoded,
            cond=cond,
            uncond=uncond,
        )

    @pipeline_trace
    def _execute_base_sampling(self, processing: CodexProcessingTxt2Img, state: PrepareState) -> SamplingOutput:
        base_samples = state.init_latents
        decoded_samples = state.init_decoded

        if base_samples is None and decoded_samples is None:
            tiling_applied, previous_tiling = apply_tiling_if_requested(processing, state.prompt_context.controls)
            try:
                base_samples = execute_sampling(
                    processing,
                    state.sampling_plan,
                    state.payload,
                    state.prompt_context,
                    state.prompt_context.loras,
                    state.prompt_context.controls,
                    rng=state.rng,
                )
                decoded_samples = maybe_decode_for_hr(processing, base_samples)
            finally:
                finalize_tiling(tiling_applied, previous_tiling)
        elif base_samples is None and decoded_samples is not None:
            tensor = decoded_samples.to(devices.default_device(), dtype=torch.float32)
            base_samples = processing.sd_model.encode_first_stage(tensor)

        if base_samples is None:
            raise RuntimeError("txt2img failed to produce initial samples")

        self._log_tensor_stats("base_samples", base_samples)
        self._log_tensor_stats("base_decoded", decoded_samples)

        return SamplingOutput(samples=base_samples, decoded=decoded_samples)

    @pipeline_trace
    def _reload_for_hires(self, processing: CodexProcessingTxt2Img, state: PrepareState) -> None:
        assert state.hires_plan is not None
        with SkipWritingToConfig():
            checkpoint_before = getattr(codex_main._SELECTIONS, "checkpoint_name")
            modules_before = list(getattr(codex_main._SELECTIONS, "additional_modules"))

            reload_required = False
            if (
                getattr(processing, "hr_additional_modules", None) is not None
                and "Use same choices" not in processing.hr_additional_modules
            ):
                modules_changed = codex_main.modules_change(
                    processing.hr_additional_modules, save=False, refresh=False
                )
                reload_required = reload_required or modules_changed

            if (
                processing.hr_checkpoint_name
                and processing.hr_checkpoint_name != "Use same checkpoint"
            ):
                checkpoint_changed = codex_main.checkpoint_change(
                    processing.hr_checkpoint_name, save=False, refresh=False
                )
                if checkpoint_changed:
                    processing.firstpass_use_distilled_cfg_scale = processing.sd_model.use_distilled_cfg_scale
                    reload_required = True

            if reload_required:
                try:
                    codex_main.refresh_model_loading_parameters()
                    load_opts = EngineLoadOptions(
                        device=None,
                        dtype=None,
                        attention_backend=os.getenv("CODEX_ATTENTION_BACKEND"),
                        accelerator=os.getenv("CODEX_ACCELERATOR"),
                        vae_path=None,
                    )
                    new_engine = _load_engine(processing.hr_checkpoint_name, options=load_opts)
                    processing.sd_model = new_engine
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"Failed to load hires checkpoint '{processing.hr_checkpoint_name}': {exc}"
                    ) from exc
                finally:
                    codex_main.modules_change(modules_before, save=False, refresh=False)
                    codex_main.checkpoint_change(checkpoint_before, save=False, refresh=False)
                    codex_main.refresh_model_loading_parameters()

            if processing.sd_model.use_distilled_cfg_scale:
                processing.extra_generation_params["Hires Distilled CFG Scale"] = processing.hr_distilled_cfg

    @pipeline_trace
    def _run_hires_pass(
        self,
        processing: CodexProcessingTxt2Img,
        state: PrepareState,
        base_result: SamplingOutput,
    ) -> torch.Tensor:
        assert state.hires_plan is not None
        hires_cfg = processing.hires
        processing.ensure_hires_prompts()

        target_width = hires_cfg.resize_x or int(processing.width * hires_cfg.scale)
        target_height = hires_cfg.resize_y or int(processing.height * hires_cfg.scale)
        steps = hires_cfg.second_pass_steps or processing.steps
        denoise = float(hires_cfg.denoise)

        original_attrs = {
            "prompts": processing.prompts,
            "negative_prompts": getattr(processing, "negative_prompts", []),
            "width": processing.width,
            "height": processing.height,
            "guidance_scale": processing.guidance_scale,
            "steps": processing.steps,
        }

        hr_prompts_source = (
            processing.hr_prompts
            or processing.all_prompts
            or state.prompt_context.prompts
        )
        hr_cleaned_prompts, hr_prompt_loras, hr_prompt_controls = parse_prompts_with_extras(list(hr_prompts_source))
        hr_negative = processing.hr_negative_prompts or original_attrs["negative_prompts"]
        hires_prompt_context = PromptContext(
            prompts=hr_cleaned_prompts,
            negative_prompts=hr_negative,
            loras=hr_prompt_loras,
            controls=dict(hr_prompt_controls),
        )

        processing.prompts = hires_prompt_context.prompts
        processing.negative_prompts = hires_prompt_context.negative_prompts
        processing.width = target_width
        processing.height = target_height
        processing.guidance_scale = float(hires_cfg.cfg or processing.guidance_scale)
        processing.cfg_scale = processing.guidance_scale
        processing.steps = int(steps)
        processing.prepare_prompt_data()

        if getattr(processing, "latent_scale_mode", None) is not None:
            mode = processing.latent_scale_mode.get("mode", "bilinear")
            antialias = bool(processing.latent_scale_mode.get("antialias", False))
            latents = torch.nn.functional.interpolate(
                base_result.samples,
                size=(target_height // 8, target_width // 8),
                mode=mode,
                antialias=antialias,
            )
            tensor = decode_latent_batch(processing.sd_model, latents)
            image_conditioning = img2img_conditioning(
                processing.sd_model,
                tensor,
                latents,
                image_mask=getattr(processing, "image_mask", None),
                round_mask=getattr(processing, "round_image_mask", True),
            )
        else:
            decoded_samples = base_result.decoded
            if decoded_samples is None:
                decoded_samples = decode_latent_batch(processing.sd_model, base_result.samples)
            pil_images = latents_to_pil(decoded_samples)
            upscaled = [img.resize((target_width, target_height), _RESAMPLE_LANCZOS) for img in pil_images]
            tensor = pil_to_tensor(upscaled)
            latents = processing.sd_model.encode_first_stage(tensor)
            image_conditioning = img2img_conditioning(
                processing.sd_model,
                tensor,
                latents,
                image_mask=getattr(processing, "image_mask", None),
                round_mask=getattr(processing, "round_image_mask", True),
            )

        hires_settings = state.sampling_plan.noise_settings
        rng_hr = ImageRNG(
            (latents.shape[1], latents.shape[2], latents.shape[3]),
            state.sampling_plan.seeds,
            subseeds=state.sampling_plan.subseeds,
            subseed_strength=state.sampling_plan.subseed_strength,
            seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
            settings=hires_settings,
        )
        noise = rng_hr.next().to(latents)
        start_index = max(0, min(int(round(denoise * processing.steps)), processing.steps - 1))

        hires_plan = replace(
            state.sampling_plan,
            steps=int(processing.steps),
            guidance_scale=float(processing.guidance_scale),
        )

        # Recompute conditioning for hires pass with updated width/height/targets (SDXL parity).
        cond_hr, uncond_hr = self._compute_conditioning(processing, hires_prompt_context)
        if cond_hr is not None and uncond_hr is not None:
            state.payload = ConditioningPayload(conditioning=cond_hr, unconditional=uncond_hr)
            self._log_conditioning(cond_hr, uncond_hr)

        samples = execute_sampling(
            processing,
            hires_plan,
            state.payload,
            hires_prompt_context,
            hires_prompt_context.loras,
            hires_prompt_context.controls,
            rng=rng_hr,
            noise=noise,
            image_conditioning=image_conditioning,
            init_latent=latents,
            start_at_step=start_index,
        )

        processing.prompts = original_attrs["prompts"]
        processing.negative_prompts = original_attrs["negative_prompts"]
        processing.width = original_attrs["width"]
        processing.height = original_attrs["height"]
        processing.guidance_scale = original_attrs["guidance_scale"]
        processing.cfg_scale = processing.guidance_scale
        processing.steps = original_attrs["steps"]
        processing.prepare_prompt_data()

        return samples

    # ------------------------------------------------------------------ helpers
    @pipeline_trace
    def _prepare_first_pass_from_image(
        self, processing: CodexProcessingTxt2Img
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        image = processing.firstpass_image
        if image is None or not processing.enable_hr:
            return None, None

        if processing.latent_scale_mode is None:
            array = np.array(image).astype(np.float32) / 255.0
            array = array * 2.0 - 1.0
            array = np.moveaxis(array, 2, 0)
            decoded_samples = torch.from_numpy(np.expand_dims(array, 0))
            return None, decoded_samples

        array = np.array(image).astype(np.float32) / 255.0
        array = np.moveaxis(array, 2, 0)
        tensor = torch.from_numpy(np.expand_dims(array, axis=0))
        tensor = tensor.to(devices.default_device(), dtype=torch.float32)

        samples = processing.sd_model.encode_first_stage(tensor)
        devices.torch_gc()
        return samples, None

    @pipeline_trace
    def _build_hires_plan(self, processing: CodexProcessingTxt2Img) -> HiResPlan | None:
        if not getattr(processing, "enable_hr", False):
            return None

        hi_cfg = processing.hires
        target_width = hi_cfg.resize_x or int(processing.width * hi_cfg.scale)
        target_height = hi_cfg.resize_y or int(processing.height * hi_cfg.scale)
        steps = hi_cfg.second_pass_steps or processing.steps
        denoise = float(hi_cfg.denoise)
        cfg_scale = hi_cfg.cfg
        return HiResPlan(
            enabled=True,
            target_width=target_width,
            target_height=target_height,
            steps=int(steps),
            denoise=denoise,
            cfg_scale=float(cfg_scale) if cfg_scale is not None else None,
            checkpoint_name=getattr(processing, "hr_checkpoint_name", None),
            additional_modules=getattr(processing, "hr_additional_modules", None),
        )

    # ------------------------------------------------------------------ diagnostics
    def _sd_model_device_info(
        self, processing: CodexProcessingTxt2Img
    ) -> tuple[torch.device | None, torch.dtype | None]:
        model = getattr(processing, "sd_model", None)
        if model is None:
            return None, None
        tensor = None
        try:
            params = model.parameters()
            if params is not None:
                tensor = next(params)
        except StopIteration:
            tensor = None
        except Exception:
            tensor = None
        if tensor is None:
            candidate = getattr(model, "weight", None)
            if isinstance(candidate, torch.Tensor):
                tensor = candidate
        if tensor is not None and isinstance(tensor, torch.Tensor):
            return tensor.device, tensor.dtype
        device_attr = getattr(model, "device", None)
        if isinstance(device_attr, torch.device):
            return device_attr, None
        return None, None
