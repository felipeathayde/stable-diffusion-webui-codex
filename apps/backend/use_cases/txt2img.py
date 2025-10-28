from __future__ import annotations

from typing import Sequence

import logging

import numpy as np
import torch
import torch.nn.functional as F
import os

from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG, NoiseSettings, NoiseSourceKind
from apps.backend.core.state import state as backend_state
from apps.backend.runtime.sampling.driver import CodexSampler
from apps.backend.runtime.sampling.context import build_sampling_context
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.backend.runtime.processing.conditioners import (
    decode_latent_batch,
    txt2img_conditioning,
    img2img_conditioning,
)
from apps.backend.runtime.processing.models import CodexProcessingTxt2Img

from apps.backend.patchers.token_merging import apply_token_merging, SkipWritingToConfig
from apps.backend.codex import main as codex_main
from loader import load_engine as _load_engine, EngineLoadOptions
from apps.backend.codex import lora as codex_lora
from apps.backend.patchers.lora_apply import apply_loras_to_engine
from PIL import Image

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


class _ExtraNetworksShim:
    @staticmethod
    def activate(processing, data):
        raise NotImplementedError("Extra networks activation is not implemented natively yet")


def _prepare_first_pass_from_image(processing) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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

    # Encode the image to latents using native engine VAE
    sample_in = tensor
    samples = processing.sd_model.encode_first_stage(sample_in)
    devices.torch_gc()
    return samples, None


def _reload_for_hires(processing) -> None:
    with SkipWritingToConfig():
        from apps.backend.codex import main as _codex
        checkpoint_before = getattr(_codex, "_SELECTIONS").checkpoint_name
        modules_before = list(getattr(_codex, "_SELECTIONS").additional_modules)

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
                processing.firstpass_use_distilled_cfg_scale = (
                    processing.sd_model.use_distilled_cfg_scale
                )
                reload_required = True

        if reload_required:
            try:
                codex_main.refresh_model_loading_parameters()
                # Native model reload for hires with runtime options derived from env/processing
                try:
                    load_opts = EngineLoadOptions(
                        device=None,  # auto
                        dtype=None,
                        attention_backend=os.getenv("CODEX_ATTENTION_BACKEND"),
                        accelerator=os.getenv("CODEX_ACCELERATOR"),
                        vae_path=None,
                    )
                    new_engine = _load_engine(processing.hr_checkpoint_name, options=load_opts)
                    processing.sd_model = new_engine
                except Exception as exc:
                    raise RuntimeError(f"Failed to load hires checkpoint '{processing.hr_checkpoint_name}': {exc}")
            finally:
                codex_main.modules_change(modules_before, save=False, refresh=False)
                codex_main.checkpoint_change(checkpoint_before, save=False, refresh=False)
                codex_main.refresh_model_loading_parameters()

        if processing.sd_model.use_distilled_cfg_scale:
            processing.extra_generation_params["Hires Distilled CFG Scale"] = (
                processing.hr_distilled_cfg
            )


class Txt2ImgRuntime:
    """Encapsulates txt2img sampling so that the orchestration can be tested in isolation."""

    def __init__(
        self,
        processing,
        conditioning,
        unconditional_conditioning,
        seeds: Sequence[int],
        subseeds: Sequence[int],
        subseed_strength: float,
        prompts: Sequence[str],
    ) -> None:
        if not isinstance(processing, CodexProcessingTxt2Img):
            raise TypeError("Txt2ImgRuntime expects CodexProcessingTxt2Img")
        self.processing = processing
        self.processing.cfg_scale = getattr(self.processing, "guidance_scale", 7.0)
        self.conditioning = conditioning
        self.unconditional_conditioning = unconditional_conditioning
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.prompts = prompts
        self._noise_settings: NoiseSettings | None = None

    def _resolve_noise_settings(self) -> NoiseSettings:
        source = None
        eta_delta = 0
        overrides = getattr(self.processing, "override_settings", {})
        if isinstance(overrides, dict):
            source = overrides.get("randn_source") or overrides.get("noise_source")
            eta_delta = overrides.get("eta_noise_seed_delta", eta_delta)
        metadata = getattr(self.processing, "metadata", {})
        if isinstance(metadata, dict):
            source = metadata.get("randn_source", source)
        if hasattr(self.processing, "noise_source") and getattr(self.processing, "noise_source"):
            source = getattr(self.processing, "noise_source")
        env_source = os.getenv("CODEX_NOISE_SOURCE")
        if source is None and env_source:
            source = env_source

        try:
            source_kind = NoiseSourceKind.from_string(source) if source else NoiseSourceKind.GPU
        except ValueError:
            source_kind = NoiseSourceKind.GPU
        settings = NoiseSettings(
            source=source_kind,
            eta_noise_seed_delta=int(getattr(self.processing, "eta_noise_seed_delta", eta_delta) or eta_delta or 0),
        )
        self.processing.eta_noise_seed_delta = settings.eta_noise_seed_delta
        return settings

    def generate(self):
        # Parse extra-network tags and clean prompts first (may override size)
        cleaned_prompts, prompt_loras, prompt_controls = parse_prompts_with_extras(list(self.prompts))
        self.processing.prompts = cleaned_prompts
        # Apply width/height before sampler/rng setup
        # Validate width/height explicitly (JSON error on invalid)
        if 'width' in prompt_controls:
            w = int(prompt_controls['width'])
            if w % 8 != 0 or w < 8 or w > 8192:
                raise ValueError('Invalid <width>: must be multiple of 8 and in [8,8192]')
            self.processing.width = w
        if 'height' in prompt_controls:
            h = int(prompt_controls['height'])
            if h % 8 != 0 or h < 8 or h > 8192:
                raise ValueError('Invalid <height>: must be multiple of 8 and in [8,8192]')
            self.processing.height = h

        self._ensure_sampler()
        self.processing.seeds = list(self.seeds)
        self.processing.subseeds = list(self.subseeds)
        self.processing.negative_prompts = getattr(
            self.processing, "negative_prompts", [getattr(self.processing, "negative_prompt", "")]
        )
        self.processing.prepare_prompt_data()
        self._run_process_scripts()

        samples, decoded_samples = _prepare_first_pass_from_image(self.processing)
        # Apply prompt-level controls for cfg/steps/seed if present
        try:
            if 'cfg' in prompt_controls:
                self.processing.guidance_scale = float(prompt_controls['cfg'])
                self.processing.cfg_scale = self.processing.guidance_scale
            if 'steps' in prompt_controls:
                self.processing.steps = int(float(prompt_controls['steps']))
            if 'seed' in prompt_controls:
                s = int(float(prompt_controls['seed']))
                self.seeds = [s]
                self.processing.seeds = [s]
        except Exception:
            pass

        if samples is None and decoded_samples is None:
            # Optional tiling control for decode passes
            from apps.backend.runtime.memory import memory_management
            _old_tiled = memory_management.VAE_ALWAYS_TILED
            try:
                if prompt_controls.get('tiling') is True:
                    memory_management.VAE_ALWAYS_TILED = True
                samples = self._run_sampling(prompt_loras, prompt_controls)
                decoded_samples = self._maybe_decode_for_hr(samples)
            finally:
                memory_management.VAE_ALWAYS_TILED = _old_tiled

        if not self.processing.enable_hr:
            return samples

        _reload_for_hires(self.processing)

        return self._run_hires_pass(
            samples,
            decoded_samples,
            prompt_loras,
            prompt_controls,
        )

    def _ensure_sampler(self) -> None:
        algo = getattr(self.processing, "sampler_name", None)
        self.processing.sampler = CodexSampler(self.processing.sd_model, algorithm=algo)
        latent_channels = getattr(
            self.processing.sd_model.forge_objects_after_applying_lora.vae,
            "latent_channels",
            4,
        )
        shape = (
            latent_channels,
            self.processing.height // 8,
            self.processing.width // 8,
        )
        self._noise_settings = self._resolve_noise_settings()
        self.processing.rng = ImageRNG(
            shape,
            self.seeds,
            subseeds=self.subseeds,
            subseed_strength=self.subseed_strength,
            seed_resize_from_h=getattr(self.processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(self.processing, "seed_resize_from_w", 0),
            settings=self._noise_settings,
        )

    def _run_sampling(
        self,
        prompt_loras,
        prompt_controls,
        *,
        noise: torch.Tensor | None = None,
        image_conditioning: torch.Tensor | None = None,
        init_latent: torch.Tensor | None = None,
        start_at_step: int | None = None,
    ) -> torch.Tensor:
        if noise is None:
            if self.processing.rng is None:
                raise RuntimeError("RNG not initialised for sampling")
            noise = self.processing.rng.next()

        model = self.processing.sd_model

        if hasattr(model, "forge_objects_original") and model.forge_objects_original is not None:
            model.forge_objects = model.forge_objects_original.shallow_copy()

        self._run_before_and_process_batch_hooks()

        selections = codex_lora.get_selections() + prompt_loras
        seen = set(); merged = []
        for sel in selections:
            if sel.path in seen:
                continue
            seen.add(sel.path); merged.append(sel)
        if merged:
            stats = apply_loras_to_engine(model, merged)
            logging.info(
                "[native] Applied %d LoRA(s), %d params touched", stats.files, stats.params_touched
            )
        model.forge_objects = model.forge_objects_after_applying_lora.shallow_copy()

        clip_skip = int(prompt_controls.get('clip_skip')) if 'clip_skip' in prompt_controls else None
        if clip_skip is not None and hasattr(self.processing.sd_model, 'set_clip_skip'):
            self.processing.sd_model.set_clip_skip(clip_skip)
        if 'sampler' in prompt_controls:
            self.processing.sampler_name = str(prompt_controls['sampler'])

        strategy = prompt_controls.get('token_merge_strategy') or getattr(
            self.processing, 'get_token_merging_strategy', lambda: None
        )()
        if not strategy:
            strategy = os.getenv('CODEX_TOKEN_MERGE_STRATEGY', 'avg')
        ratio_override = prompt_controls.get('token_merge_ratio', None)
        ratio = float(ratio_override) if ratio_override is not None else float(
            self.processing.get_token_merging_ratio(for_hires=bool(init_latent is not None))
        )
        apply_token_merging(model, ratio, strategy=strategy)

        if self.processing.scripts is not None:
            self.processing.scripts.process_before_every_sampling(
                self.processing,
                x=noise,
                noise=noise,
                c=self.conditioning,
                uc=self.unconditional_conditioning,
            )

        if self.processing.modified_noise is not None:
            noise = self.processing.modified_noise
            self.processing.modified_noise = None

        def _preview_cb(denoised_latent: torch.Tensor, step: int, total: int) -> None:
            img = decode_latent_batch(self.processing.sd_model, denoised_latent, target_device=devices.cpu())
            arr = img[0].detach().float().cpu().clamp(-1, 1)
            arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
            backend_state.set_current_image(Image.fromarray(arr, mode='RGB'))

        if image_conditioning is None:
            image_conditioning = txt2img_conditioning(
                self.processing.sd_model,
                noise,
                self.processing.width,
                self.processing.height,
            )

        sampler_name = getattr(self.processing, "sampler_name", None) or self.processing.sampler.algorithm
        scheduler_name = getattr(self.processing, "scheduler", None)
        context = build_sampling_context(
            self.processing.sd_model,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            steps=int(getattr(self.processing, "steps", 20) or 20),
            noise_source=(self._noise_settings.source.value if self._noise_settings else None),
            eta_noise_seed_delta=(self._noise_settings.eta_noise_seed_delta if self._noise_settings else int(getattr(self.processing, "eta_noise_seed_delta", 0) or 0)),
            device=noise.device,
            dtype=noise.dtype,
        )

        samples = self.processing.sampler.sample(
            self.processing,
            noise,
            self.conditioning,
            self.unconditional_conditioning,
            image_conditioning=image_conditioning,
            init_latent=init_latent,
            start_at_step=start_at_step,
            preview_callback=_preview_cb,
            context=context,
        )

        samples = self._run_post_sample_hooks(samples)
        devices.torch_gc()
        return samples


    def _maybe_decode_for_hr(self, samples):
        if not self.processing.enable_hr:
            return None

        devices.torch_gc()

        if self.processing.latent_scale_mode is None:
            return _decode_latent_batch(
                self.processing.sd_model, samples, target_device=devices.cpu()
            ).to(dtype=torch.float32)

        return None

    def _run_post_sample_hooks(self, samples):
        script_runner = getattr(self.processing, "scripts", None)
        if script_runner is None or not hasattr(script_runner, "post_sample"):
            return samples

        class _PostSampleArgs:
            def __init__(self, samples):
                self.samples = samples
        args = _PostSampleArgs(samples)
        script_runner.post_sample(self.processing, args)
        return getattr(args, "samples", samples)

    def _run_process_scripts(self):
        script_runner = getattr(self.processing, "scripts", None)
        if script_runner is None or not hasattr(script_runner, "process"):
            return
        script_runner.process(self.processing)

    def _run_before_and_process_batch_hooks(self):
        script_runner = getattr(self.processing, "scripts", None)
        if script_runner is None:
            self._activate_extra_networks()
            return

        hook_kwargs = {
            "batch_number": 0,
            "prompts": getattr(self.processing, "prompts", self.prompts),
            "seeds": getattr(self.processing, "seeds", self.seeds),
            "subseeds": getattr(self.processing, "subseeds", self.subseeds),
            "negative_prompts": getattr(
                self.processing,
                "negative_prompts",
                [getattr(self.processing, "negative_prompt", "")],
            ),
        }

        if hasattr(script_runner, "before_process_batch"):
            script_runner.before_process_batch(self.processing, **hook_kwargs)

        if hasattr(script_runner, "process_batch"):
            script_runner.process_batch(self.processing, **hook_kwargs)

        self._activate_extra_networks()
        self._set_shared_job()

    def _activate_extra_networks(self):
        """Apply native extras (LoRA etc.) if selections are configured.

        No legacy extra_networks usage; we rely on explicit LoRA selections
        stored via Codex options.
        """
        if getattr(self.processing, "disable_extra_networks", False):
            return
        try:
            selections = codex_lora.get_selections()
        except Exception:
            selections = []
        if selections:
            stats = apply_loras_to_engine(self.processing.sd_model, selections)
            logging.info("[native] Applied %d LoRA(s), %d params touched", stats.files, stats.params_touched)

    def _set_shared_job(self):
        if getattr(self.processing, "iterations", 1) <= 1:
            return
        backend_state.begin(job=f"Batch 1 out of {self.processing.iterations}")

    def _latents_to_pil(self, decoded: torch.Tensor) -> list[Image.Image]:
        images: list[Image.Image] = []
        for sample in decoded:
            arr = sample.detach().float().cpu().clamp(-1, 1)
            arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
            images.append(Image.fromarray(arr, mode='RGB'))
        return images

    def _pil_to_tensor(self, images: list[Image.Image]) -> torch.Tensor:
        arrays = []
        for img in images:
            arr = np.array(img.convert('RGB')).astype(np.float32) / 255.0
            arr = arr * 2.0 - 1.0
            arr = np.moveaxis(arr, 2, 0)
            arrays.append(arr)
        tensor = torch.from_numpy(np.stack(arrays, axis=0))
        return tensor.to(devices.default_device(), dtype=torch.float32)

    def _run_hires_pass(
        self,
        base_samples: torch.Tensor,
        decoded_samples: torch.Tensor | None,
        base_prompt_loras,
        base_prompt_controls,
    ) -> torch.Tensor:
        processing = self.processing
        hi_cfg = processing.hires
        processing.ensure_hires_prompts()

        target_width = hi_cfg.resize_x or int(processing.width * hi_cfg.scale)
        target_height = hi_cfg.resize_y or int(processing.height * hi_cfg.scale)
        steps = hi_cfg.second_pass_steps or processing.steps
        denoise = float(hi_cfg.denoise)

        hr_cleaned_prompts, hr_prompt_loras, hr_prompt_controls = parse_prompts_with_extras(
            list(processing.hr_prompts or processing.all_prompts or self.prompts)
        )

        original = {
            "prompts": processing.prompts,
            "negative_prompts": getattr(processing, "negative_prompts", []),
            "width": processing.width,
            "height": processing.height,
            "guidance_scale": processing.guidance_scale,
            "steps": processing.steps,
        }

        processing.prompts = hr_cleaned_prompts
        processing.negative_prompts = processing.hr_negative_prompts or original["negative_prompts"]
        processing.width = target_width
        processing.height = target_height
        processing.guidance_scale = hi_cfg.cfg or processing.guidance_scale
        processing.cfg_scale = processing.guidance_scale
        processing.steps = steps
        processing.prepare_prompt_data()

        if processing.latent_scale_mode is not None:
            mode = processing.latent_scale_mode.get("mode", "bilinear")
            antialias = bool(processing.latent_scale_mode.get("antialias", False))
            latents = F.interpolate(
                base_samples,
                size=(target_height // 8, target_width // 8),
                mode=mode,
                align_corners=False if mode in {"bilinear", "bicubic"} else None,
                antialias=antialias,
            )
            tensor = decode_latent_batch(processing.sd_model, latents, target_device=devices.cpu())
            image_conditioning = txt2img_conditioning(
                processing.sd_model,
                latents,
                target_width,
                target_height,
            )
        else:
            if decoded_samples is None:
                decoded_samples = decode_latent_batch(processing.sd_model, base_samples, target_device=devices.cpu())
            pil_images = self._latents_to_pil(decoded_samples)
            upscaled = [img.resize((target_width, target_height), _RESAMPLE_LANCZOS) for img in pil_images]
            tensor = self._pil_to_tensor(upscaled)
            latents = processing.sd_model.encode_first_stage(tensor)
            image_conditioning = img2img_conditioning(
                processing.sd_model,
                tensor,
                latents,
                image_mask=getattr(processing, "image_mask", None),
                round_mask=getattr(processing, "round_image_mask", True),
            )

        shape = (latents.shape[1], latents.shape[2], latents.shape[3])
        hires_settings = self._noise_settings or self._resolve_noise_settings()
        rng = ImageRNG(
            shape,
            self.seeds,
            subseeds=self.subseeds,
            subseed_strength=self.subseed_strength,
            seed_resize_from_h=getattr(processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(processing, "seed_resize_from_w", 0),
            settings=hires_settings,
        )
        noise = rng.next().to(latents)

        start_index = max(0, min(int(round(denoise * steps)), steps - 1))

        samples = self._run_sampling(
            hr_prompt_loras,
            hr_prompt_controls,
            noise=noise,
            image_conditioning=image_conditioning,
            init_latent=latents,
            start_at_step=start_index,
        )

        processing.prompts = original["prompts"]
        processing.negative_prompts = original["negative_prompts"]
        processing.width = original["width"]
        processing.height = original["height"]
        processing.guidance_scale = original["guidance_scale"]
        processing.cfg_scale = processing.guidance_scale
        processing.steps = original["steps"]
        processing.prepare_prompt_data()

        return samples


def generate_txt2img(
    processing,
    conditioning,
    unconditional_conditioning,
    seeds: Sequence[int],
    subseeds: Sequence[int],
    subseed_strength: float,
    prompts: Sequence[str],
):
    runtime = Txt2ImgRuntime(
        processing,
        conditioning,
        unconditional_conditioning,
        seeds,
        subseeds,
        subseed_strength,
        prompts,
    )

    return runtime.generate()
