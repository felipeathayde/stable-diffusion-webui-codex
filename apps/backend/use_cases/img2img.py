from __future__ import annotations

from typing import Any, Sequence

import os

import numpy as np
import torch

from apps.backend.core import devices
from apps.backend.core.rng import ImageRNG, NoiseSettings, NoiseSourceKind
from apps.backend.runtime.sampling.driver import CodexSampler
from apps.backend.runtime.sampling.context import build_sampling_context
from apps.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.backend.runtime.processing.conditioners import img2img_conditioning, decode_latent_batch
from apps.backend.runtime.processing.models import CodexProcessingImg2Img
from apps.backend.patchers.token_merging import apply_token_merging
from apps.backend.codex import lora as codex_lora
from apps.backend.patchers.lora_apply import apply_loras_to_engine
from apps.backend.core.state import state as backend_state
from PIL import Image

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


"""Image→Image task runtime (skeleton).

Mirrors the organization of txt2img.py while keeping responsibilities local to
the engines that call it. Engines can import helpers from here to avoid large
conditionals.
"""


class Img2ImgRuntime:
    def __init__(self, processing: Any, conditioning: Any, unconditional_conditioning: Any, *, prompts: Sequence[str]) -> None:
        if not isinstance(processing, CodexProcessingImg2Img):
            raise TypeError("Img2ImgRuntime expects CodexProcessingImg2Img")
        self.processing = processing
        self.conditioning = conditioning
        self.unconditional_conditioning = unconditional_conditioning
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

        eta_value = int(getattr(self.processing, "eta_noise_seed_delta", eta_delta) or eta_delta or 0)
        settings = NoiseSettings(source=source_kind, eta_noise_seed_delta=eta_value)
        self.processing.eta_noise_seed_delta = settings.eta_noise_seed_delta
        return settings

    def _encode_image_to_latent(self) -> tuple[torch.Tensor, torch.Tensor]:
        image = getattr(self.processing, "init_image", None)
        if image is None:
            raise ValueError("img2img requires processing.init_image")
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.moveaxis(arr, 2, 0)
        tensor = torch.from_numpy(np.expand_dims(arr, axis=0)).to(devices.default_device(), dtype=torch.float32)
        latent = self.processing.sd_model.encode_first_stage(tensor)
        return latent, tensor

    def _apply_extras(self) -> None:
        if getattr(self.processing, "disable_extra_networks", False):
            return
        try:
            selections = codex_lora.get_selections()
        except Exception:
            selections = []
        if selections:
            stats = apply_loras_to_engine(self.processing.sd_model, selections)
            # best-effort log through processing if available
            try:
                print(f"[native] img2img applied {stats.files} LoRA(s), {stats.params_touched} params touched")
            except Exception:
                pass

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

    def generate(self) -> torch.Tensor:
        # Parse extra-network tags and clean prompts
        if hasattr(self.processing, 'prompts'):
            cleaned_prompts, prompt_loras, prompt_controls = parse_prompts_with_extras(list(getattr(self.processing, 'prompts', self.prompts)))
            self.processing.prompts = cleaned_prompts
        else:
            prompt_loras, prompt_controls = [], {}

        # Apply width/height before preparing sampler
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

        # Prepare sampler
        algo = getattr(self.processing, "sampler_name", None)
        self.processing.sampler = CodexSampler(self.processing.sd_model, algorithm=algo)
        latent_channels = getattr(self.processing.sd_model.codex_objects_after_applying_lora.vae, "latent_channels", 4)
        shape = (latent_channels, self.processing.height // 8, self.processing.width // 8)
        self._noise_settings = self._resolve_noise_settings()

        # Encode input image to latent
        init_latent, source_tensor = self._encode_image_to_latent()
        init_latent = init_latent.to(devices.default_device())
        noise_rng = ImageRNG(
            shape,
            [getattr(self.processing, "seed", -1) or -1],
            subseeds=getattr(self.processing, "subseeds", []),
            subseed_strength=getattr(self.processing, "subseed_strength", 0.0),
            seed_resize_from_h=getattr(self.processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(self.processing, "seed_resize_from_w", 0),
            settings=self._noise_settings,
        )
        noise = noise_rng.next().to(init_latent)

        # Apply LoRA and token merging
        if hasattr(self.processing.sd_model, "codex_objects_original") and self.processing.sd_model.codex_objects_original is not None:
            self.processing.sd_model.codex_objects = self.processing.sd_model.codex_objects_original.shallow_copy()
        # Merge global selections with prompt-local LoRAs
        try:
            selections = codex_lora.get_selections() + prompt_loras
            seen = set(); merged = []
            for s in selections:
                if s.path in seen:
                    continue
                seen.add(s.path); merged.append(s)
        except Exception:
            merged = prompt_loras
        if merged:
            stats = apply_loras_to_engine(self.processing.sd_model, merged)
            try:
                print(f"[native] img2img applied {stats.files} LoRA(s), {stats.params_touched} params touched")
            except Exception:
                pass
        self.processing.sd_model.codex_objects = self.processing.sd_model.codex_objects_after_applying_lora.shallow_copy()
        # Controls: clip_skip, sampler, scheduler, token merge
        try:
            cs = int(prompt_controls.get('clip_skip')) if 'clip_skip' in prompt_controls else None
            if cs is not None and hasattr(self.processing.sd_model, 'set_clip_skip'):
                self.processing.sd_model.set_clip_skip(cs)
        except Exception:
            pass
        if 'sampler' in prompt_controls:
            self.processing.sampler_name = str(prompt_controls['sampler'])
        strat = prompt_controls.get('token_merge_strategy') or getattr(self.processing, 'get_token_merging_strategy', lambda: None)()
        if not strat:
            strat = os.getenv('CODEX_TOKEN_MERGE_STRATEGY', 'avg')
        ratio_override = prompt_controls.get('token_merge_ratio', None)
        ratio = float(ratio_override) if ratio_override is not None else float(getattr(self.processing, "token_merging_ratio", 0.0))
        apply_token_merging(self.processing.sd_model, ratio, strategy=strat)

        # Denoising strength controls where we start in the sigma schedule
        steps = int(getattr(self.processing, "steps", 20) or 20)
        strength = float(getattr(self.processing, "denoising_strength", 0.5) or 0.5)
        start_step = max(0, min(int(round(strength * steps)), steps - 1))

        def _preview_cb(denoised_latent: torch.Tensor, step: int, total: int) -> None:
            try:
                sample = self.processing.sd_model.decode_first_stage(denoised_latent.to(devices.default_device()))
                arr = sample[0].detach().float().cpu().clamp(-1, 1)
                arr = ((arr + 1.0) * 0.5).mul(255.0).byte().movedim(0, -1).numpy()
                backend_state.set_current_image(Image.fromarray(arr, mode='RGB'))
            except Exception:
                pass

        # Build c_concat from mask when available
        img_cond = None
        try:
            img_cond = img2img_conditioning(
                self.processing.sd_model,
                source_image=source_tensor,
                latent_image=init_latent,
                image_mask=getattr(self.processing, "image_mask", None),
                round_mask=getattr(self.processing, "round_image_mask", True),
            )
        except Exception:
            img_cond = None

        # Apply prompt-level controls for cfg/steps/seed/denoise if present
        try:
            if 'cfg' in prompt_controls:
                self.processing.cfg_scale = float(prompt_controls['cfg'])
            if 'steps' in prompt_controls:
                self.processing.steps = int(float(prompt_controls['steps']))
            if 'seed' in prompt_controls:
                s = int(float(prompt_controls['seed']))
                # Not overriding batch seeds here; use main seed when present
                self.processing.seed = s
            if 'denoise' in prompt_controls:
                self.processing.denoising_strength = float(prompt_controls['denoise'])
        except Exception:
            pass

        # Optional tiling control for decode passes
        from apps.backend.runtime.memory import memory_management
        _old_tiled = memory_management.VAE_ALWAYS_TILED
        try:
            if prompt_controls.get('tiling') is True:
                memory_management.VAE_ALWAYS_TILED = True
            sampler_name = getattr(self.processing, "sampler_name", None) or self.processing.sampler.algorithm
            scheduler_name = getattr(self.processing, "scheduler", None)
            context = build_sampling_context(
                self.processing.sd_model,
                sampler_name=sampler_name,
                scheduler_name=scheduler_name,
                steps=steps,
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
                image_conditioning=img_cond,
                init_latent=init_latent,
                start_at_step=start_step,
                preview_callback=_preview_cb,
                context=context,
            )
        finally:
            memory_management.VAE_ALWAYS_TILED = _old_tiled

        if getattr(self.processing.hires, "enabled", False):
            return self._run_hires_pass(samples, prompt_loras, prompt_controls)

        return samples

    def _run_hires_pass(self, base_samples: torch.Tensor, base_prompt_loras, base_prompt_controls) -> torch.Tensor:
        hi_cfg = self.processing.hires
        target_width = hi_cfg.resize_x or int(self.processing.width * hi_cfg.scale)
        target_height = hi_cfg.resize_y or int(self.processing.height * hi_cfg.scale)
        steps = hi_cfg.second_pass_steps or self.processing.steps
        denoise = float(hi_cfg.denoise)

        hr_prompts_source = hi_cfg.prompt if hi_cfg.prompt else self.prompts
        hr_prompts_raw = [hr_prompts_source] if isinstance(hr_prompts_source, str) else list(hr_prompts_source)
        hr_cleaned_prompts, hr_prompt_loras, hr_prompt_controls = parse_prompts_with_extras(hr_prompts_raw)

        decoded = decode_latent_batch(self.processing.sd_model, base_samples, target_device=devices.cpu())
        pil_images = self._latents_to_pil(decoded)
        upscaled = [img.resize((target_width, target_height), _RESAMPLE_LANCZOS) for img in pil_images]
        tensor = self._pil_to_tensor(upscaled)
        latents = self.processing.sd_model.encode_first_stage(tensor)
        image_conditioning = img2img_conditioning(
            self.processing.sd_model,
            tensor,
            latents,
            image_mask=getattr(self.processing, "image_mask", None),
            round_mask=getattr(self.processing, "round_image_mask", True),
        )

        shape = (latents.shape[1], latents.shape[2], latents.shape[3])
        hires_settings = self._noise_settings or self._resolve_noise_settings()
        rng = ImageRNG(
            shape,
            getattr(self.processing, "seeds", [getattr(self.processing, "seed", -1) or -1]),
            subseeds=getattr(self.processing, "subseeds", []),
            subseed_strength=getattr(self.processing, "subseed_strength", 0.0),
            seed_resize_from_h=getattr(self.processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(self.processing, "seed_resize_from_w", 0),
            settings=hires_settings,
        )
        noise = rng.next().to(latents)

        start_index = max(0, min(int(round(denoise * steps)), steps - 1))

        original = {
            "prompts": getattr(self.processing, 'prompts', []),
            "negative": getattr(self.processing, 'negative_prompts', []),
            "width": self.processing.width,
            "height": self.processing.height,
            "steps": self.processing.steps,
            "denoise": getattr(self.processing, 'denoising_strength', 0.5),
        }

        self.processing.prompts = hr_cleaned_prompts
        self.processing.negative_prompts = [hi_cfg.negative_prompt] if hi_cfg.negative_prompt else original["negative"]
        self.processing.width = target_width
        self.processing.height = target_height
        self.processing.steps = steps
        self.processing.denoising_strength = denoise

        self._apply_extras()

        sampler_name = getattr(self.processing, "sampler_name", None) or self.processing.sampler.algorithm
        scheduler_name = getattr(self.processing, "scheduler", None)
        context = build_sampling_context(
            self.processing.sd_model,
            sampler_name=sampler_name,
            scheduler_name=scheduler_name,
            steps=steps,
            noise_source=(hires_settings.source.value if hires_settings else None),
            eta_noise_seed_delta=(hires_settings.eta_noise_seed_delta if hires_settings else int(getattr(self.processing, "eta_noise_seed_delta", 0) or 0)),
            device=noise.device,
            dtype=noise.dtype,
        )

        samples = self.processing.sampler.sample(
            self.processing,
            noise,
            self.conditioning,
            self.unconditional_conditioning,
            image_conditioning=image_conditioning,
            init_latent=latents,
            start_at_step=start_index,
            context=context,
        )

        self.processing.prompts = original["prompts"]
        self.processing.negative_prompts = original["negative"]
        self.processing.width = original["width"]
        self.processing.height = original["height"]
        self.processing.steps = original["steps"]
        self.processing.denoising_strength = original["denoise"]

        return samples


def run_img2img(*, engine, processing: Any, conditioning: Any, unconditional_conditioning: Any, prompts: Sequence[str]) -> Any:
    runtime = Img2ImgRuntime(processing, conditioning, unconditional_conditioning, prompts=prompts)
    return runtime.generate()
