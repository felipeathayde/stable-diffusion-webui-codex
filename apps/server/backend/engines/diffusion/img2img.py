from __future__ import annotations

"""Image→Image task runtime (skeleton).

Mirrors the organization of txt2img.py while keeping responsibilities local to
the engines that call it. Engines can import helpers from here to avoid large
conditionals.
"""

from typing import Any, Sequence

import numpy as np
import torch

from apps.server.backend.core import devices
from apps.server.backend.core.rng import ImageRNG
from apps.server.backend.runtime.sampling.driver import CodexSampler
from apps.server.backend.runtime.text_processing.extra_nets import parse_prompts_with_extras
from apps.server.backend.patchers.token_merging import apply_token_merging, SkipWritingToConfig
from apps.server.backend.codex import lora as codex_lora
from apps.server.backend.patchers.lora_apply import apply_loras_to_engine
from apps.server.backend.core.state import state as backend_state
from PIL import Image


class Img2ImgRuntime:
    def __init__(self, processing: Any, conditioning: Any, unconditional_conditioning: Any, *, prompts: Sequence[str]) -> None:
        self.processing = processing
        self.conditioning = conditioning
        self.unconditional_conditioning = unconditional_conditioning
        self.prompts = prompts

    def _encode_image_to_latent(self) -> torch.Tensor:
        image = getattr(self.processing, "init_image", None)
        if image is None:
            raise ValueError("img2img requires processing.init_image")
        arr = np.array(image).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        arr = np.moveaxis(arr, 2, 0)
        tensor = torch.from_numpy(np.expand_dims(arr, axis=0))
        tensor = tensor.to(devices.default_device(), dtype=torch.float32)
        latent = self.processing.sd_model.encode_first_stage(tensor)
        return latent

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

    def generate(self) -> torch.Tensor:
        # Parse extra-network tags and clean prompts
        if hasattr(self.processing, 'prompts'):
            cleaned_prompts, prompt_loras, prompt_controls = parse_prompts_with_extras(list(getattr(self.processing, 'prompts', self.prompts)))
            self.processing.prompts = cleaned_prompts
        else:
            prompt_loras, prompt_controls = [], {}

        # Apply width/height before preparing sampler
        try:
            if 'width' in prompt_controls:
                self.processing.width = int(prompt_controls['width'])
            if 'height' in prompt_controls:
                self.processing.height = int(prompt_controls['height'])
        except Exception:
            pass

        # Prepare sampler
        algo = getattr(self.processing, "sampler_name", None)
        self.processing.sampler = CodexSampler(self.processing.sd_model, algorithm=algo)
        latent_channels = getattr(self.processing.sd_model.forge_objects_after_applying_lora.vae, "latent_channels", 4)
        shape = (latent_channels, self.processing.height // 8, self.processing.width // 8)

        # Encode input image to latent
        init_latent = self._encode_image_to_latent().to(devices.default_device())
        noise = ImageRNG(shape, [getattr(self.processing, "seed", -1) or -1]).next().to(init_latent)

        # Apply LoRA and token merging
        if hasattr(self.processing.sd_model, "forge_objects_original") and self.processing.sd_model.forge_objects_original is not None:
            self.processing.sd_model.forge_objects = self.processing.sd_model.forge_objects_original.shallow_copy()
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
        self.processing.sd_model.forge_objects = self.processing.sd_model.forge_objects_after_applying_lora.shallow_copy()
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
            if hasattr(self.processing, 'image_mask') and self.processing.image_mask is not None:
                import numpy as np
                mask = np.array(self.processing.image_mask).astype(np.float32) / 255.0
                if getattr(self.processing, 'round_image_mask', True):
                    mask = (mask > 0.5).astype(np.float32)
                mask = np.mean(mask, axis=2) if mask.ndim == 3 else mask  # single channel
                mask_t = torch.from_numpy(mask).to(init_latent).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                # resize to latent size
                mask_t = torch.nn.functional.interpolate(mask_t, size=(init_latent.shape[2], init_latent.shape[3]), mode='nearest')
                img_cond = mask_t
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

        samples = self.processing.sampler.sample(
            self.processing,
            noise,
            self.conditioning,
            self.unconditional_conditioning,
            image_conditioning=img_cond,
            init_latent=init_latent,
            start_at_step=start_step,
            preview_callback=_preview_cb,
        )

        return samples


def run_img2img(*, engine, processing: Any, conditioning: Any, unconditional_conditioning: Any, prompts: Sequence[str]) -> Any:
    runtime = Img2ImgRuntime(processing, conditioning, unconditional_conditioning, prompts=prompts)
    return runtime.generate()
