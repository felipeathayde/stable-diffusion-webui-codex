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
        self._apply_extras()
        self.processing.sd_model.forge_objects = self.processing.sd_model.forge_objects_after_applying_lora.shallow_copy()
        apply_token_merging(self.processing.sd_model, getattr(self.processing, "token_merging_ratio", 0.0))

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

        samples = self.processing.sampler.sample(
            self.processing,
            noise,
            self.conditioning,
            self.unconditional_conditioning,
            image_conditioning=getattr(self.processing, 'img2img_image_conditioning', lambda *_: None)(None, None, None),
            init_latent=init_latent,
            start_at_step=start_step,
            preview_callback=_preview_cb,
        )

        return samples


def run_img2img(*, engine, processing: Any, conditioning: Any, unconditional_conditioning: Any, prompts: Sequence[str]) -> Any:
    runtime = Img2ImgRuntime(processing, conditioning, unconditional_conditioning, prompts=prompts)
    return runtime.generate()
