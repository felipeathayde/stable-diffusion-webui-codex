from __future__ import annotations

from typing import Sequence

import logging

import numpy as np
import torch

from apps.server.backend.core import devices
from apps.server.backend.core.rng import ImageRNG
from apps.server.backend.runtime.sampling.driver import CodexSampler


class _ExtraNetworksShim:
    @staticmethod
    def activate(processing, data):
        raise NotImplementedError("Extra networks activation is not implemented natively yet")

try:  # optional – only present when LoRA extension is available
    from extensions_builtin.sd_forge_lora import networks as lora_networks
except Exception:  # pragma: no cover - optional dependency
    lora_networks = None
from apps.server.backend.patchers.token_merging import apply_token_merging, SkipWritingToConfig
from apps.server.backend.codex import main as codex_main
from apps.server.backend.codex.loader import load_engine as _load_engine, EngineLoadOptions


def _decode_latent_batch(model, batch, target_device=None) -> torch.Tensor:
    """Decode latents using the engine VAE (native)."""
    decoded = model.decode_first_stage(batch)
    if target_device is not None:
        decoded = decoded.to(target_device)
    return decoded


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
        from apps.server.backend.codex import main as _codex
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
        self.processing = processing
        self.conditioning = conditioning
        self.unconditional_conditioning = unconditional_conditioning
        self.seeds = seeds
        self.subseeds = subseeds
        self.subseed_strength = subseed_strength
        self.prompts = prompts

    def generate(self):
        self._ensure_sampler()

        self.processing.prompts = list(self.prompts)
        self.processing.seeds = list(self.seeds)
        self.processing.subseeds = list(self.subseeds)
        self.processing.negative_prompts = getattr(
            self.processing, "negative_prompts", [getattr(self.processing, "negative_prompt", "")]
        )
        self._run_process_scripts()

        samples, decoded_samples = _prepare_first_pass_from_image(self.processing)

        if samples is None and decoded_samples is None:
            samples = self._run_base_sampling()
            decoded_samples = self._maybe_decode_for_hr(samples)

        if not self.processing.enable_hr:
            return samples

        _reload_for_hires(self.processing)

        return self.processing.sample_hr_pass(
            samples,
            decoded_samples,
            self.seeds,
            self.subseeds,
            self.subseed_strength,
            self.prompts,
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
        self.processing.rng = ImageRNG(
            shape,
            self.seeds,
            subseeds=self.subseeds,
            subseed_strength=self.subseed_strength,
            seed_resize_from_h=getattr(self.processing, "seed_resize_from_h", 0),
            seed_resize_from_w=getattr(self.processing, "seed_resize_from_w", 0),
        )

    def _run_base_sampling(self):
        noise = self.processing.rng.next()

        model = self.processing.sd_model

        if hasattr(model, "forge_objects_original") and model.forge_objects_original is not None:
            model.forge_objects = model.forge_objects_original.shallow_copy()

        self._run_before_and_process_batch_hooks()

        model.forge_objects = model.forge_objects_after_applying_lora.shallow_copy()
        apply_token_merging(
            model, self.processing.get_token_merging_ratio()
        )

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

        samples = self.processing.sampler.sample(
            self.processing,
            noise,
            self.conditioning,
            self.unconditional_conditioning,
            image_conditioning=self.processing.txt2img_image_conditioning(noise),
        )

        samples = self._run_post_sample_hooks(samples)

        # Native backend state is updated by services; no legacy shared.state here.

        del noise
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
        try:
            args = _PostSampleArgs(samples)
            script_runner.post_sample(self.processing, args)
            return getattr(args, "samples", samples)
        except Exception:
            return samples

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
        if getattr(self.processing, "disable_extra_networks", False):
            return
        bridge = None
        if bridge is not None:
            bridge.ensure_lora_registry()
        elif lora_networks is not None:
            try:
                print(
                    "[runtime] scanning lora dir:",
                    "<unset>",
                )
                lora_networks.list_available_networks()
            except Exception as exc:
                print("[runtime] LoRA listing failed:", exc)
        if hasattr(self.processing, "parse_extra_network_prompts"):
            self.processing.parse_extra_network_prompts()
        data = getattr(self.processing, "extra_network_data", None)
        if data:
            if lora_networks is not None:
                print(
                    "[runtime] activating extra networks:",
                    list(data.keys()),
                    "lotr entries",
                    len(getattr(lora_networks, "available_networks", {})),
                    "aliases",
                    len(getattr(lora_networks, "available_network_aliases", {})),
                )
            extra_networks.activate(self.processing, data)

    def _set_shared_job(self):
        if getattr(self.processing, "n_iter", 1) <= 1:
            return
        state = getattr(shared, "state", None)
        if state is not None:
            state.job = f"Batch 1 out of {self.processing.n_iter}"


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
