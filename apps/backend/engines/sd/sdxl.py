import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.patchers.clip import CLIP
from apps.backend.patchers.vae import VAE
from apps.backend.patchers.unet import UnetPatcher
from apps.backend.runtime.text_processing.classic_engine import ClassicTextProcessingEngine
from apps.backend.infra.config.args import dynamic_args
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.common.nn.unet import Timestep

import safetensors.torch as sf
from apps.backend.runtime import utils

def _opts():
    # Minimal option shim for SDXL-specific parameters without legacy deps
    from types import SimpleNamespace
    return SimpleNamespace(
        sdxl_crop_left=0,
        sdxl_crop_top=0,
        sdxl_refiner_low_aesthetic_score=2.5,
        sdxl_refiner_high_aesthetic_score=6.0,
    )


class StableDiffusionXL(CodexDiffusionEngine):
    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)

        clip = CLIP(
            model_dict={
                'clip_l': codex_components['text_encoder'],
                'clip_g': codex_components['text_encoder_2']
            },
            tokenizer_dict={
                'clip_l': codex_components['tokenizer'],
                'clip_g': codex_components['tokenizer_2']
            },
            model_config=estimated_config,
        )

        vae = VAE(model=codex_components['vae'])

        unet = UnetPatcher.from_model(
            model=codex_components['unet'],
            diffusers_scheduler=codex_components['scheduler'],
            config=estimated_config
        )

        self.text_processing_engine_l = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_l,
            tokenizer=clip.tokenizer.clip_l,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_l',
            embedding_expected_shape=2048,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=False,
            minimal_clip_skip=2,
            clip_skip=2,
            return_pooled=False,
            final_layer_norm=False,
        )

        self.text_processing_engine_g = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_g,
            tokenizer=clip.tokenizer.clip_g,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_g',
            embedding_expected_shape=2048,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=True,
            minimal_clip_skip=2,
            clip_skip=2,
            return_pooled=True,
            final_layer_norm=False,
        )

        self.embedder = Timestep(256)

        self.codex_objects = CodexObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.codex_objects_original = self.codex_objects.shallow_copy()
        self.codex_objects_after_applying_lora = self.codex_objects.shallow_copy()

        # WebUI Legacy
        self.is_sdxl = True

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine_l.clip_skip = clip_skip
        self.text_processing_engine_g.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)

        cond_l = self.text_processing_engine_l(prompt)
        cond_g, clip_pooled = self.text_processing_engine_g(prompt)

        width = getattr(prompt, 'width', 1024) or 1024
        height = getattr(prompt, 'height', 1024) or 1024
        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

        crop_w = _opts().sdxl_crop_left
        crop_h = _opts().sdxl_crop_top
        target_width = width
        target_height = height

        out = [
            self.embedder(torch.Tensor([height])), self.embedder(torch.Tensor([width])),
            self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
            self.embedder(torch.Tensor([target_height])), self.embedder(torch.Tensor([target_width]))
        ]

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1).to(clip_pooled)

        force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in prompt)

        if force_zero_negative_prompt:
            clip_pooled = torch.zeros_like(clip_pooled)
            cond_l = torch.zeros_like(cond_l)
            cond_g = torch.zeros_like(cond_g)

        cond = dict(
            crossattn=torch.cat([cond_l, cond_g], dim=2),
            vector=torch.cat([clip_pooled, flat], dim=1),
        )

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine_l.process_texts([prompt])
        return token_count, self.text_processing_engine_l.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.codex_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.codex_objects.vae.first_stage_model.process_out(x)
        sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    def save_checkpoint(self, filename):
        sd = {}
        sd.update(
            utils.get_state_dict_after_quant(self.codex_objects.unet.model.diffusion_model, prefix='model.diffusion_model.')
        )
        sd.update(
            model_list.SDXL.process_clip_state_dict_for_saving(self,
                utils.get_state_dict_after_quant(self.codex_objects.clip.cond_stage_model, prefix='')
            )
        )
        sd.update(
            utils.get_state_dict_after_quant(self.codex_objects.vae.first_stage_model, prefix='first_stage_model.')
        )
        sf.save_file(sd, filename)
        return filename


class StableDiffusionXLRefiner(CodexDiffusionEngine):
    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)

        clip = CLIP(
            model_dict={'clip_g': codex_components['text_encoder']},
            tokenizer_dict={'clip_g': codex_components['tokenizer']},
            model_config=estimated_config,
        )

        vae = VAE(model=codex_components['vae'])

        unet = UnetPatcher.from_model(
            model=codex_components['unet'],
            diffusers_scheduler=codex_components['scheduler'],
            config=estimated_config
        )

        self.text_processing_engine_g = ClassicTextProcessingEngine(
            text_encoder=clip.cond_stage_model.clip_g,
            tokenizer=clip.tokenizer.clip_g,
            embedding_dir=dynamic_args['embedding_dir'],
            embedding_key='clip_g',
            embedding_expected_shape=2048,
            emphasis_name=dynamic_args['emphasis_name'],
            text_projection=True,
            minimal_clip_skip=2,
            clip_skip=2,
            return_pooled=True,
            final_layer_norm=False,
        )

        self.embedder = Timestep(256)

        self.codex_objects = CodexObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.codex_objects_original = self.codex_objects.shallow_copy()
        self.codex_objects_after_applying_lora = self.codex_objects.shallow_copy()

        # WebUI Legacy
        self.is_sdxl = True

    def set_clip_skip(self, clip_skip):
        self.text_processing_engine_g.clip_skip = clip_skip

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)

        cond_g, clip_pooled = self.text_processing_engine_g(prompt)

        width = getattr(prompt, 'width', 1024) or 1024
        height = getattr(prompt, 'height', 1024) or 1024
        is_negative_prompt = getattr(prompt, 'is_negative_prompt', False)

        crop_w = _opts().sdxl_crop_left
        crop_h = _opts().sdxl_crop_top
        aesthetic = _opts().sdxl_refiner_low_aesthetic_score if is_negative_prompt else _opts().sdxl_refiner_high_aesthetic_score

        out = [
            self.embedder(torch.Tensor([height])), self.embedder(torch.Tensor([width])),
            self.embedder(torch.Tensor([crop_h])), self.embedder(torch.Tensor([crop_w])),
            self.embedder(torch.Tensor([aesthetic]))
        ]

        flat = torch.flatten(torch.cat(out)).unsqueeze(dim=0).repeat(clip_pooled.shape[0], 1).to(clip_pooled)

        force_zero_negative_prompt = is_negative_prompt and all(x == '' for x in prompt)

        if force_zero_negative_prompt:
            clip_pooled = torch.zeros_like(clip_pooled)
            cond_g = torch.zeros_like(cond_g)

        cond = dict(
            crossattn=cond_g,
            vector=torch.cat([clip_pooled, flat], dim=1),
        )

        return cond

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        _, token_count = self.text_processing_engine_g.process_texts([prompt])
        return token_count, self.text_processing_engine_g.get_target_prompt_token_count(token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.codex_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.codex_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.codex_objects.vae.first_stage_model.process_out(x)
        sample = self.codex_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)

    def save_checkpoint(self, filename):
        sd = {}
        sd.update(
            utils.get_state_dict_after_quant(self.codex_objects.unet.model.diffusion_model, prefix='model.diffusion_model.')
        )
        sd.update(
            model_list.SDXLRefiner.process_clip_state_dict_for_saving(self,
                utils.get_state_dict_after_quant(self.codex_objects.clip.cond_stage_model, prefix='')
            )
        )
        sd.update(
            utils.get_state_dict_after_quant(self.codex_objects.vae.first_stage_model, prefix='first_stage_model.')
        )
        sf.save_file(sd, filename)
        return filename
