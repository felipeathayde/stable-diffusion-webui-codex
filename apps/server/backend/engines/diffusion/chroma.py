import torch

from huggingface_guess import model_list
from .base import ForgeDiffusionEngine, ForgeObjects
from apps.server.backend.patchers.clip import CLIP
from apps.server.backend.patchers.vae import VAE
from apps.server.backend.patchers.unet import UnetPatcher
from apps.server.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine
from apps.server.backend.config.args import dynamic_args
from apps.server.backend.runtime.modules.k_prediction import FlowMatchEulerPrediction
from apps.server.backend.runtime.memory import memory_management

class Chroma(ForgeDiffusionEngine):
    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={
                't5xxl': huggingface_components['text_encoder']
            },
            tokenizer_dict={
                't5xxl': huggingface_components['tokenizer']
            }
        )

        vae = VAE(model=huggingface_components['vae'])
        k_predictor = FlowMatchEulerPrediction(
            mu=1.0
        )
        unet = UnetPatcher.from_model(
            model=huggingface_components['transformer'],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config
        )

        self.text_processing_engine_t5 = T5TextProcessingEngine(
            text_encoder=clip.cond_stage_model.t5xxl,
            tokenizer=clip.tokenizer.t5xxl,
            emphasis_name=dynamic_args['emphasis_name'],
            min_length=1
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        pass
        
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x):
        sample = self.forge_objects.vae.encode(x.movedim(1, -1) * 0.5 + 0.5)
        sample = self.forge_objects.vae.first_stage_model.process_in(sample)
        return sample.to(x)

    @torch.inference_mode()
    def decode_first_stage(self, x):
        sample = self.forge_objects.vae.first_stage_model.process_out(x)
        sample = self.forge_objects.vae.decode(sample).movedim(-1, 1) * 2.0 - 1.0
        return sample.to(x)        
