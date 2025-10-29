import torch

from apps.backend.engines.common.base import CodexDiffusionEngine, CodexObjects
from apps.backend.patchers.clip import CLIP
from apps.backend.patchers.vae import VAE
from apps.backend.patchers.unet import UnetPatcher
from apps.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine
from apps.backend.infra.config.args import dynamic_args
from apps.backend.runtime.modules.k_prediction import FlowMatchEulerPrediction
from apps.backend.runtime.memory import memory_management


class Chroma(CodexDiffusionEngine):
    def __init__(self, estimated_config, codex_components):
        super().__init__(estimated_config, codex_components)
        self.is_inpaint = False

        clip = CLIP(
            model_dict={'t5xxl': codex_components['text_encoder']},
            tokenizer_dict={'t5xxl': codex_components['tokenizer']},
            model_config=estimated_config,
        )

        vae = VAE(model=codex_components['vae'])
        k_predictor = FlowMatchEulerPrediction(
            mu=1.0
        )
        unet = UnetPatcher.from_model(
            model=codex_components['transformer'],
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

        self.codex_objects = CodexObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.codex_objects_original = self.codex_objects.shallow_copy()
        self.codex_objects_after_applying_lora = self.codex_objects.shallow_copy()

    def set_clip_skip(self, clip_skip):
        pass
        
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.codex_objects.clip.patcher)
        return self.text_processing_engine_t5(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt):
        token_count = len(self.text_processing_engine_t5.tokenize([prompt])[0])
        return token_count, max(255, token_count)

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
