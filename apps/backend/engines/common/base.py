import torch
import safetensors.torch as sf

from apps.backend.runtime.adapters import utils

class CodexObjects:
    def __init__(self, unet, clip, vae, clipvision):
        self.unet = unet
        self.clip = clip
        self.vae = vae
        self.clipvision = clipvision

    def shallow_copy(self):
        return CodexObjects(
            self.unet,
            self.clip,
            self.vae,
            self.clipvision
        )


class CodexDiffusionEngine:
    matched_guesses: tuple = ()

    def __init__(self, estimated_config, codex_components):
        self.model_config = estimated_config
        self.is_inpaint = estimated_config.inpaint_model()

        self.codex_objects = None
        self.codex_objects_original = None
        self.codex_objects_after_applying_lora = None

        self.current_lora_hash = str([])

        self.fix_for_webui_backward_compatibility()

    def set_clip_skip(self, clip_skip):
        pass

    def get_first_stage_encoding(self, x):
        return x  # legacy code, do not change

    def get_learned_conditioning(self, prompt: list[str]):
        pass

    def encode_first_stage(self, x):
        pass

    def decode_first_stage(self, x):
        pass

    def get_prompt_lengths_on_ui(self, prompt):
        return 0, 75

    def is_webui_legacy_model(self):
        return self.is_sd1 or self.is_sd2 or self.is_sdxl or self.is_sd3

    def fix_for_webui_backward_compatibility(self):
        self.tiling_enabled = False
        self.first_stage_model = None
        self.cond_stage_model = None
        self.use_distilled_cfg_scale = False
        self.is_sd1 = False
        self.is_sd2 = False
        self.is_sdxl = False
        self.is_sd3 = False
        return

    def save_unet(self, filename):
        sd = utils.get_state_dict_after_quant(self.codex_objects.unet.model.diffusion_model)
        sf.save_file(sd, filename)
        return filename
