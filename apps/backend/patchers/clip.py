from apps.backend.runtime.memory import memory_management
from .base import ModelPatcher
from apps.backend.runtime.nn import ModuleDict, ObjectDict


class JointTextEncoder(ModuleDict):
    pass


class CLIP:
    def __init__(self, model_dict=None, tokenizer_dict=None, *, model_config=None, no_init=False):
        model_dict = model_dict or {}
        tokenizer_dict = tokenizer_dict or {}
        if no_init:
            return

        load_device = memory_management.text_encoder_device()
        offload_device = memory_management.text_encoder_offload_device()

        self.cond_stage_model = JointTextEncoder(model_dict)
        if model_config is not None:
            setattr(self.cond_stage_model, "model_config", model_config)
        self.tokenizer = ObjectDict(tokenizer_dict)
        self.patcher = ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)

    def clone(self):
        n = CLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        return n

    def add_patches(self, *arg, **kwargs):
        return self.patcher.add_patches(*arg, **kwargs)
