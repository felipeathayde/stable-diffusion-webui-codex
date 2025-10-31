from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Mapping, Optional

from apps.backend.patchers.clip import CLIP
from apps.backend.patchers.unet import UnetPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.modules.k_prediction import FlowMatchEulerPrediction
from apps.backend.runtime.text_processing.classic_engine import ClassicTextProcessingEngine
from apps.backend.runtime.text_processing.t5_engine import T5TextProcessingEngine
from apps.backend.runtime.memory import memory_management
from apps.backend.infra.config.args import dynamic_args

logger = logging.getLogger("backend.engines.flux.spec")


@dataclass(frozen=True)
class FluxTextPipelines:
    clip_text: Optional[ClassicTextProcessingEngine]
    t5_text: T5TextProcessingEngine


@dataclass(frozen=True)
class FluxEngineRuntime:
    clip: CLIP
    vae: VAE
    unet: UnetPatcher
    text: FluxTextPipelines
    use_distilled_cfg: bool

    def set_clip_skip(self, clip_skip: int) -> None:
        if self.text.clip_text is not None:
            if clip_skip < 1:
                raise ValueError("clip_skip must be >= 1 for Flux CLIP branch")
            self.text.clip_text.clip_skip = clip_skip


@dataclass(frozen=True)
class FluxEngineSpec:
    name: str
    uses_clip_branch: bool
    distilled_cfg_scale_default: float = 3.5
    schnell_threshold: Callable[[str], bool] | None = None

    def is_schnell(self, repo: str) -> bool:
        if self.schnell_threshold is None:
            return False
        return self.schnell_threshold(repo)


def _k_predictor(repo: str, is_schnell: bool) -> FlowMatchEulerPrediction:
    if is_schnell:
        logger.debug("Using FlowMatch predictor for schnell repo=%s", repo)
        return FlowMatchEulerPrediction(mu=1.0)
    logger.debug("Using FlowMatch predictor with seq_len scheduling for repo=%s", repo)
    return FlowMatchEulerPrediction(
        seq_len=4096,
        base_seq_len=256,
        max_seq_len=4096,
        base_shift=0.5,
        max_shift=1.15,
    )


def assemble_flux_runtime(
    *,
    spec: FluxEngineSpec,
    estimated_config,
    codex_components: Mapping[str, object],
) -> FluxEngineRuntime:
    logger.debug("Assembling %s engine", spec.name)

    clip_keys = {"clip_l": "text_encoder", "t5xxl": "text_encoder_2"} if spec.uses_clip_branch else {"t5xxl": "text_encoder"}
    tokenizer_keys = {"clip_l": "tokenizer", "t5xxl": "tokenizer_2"} if spec.uses_clip_branch else {"t5xxl": "tokenizer"}

    model_dict = {alias: codex_components[key] for alias, key in clip_keys.items()}
    tokenizer_dict = {alias: codex_components[key] for alias, key in tokenizer_keys.items()}
    clip = CLIP(model_dict=model_dict, tokenizer_dict=tokenizer_dict, model_config=estimated_config)
    vae = VAE(model=codex_components["vae"])

    repo = getattr(estimated_config, "huggingface_repo", "" ) or ""
    schnell = spec.is_schnell(repo)
    k_predictor = _k_predictor(repo, schnell)
    if not schnell:
        logger.debug("Distilled CFG scale enabled for %s", spec.name)
    use_distilled_cfg = not schnell

    unet = UnetPatcher.from_model(
        model=codex_components["transformer"],
        diffusers_scheduler=None,
        k_predictor=k_predictor,
        config=estimated_config,
    )

    embedding_dir = dynamic_args["embedding_dir"]
    emphasis_name = dynamic_args["emphasis_name"]

    if spec.uses_clip_branch:
        clip_l = clip.cond_stage_model.clip_l
        tokenizer_l = clip.tokenizer.clip_l
        clip_engine = ClassicTextProcessingEngine(
            text_encoder=clip_l,
            tokenizer=tokenizer_l,
            embedding_dir=embedding_dir,
            embedding_key="clip_l",
            embedding_expected_shape=768,
            emphasis_name=emphasis_name,
            text_projection=False,
            minimal_clip_skip=1,
            clip_skip=1,
            return_pooled=True,
            final_layer_norm=True,
        )
    else:
        clip_engine = None

    t5_attr = "t5xxl"
    t5_encoder = getattr(clip.cond_stage_model, t5_attr)
    t5_tokenizer = getattr(clip.tokenizer, t5_attr)
    t5_engine = T5TextProcessingEngine(
        text_encoder=t5_encoder,
        tokenizer=t5_tokenizer,
        emphasis_name=emphasis_name,
        min_length=1,
    )

    logger.debug("Flux runtime assembled (clip branch: %s, distilled cfg: %s)", spec.uses_clip_branch, use_distilled_cfg)

    return FluxEngineRuntime(
        clip=clip,
        vae=vae,
        unet=unet,
        text=FluxTextPipelines(clip_text=clip_engine, t5_text=t5_engine),
        use_distilled_cfg=use_distilled_cfg,
    )


FLUX_SPEC = FluxEngineSpec(
    name="flux",
    uses_clip_branch=True,
    distilled_cfg_scale_default=3.5,
    schnell_threshold=lambda repo: "schnell" in repo.lower(),
)

CHROMA_SPEC = FluxEngineSpec(
    name="chroma",
    uses_clip_branch=False,
    distilled_cfg_scale_default=1.0,
)

__all__ = [
    "FluxTextPipelines",
    "FluxEngineRuntime",
    "FluxEngineSpec",
    "assemble_flux_runtime",
    "FLUX_SPEC",
    "CHROMA_SPEC",
]
