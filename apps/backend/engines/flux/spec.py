from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from apps.backend.runtime.flux.model import FluxTransformer2DModel
from apps.backend.runtime.flux.streaming import (
    StreamingConfig,
    StreamedFluxCore,
    StreamingPolicy,
    CoreController,
    trace_execution_plan,
)
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


def _maybe_enable_streaming_core(
    transformer: object,
    *,
    spec: FluxEngineSpec,
    engine_options: Mapping[str, Any] | None,
) -> object:
    """Optionally wrap a Flux core with StreamedFluxCore based on engine options and VRAM state.

    Streaming is currently only applied to the Flux engine (not Chroma) and only when
    the transformer is a FluxTransformer2DModel instance.
    """
    if spec.name != "flux":
        return transformer
    if not isinstance(transformer, FluxTransformer2DModel):
        return transformer

    options = dict(engine_options or {})
    streaming_config = StreamingConfig.from_options(options)

    from apps.backend.runtime.memory import memory_management

    core_device = memory_management.get_torch_device()
    try:
        free_bytes = memory_management.get_free_memory(core_device)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Flux streaming: failed to probe free memory (%s); disabling streaming", exc)
        return transformer

    try:
        free_mb = int(free_bytes // (1024 * 1024))
    except Exception:
        free_mb = 0

    if not streaming_config.should_enable(free_mb):
        logger.debug("Flux streaming disabled (enabled=%s, free_vram_mb=%d)", streaming_config.enabled, free_mb)
        return transformer

    try:
        plan = trace_execution_plan(transformer, blocks_per_segment=streaming_config.blocks_per_segment)
        storage_device = memory_management.core_offload_device()

        controller = CoreController(
            storage_device=storage_device,
            compute_device=core_device,
            policy=StreamingPolicy(streaming_config.policy),
            window_size=streaming_config.window_size,
        )

        streamed = StreamedFluxCore(transformer, plan, controller)

        # Preserve loader metadata expected by KModel / patchers.
        for attr in (
            "storage_dtype",
            "computation_dtype",
            "load_device",
            "initial_device",
            "offload_device",
            "architecture",
        ):
            if hasattr(transformer, attr):
                setattr(streamed, attr, getattr(transformer, attr))
        if hasattr(transformer, "codex_config"):
            streamed.codex_config = transformer.codex_config

        logger.info(
            "Flux streaming enabled (policy=%s, blocks_per_segment=%d, window_size=%d, free_vram_mb=%d)",
            streaming_config.policy,
            streaming_config.blocks_per_segment,
            streaming_config.window_size,
            free_mb,
        )
        return streamed
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to enable Flux streaming; falling back to non-streaming core: %s", exc, exc_info=True
        )
        return transformer


def assemble_flux_runtime(
    *,
    spec: FluxEngineSpec,
    estimated_config,
    codex_components: Mapping[str, object],
    engine_options: Mapping[str, Any] | None = None,
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

    transformer = codex_components["transformer"]
    transformer = _maybe_enable_streaming_core(transformer, spec=spec, engine_options=engine_options)

    unet = UnetPatcher.from_model(
        model=transformer,
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
