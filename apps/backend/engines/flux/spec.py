"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flux engine spec + runtime assembly (components + text pipelines + optional streaming core).
Defines the Flux engine runtime containers (denoiser/CLIP/T5/VAE + streaming policy) and assembles a runnable runtime from selected models,
with strict validation (no implicit fallbacks) and optional streamed core execution.

Symbols (top-level; keep in sync; no ghosts):
- `FluxTextPipelines` (dataclass): Holds the text processing engines used by Flux (optional CLIP classic + required T5).
- `FluxEngineRuntime` (dataclass): Fully assembled runtime components for Flux (CLIP, VAE, denoiser patcher, text pipelines, distilled CFG flag).
- `FluxEngineSpec` (dataclass): Spec/config holder for a Flux runtime build (repo/model selection + streaming policy/config).
- `_k_predictor` (function): Builds the FlowMatchEuler predictor for the selected Flux variant (Schnell vs dev).
- `_maybe_enable_streaming_core` (function): Wraps a core transformer with streaming support based on policy/config and runtime flags.
- `_is_clip_encoder` (function): Type guard for identifying CLIP text encoder models in a mixed component set.
- `_is_t5_encoder` (function): Type guard for identifying T5 text encoder models in a mixed component set.
- `assemble_flux_runtime` (function): Assembles a validated `FluxEngineRuntime` from selected components, applying device/memory policies
  and streaming options (contains nested helpers for controller setup and trace planning).
"""

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
from apps.backend.patchers.denoiser import DenoiserPatcher
from apps.backend.patchers.vae import VAE
from apps.backend.runtime.model_registry.specs import ModelFamily
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
    denoiser: DenoiserPatcher
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

    streamed: StreamedFluxCore | None = None
    base_core: FluxTransformer2DModel | None = None
    if isinstance(transformer, StreamedFluxCore):
        streamed = transformer
        base_core = transformer.base_core
    elif isinstance(transformer, FluxTransformer2DModel):
        base_core = transformer
    else:
        return transformer

    options = dict(engine_options or {})
    streaming_config = StreamingConfig.from_options(options)

    from apps.backend.runtime.memory import memory_management

    core_device = memory_management.get_torch_device()
    free_mb: int | None = None
    try:
        free_bytes = memory_management.get_free_memory(core_device)
        free_mb = int(free_bytes // (1024 * 1024))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Flux streaming: failed to probe free memory (%s)", exc)

    should_stream = bool(streaming_config.enabled)
    if not should_stream and streaming_config.auto_enable_threshold_mb > 0:
        if free_mb is not None:
            should_stream = streaming_config.should_enable(free_mb)

    if not should_stream:
        if free_mb is None:
            logger.debug("Flux streaming disabled (enabled=%s)", streaming_config.enabled)
        else:
            logger.debug("Flux streaming disabled (enabled=%s, free_vram_mb=%d)", streaming_config.enabled, free_mb)
        if streamed is not None:
            try:
                streamed.controller.compute_device = core_device
                streamed.move_all_to_compute()
                streamed.controller.reset()
                logger.info("Flux streaming disabled; reverted StreamedFluxCore -> FluxTransformer2DModel")
                return streamed.base_core
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Flux streaming: failed to disable streamed core; keeping StreamedFluxCore active: %s",
                    exc,
                    exc_info=True,
                )
        return transformer

    # Streaming is desired; if already wrapped, keep the existing wrapper.
    if streamed is not None:
        return streamed

    try:
        plan = trace_execution_plan(base_core, blocks_per_segment=streaming_config.blocks_per_segment)
        storage_device = memory_management.core_offload_device()

        controller = CoreController(
            storage_device=storage_device,
            compute_device=core_device,
            policy=StreamingPolicy(streaming_config.policy),
            window_size=streaming_config.window_size,
        )

        streamed = StreamedFluxCore(base_core, plan, controller)

        # Preserve loader metadata expected by KModel / patchers.
        for attr in (
            "storage_dtype",
            "computation_dtype",
            "load_device",
            "initial_device",
            "offload_device",
            "architecture",
        ):
            if hasattr(base_core, attr):
                setattr(streamed, attr, getattr(base_core, attr))
        if hasattr(base_core, "codex_config"):
            streamed.codex_config = base_core.codex_config

        logger.info(
            "Flux streaming enabled (policy=%s, blocks_per_segment=%d, window_size=%d, free_vram_mb=%d)",
            streaming_config.policy,
            streaming_config.blocks_per_segment,
            streaming_config.window_size,
            free_mb or 0,
        )
        return streamed
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Failed to enable Flux streaming; falling back to non-streaming core: %s", exc, exc_info=True
        )
        return transformer


def _is_clip_encoder(model: object) -> bool:
    """Detect if a text encoder is CLIP (has text_model attribute) vs T5."""
    if model is None:
        return False
    # Check for CLIP-specific structure
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        if hasattr(transformer, 'text_model'):
            return True
    if hasattr(model, 'text_model'):
        return True
    # Check class name as fallback
    cls_name = type(model).__name__
    if 'CLIP' in cls_name or 'clip' in cls_name.lower():
        return True
    return False


def _is_t5_encoder(model: object) -> bool:
    """Detect if a text encoder is T5 (has encoder.block structure)."""
    if model is None:
        return False
    # Check for T5-specific structure
    if hasattr(model, 'transformer'):
        transformer = model.transformer
        if hasattr(transformer, 'encoder') and hasattr(transformer.encoder, 'block'):
            return True
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        return True
    # Check class name as fallback
    cls_name = type(model).__name__
    if 'T5' in cls_name or 't5' in cls_name.lower():
        return True
    return False


def assemble_flux_runtime(
    *,
    spec: FluxEngineSpec,
    estimated_config,
    codex_components: Mapping[str, object],
    engine_options: Mapping[str, Any] | None = None,
) -> FluxEngineRuntime:
    logger.debug("Assembling %s engine", spec.name)

    # Detect encoder types dynamically instead of assuming by slot position
    # This handles GGUF models that may have encoders in swapped positions
    te1 = codex_components.get("text_encoder")
    te2 = codex_components.get("text_encoder_2")
    tok1 = codex_components.get("tokenizer")
    tok2 = codex_components.get("tokenizer_2")
    
    if spec.uses_clip_branch:
        # Flux needs both CLIP and T5 - detect which is which by model structure
        clip_encoder, clip_tokenizer = None, None
        t5_encoder, t5_tokenizer = None, None
        
        # Also detect tokenizer types - CLIPTokenizer uses 'eos_token', T5 uses 'unk_token'
        def _is_clip_tokenizer(tok):
            if tok is None:
                return False
            cls_name = type(tok).__name__
            return 'CLIP' in cls_name or 'clip' in cls_name.lower()
        
        def _is_t5_tokenizer(tok):
            if tok is None:
                return False
            cls_name = type(tok).__name__
            return 'T5' in cls_name or 't5' in cls_name.lower()
        
        # Match tokenizers by type, not by slot
        clip_tok_candidate = tok1 if _is_clip_tokenizer(tok1) else (tok2 if _is_clip_tokenizer(tok2) else tok1)
        t5_tok_candidate = tok2 if _is_t5_tokenizer(tok2) else (tok1 if _is_t5_tokenizer(tok1) else tok2)
        
        # Check te1
        if _is_clip_encoder(te1):
            clip_encoder = te1
        elif _is_t5_encoder(te1):
            t5_encoder = te1
        
        # Check te2
        if _is_clip_encoder(te2):
            clip_encoder = te2
        elif _is_t5_encoder(te2):
            t5_encoder = te2
        
        # Assign tokenizers by type, not by slot position
        clip_tokenizer = clip_tok_candidate
        t5_tokenizer = t5_tok_candidate
        
        # Fallback to position-based if detection fails
        if clip_encoder is None and t5_encoder is None:
            logger.warning("Could not detect encoder types; falling back to position-based assignment")
            clip_encoder, clip_tokenizer = te1, tok1
            t5_encoder, t5_tokenizer = te2, tok2
        elif clip_encoder is None:
            # T5 detected but no CLIP - check if there's another encoder
            clip_encoder = te1 if t5_encoder is not te1 else te2
        elif t5_encoder is None:
            # CLIP detected but no T5 - check if there's another encoder
            t5_encoder = te1 if clip_encoder is not te1 else te2
        
        logger.debug(
            "Encoder detection: CLIP=%s (tok=%s) T5=%s (tok=%s)", 
            type(clip_encoder).__name__ if clip_encoder else None,
            type(clip_tokenizer).__name__ if clip_tokenizer else None,
            type(t5_encoder).__name__ if t5_encoder else None,
            type(t5_tokenizer).__name__ if t5_tokenizer else None,
        )
        
        model_dict = {"clip_l": clip_encoder, "t5xxl": t5_encoder}
        tokenizer_dict = {"clip_l": clip_tokenizer, "t5xxl": t5_tokenizer}
    else:
        # Chroma: only T5, no CLIP
        model_dict = {"t5xxl": te1}
        tokenizer_dict = {"t5xxl": tok1}

    clip = CLIP(model_dict=model_dict, tokenizer_dict=tokenizer_dict, model_config=estimated_config)
    vae_family = ModelFamily.FLUX if spec.name == "flux" else ModelFamily.CHROMA
    vae = VAE(model=codex_components["vae"], family=vae_family)

    repo = getattr(estimated_config, "huggingface_repo", "" ) or ""
    schnell = spec.is_schnell(repo)
    k_predictor = _k_predictor(repo, schnell)
    if not schnell:
        logger.debug("Distilled CFG scale enabled for %s", spec.name)
    use_distilled_cfg = not schnell

    transformer = codex_components["transformer"]
    transformer = _maybe_enable_streaming_core(transformer, spec=spec, engine_options=engine_options)

    denoiser = DenoiserPatcher.from_model(
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
    
    # Get t5_min_length from family spec
    from apps.backend.runtime.model_registry import FAMILY_RUNTIME_SPECS
    flux_family = ModelFamily.CHROMA if spec.name == "chroma" else ModelFamily.FLUX
    family_spec = FAMILY_RUNTIME_SPECS.get(flux_family)
    t5_min_len = family_spec.t5_min_length if family_spec and family_spec.t5_min_length else 256
    
    t5_engine = T5TextProcessingEngine(
        text_encoder=t5_encoder,
        tokenizer=t5_tokenizer,
        emphasis_name=emphasis_name,
        min_length=t5_min_len,
    )

    logger.debug("Flux runtime assembled (clip branch: %s, distilled cfg: %s)", spec.uses_clip_branch, use_distilled_cfg)

    return FluxEngineRuntime(
        clip=clip,
        vae=vae,
        denoiser=denoiser,
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
