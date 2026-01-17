"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Central model loader for diffusion engines (checkpoint/diffusers parsing, component assembly, and runtime-friendly overrides).
This module resolves TE/VAE overrides (including `tenc_path` shorthand), normalizes state_dict layouts, chooses compatible dtypes, and returns a `DiffusionModelBundle`
ready for orchestrator/engine execution (with memory-management hooks).

Symbols (top-level; keep in sync; no ghosts):
- `ParsedCheckpoint` (dataclass): Parsed checkpoint bundle (primary path + optional additional modules + extracted configs/metadata).
- `DiffusionModelBundle` (dataclass): Loaded model components and configs (UNet/VAE/text encoders + signature + quant/layout info).
- `_supported_inference_dtypes` (function): Returns supported inference dtypes for a given model family.
- `_prediction_type_value` (function): Converts a `PredictionKind` into the string value expected by configs/pipelines.
- `_load_state_dict` (function): Loads a state dict from disk (handles supported formats) for downstream parsing.
- `_read_json` (function): Reads a required JSON metadata file with explicit errors (used by vendored-HF signature builders).
- `_zimage_signature_from_vendored_hf` (function): Builds a Z-Image `ModelSignature` from vendored HF metadata (no state-dict detector).
- `_flux_signature_from_vendored_hf` (function): Builds a Flux `ModelSignature` from vendored HF metadata (no state-dict detector).
- `_parse_checkpoint` (function): Parses one checkpoint (plus optional addons) into `ParsedCheckpoint` for bundle assembly.
- `_build_diffusion_bundle` (function): Assembles a `DiffusionModelBundle` from a parsed checkpoint and loader options.
- `_load_component_config` (function): Loads a component config dict from a diffusers component directory.
- `_resolve_vae_class` (function): Picks the VAE class/loader path based on model signature and layout (`diffusers` vs legacy layouts).
- `_maybe_convert_sdxl_vae_state_dict` (function): Applies SDXL-specific VAE key conversions when the checkpoint layout requires it.
- `_detect_vae_layout` (function): Detects VAE state dict layout (used to choose conversion/loading strategy).
- `_SAFETENSORS_DTYPE_LABEL_TO_TORCH` (constant): Maps safetensors dtype labels to torch dtypes for default precision selection.
- `_maybe_default_dtype_from_weights` (function): Picks a default dtype from safetensors header dtype hints when no dtype was forced.
- `_load_huggingface_component` (function): Loads a diffusers component/pipeline from a local HF-style repo directory.
- `_apply_prediction_type` (function): Applies prediction-type overrides to loaded components/configs when specified.
- `codex_loader` (function): Primary loader entrypoint; coordinates checkpoint parsing, TE override resolution (incl. `tenc_path` shorthand),
  VAE layout handling, dtype selection, and memory-management integration to produce a `DiffusionModelBundle`.
- `_SimpleEstimated` (class): Minimal estimate container used for config detection/compat when only partial metadata is available.
- `_detect_engine_from_config` (function): Detects engine identifier from a diffusers config dict.
- `load_engine_from_diffusers` (function): Loads a `DiffusionModelBundle` directly from a diffusers repo directory.
- `resolve_diffusion_bundle` (function): Resolves and loads a diffusion bundle from either checkpoint paths or diffusers repos based on inputs.
"""

import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from diffusers import AutoencoderKL, DiffusionPipeline
from transformers import modeling_utils

from apps.backend.huggingface.assets import ensure_repo_minimal_files
from apps.backend.infra.config.args import args
from apps.backend.runtime import trace as _trace
from apps.backend.runtime.common.nn.clip import IntegratedCLIP
from apps.backend.runtime.common.nn.t5 import IntegratedT5
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.model_parser import parse_state_dict
from apps.backend.runtime.model_parser.quantization import detect_state_dict_dtype
from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.model_registry.errors import ModelRegistryError
from apps.backend.runtime.model_registry.loader import detect_from_state_dict as registry_detect
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    CodexCoreSignature,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationHint,
    LatentFormat,
    QuantizationKind,
    TextEncoderSignature,
    VAESignature,
)
from apps.backend.runtime.models import api as model_api
from apps.backend.runtime.models.key_normalization import _normalize_unet_state_dict, _strip_transformer_prefixes
from apps.backend.runtime.models.text_encoder_overrides import (
    _canonical_override_family,
    TextEncoderOverrideConfig,
    TextEncoderOverrideError,
    resolve_text_encoder_override_paths,
)
from apps.backend.runtime.models.state_dict import load_state_dict, transformers_convert
from apps.backend.runtime.ops import using_codex_operations
from apps.backend.runtime.checkpoint_io import load_torch_file, read_arbitrary_config
from apps.backend.runtime.state_dict_tools import beautiful_print_gguf_state_dict_statics
from apps.backend.runtime.wan22.vae import AutoencoderKLWan
from apps.backend.runtime.models.registry import _detect_safetensors_primary_dtype

LOGGER = logging.getLogger(__name__)
CLIP_LOG = logging.getLogger(__name__ + ".clip")
_LOG = logging.getLogger(__name__)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_INFERENCE_DTYPES: Dict[ModelFamily, tuple[torch.dtype, ...]] = {
    ModelFamily.FLUX: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.FLUX_KONTEXT: (torch.bfloat16, torch.float16, torch.float32),
    ModelFamily.CHROMA: (torch.bfloat16, torch.float16, torch.float32),
}
DEFAULT_SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)

_CORE_ARCH_LABELS: Dict[CodexCoreArchitecture, str] = {
    CodexCoreArchitecture.UNET: "UNet",
    CodexCoreArchitecture.DIT: "DiT",
    CodexCoreArchitecture.TRANSFORMER: "Transformer",
    CodexCoreArchitecture.FLOW_TRANSFORMER: "FlowTransformer",
}

PREDICTION_TYPE_MAP = {
    PredictionKind.EPSILON: "epsilon",
    PredictionKind.V_PREDICTION: "v_prediction",
    PredictionKind.EDM: "edm",
    PredictionKind.FLOW: "flow",
}


@dataclass
class ParsedCheckpoint:
    signature: ModelSignature
    config: CodexEstimatedConfig


@dataclass(frozen=True, slots=True)
class DiffusionModelBundle:
    """Fully materialised diffusion checkpoint ready for engine binding."""

    model_ref: str
    family: ModelFamily
    estimated_config: Any
    components: Dict[str, Any]
    signature: Optional[ModelSignature] = None
    source: str = "state_dict"
    metadata: Dict[str, Any] = field(default_factory=dict)


ENGINE_KEY_TO_FAMILY: Dict[str, ModelFamily] = {
    "sdxl": ModelFamily.SDXL,
    "sdxl_refiner": ModelFamily.SDXL_REFINER,
    "flux1": ModelFamily.FLUX,
    "flux1_kontext": ModelFamily.FLUX_KONTEXT,
    "sd35": ModelFamily.SD35,
    "sd3": ModelFamily.SD3,
    "flux1_chroma": ModelFamily.CHROMA,
    "sd20": ModelFamily.SD20,
    "sd15": ModelFamily.SD15,
    "wan22_14b": ModelFamily.WAN22,
    "wan22_5b": ModelFamily.WAN22,
}

FAMILY_TO_ENGINE_KEY: Dict[ModelFamily, str] = {
    ModelFamily.SDXL_REFINER: "sdxl_refiner",
    ModelFamily.SDXL: "sdxl",
    ModelFamily.FLUX: "flux1",
    ModelFamily.FLUX_KONTEXT: "flux1_kontext",
    ModelFamily.SD35: "sd35",
    ModelFamily.SD3: "sd35",
    ModelFamily.CHROMA: "flux1_chroma",
    ModelFamily.SD20: "sd20",
    ModelFamily.SD15: "sd15",
    ModelFamily.WAN22: "wan22_14b",
}


def _supported_inference_dtypes(family: ModelFamily) -> tuple[torch.dtype, ...]:
    return SUPPORTED_INFERENCE_DTYPES.get(family, DEFAULT_SUPPORTED_DTYPES)


def _prediction_type_value(prediction: PredictionKind) -> str:
    return PREDICTION_TYPE_MAP.get(prediction, "epsilon")


def _load_state_dict(path: str) -> Mapping[str, Any]:
    _trace.event("load_torch_file_start", path=str(path))
    # Resolve the initial load device explicitly (no 'auto' fallback)
    initial_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
    sd = load_torch_file(path, device=initial_device)
    try:
        tensor_count = len(sd.keys())  # type: ignore[attr-defined]
    except Exception:
        tensor_count = -1
    _trace.event("load_torch_file_done", path=str(path), type=type(sd).__name__, tensors=tensor_count)
    return sd

def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required metadata file missing: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to parse JSON metadata file: {path}") from exc
    if not isinstance(data, dict):
        raise TypeError(f"Expected JSON object at {path}, got {type(data).__name__}")
    return data


def _zimage_signature_from_vendored_hf(*, model_path: str) -> ModelSignature:
    """Build a Z-Image `ModelSignature` from vendored HF metadata (no state-dict detection)."""

    vendor_root = _BACKEND_ROOT / "huggingface" / "Alibaba-TongYi" / "Z-Image-Turbo"
    transformer_cfg = _read_json(vendor_root / "transformer" / "config.json")
    text_encoder_cfg = _read_json(vendor_root / "text_encoder" / "config.json")

    patch_size = transformer_cfg.get("all_patch_size")
    if isinstance(patch_size, list) and patch_size:
        patch_size = patch_size[0]
    patch_size = int(patch_size) if isinstance(patch_size, (int, float, str)) and str(patch_size).strip() else 2
    patch_area = int(max(1, patch_size) * max(1, patch_size))

    latent_channels_raw = transformer_cfg.get("in_channels", 16)
    latent_channels = int(latent_channels_raw) if isinstance(latent_channels_raw, (int, float, str)) else 16

    hidden_dim_raw = transformer_cfg.get("dim", 3840)
    hidden_dim = int(hidden_dim_raw) if isinstance(hidden_dim_raw, (int, float, str)) else 3840

    context_dim_raw = transformer_cfg.get("cap_feat_dim")
    if context_dim_raw is None:
        context_dim_raw = text_encoder_cfg.get("hidden_size", 2560)
    context_dim = int(context_dim_raw) if isinstance(context_dim_raw, (int, float, str)) else 2560

    num_layers_raw = transformer_cfg.get("n_layers", 30)
    num_layers = int(num_layers_raw) if isinstance(num_layers_raw, (int, float, str)) else 30

    num_refiner_layers_raw = transformer_cfg.get("n_refiner_layers", 2)
    num_refiner_layers = int(num_refiner_layers_raw) if isinstance(num_refiner_layers_raw, (int, float, str)) else 2

    num_heads_raw = transformer_cfg.get("n_heads", 30)
    num_heads = int(num_heads_raw) if isinstance(num_heads_raw, (int, float, str)) else 30

    suffix = Path(model_path).suffix.lower()
    quantization = QuantizationHint()
    gguf_core_only = False
    if suffix == ".gguf":
        quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="file_extension")
        gguf_core_only = True

    channels_out = latent_channels
    channels_in = latent_channels * patch_area

    extras = {
        "hidden_dim": hidden_dim,
        "context_dim": context_dim,
        "num_layers": num_layers,
        "num_refiner_layers": num_refiner_layers,
        "num_heads": num_heads,
        "latent_channels": latent_channels,
        "guidance_embeds": False,
        "gguf_core_only": gguf_core_only,
        "signature_source": "vendored_hf",
    }

    text_encoders = [
        TextEncoderSignature(
            name="qwen3_4b",
            key_prefix="text_encoder.",
            expected_dim=context_dim,
            tokenizer_hint="Qwen/Qwen3-4B",
        )
    ]

    # Z-Image supports embedded VAE/text-encoder in non-GGUF exports, but the engine
    # decides core-only vs full based on parsed components. Keep VAE signature
    # optional here so metadata drives the pipeline without constraining assets.
    vae: VAESignature | None = None if gguf_core_only else VAESignature(key_prefix="vae.", latent_channels=latent_channels)

    return ModelSignature(
        family=ModelFamily.ZIMAGE,
        repo_hint="Alibaba-TongYi/Z-Image-Turbo",
        prediction=PredictionKind.FLOW,
        latent_format=LatentFormat.ZIMAGE,
        quantization=quantization,
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.DIT,
            channels_in=channels_in,
            channels_out=channels_out,
            context_dim=context_dim,
            temporal=False,
            depth=num_layers,
            key_prefixes=["layers."],
        ),
        text_encoders=text_encoders,
        vae=vae,
        extras=extras,
    )


def _flux_signature_from_vendored_hf(
    *,
    model_path: str,
    expected_family: ModelFamily,
    has_guidance: bool | None = None,
) -> ModelSignature:
    """Build a Flux `ModelSignature` from vendored HF metadata (no state-dict detection)."""

    vendor_root = _BACKEND_ROOT / "huggingface" / "black-forest-labs"
    if expected_family is ModelFamily.FLUX_KONTEXT:
        repo_hint = "black-forest-labs/FLUX.1-Kontext-dev"
        transformer_cfg = _read_json(vendor_root / "FLUX.1-Kontext-dev" / "transformer" / "config.json")
    else:
        # Default Flux core signature (dev/schnell share the same architecture, but guidance differs).
        if has_guidance is False:
            repo_hint = "black-forest-labs/FLUX.1-schnell"
            transformer_cfg = _read_json(vendor_root / "FLUX.1-schnell" / "transformer" / "config.json")
        else:
            repo_hint = "black-forest-labs/FLUX.1-dev"
            transformer_cfg = _read_json(vendor_root / "FLUX.1-dev" / "transformer" / "config.json")

    latent_channels_raw = transformer_cfg.get("in_channels", 64)
    latent_channels = int(latent_channels_raw) if isinstance(latent_channels_raw, (int, float, str)) else 64

    context_dim_raw = transformer_cfg.get("joint_attention_dim", 4096)
    context_dim = int(context_dim_raw) if isinstance(context_dim_raw, (int, float, str)) else 4096

    double_layers_raw = transformer_cfg.get("num_layers", 19)
    double_layers = int(double_layers_raw) if isinstance(double_layers_raw, (int, float, str)) else 19

    single_layers_raw = transformer_cfg.get("num_single_layers", 38)
    single_layers = int(single_layers_raw) if isinstance(single_layers_raw, (int, float, str)) else 38

    guidance_embed = bool(transformer_cfg.get("guidance_embeds", False))

    suffix = Path(model_path).suffix.lower()
    quantization = QuantizationHint()
    gguf_core_only = False
    if suffix == ".gguf":
        quantization = QuantizationHint(kind=QuantizationKind.GGUF, detail="file_extension")
        gguf_core_only = True

    extras = {
        "flow_double_layers": double_layers,
        "flow_single_layers": single_layers,
        "guidance_embed": guidance_embed,
        "gguf_core_only": gguf_core_only,
        "signature_source": "vendored_hf",
    }

    text_encoders = [
        TextEncoderSignature(
            name="clip_l",
            key_prefix="text_encoders.clip_l.",
            expected_dim=768,
            tokenizer_hint=f"{repo_hint}/tokenizer",
        ),
        TextEncoderSignature(
            name="t5xxl",
            key_prefix="text_encoders.t5xxl.",
            expected_dim=4096,
            tokenizer_hint=f"{repo_hint}/tokenizer_2",
        ),
    ]

    # Flux core-only checkpoints require external VAE in the engine loader path.
    vae: VAESignature | None = None if gguf_core_only else VAESignature(key_prefix="vae.", latent_channels=16)

    return ModelSignature(
        family=expected_family,
        repo_hint=repo_hint,
        prediction=PredictionKind.FLOW,
        latent_format=LatentFormat.FLOW16,
        quantization=quantization,
        core=CodexCoreSignature(
            architecture=CodexCoreArchitecture.FLOW_TRANSFORMER,
            channels_in=latent_channels,
            channels_out=latent_channels,
            context_dim=context_dim,
            temporal=False,
            depth=double_layers + single_layers,
            key_prefixes=["transformer.", "model.diffusion_model."],
        ),
        text_encoders=text_encoders,
        vae=vae,
        extras=extras,
    )


def _parse_checkpoint(
    primary_path: str,
    additional_paths: list[str] | None,
    *,
    expected_family: ModelFamily | None = None,
) -> ParsedCheckpoint:
    base_state = _load_state_dict(primary_path)
    if expected_family is ModelFamily.ZIMAGE:
        signature = _zimage_signature_from_vendored_hf(model_path=primary_path)
    elif expected_family in {ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT}:
        guidance_key = "guidance_in.in_layer.weight"
        has_guidance = any(
            k in base_state for k in (guidance_key, f"transformer.{guidance_key}", f"model.diffusion_model.{guidance_key}")
        )
        signature = _flux_signature_from_vendored_hf(
            model_path=primary_path,
            expected_family=expected_family,
            has_guidance=has_guidance,
        )
    else:
        signature = registry_detect(base_state)
    config = parse_state_dict(base_state, signature)

    if additional_paths:
        replacements: Dict[str, Mapping[str, Any]] = {}
        for extra in additional_paths:
            extra_state = _load_state_dict(extra)
            if expected_family is ModelFamily.ZIMAGE:
                extra_signature = _zimage_signature_from_vendored_hf(model_path=extra)
            elif expected_family in {ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT}:
                guidance_key = "guidance_in.in_layer.weight"
                has_guidance = any(
                    k in extra_state for k in (guidance_key, f"transformer.{guidance_key}", f"model.diffusion_model.{guidance_key}")
                )
                extra_signature = _flux_signature_from_vendored_hf(
                    model_path=extra,
                    expected_family=expected_family,
                    has_guidance=has_guidance,
                )
            else:
                extra_signature = registry_detect(extra_state)
            extra_config = parse_state_dict(extra_state, extra_signature)
            for name, component in extra_config.components.items():
                replacements[name] = component.state_dict
        if replacements:
            config = config.replace_components(replacements)

    return ParsedCheckpoint(signature=signature, config=config)


def _build_diffusion_bundle(
    *,
    model_ref: str,
    family: ModelFamily,
    estimated_config: Any,
    components: Dict[str, Any],
    signature: Optional[ModelSignature] = None,
    source: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> DiffusionModelBundle:
    return DiffusionModelBundle(
        model_ref=model_ref,
        family=family,
        estimated_config=estimated_config,
        components=dict(components),
        signature=signature,
        source=source,
        metadata=dict(metadata or {}),
    )


def _load_component_config(component_path: str) -> Dict[str, Any]:
    config_file = os.path.join(component_path, "config.json")
    if os.path.isfile(config_file):
        with open(config_file, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def _resolve_vae_class(signature: ModelSignature | None, *, layout: str = "diffusers"):
    """Select the appropriate VAE class.

    - WAN22 family always uses AutoencoderKLWan.
    - All other families use diffusers AutoencoderKL regardless of VAE layout.
    """

    family = getattr(signature, "family", None)
    if family is ModelFamily.WAN22:
        return AutoencoderKLWan
    return AutoencoderKL


def _maybe_convert_sdxl_vae_state_dict(
    state_dict: Mapping[str, Any],
    signature: Optional[ModelSignature],
) -> Mapping[str, Any]:
    """Normalise SDXL VAE weights that use the original LDM naming.

    Some SDXL checkpoints (including some exported variants) store VAE weights
    under ``first_stage_model.*`` with keys like::

        encoder.down.{i}.block.{j}.*
        encoder.down.{i}.downsample.*
        decoder.up.{k}.block.{j}.*
        decoder.up.{k}.upsample.*
        decoder.mid.block_{1,2}.*
        decoder.mid.attn_1.{q,k,v,proj_out,norm}.*

    Diffusers ``AutoencoderKL`` expects the canonical SDXL layout instead::

        encoder.down_blocks.{i}.resnets.{j}.*
        encoder.down_blocks.{i}.downsamplers.0.*
        decoder.up_blocks.{i}.resnets.{j}.*
        decoder.up_blocks.{i}.upsamplers.0.*
        mid_block.resnets.{0,1}.*
        mid_block.attentions.0.*

    This helper rewrites keys for SDXL families only and leaves other layouts
    untouched. It also flattens 1x1 conv attention weights into linear weights
    where necessary (q/k/v/proj_out) so they match diffusers expectations.
    """

    family = getattr(signature, "family", None) if signature is not None else None
    # Support SDXL, FLUX and ZIMAGE families (all share the same VAE key layout)
    if family not in (ModelFamily.SDXL, ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT, ModelFamily.ZIMAGE):
        return state_dict

    keys = list(state_dict.keys())
    # Already in diffusers-style layout; nothing to do.
    if any(isinstance(k, str) and k.startswith("encoder.down_blocks.") for k in keys):
        return state_dict
    # Only touch classic SDXL/LDM style VAEs.
    if not any(isinstance(k, str) and k.startswith("encoder.down.") for k in keys):
        return state_dict

    # Materialise the VAE tensors before mutating them. Lazy SafeTensors views
    # (LazySafetensorsDict/FilterPrefixView) reopen the file for every
    # __getitem__, and calling .contiguous() on those per-key views has been
    # observed to crash torch_cpu.dll on Windows. A single streaming
    # materialisation detaches the tensors from the underlying file before we
    # reshape mid-attention weights.
    materialize = getattr(state_dict, "materialize", None)
    if callable(materialize):
        try:
            state_dict = materialize()
        except TypeError:
            state_dict = materialize(prefix="", new_prefix="")
    elif not isinstance(state_dict, dict):
        state_dict = dict(state_dict)

    def _flatten_conv_to_linear(tensor: torch.Tensor) -> torch.Tensor:
        """Convert 1x1 conv weights to linear weights when safe.

        Some SDXL checkpoints store mid attention projections (q/k/v/proj_out)
        as Conv2d weights with shape [C_out, C_in, 1, 1], while diffusers
        AutoencoderKL expects Linear weights [C_out, C_in].
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
        if tensor.ndim == 4 and tensor.shape[-2:] == (1, 1):
            return tensor[:, :, 0, 0].contiguous()
        return tensor

    converted: Dict[str, Any] = {}

    for raw_key, value in state_dict.items():
        key = str(raw_key)
        new_key = key
        tensor = value

        # Encoder down blocks: encoder.down.{i}.block.{j}.*
        if key.startswith("encoder.down."):
            parts = key.split(".")
            # encoder.down.{i}.block.{j}.rest...
            if len(parts) >= 6 and parts[2].isdigit() and parts[4].isdigit() and parts[3] == "block":
                i = int(parts[2])
                j = int(parts[4])
                rest = ".".join(parts[5:])
                new_key = f"encoder.down_blocks.{i}.resnets.{j}.{rest}"
            # encoder.down.{i}.downsample.*
            elif len(parts) >= 4 and parts[2].isdigit() and parts[3] == "downsample":
                i = int(parts[2])
                rest = ".".join(parts[4:])
                new_key = f"encoder.down_blocks.{i}.downsamplers.0.{rest}"

        # Decoder up blocks: decoder.up.{k}.block.{j}.*
        elif key.startswith("decoder.up."):
            parts = key.split(".")
            # decoder.up.{k}.block.{j}.rest...
            if len(parts) >= 6 and parts[2].isdigit() and parts[4].isdigit() and parts[3] == "block":
                k = int(parts[2])
                j = int(parts[4])
                # SDXL VAE indexes up_blocks in reverse order vs. LDM-style up.{k}
                i = 3 - k
                rest = ".".join(parts[5:])
                new_key = f"decoder.up_blocks.{i}.resnets.{j}.{rest}"
            # decoder.up.{k}.upsample.*
            elif len(parts) >= 4 and parts[2].isdigit() and parts[3] == "upsample":
                k = int(parts[2])
                i = 3 - k
                rest = ".".join(parts[4:])
                new_key = f"decoder.up_blocks.{i}.upsamplers.0.{rest}"

        # Mid blocks (resnets): encoder.mid.block_1/2.*, decoder.mid.block_1/2.*
        elif key.startswith("encoder.mid.block_") or key.startswith("decoder.mid.block_"):
            parts = key.split(".")
            # {encoder,decoder}.mid.block_{1,2}.rest...
            if len(parts) >= 4 and parts[2].startswith("block_"):
                try:
                    block_index = int(parts[2].split("_", 1)[1]) - 1
                except (IndexError, ValueError):
                    block_index = 0
                rest = ".".join(parts[3:])
                prefix = "encoder" if parts[0] == "encoder" else "decoder"
                new_key = f"{prefix}.mid_block.resnets.{block_index}.{rest}"

        # Mid attention: encoder/decoder.mid.attn_1.{q,k,v,proj_out,norm,query,key,value,proj_attn}.*
        elif key.startswith("encoder.mid.attn_1.") or key.startswith("decoder.mid.attn_1."):
            is_encoder = key.startswith("encoder.")
            base = "encoder.mid.attn_1." if is_encoder else "decoder.mid.attn_1."
            suffix = key[len(base) :]
            prefix = "encoder" if is_encoder else "decoder"

            try:
                head, rest = suffix.split(".", 1)
            except ValueError:
                head, rest = suffix, ""

            table = {
                "q": "to_q",
                "k": "to_k",
                "v": "to_v",
                "proj_out": "to_out.0",
                "norm": "group_norm",
                # older / alternative naming
                "query": "to_q",
                "key": "to_k",
                "value": "to_v",
                "proj_attn": "to_out.0",
            }

            mapped = table.get(head)
            if mapped is not None and rest:
                new_key = f"{prefix}.mid_block.attentions.0.{mapped}.{rest}"
                if mapped in ("to_q", "to_k", "to_v", "to_out.0") and rest.startswith("weight"):
                    tensor = _flatten_conv_to_linear(tensor)

        # Conv shortcuts: nin_shortcut (LDM) -> conv_shortcut (diffusers)
        if "nin_shortcut." in key:
            parts = key.split(".")
            # Handle encoder.down.{i}.block.0.nin_shortcut.{weight,bias}
            if key.startswith("encoder.down.") and len(parts) >= 7 and parts[2].isdigit() and parts[3] == "block" and parts[4] == "0":
                i = int(parts[2])
                rest = ".".join(parts[6:])
                new_key = f"encoder.down_blocks.{i}.resnets.0.conv_shortcut.{rest}"
            # Handle decoder.up.{k}.block.0.nin_shortcut.{weight,bias} (map k -> up_blocks.{3-k})
            elif key.startswith("decoder.up.") and len(parts) >= 7 and parts[2].isdigit() and parts[3] == "block" and parts[4] == "0":
                k = int(parts[2])
                i = 3 - k
                rest = ".".join(parts[6:])
                new_key = f"decoder.up_blocks.{i}.resnets.0.conv_shortcut.{rest}"

        # Output norm: norm_out (LDM) -> conv_norm_out (diffusers)
        if key.startswith("encoder.norm_out."):
            rest = key[len("encoder.norm_out.") :]
            new_key = f"encoder.conv_norm_out.{rest}"
        elif key.startswith("decoder.norm_out."):
            rest = key[len("decoder.norm_out.") :]
            new_key = f"decoder.conv_norm_out.{rest}"

        converted[new_key] = tensor

    return converted


def _detect_vae_layout(sd: Mapping[str, Any]) -> str:
    """Return 'ldm' if keys look like LDM VAE (encoder.down.*), else 'diffusers'."""

    if not hasattr(sd, "keys"):
        return "diffusers"
    keys = list(sd.keys())
    for k in keys:
        if isinstance(k, str) and (k.startswith("encoder.down.") or k.startswith("decoder.conv_in")):
            return "ldm"
    return "diffusers"


_SAFETENSORS_DTYPE_LABEL_TO_TORCH: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
}


def _maybe_default_dtype_from_weights(
    *,
    role: DeviceRole,
    current: torch.dtype,
    weights_path: str | None,
    supported: tuple[torch.dtype, ...] = (torch.float16, torch.bfloat16, torch.float32),
) -> torch.dtype:
    if not weights_path:
        return current
    try:
        forced = memory_management.manager.config.component_policy(role).forced_dtype
    except Exception:
        forced = None
    if forced:
        return current

    label = _detect_safetensors_primary_dtype(Path(weights_path))
    if not label:
        return current
    detected = _SAFETENSORS_DTYPE_LABEL_TO_TORCH.get(label)
    if detected is None:
        return current
    if detected not in supported:
        return current
    return detected


def _load_huggingface_component(
    parsed: ParsedCheckpoint,
    component_name: str,
    lib_name: str,
    cls_name: str,
    repo_path: str,
    state_dict: Mapping[str, Any] | None,
    *,
    weights_path: str | None = None,
):
    family = parsed.signature.family
    config = parsed.config
    component_path = os.path.join(repo_path, component_name)

    if component_name in {"feature_extractor", "safety_checker"}:
        return None

    if lib_name in {"transformers", "diffusers"} and component_name == "scheduler":
        cls = getattr(importlib.import_module(lib_name), cls_name)
        _trace.event("component_from_pretrained", name=component_name, lib=lib_name, cls=cls_name)
        return cls.from_pretrained(os.path.join(repo_path, component_name))

    if lib_name in {"transformers", "diffusers"} and component_name.startswith("tokenizer"):
        cls = getattr(importlib.import_module(lib_name), cls_name)
        _trace.event("component_from_pretrained", name=component_name, lib=lib_name, cls=cls_name)
        tokenizer = cls.from_pretrained(os.path.join(repo_path, component_name))
        if hasattr(tokenizer, "_eventual_warn_about_too_long_sequence"):
            tokenizer._eventual_warn_about_too_long_sequence = lambda *_, **__: None
        return tokenizer

    if cls_name == "AutoencoderKL":
        if state_dict is None:
            # For SDXL (and refiner) a VAE is mandatory; fail fast instead of
            # attempting to proceed without it.
            if family in (ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
                raise RuntimeError(
                    "No VAE detected in checkpoint for SDXL. Provide a VAE override (vae_path) "
                    "or use a checkpoint with an embedded SDXL VAE."
                )
            # Flux GGUF core-only checkpoints carry only the rectified-flow backbone;
            # they must be composed with an explicit external VAE (sha-selected).
            signature = getattr(parsed, "signature", None)
            quant = getattr(signature, "quantization", None)
            extras = getattr(signature, "extras", {}) or {}
            is_flux_core_gguf = (
                isinstance(signature, ModelSignature)
                and signature.family in (ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT)
                and getattr(quant, "kind", None) is QuantizationKind.GGUF
                and bool(extras.get("gguf_core_only"))
            )
            if is_flux_core_gguf:
                raise RuntimeError(
                    "Flux GGUF core-only checkpoint is missing a VAE. "
                    "Provide one explicitly (request extras.vae_sha), so the API passes a valid vae_path to the loader."
                )
            return None

        # Unwrap common packing shapes (e.g., {'state_dict': {...}})
        if isinstance(state_dict, Mapping) and len(state_dict) == 1 and "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        if not isinstance(state_dict, Mapping):
            raise RuntimeError(
                f"VAE state_dict must be a mapping; got {type(state_dict).__name__}. "
                "Checkpoint may be malformed or require manual VAE extraction."
            )

        LOGGER.debug(
            "VAE state_dict type=%s len=%d sample_keys=%s",
            type(state_dict).__name__,
            len(state_dict.keys()) if hasattr(state_dict, "keys") else -1,
            list(state_dict.keys())[:5] if hasattr(state_dict, "keys") else None,
        )

        # Normalise SDXL VAE layouts that use the original LDM-style naming before
        # we inspect layout or strip prefixes. This keeps UNet/CLIP detection
        # independent from how the VAE block was stored in the checkpoint.
        state_dict = _maybe_convert_sdxl_vae_state_dict(state_dict, getattr(parsed, "signature", None))

        vae_layout = _detect_vae_layout(state_dict)
        LOGGER.debug("VAE layout detected=%s", vae_layout)

        def _strip_prefixes(sd: Mapping[str, Any]) -> Mapping[str, Any]:
            """Return a lazy view with VAE prefixes stripped.

            Uses RemapKeysView so tensors load on demand; avoids materialising
            the entire VAE just to rename keys (important for large XL VAEs).
            """
            from apps.backend.runtime.state_dict_views import RemapKeysView

            prefixes = (
                "first_stage_model.",
                "vae.",
                "model.",
            )

            mapping: Dict[str, str] = {}
            for raw_key in sd.keys():  # only touch key names, not tensors
                key = str(raw_key)
                new_key = key
                changed = True
                while changed:
                    changed = False
                    for prefix in prefixes:
                        if new_key.startswith(prefix):
                            new_key = new_key[len(prefix) :]
                            changed = True
                            break
                mapping[new_key] = key

            if not mapping:
                return sd
            return RemapKeysView(sd, mapping)

        state_dict = _strip_prefixes(state_dict)
        # Convert LDM-style VAE keys to diffusers-style for SDXL/FLUX
        state_dict = _maybe_convert_sdxl_vae_state_dict(state_dict, getattr(parsed, "signature", None))

        vae_cls = _resolve_vae_class(getattr(parsed, "signature", None), layout=vae_layout)
        try:
            config_json = vae_cls.load_config(component_path)
        except Exception:
            config_json = _load_component_config(component_path)
        vae_device = memory_management.manager.get_device(DeviceRole.VAE)
        vae_dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE)
        vae_dtype = _maybe_default_dtype_from_weights(role=DeviceRole.VAE, current=vae_dtype, weights_path=weights_path)
        _trace.event("vae_construct", device=str(vae_device), dtype=str(vae_dtype), cls=vae_cls.__name__)

        with using_codex_operations(device=vae_device, dtype=vae_dtype, manual_cast_enabled=True):
            model = vae_cls.from_config(config_json)

        _trace.event("load_state_dict", module="vae", tensors=len(state_dict))
        from .state_dict import safe_load_state_dict as _safe_load
        expected_total = len(model.state_dict())
        family_name = getattr(getattr(parsed, "signature", None), "family", "unknown")

        try:
            missing, unexpected = _safe_load(model, state_dict, log_name="VAE")
        except Exception as exc:  # no silent fallbacks for VAE
            raise RuntimeError(
                f"Failed to load VAE weights for family {family_name}: {exc!s}"
            ) from exc

        if missing:
            sample = missing[:10]
            LOGGER.error(
                "VAE load failed: missing %d/%d keys for family=%s sample=%s",
                len(missing),
                expected_total,
                family_name,
                sample,
            )
            raise RuntimeError(
                "VAE state_dict missing %d/%d keys for family %s. "
                "Checkpoint likely lacks a compatible VAE; supply an SDXL VAE ou separate VAE weights. "
                "Sample missing keys: %s"
                % (len(missing), expected_total, family_name, sample)
            )

        if unexpected:
            LOGGER.warning(
                "VAE load: unexpected %d keys (sample=%s)", len(unexpected), unexpected[:10]
            )
        return model

    if cls_name in {"CLIPTextModel", "CLIPTextModelWithProjection"}:
        if state_dict is None:
            return None
        
        # Detect T5 state dict keys - if found, load as T5 instead of CLIP
        # This handles GGUF models that may have T5 bundled as text_encoder
        _T5_KEY_PATTERNS = ("encoder.block.", "encoder.final_layer_norm.", "shared.weight")
        _is_actually_t5 = any(
            any(pattern in k for pattern in _T5_KEY_PATTERNS)
            for k in list(state_dict.keys())[:100]
        )
        if _is_actually_t5:
            # Load as T5 using T5 handler but keep in same slot
            # The spec.py will detect the correct type later
            _LOG.info(
                "Detected T5 state dict in %s (expected CLIP); loading as T5 model in same slot",
                component_name
            )
            # Use T5EncoderModel handler but keep this component_name (not redirecting)
            return _load_huggingface_component(
                parsed, component_name, lib_name, "T5EncoderModel", repo_path, state_dict, weights_path=weights_path
            )
        # Build native Codex CLIP instead of HF; normalise state dict beforehand
        te_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        te_dtype = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
        te_dtype = _maybe_default_dtype_from_weights(role=DeviceRole.TEXT_ENCODER, current=te_dtype, weights_path=weights_path)
        to_args = dict(device=te_device, dtype=te_dtype)
        add_proj = component_name in {"text_encoder_2", "text_encoder_3"}
        from .state_dict import safe_load_state_dict
        from apps.backend.runtime.common.nn.clip_text_cx import CodexCLIPTextConfig, CodexCLIPTextModel
        _CLIP_PREFIXES = (
            "conditioner.embedders.0.transformer.",
            "conditioner.embedders.0.model.",
            "conditioner.embedders.0.",
            "conditioner.embedders.1.transformer.",
            "conditioner.embedders.1.model.",
            "conditioner.embedders.1.",
            "cond_stage_model.model.",
            "cond_stage_model.",
            "text_encoders.clip_l.",
            "text_encoders.clip_g.",
            "clip_l.",
            "clip_g.",
            "clip_h.",
            "model.text_model.",
            "model.",
        )

        def _strip_known_prefixes(sd: Mapping[str, Any]) -> Dict[str, Any]:
            stripped: Dict[str, Any] = {}
            for raw_key, value in sd.items():
                key = str(raw_key)
                changed = True
                while changed:
                    changed = False
                    for prefix in _CLIP_PREFIXES:
                        if key.startswith(prefix):
                            key = key[len(prefix):]
                            changed = True
                            break
                stripped[key] = value
            return stripped

        def _ensure_position_ids_long(work: Dict[str, Any]) -> None:
            key = "transformer.text_model.embeddings.position_ids"
            value = work.get(key)
            if isinstance(value, torch.Tensor) and value.dtype != torch.long:
                work[key] = value.round().to(torch.long)

        def _normalize_text_projection(work: Dict[str, Any], *, keep_projection: bool, transpose: bool) -> None:
            def _assign(tensor: Any) -> None:
                if not keep_projection:
                    return
                if isinstance(tensor, torch.Tensor) and transpose:
                    tensor = tensor.transpose(0, 1).contiguous()
                # Provide both aliases expected across Codex/HF CLIP implementations.
                work["text_projection.weight"] = tensor
                work["transformer.text_projection.weight"] = tensor

            for key in (
                "transformer.text_projection.weight",
                "transformer.text_projection",
                "text_projection.weight",
                "text_projection",
            ):
                if key in work:
                    value = work.pop(key)
                    _assign(value)
            if not keep_projection:
                work.pop("text_projection.weight", None)
                work.pop("transformer.text_projection.weight", None)

        _ESSENTIAL_KEYS = (
            "transformer.text_model.embeddings.token_embedding.weight",
            "transformer.text_model.embeddings.position_embedding.weight",
            "transformer.text_model.encoder.layers.0.self_attn.q_proj.weight",
            "transformer.text_model.final_layer_norm.weight",
        )

        def _has_essentials(work: Mapping[str, Any]) -> bool:
            return all(key in work for key in _ESSENTIAL_KEYS)

        # CLIP-ViT-Large-patch14 default config (used by Flux CLIP-L and SD1.x/2.x)
        _CLIP_L_DEFAULT_CONFIG = {
            "hidden_size": 768,
            "intermediate_size": 3072,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "hidden_act": "quick_gelu",
            "max_position_embeddings": 77,
            "layer_norm_eps": 1e-05,
            "vocab_size": 49408,
            "projection_dim": 768,
        }

        # OpenCLIP-ViT-bigG config (used by SDXL text_encoder_2)
        _CLIP_G_DEFAULT_CONFIG = {
            "hidden_size": 1280,
            "intermediate_size": 5120,
            "num_hidden_layers": 32,
            "num_attention_heads": 20,
            "hidden_act": "gelu",
            "max_position_embeddings": 77,
            "layer_norm_eps": 1e-05,
            "vocab_size": 49408,
            "projection_dim": 1280,
        }
        
        # Try to infer the correct config from state_dict shapes
        def _infer_clip_config_from_state(sd: Mapping[str, Any]) -> dict | None:
            """Infer CLIP config variant from state_dict tensor shapes."""
            # Check embedding weight shape to determine hidden_size
            embedding_keys = [
                "transformer.text_model.embeddings.token_embedding.weight",
                "text_model.embeddings.token_embedding.weight",
            ]
            for key in embedding_keys:
                if key in sd:
                    t = sd[key]
                    if hasattr(t, "shape"):
                        hidden_size = t.shape[1] if len(t.shape) > 1 else None
                        if hidden_size == 1280:
                            _LOG.info("Detected OpenCLIP-G variant (hidden_size=1280)")
                            return _CLIP_G_DEFAULT_CONFIG
                        elif hidden_size == 768:
                            _LOG.info("Detected CLIP-L variant (hidden_size=768)")
                            return _CLIP_L_DEFAULT_CONFIG
            return None
        
        # Determine if we should use default CLIP config
        use_default_config = False
        inferred_config = None
        
        # Case 1: CLIP in T5 slot (text_encoder_2 is normally T5 for Flux)
        # But for SDXL, text_encoder_2 is OpenCLIP-G, so we need to infer from state_dict
        if component_name == "text_encoder_2":
            inferred_config = _infer_clip_config_from_state(state_dict)
            if inferred_config:
                _LOG.info("CLIP loading in T5 slot (%s); using inferred config", component_name)
                use_default_config = True
            else:
                _LOG.info("CLIP loading in T5 slot (%s); using CLIP-L default config", component_name)
                inferred_config = _CLIP_L_DEFAULT_CONFIG
                use_default_config = True
        
        if use_default_config:
            config_json = inferred_config or _CLIP_L_DEFAULT_CONFIG
        else:
            try:
                config_json = read_arbitrary_config(component_path)
                # Validate it's actually a CLIP config
                if "hidden_size" not in config_json:
                    _LOG.warning(
                        "Config at %s missing 'hidden_size' (got %s); using CLIP-L defaults",
                        component_path, list(config_json.keys())[:5]
                    )
                    config_json = _CLIP_L_DEFAULT_CONFIG
            except FileNotFoundError:
                _LOG.info(
                    "No config.json found at %s; using default CLIP-L configuration",
                    component_path
                )
                config_json = _CLIP_L_DEFAULT_CONFIG
        
        cfg = CodexCLIPTextConfig.from_dict(config_json)

        def _normalize_clip_state(
            component: str,
            sd: Mapping[str, Any],
            *,
            num_layers: int,
            keep_projection: bool,
            transpose_projection: bool,
        ) -> Dict[str, Any]:
            work = _strip_known_prefixes(sd)
            work = dict(work)

            # Add transformer. prefix to keys that start with text_model.
            # This handles CLIP files that use text_model.* instead of transformer.text_model.*
            keys_to_rename = [(k, f"transformer.{k}") for k in list(work.keys()) 
                              if k.startswith("text_model.") and not k.startswith("transformer.")]
            for old_key, new_key in keys_to_rename:
                work[new_key] = work.pop(old_key)

            transformers_convert(work, "transformer.", "transformer.text_model.", num_layers)
            transformers_convert(work, "", "transformer.text_model.", num_layers)

            _ensure_position_ids_long(work)
            _normalize_text_projection(work, keep_projection=keep_projection, transpose=transpose_projection)
            # Keep logit_scale for all CLIPs; inject default if missing.
            has_logit = any(
                k in work for k in ("logit_scale", "transformer.logit_scale", "transformer.text_model.logit_scale")
            )
            if not has_logit:
                # Match IntegratedCLIP default (ln 100).
                default_logit = torch.tensor(4.605170185988092)
                for key in ("logit_scale", "transformer.logit_scale", "transformer.text_model.logit_scale"):
                    work[key] = default_logit
            # Ensure text_projection alias for models with projection; if projection absent (CLIP-L), drop both.
            if keep_projection:
                if "transformer.text_projection.weight" in work and "text_projection.weight" not in work:
                    work["text_projection.weight"] = work["transformer.text_projection.weight"]
                if "text_projection.weight" in work and "transformer.text_projection.weight" not in work:
                    work["transformer.text_projection.weight"] = work["text_projection.weight"]
            else:
                work.pop("text_projection.weight", None)
                work.pop("transformer.text_projection.weight", None)

            # Check for essential keys with or without transformer prefix
            _ESSENTIAL_KEYS_ALT = (
                "text_model.embeddings.token_embedding.weight",
                "text_model.embeddings.position_embedding.weight",
            )
            has_essentials = _has_essentials(work) or all(
                any(k.endswith(key.split(".")[-1]) for k in work.keys()) for key in _ESSENTIAL_KEYS
            )
            if not has_essentials:
                sample_keys = list(sorted(work.keys()))[:10]
                raise RuntimeError(
                    "CLIP state dict normalisation failed for %s; missing essential tensors. Sample keys: %s"
                    % (component, sample_keys)
                )

            return work

        state_dict = _normalize_clip_state(
            component_name,
            state_dict,
            num_layers=cfg.num_hidden_layers,
            keep_projection=add_proj,
            transpose_projection=component_name in {"text_encoder_2", "text_encoder_3"},
        )

        with using_codex_operations(**to_args, manual_cast_enabled=True):
            model = IntegratedCLIP(CodexCLIPTextModel, cfg, add_text_projection=add_proj).to(**to_args)

        missing, unexpected = safe_load_state_dict(model, state_dict, log_name=cls_name)
        if missing:
            CLIP_LOG.warning("CLIP missing (%s): %s", component_name, missing[:10])
        if unexpected:
            CLIP_LOG.debug("CLIP unexpected (%s): %s", component_name, unexpected[:10])
        return model

    if cls_name == "T5EncoderModel":
        if state_dict is None:
            return None
        
        # Detect CLIP state dict keys - if found, load as CLIP instead of T5
        # This handles checkpoints that may have CLIP bundled as text_encoder_2
        _CLIP_KEY_PATTERNS = ("text_model.embeddings.", "text_model.encoder.layers.", "logit_scale")
        _is_actually_clip = any(
            any(pattern in k for pattern in _CLIP_KEY_PATTERNS)
            for k in list(state_dict.keys())[:100]
        )
        if _is_actually_clip:
            # Load as CLIP using CLIP handler but keep in same slot
            # The spec.py will detect the correct type later
            _LOG.info(
                "Detected CLIP state dict in %s (expected T5); loading as CLIP model in same slot",
                component_name
            )
            # Use CLIPTextModel handler but keep this component_name (not redirecting)
            return _load_huggingface_component(
                parsed, component_name, lib_name, "CLIPTextModel", repo_path, state_dict, weights_path=weights_path
            )
        # T5-XXL config (google/t5-v1_1-xxl) - used by Flux
        _T5_XXL_DEFAULT_CONFIG = {
            "d_ff": 10240,
            "d_kv": 64,
            "d_model": 4096,
            "dense_act_fn": "gelu_new",
            "is_gated_act": True,
            "model_type": "t5",
            "num_heads": 64,
            "num_layers": 24,
            "vocab_size": 32128,
        }
        
        # Determine if we should use default T5 config:
        # 1. T5 loaded in CLIP slot via redirect (component_path has CLIP config)
        # 2. No config.json exists
        # 3. Config exists but is wrong type (e.g., CLIP config with num_hidden_layers)
        use_default_config = False
        
        # Case 1: T5 in CLIP slot (text_encoder is normally CLIP for Flux)
        if component_name == "text_encoder":
            _LOG.info("T5 loading in CLIP slot (%s); using T5-XXL default config", component_name)
            use_default_config = True
        
        if use_default_config:
            t5_config = _T5_XXL_DEFAULT_CONFIG
        else:
            try:
                t5_config = read_arbitrary_config(component_path)
                # Validate it's actually a T5 config
                if "num_layers" not in t5_config:
                    _LOG.warning(
                        "Config at %s missing 'num_layers' (got %s); using T5-XXL defaults",
                        component_path, list(t5_config.keys())[:5]
                    )
                    t5_config = _T5_XXL_DEFAULT_CONFIG
            except FileNotFoundError:
                _LOG.info(
                    "No config.json found at %s; using default T5-XXL configuration",
                    component_path
                )
                t5_config = _T5_XXL_DEFAULT_CONFIG
        te_device = memory_management.manager.get_device(DeviceRole.TEXT_ENCODER)
        storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER)
        storage_dtype = _maybe_default_dtype_from_weights(
            role=DeviceRole.TEXT_ENCODER,
            current=storage_dtype,
            weights_path=weights_path,
        )
        state_dict_dtype = detect_state_dict_dtype(state_dict)
        if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
            LOGGER.info("Using Detected T5 Data Type: %s", state_dict_dtype)
            storage_dtype = state_dict_dtype
            if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                LOGGER.info("Using pre-quant state dict!")
                if state_dict_dtype == "gguf":
                    beautiful_print_gguf_state_dict_statics(state_dict)
        else:
            LOGGER.info("Using Default T5 Data Type: %s", storage_dtype)

        if storage_dtype in ["nf4", "fp4", "gguf"]:
            with modeling_utils.no_init_weights():
                with using_codex_operations(
                    device=te_device,
                    dtype=memory_management.manager.dtype_for_role(DeviceRole.TEXT_ENCODER),
                    manual_cast_enabled=False,
                    bnb_dtype=storage_dtype,
                ):
                    model = IntegratedT5(t5_config)
        else:
            with modeling_utils.no_init_weights():
                with using_codex_operations(device=te_device, dtype=storage_dtype, manual_cast_enabled=True):
                    model = IntegratedT5(t5_config)

        # Normalize T5 state dict keys: add transformer. prefix if missing
        # T5 files often have keys like encoder.block.* but model expects transformer.encoder.block.*
        if hasattr(state_dict, 'keys'):
            keys_to_check = list(state_dict.keys())
            needs_prefix = any(k.startswith("encoder.") or k == "shared.weight" for k in keys_to_check[:50])
            if needs_prefix:
                # Create a new dict with normalized keys
                normalized_sd = {}
                for k, v in (state_dict.items() if hasattr(state_dict, 'items') else [(k, state_dict[k]) for k in keys_to_check]):
                    if k.startswith("encoder.") or k == "shared.weight" or k.startswith("embed_tokens"):
                        new_key = f"transformer.{k}"
                        normalized_sd[new_key] = v
                    else:
                        normalized_sd[k] = v
                state_dict = normalized_sd

        load_state_dict(
            model,
            state_dict,
            log_name=cls_name,
            ignore_errors=["transformer.encoder.embed_tokens.weight", "logit_scale"],
        )
        return model

    if cls_name in {"UNet2DConditionModel", "FluxTransformer2DModel", "SD3Transformer2DModel", "ChromaTransformer2DModel", "ZImageTransformer2DModel"}:
        if state_dict is None:
            return None
        # Choose configuration source per family/model class
        if cls_name == "UNet2DConditionModel" and family in (ModelFamily.SD15, ModelFamily.SD20, ModelFamily.SDXL, ModelFamily.SDXL_REFINER):
            # Start from parser-provided core_config (LDM-style) and drop unknown keys for legacy UNet
            raw_cfg = dict(config.core_config or {})
            allowed = {
                "in_channels",
                "model_channels",
                "out_channels",
                "num_res_blocks",
                "dropout",
                "channel_mult",
                "conv_resample",
                "dims",
                "num_classes",
                "use_checkpoint",
                "num_heads",
                "num_head_channels",
                "use_scale_shift_norm",
                "resblock_updown",
                "use_spatial_transformer",
                "transformer_depth",
                "context_dim",
                "disable_self_attentions",
                "num_attention_blocks",
                "disable_middle_self_attn",
                "use_linear_in_transformer",
                "adm_in_channels",
                "transformer_depth_middle",
                "transformer_depth_output",
            }
            config_json = {k: v for k, v in raw_cfg.items() if k in allowed}
        else:
            config_json = _load_component_config(component_path)
        core_arch = config.signature.core.architecture
        core_label = _CORE_ARCH_LABELS.get(core_arch, "Core")
        architecture_value = core_arch.value
        module_name = component_name or ("unet" if core_arch is CodexCoreArchitecture.UNET else "transformer")

        if cls_name == "UNet2DConditionModel":
            # For SD15/SD20/SDXL families use Codex legacy UNet with LDM-style config
            from apps.backend.runtime.common.nn.unet.model import UNet2DConditionModel as _CodexUNet

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return _CodexUNet.from_config(cfg)

        elif cls_name == "FluxTransformer2DModel":
            from apps.backend.runtime.flux.flux import FluxTransformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return FluxTransformer2DModel(**dict(cfg))

        elif cls_name == "ChromaTransformer2DModel":
            from apps.backend.runtime.chroma.chroma import ChromaTransformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return ChromaTransformer2DModel(**dict(cfg))

        elif cls_name == "ZImageTransformer2DModel":
            from apps.backend.runtime.zimage.model import ZImageTransformer2DModel
            # Filter out HuggingFace metadata keys (starting with _)

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                filtered = {k: v for k, v in cfg.items() if not k.startswith("_")}
                return ZImageTransformer2DModel(**filtered)

        else:
            from apps.backend.runtime.sd.mmditx import SD3Transformer2DModel

            def model_ctor(cfg: Mapping[str, Any]) -> Any:
                return SD3Transformer2DModel(**dict(cfg))

        supported_dtypes = _supported_inference_dtypes(family)
        quant_kind = config.quantization.kind
        storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE, supported=supported_dtypes)
        storage_dtype = _maybe_default_dtype_from_weights(
            role=DeviceRole.CORE,
            current=storage_dtype,
            weights_path=weights_path,
            supported=tuple(supported_dtypes),
        )
        if quant_kind == QuantizationKind.NF4:
            storage_dtype = "nf4"
        elif quant_kind == QuantizationKind.FP4:
            storage_dtype = "fp4"
        elif quant_kind == QuantizationKind.GGUF:
            storage_dtype = "gguf"

        load_device = memory_management.manager.get_device(DeviceRole.CORE)
        offload_device = memory_management.manager.get_offload_device(DeviceRole.CORE)

        mem_config = memory_management.manager.config

        if storage_dtype in ["nf4", "fp4", "gguf"]:
            # For quantized models, compute computation_dtype based on the actual load device
            # This will be the GPU device where computations happen
            computation_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE, supported=supported_dtypes)
            
            initial_device = memory_management.manager.get_offload_device(DeviceRole.CORE)
            # Smart offload: load transformers to CPU to prevent OOM
            # The engine will move it to GPU on demand via streaming or memory management
            from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
            _SMART_OFFLOAD_TRANSFORMERS = {
                "FluxTransformer2DModel",
                "ZImageTransformer2DModel",
                "ChromaTransformer2DModel",
                "SD3Transformer2DModel",
            }
            if cls_name in _SMART_OFFLOAD_TRANSFORMERS and smart_offload_enabled():
                initial_device = torch.device("cpu")
                LOGGER.info("[loader] Smart offload: loading %s to CPU (will stream to GPU)", cls_name)
            
            # For GGUF on CPU, use bfloat16 for storage to avoid unnecessary upcasting
            # The model will automatically cast to appropriate dtype during forward pass on GPU
            construct_dtype = torch.bfloat16 if initial_device.type == "cpu" else computation_dtype
            
            with using_codex_operations(device=initial_device, dtype=construct_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                model = model_ctor(config_json)
        else:
            # Non-quantized models: compute computation_dtype for non-quantized path
            computation_dtype = memory_management.manager.dtype_for_role(DeviceRole.CORE, supported=supported_dtypes)
            if isinstance(storage_dtype, torch.dtype):
                computation_dtype = storage_dtype
            
            prefer_gpu = bool(getattr(mem_config, "gpu_prefer_construct", False))
            construct_device = load_device if prefer_gpu else memory_management.manager.get_offload_device(DeviceRole.CORE)
            initial_device = construct_device
            construct_dtype = storage_dtype
            if construct_device.type == "cpu" and construct_dtype in (torch.bfloat16, torch.float16):
                _trace.event(
                    "construct_cpu_cast_override",
                    dtype=str(construct_dtype),
                    to="torch.float32",
                    component=module_name,
                    architecture=architecture_value,
                )
                construct_dtype = torch.float32

            need_manual_cast = construct_dtype != computation_dtype
            to_args = dict(device=construct_device, dtype=construct_dtype)
            _trace.event(
                "core_construct",
                component=module_name,
                architecture=architecture_value,
                device=str(construct_device),
                storage=str(construct_dtype),
                compute=str(computation_dtype),
            )
            try:
                with using_codex_operations(**to_args, manual_cast_enabled=need_manual_cast):
                    model = model_ctor(config_json).to(**to_args)
            except memory_management.manager.oom_exception as exc:
                policy = getattr(mem_config.swap, "policy", None)
                if hasattr(policy, "value"):
                    policy_value = policy.value
                elif policy is not None:
                    policy_value = str(policy)
                else:
                    policy_value = "cpu"
                _trace.event("construct_oom", policy=policy, component=module_name, architecture=architecture_value)
                raise RuntimeError(
                    "Core construction OOM for component={comp} (architecture={arch}) on device={dev} with dtype={dtype}. "
                    "Automatic fallback/offload is disabled. Reduce model precision/size or free VRAM and retry. "
                    "(swap_policy={policy}, gpu_prefer_construct={prefer})"
                .format(
                    comp=module_name,
                    arch=architecture_value,
                    dev=str(construct_device),
                    dtype=str(construct_dtype),
                    policy=str(policy_value),
                    prefer=str(bool(getattr(mem_config, "gpu_prefer_construct", False))),
                )) from exc

        if cls_name == "UNet2DConditionModel":
            state_dict = _normalize_unet_state_dict(state_dict, config_json)
        elif cls_name in {"FluxTransformer2DModel", "ChromaTransformer2DModel", "SD3Transformer2DModel", "ZImageTransformer2DModel"}:
            # Strip common prefixes from transformer state dict keys (similar to UNet normalization)
            state_dict = _strip_transformer_prefixes(state_dict)

            # NOTE: GGUF key names now match Codex model names directly
            # No remapping needed - components.py uses .lin and .adaLN_modulation


        _trace.event("load_state_dict", module=module_name, architecture=architecture_value, tensors=len(state_dict))
        
        # GGUF models require PyTorch's load_state_dict to trigger _load_from_state_dict hooks
        # in CodexOperationsGGUF.Linear which handle the .weight/.bias mapping from GGUF format
        if storage_dtype == "gguf":
            LOGGER.debug("Using PyTorch load_state_dict for GGUF model")
            missing, unexpected = model.load_state_dict(dict(state_dict), strict=False)
            if missing:
                LOGGER.warning('%s Missing: %d keys: %s', core_label, len(missing), missing[:10])
            if unexpected:
                LOGGER.warning('%s Unexpected: %d keys: %s', core_label, len(unexpected), unexpected[:10])
        else:
            try:
                from .state_dict import safe_load_state_dict as _safe_load
                _safe_load(model, state_dict, log_name=core_label)
            except Exception:
                load_state_dict(model, state_dict, log_name=core_label)

        # Avoid assigning to model.config (read-only on diffusers models)
        model.storage_dtype = storage_dtype
        model.computation_dtype = computation_dtype
        model.load_device = load_device
        model.initial_device = initial_device
        model.offload_device = offload_device
        model.architecture = core_arch

        return model

    _LOG.debug("Skipping component %s (%s.%s)", component_name, lib_name, cls_name)
    return None


def _apply_prediction_type(codex_components: Dict[str, Any], parsed: ParsedCheckpoint, yaml_prediction: str | None) -> None:
    scheduler = codex_components.get("scheduler")
    if not scheduler or not hasattr(scheduler, "config"):
        return
    desired = _prediction_type_value(parsed.signature.prediction)
    current = getattr(scheduler.config, "prediction_type", None)
    if yaml_prediction:
        scheduler.config.prediction_type = yaml_prediction
        _LOG.info("prediction_type overridden by YAML: %s -> %s", current, yaml_prediction)
        return
    if current and current != desired:
        _LOG.warning(
            "Scheduler prediction_type=%s differs from signature=%s; keeping scheduler value.",
            current,
            desired,
        )
        setattr(scheduler.config, "codex_signature_prediction_type", desired)
        return
    scheduler.config.prediction_type = desired or current or "epsilon"


@torch.inference_mode()
def codex_loader(
    sd_path: str,
    additional_state_dicts=None,
    text_encoder_override: TextEncoderOverrideConfig | None = None,
    vae_path: str | None = None,
    tenc_path: str | list[str] | None = None,
    expected_family: ModelFamily | None = None,
) -> DiffusionModelBundle:
    try:
        parsed = _parse_checkpoint(
            sd_path,
            additional_state_dicts or [],
            expected_family=expected_family,
        )
    except ModelRegistryError as exc:
        raise ValueError("Failed to recognize model type!") from exc

    config = parsed.config
    signature = getattr(parsed, "signature", None)

    quant = getattr(signature, "quantization", None)
    extras = getattr(signature, "extras", {}) or {}
    is_flux_core_gguf = (
        isinstance(signature, ModelSignature)
        and signature.family in (ModelFamily.FLUX, ModelFamily.FLUX_KONTEXT)
        and getattr(quant, "kind", None) is QuantizationKind.GGUF
        and bool(extras.get("gguf_core_only"))
    )
    if is_flux_core_gguf:
        if not isinstance(vae_path, str) or not vae_path.strip():
            raise RuntimeError(
                "Flux GGUF core-only checkpoint requires an external VAE (sha-selected). "
                "Provide a VAE via request extras.vae_sha so the API can pass a valid vae_path."
            )
        vae_path = os.path.expanduser(vae_path.strip())
        if not os.path.isfile(vae_path):
            raise RuntimeError(f"Flux GGUF core-only VAE path not found: {vae_path}")

    te_override_cfg = text_encoder_override
    if te_override_cfg is None and tenc_path is not None and parsed.signature.family is not ModelFamily.ZIMAGE:
        # Shorthand: map `tenc_path` entries onto the signature-declared text encoders in order.
        if isinstance(tenc_path, str):
            paths = [tenc_path.strip()] if tenc_path.strip() else []
        elif isinstance(tenc_path, list):
            paths = []
            for entry in tenc_path:
                if not isinstance(entry, str):
                    raise TypeError("tenc_path must be a string or list[str] when provided.")
                item = entry.strip()
                if item:
                    paths.append(item)
        else:
            raise TypeError("tenc_path must be a string or list[str] when provided.")
        if not paths:
            raise RuntimeError("tenc_path was provided but empty after trimming.")

        aliases = tuple(te.name for te in parsed.signature.text_encoders)
        if not aliases:
            raise RuntimeError(
                "tenc_path override was provided, but this checkpoint declares no text encoders in the signature."
            )
        if len(paths) != len(aliases):
            raise RuntimeError(
                "tenc_path override expects exactly %d paths for this checkpoint (encoders=%s); got %d."
                % (len(aliases), ", ".join(aliases), len(paths))
            )

        te_override_cfg = TextEncoderOverrideConfig(
            family=_canonical_override_family(parsed.signature.family),
            root_label="tenc_path",
            components=aliases,
            explicit_paths={alias: path for alias, path in zip(aliases, paths, strict=True)},
        )

    try:
        te_override_paths = resolve_text_encoder_override_paths(
            signature=parsed.signature,
            estimated_config=config,
            override=te_override_cfg,
        )
    except TextEncoderOverrideError as exc:
        # Keep the surface error explicit and actionable; do not fall back silently.
        raise RuntimeError(str(exc)) from exc

    component_states = {name: comp.state_dict for name, comp in config.components.items()}

    repo_name = config.repo_id
    if not isinstance(repo_name, str) or not repo_name:
        raise ValueError("Codex model parser did not resolve a repository id")

    local_repo_path = os.path.join(str(_BACKEND_ROOT), "huggingface", repo_name)
    offline = bool(args.disable_online_tokenizer)
    include = ("config", "tokenizer", "scheduler")  # strictly minimal; no weights
    ensure_repo_minimal_files(repo_name, local_repo_path, offline=offline, include=include)

    pipeline_config = DiffusionPipeline.load_config(local_repo_path)
    codex_components: Dict[str, Any] = {}

    for component_name, component_info in pipeline_config.items():
        if not (isinstance(component_info, list) and len(component_info) == 2):
            continue
        lib_name, cls_name = component_info
        component_sd = component_states.get(component_name)

        if component_sd is None and is_flux_core_gguf and cls_name == "AutoencoderKL":
            component_sd = _load_state_dict(vae_path)

        override_path = te_override_paths.get(component_name)
        if override_path is not None:
            if not os.path.isfile(override_path):
                raise RuntimeError(
                    "Text encoder override path for component %s does not exist: %s"
                    % (component_name, override_path)
                )
            component_sd = _load_state_dict(override_path)

        component_obj = _load_huggingface_component(
            parsed,
            component_name,
            lib_name,
            cls_name,
            local_repo_path,
            component_sd,
            weights_path=sd_path if override_path is None and not (is_flux_core_gguf and cls_name == "AutoencoderKL") else (override_path or vae_path),
        )
        if component_sd is not None:
            component_states.pop(component_name, None)
        if component_obj is not None:
            codex_components[component_name] = component_obj
            # Smart offload: move text encoders to CPU immediately after loading
            # This frees VRAM so the transformer can fit later
            from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
            if smart_offload_enabled() and component_name in ("text_encoder", "text_encoder_2"):
                try:
                    component_obj.to("cpu")
                    memory_management.manager.soft_empty_cache(force=True)
                    LOGGER.info("[loader] Smart offload: moved %s to CPU after load", component_name)
                except Exception as e:
                    LOGGER.warning("[loader] Smart offload: failed to move %s to CPU: %s", component_name, e)

    yaml_prediction = None
    config_filename = os.path.splitext(sd_path)[0] + ".yaml"
    if os.path.isfile(config_filename):
        try:
            import yaml
            with open(config_filename, "r", encoding="utf-8") as stream:
                yaml_config = yaml.safe_load(stream)
            yaml_prediction = (
                yaml_config.get("model", {}).get("params", {}).get("parameterization", "")
                or yaml_config.get("model", {})
                .get("params", {})
                .get("denoiser_config", {})
                .get("params", {})
                .get("scaling_config", {})
                .get("target", "")
            )
            if yaml_prediction == "v" or yaml_prediction.endswith(".VScaling"):
                yaml_prediction = "v_prediction"
            elif not yaml_prediction:
                yaml_prediction = None
        except Exception as exc:  # pragma: no cover - defensive
            _LOG.warning("Failed to parse YAML config %s: %s", config_filename, exc)

    _apply_prediction_type(codex_components, parsed, yaml_prediction)

    metadata = {"repo_id": repo_name}
    if yaml_prediction:
        metadata["prediction_type"] = yaml_prediction
    # Note: VAE selection is expressed via engine options (`vae_path` + `vae_source`).

    return _build_diffusion_bundle(
        model_ref=sd_path,
        family=parsed.signature.family,
        estimated_config=config,
        components=codex_components,
        signature=parsed.signature,
        source="state_dict",
        metadata=metadata,
    )


# ------------------------------ Native diffusers repo loader (no state dict)
class _SimpleEstimated:
    def __init__(self, *, huggingface_repo: str, core_config: dict):
        self.huggingface_repo = huggingface_repo
        self.core_config = core_config

    def inpaint_model(self) -> bool:  # API parity with CodexEstimatedConfig
        return False


def _detect_engine_from_config(config: dict) -> str:
    pipeline_cls = str(config.get("_class_name") or "").strip().lower()
    if pipeline_cls == "fluxkontextpipeline":
        return "flux1_kontext"
    comps = {k: v for k, v in config.items() if isinstance(v, list) and len(v) == 2}
    cls_by_name = {k: v[1] for k, v in comps.items()}
    if "text_encoder_2" in comps and "unet" in comps:
        return "sdxl"
    if cls_by_name.get("transformer") in ("FluxTransformer2DModel",):
        return "flux1"
    if cls_by_name.get("transformer") in ("SD3Transformer2DModel",):
        return "sd35"
    if cls_by_name.get("transformer") in ("ChromaTransformer2DModel",):
        return "flux1_chroma"
    if "unet" in comps and "text_encoder" in comps and "vae" in comps:
        te_cls = cls_by_name.get("text_encoder", "")
        if te_cls.endswith("WithProjection"):
            return "sd20"
        return "sd15"
    raise ValueError("Unable to determine engine from diffusers config")


def load_engine_from_diffusers(repo_dir: str) -> DiffusionModelBundle:
    config: dict = DiffusionPipeline.load_config(repo_dir)
    comps = {}
    for name, (lib_name, cls_name) in (
        (k, v) for k, v in config.items() if isinstance(v, list) and len(v) == 2
    ):
        # Optional components are represented as [null, null] in model_index.json
        if lib_name is None and cls_name is None:
            continue
        if not isinstance(lib_name, str) or not isinstance(cls_name, str):
            raise TypeError(
                f"Invalid diffusers component spec for '{name}': expected [str, str] or [null, null], "
                f"got [{type(lib_name).__name__}, {type(cls_name).__name__}]"
            )
        cls = getattr(importlib.import_module(lib_name), cls_name)
        comps[name] = cls.from_pretrained(os.path.join(repo_dir, name), local_files_only=True)

    engine_key = _detect_engine_from_config(config)
    family = ENGINE_KEY_TO_FAMILY.get(engine_key)
    if family is None:
        raise ValueError(f"Unsupported engine key from diffusers config: {engine_key}")
    core_config = {}
    try:
        for k in ("unet", "transformer"):
            cfg_dir = os.path.join(repo_dir, k)
            if os.path.isdir(cfg_dir):
                cfg_path = os.path.join(cfg_dir, "config.json")
                if os.path.isfile(cfg_path):
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        core_config = json.load(fh)
                    break
    except Exception:
        core_config = {}

    est = _SimpleEstimated(huggingface_repo=os.path.basename(repo_dir), core_config=core_config)

    return _build_diffusion_bundle(
        model_ref=repo_dir,
        family=family,
        estimated_config=est,
        components=comps,
        source="diffusers",
        metadata={"engine_key": engine_key, "core_config": core_config},
    )


def resolve_diffusion_bundle(
    model_ref: str,
    *,
    additional_state_dicts: Optional[list[str]] = None,
    text_encoder_override: TextEncoderOverrideConfig | None = None,
    vae_path: str | None = None,
    tenc_path: str | list[str] | None = None,
    expected_family: ModelFamily | None = None,
) -> DiffusionModelBundle:
    """Resolve a diffusion model reference into a fully loaded bundle."""
    if os.path.isdir(model_ref):
        index = os.path.join(model_ref, "model_index.json")
        if os.path.isfile(index):
            return load_engine_from_diffusers(model_ref)
        raise ValueError(f"Not a diffusers repository (missing model_index.json): {model_ref}")

    if os.path.isfile(model_ref):
        return codex_loader(
            model_ref,
            additional_state_dicts=additional_state_dicts,
            text_encoder_override=text_encoder_override,
            vae_path=vae_path,
            tenc_path=tenc_path,
            expected_family=expected_family,
        )

    record = model_api.find_checkpoint(model_ref)
    if record is None:
        raise ValueError(f"Checkpoint not found: {model_ref}")

    # Determine format via metadata or filesystem inspection
    metadata = getattr(record, "metadata", {}) or {}
    if isinstance(metadata, dict) and metadata.get("format") == "diffusers":
        return load_engine_from_diffusers(record.path)

    repo_index = os.path.join(record.path, "model_index.json")
    if os.path.isfile(repo_index):
        return load_engine_from_diffusers(record.path)
    return codex_loader(
        record.filename,
        additional_state_dicts=additional_state_dicts,
        text_encoder_override=text_encoder_override,
        vae_path=vae_path,
        tenc_path=tenc_path,
        expected_family=expected_family,
    )
