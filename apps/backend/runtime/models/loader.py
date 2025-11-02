import importlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
from diffusers import DiffusionPipeline
from transformers import modeling_utils

from apps.backend.infra.config.args import args
from apps.backend.runtime import trace as _trace
from apps.backend.runtime.common.nn.clip import IntegratedCLIP
from apps.backend.runtime.common.nn.t5 import IntegratedT5
from apps.backend.runtime.common.nn.unet import UNet2DConditionModel
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import SwapPolicy
from apps.backend.runtime.model_parser import parse_state_dict
from apps.backend.runtime.model_parser.specs import CodexEstimatedConfig
from apps.backend.runtime.model_registry import detect_from_state_dict as registry_detect
from apps.backend.runtime.model_registry.errors import ModelRegistryError
from apps.backend.runtime.model_registry.specs import (
    CodexCoreArchitecture,
    ModelFamily,
    ModelSignature,
    PredictionKind,
    QuantizationKind,
)
from apps.backend.runtime.models.state_dict import load_state_dict
from apps.backend.runtime.ops import using_codex_operations
from apps.backend.runtime.utils import (
    beautiful_print_gguf_state_dict_statics,
    load_torch_file,
    read_arbitrary_config,
)
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.wan22.vae import AutoencoderKLWan
from apps.backend.huggingface.assets import ensure_repo_minimal_files
from apps.backend.runtime.models import api as model_api

_LOG = logging.getLogger(__name__)
_BACKEND_ROOT = Path(__file__).resolve().parents[2]

SUPPORTED_INFERENCE_DTYPES: Dict[ModelFamily, tuple[torch.dtype, ...]] = {
    ModelFamily.FLUX: (torch.bfloat16, torch.float16, torch.float32),
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
    "flux": ModelFamily.FLUX,
    "sd35": ModelFamily.SD35,
    "sd3": ModelFamily.SD3,
    "chroma": ModelFamily.CHROMA,
    "sd20": ModelFamily.SD20,
    "sd15": ModelFamily.SD15,
}

FAMILY_TO_ENGINE_KEY: Dict[ModelFamily, str] = {
    ModelFamily.SDXL_REFINER: "sdxl_refiner",
    ModelFamily.SDXL: "sdxl",
    ModelFamily.FLUX: "flux",
    ModelFamily.SD35: "sd35",
    ModelFamily.SD3: "sd35",
    ModelFamily.CHROMA: "chroma",
    ModelFamily.SD20: "sd20",
    ModelFamily.SD15: "sd15",
}


def _supported_inference_dtypes(family: ModelFamily) -> tuple[torch.dtype, ...]:
    return SUPPORTED_INFERENCE_DTYPES.get(family, DEFAULT_SUPPORTED_DTYPES)


def _prediction_type_value(prediction: PredictionKind) -> str:
    return PREDICTION_TYPE_MAP.get(prediction, "epsilon")


def _load_state_dict(path: str) -> Mapping[str, Any]:
    print(f"[loader] load_torch_file_start path='{path}'", flush=True)
    _trace.event("load_torch_file_start", path=str(path))
    # Resolve the initial load device explicitly (no 'auto' fallback)
    initial_device = memory_management.core_initial_load_device(parameters=0, dtype=None)
    sd = load_torch_file(path, device=initial_device)
    try:
        tensor_count = len(sd.keys())  # type: ignore[attr-defined]
    except Exception:
        tensor_count = -1
    print(
        f"[loader] load_torch_file_done path='{path}' type='{type(sd).__name__}' tensors={tensor_count} map_device='{getattr(initial_device, 'type', 'unknown')}'",
        flush=True,
    )
    _trace.event("load_torch_file_done", path=str(path), type=type(sd).__name__, tensors=tensor_count)
    return sd


def _parse_checkpoint(primary_path: str, additional_paths: list[str] | None) -> ParsedCheckpoint:
    print(f"[loader] parse_checkpoint start path='{primary_path}'", flush=True)
    base_state = _load_state_dict(primary_path)
    signature = registry_detect(base_state)
    print(f"[loader] registry_detect family='{getattr(signature, 'family', None)}' kind='{getattr(signature, 'kind', None)}'", flush=True)
    config = parse_state_dict(base_state, signature)
    try:
        comp_names = list(getattr(config, 'components', {}).keys())
    except Exception:
        comp_names = []
    print(f"[loader] parse_state_dict ok components={comp_names}", flush=True)

    if additional_paths:
        replacements: Dict[str, Mapping[str, Any]] = {}
        for extra in additional_paths:
            extra_state = _load_state_dict(extra)
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


def _load_huggingface_component(
    parsed: ParsedCheckpoint,
    component_name: str,
    lib_name: str,
    cls_name: str,
    repo_path: str,
    state_dict: Mapping[str, Any] | None,
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
            return None
        config_json = AutoencoderKLWan.load_config(component_path)
        vae_device = memory_management.vae_device()
        vae_dtype = memory_management.vae_dtype(device=vae_device)
        print(f"[loader] vae_construct device='{vae_device}' dtype='{vae_dtype}'", flush=True)
        _trace.event("vae_construct", device=str(vae_device), dtype=str(vae_dtype))
        with using_codex_operations(device=vae_device, dtype=vae_dtype, manual_cast_enabled=True):
            model = AutoencoderKLWan.from_config(config_json)
        print(f"[loader] vae_load_state_dict tensors={len(state_dict)}", flush=True)
        _trace.event("load_state_dict", module="vae", tensors=len(state_dict))
        try:
            from .state_dict import safe_load_state_dict as _safe_load
            _safe_load(model, state_dict, log_name="VAE")
        except Exception:
            load_state_dict(model, state_dict, ignore_start="loss.", log_name="VAE")
        return model

    if cls_name in {"CLIPTextModel", "CLIPTextModelWithProjection"}:
        if state_dict is None:
            return None
        clip_config = importlib.import_module("transformers").CLIPTextConfig.from_pretrained(component_path)
        te_device = memory_management.text_encoder_device()
        te_dtype = memory_management.text_encoder_dtype(device=te_device)
        to_args = dict(device=te_device, dtype=te_dtype)
        print(f"[loader] clip_construct device='{te_device}' dtype='{te_dtype}'", flush=True)
        with modeling_utils.no_init_weights():
            with using_codex_operations(**to_args, manual_cast_enabled=True):
                model = IntegratedCLIP(importlib.import_module("transformers").CLIPTextModel, clip_config, add_text_projection=True).to(**to_args)
        load_state_dict(
            model,
            state_dict,
            ignore_errors=[
                "transformer.text_projection.weight",
                "transformer.text_model.embeddings.position_ids",
                "logit_scale",
            ],
            log_name=cls_name,
        )
        return model

    if cls_name == "T5EncoderModel":
        if state_dict is None:
            return None
        t5_config = read_arbitrary_config(component_path)
        te_device = memory_management.text_encoder_device()
        storage_dtype = memory_management.text_encoder_dtype(device=te_device)
        state_dict_dtype = memory_management.state_dict_dtype(state_dict)
        if state_dict_dtype in [torch.float8_e4m3fn, torch.float8_e5m2, "nf4", "fp4", "gguf"]:
            print(f"Using Detected T5 Data Type: {state_dict_dtype}")
            storage_dtype = state_dict_dtype
            if state_dict_dtype in ["nf4", "fp4", "gguf"]:
                print("Using pre-quant state dict!")
                if state_dict_dtype == "gguf":
                    beautiful_print_gguf_state_dict_statics(state_dict)
        else:
            print(f"Using Default T5 Data Type: {storage_dtype}")

        if storage_dtype in ["nf4", "fp4", "gguf"]:
            with modeling_utils.no_init_weights():
                with using_codex_operations(
                    device=te_device,
                    dtype=memory_management.text_encoder_dtype(device=te_device),
                    manual_cast_enabled=False,
                    bnb_dtype=storage_dtype,
                ):
                    model = IntegratedT5(t5_config)
        else:
            with modeling_utils.no_init_weights():
                with using_codex_operations(device=te_device, dtype=storage_dtype, manual_cast_enabled=True):
                    model = IntegratedT5(t5_config)

        load_state_dict(
            model,
            state_dict,
            log_name=cls_name,
            ignore_errors=["transformer.encoder.embed_tokens.weight", "logit_scale"],
        )
        return model

    if cls_name in {"UNet2DConditionModel", "FluxTransformer2DModel", "SD3Transformer2DModel", "ChromaTransformer2DModel"}:
        if state_dict is None:
            return None
        config_json = _load_component_config(component_path)
        core_arch = config.signature.core.architecture
        core_label = _CORE_ARCH_LABELS.get(core_arch, "Core")
        architecture_value = core_arch.value
        module_name = component_name or ("unet" if core_arch is CodexCoreArchitecture.UNET else "transformer")

        if cls_name == "UNet2DConditionModel":
            model_ctor = lambda cfg: UNet2DConditionModel.from_config(cfg)
        elif cls_name == "FluxTransformer2DModel":
            from apps.backend.runtime.flux.flux import FluxTransformer2DModel
            model_ctor = lambda cfg: FluxTransformer2DModel(**cfg)
        elif cls_name == "ChromaTransformer2DModel":
            from apps.backend.runtime.chroma.chroma import ChromaTransformer2DModel
            model_ctor = lambda cfg: ChromaTransformer2DModel(**cfg)
        else:
            from apps.backend.runtime.sd.mmditx import SD3Transformer2DModel
            model_ctor = lambda cfg: SD3Transformer2DModel(**cfg)

        supported_dtypes = _supported_inference_dtypes(family)
        quant_kind = config.quantization.kind
        storage_dtype = memory_management.core_dtype(supported_dtypes=supported_dtypes)
        if quant_kind == QuantizationKind.NF4:
            storage_dtype = "nf4"
        elif quant_kind == QuantizationKind.FP4:
            storage_dtype = "fp4"
        elif quant_kind == QuantizationKind.GGUF:
            storage_dtype = "gguf"

        load_device = memory_management.get_torch_device()
        computation_dtype = memory_management.get_computation_dtype(load_device, parameters=0, supported_dtypes=supported_dtypes)
        offload_device = memory_management.core_offload_device()

        mem_config = memory_management.memory_config

        if storage_dtype in ["nf4", "fp4", "gguf"]:
            initial_device = memory_management.core_initial_load_device(parameters=0, dtype=computation_dtype)
            with using_codex_operations(device=initial_device, dtype=computation_dtype, manual_cast_enabled=False, bnb_dtype=storage_dtype):
                model = model_ctor(config_json)
        else:
            prefer_gpu = bool(getattr(mem_config, "gpu_prefer_construct", False))
            construct_device = load_device if prefer_gpu else memory_management.core_initial_load_device(parameters=0, dtype=storage_dtype)
            initial_device = construct_device
            construct_dtype = storage_dtype
            if memory_management.is_device_cpu(construct_device) and construct_dtype in (torch.bfloat16, torch.float16):
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
            except memory_management.OOM_EXCEPTION as exc:
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

        _trace.event("load_state_dict", module=module_name, architecture=architecture_value, tensors=len(state_dict))
        try:
            from .state_dict import safe_load_state_dict as _safe_load
            _safe_load(model, state_dict, log_name=core_label)
        except Exception:
            load_state_dict(model, state_dict, log_name=core_label)

        model.config = config_json
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
    if yaml_prediction:
        scheduler.config.prediction_type = yaml_prediction
        return
    scheduler.config.prediction_type = _prediction_type_value(parsed.signature.prediction)


@torch.inference_mode()
def codex_loader(sd_path: str, additional_state_dicts=None):
    print(f"[loader] codex_loader enter sd_path='{sd_path}'", flush=True)
    try:
        parsed = _parse_checkpoint(sd_path, additional_state_dicts or [])
    except ModelRegistryError as exc:
        raise ValueError("Failed to recognize model type!") from exc

    config = parsed.config
    component_states = {name: comp.state_dict for name, comp in config.components.items()}

    repo_name = config.repo_id
    if not isinstance(repo_name, str) or not repo_name:
        raise ValueError("Codex model parser did not resolve a repository id")

    local_repo_path = os.path.join(str(_BACKEND_ROOT), "huggingface", repo_name)
    offline = bool(args.disable_online_tokenizer)
    include = ("config", "tokenizer", "scheduler")  # strictly minimal; no weights
    print(
        f"[loader] ensure_repo_minimal_files repo='{repo_name}' offline={offline} include={include}",
        flush=True,
    )
    ensure_repo_minimal_files(repo_name, local_repo_path, offline=offline, include=include)

    print(f"[loader] DiffusionPipeline.load_config path='{local_repo_path}'", flush=True)
    pipeline_config = DiffusionPipeline.load_config(local_repo_path)
    print(f"[loader] pipeline_config keys={list(pipeline_config.keys())}", flush=True)
    codex_components: Dict[str, Any] = {}

    for component_name, component_info in pipeline_config.items():
        if not (isinstance(component_info, list) and len(component_info) == 2):
            continue
        lib_name, cls_name = component_info
        component_sd = component_states.get(component_name)
        print(f"[loader] load component name='{component_name}' cls='{lib_name}.{cls_name}'", flush=True)
        component_obj = _load_huggingface_component(
            parsed,
            component_name,
            lib_name,
            cls_name,
            local_repo_path,
            component_sd,
        )
        if component_sd is not None:
            component_states.pop(component_name, None)
        if component_obj is not None:
            print(f"[loader] component ok name='{component_name}'", flush=True)
            codex_components[component_name] = component_obj

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
    comps = {k: v for k, v in config.items() if isinstance(v, list) and len(v) == 2}
    cls_by_name = {k: v[1] for k, v in comps.items()}
    if "text_encoder_2" in comps and "unet" in comps:
        return "sdxl"
    if cls_by_name.get("transformer") in ("FluxTransformer2DModel",):
        return "flux"
    if cls_by_name.get("transformer") in ("SD3Transformer2DModel",):
        return "sd35"
    if cls_by_name.get("transformer") in ("ChromaTransformer2DModel",):
        return "chroma"
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
) -> DiffusionModelBundle:
    """Resolve a diffusion model reference into a fully loaded bundle."""
    print(f"[loader] resolve_diffusion_bundle model_ref='{model_ref}'", flush=True)
    if os.path.isdir(model_ref):
        index = os.path.join(model_ref, "model_index.json")
        if os.path.isfile(index):
            print(f"[loader] detected diffusers repo at '{model_ref}'", flush=True)
            return load_engine_from_diffusers(model_ref)
        raise ValueError(f"Not a diffusers repository (missing model_index.json): {model_ref}")

    if os.path.isfile(model_ref):
        print(f"[loader] detected state_dict file at '{model_ref}'", flush=True)
        return codex_loader(model_ref, additional_state_dicts=additional_state_dicts)

    record = model_api.find_checkpoint(model_ref)
    if record is None:
        raise ValueError(f"Checkpoint not found: {model_ref}")

    # Determine format via metadata or filesystem inspection
    metadata = getattr(record, "metadata", {}) or {}
    if isinstance(metadata, dict) and metadata.get("format") == "diffusers":
        print(f"[loader] registry metadata indicates diffusers for '{record.path}'", flush=True)
        return load_engine_from_diffusers(record.path)

    repo_index = os.path.join(record.path, "model_index.json")
    if os.path.isfile(repo_index):
        print(f"[loader] found model_index.json under '{record.path}'", flush=True)
        return load_engine_from_diffusers(record.path)
    print(f"[loader] falling back to state_dict filename='{record.filename}'", flush=True)
    return codex_loader(record.filename, additional_state_dicts=additional_state_dicts)
