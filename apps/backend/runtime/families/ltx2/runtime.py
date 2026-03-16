"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed bundle rehydration, native runtime assembly, and run-result contracts for the LTX2 seam.
Rebuilds the loader-produced LTX2 planning contract from a generic diffusion bundle, assembles the dedicated native
runtime from local `apps/**` modules (including optional wrapper-backed transformer-core streaming), and normalizes
execution results into the family-local
`frames + AudioExportAsset + metadata` contract consumed by the canonical video use-cases.

Symbols (top-level; keep in sync; no ghosts):
- `Ltx2RunResult` (dataclass): Family-local execution result consumed by canonical video use-cases.
- `Ltx2NativeComponents` (dataclass): Loaded LTX2 runtime components reused across txt2vid/img2vid runs.
- `build_ltx2_run_result` (function): Normalize `frames + audio_asset + metadata` into an immutable LTX2 result object.
- `build_ltx2_native_components` (function): Assemble the loaded native LTX2 runtime components from a typed bundle.
- `run_ltx2_txt2vid` (function): Execute the native LTX2 txt2vid pipeline and normalize the result contract.
- `run_ltx2_img2vid` (function): Execute the native LTX2 img2vid pipeline and normalize the result contract.
- `require_ltx2_bundle_inputs` (function): Rehydrate and validate the loader-produced LTX2 planning contract from a bundle-like object.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from apps.backend.runtime.checkpoint.io import read_arbitrary_config
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.pipeline_stages.video import AudioExportAsset, GeneratedAudioExportPolicy

from .audio import is_ltx2_wrapped_vocoder_state, materialize_ltx2_generated_audio_asset
from .model import Ltx2BundleInputs, Ltx2ComponentStates, Ltx2TextEncoderAsset, Ltx2VendorPaths
from .text_encoder import Ltx2TextEncoderRuntime, load_ltx2_text_encoder_runtime

logger = logging.getLogger("backend.runtime.families.ltx2.runtime")
_LTX2_EFFECTIVE_SAMPLER = "euler"
_LTX2_EFFECTIVE_SCHEDULER = "FlowMatchEulerDiscreteScheduler"
_LTX2_ALLOWED_SAMPLERS = frozenset({"", "euler", "uni-pc"})
_LTX2_ALLOWED_SCHEDULERS = frozenset({"", "simple"})


@dataclass(frozen=True, slots=True)
class Ltx2RunResult:
    frames: tuple[Any, ...]
    audio_asset: AudioExportAsset | None
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Ltx2NativeComponents:
    text_encoder: Any
    tokenizer: Any
    connectors: Any
    transformer: Any
    vae: Any
    audio_vae: Any
    vocoder: Any
    scheduler_config: dict[str, Any]
    requested_device_label: str
    device_label: str
    torch_dtype: torch.dtype
    dtype_label: str
    audio_sample_rate_hz: int
    runtime_impl: str
    transformers_version: str
    core_streaming_enabled: bool


def _resolve_ltx2_sampler_contract(request: Any) -> tuple[str | None, str | None, str, str]:
    sampler_requested = str(getattr(request, "sampler", "") or "").strip().lower()
    scheduler_requested = str(getattr(request, "scheduler", "") or "").strip().lower()

    if sampler_requested not in _LTX2_ALLOWED_SAMPLERS:
        raise RuntimeError(
            "LTX2 runtime currently executes on a fixed FlowMatchEulerDiscreteScheduler path. "
            "Accepted sampler values on this backend-only slice are empty, 'uni-pc' (generic-route default), "
            f"or 'euler'; got {getattr(request, 'sampler', None)!r}."
        )
    if scheduler_requested not in _LTX2_ALLOWED_SCHEDULERS:
        raise RuntimeError(
            "LTX2 runtime currently executes on a fixed FlowMatchEulerDiscreteScheduler path. "
            "Accepted scheduler values on this backend-only slice are empty or 'simple'; "
            f"got {getattr(request, 'scheduler', None)!r}."
        )

    return (
        sampler_requested or None,
        scheduler_requested or None,
        _LTX2_EFFECTIVE_SAMPLER,
        _LTX2_EFFECTIVE_SCHEDULER,
    )


def build_ltx2_run_result(
    *,
    frames: Sequence[Any],
    audio_asset: AudioExportAsset | None,
    metadata: Mapping[str, Any] | None = None,
) -> Ltx2RunResult:
    frames_tuple = tuple(frames)
    if not frames_tuple:
        raise RuntimeError("LTX2 run result requires at least one frame.")
    return Ltx2RunResult(
        frames=frames_tuple,
        audio_asset=audio_asset,
        metadata=dict(metadata or {}),
    )


def _as_torch_dtype(dtype_label: str) -> torch.dtype:
    normalized = str(dtype_label or "").strip().lower()
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    raise RuntimeError(f"LTX2 runtime dtype must be one of fp16|bf16|fp32, got {dtype_label!r}.")


def _resolve_device_name(device_label: str) -> torch.device:
    normalized = str(device_label or "").strip().lower()
    manager = getattr(memory_management, "manager", None)
    if manager is None or not hasattr(manager, "mount_device"):
        raise RuntimeError("LTX2 runtime requires an active memory manager with mount_device().")
    mount_device = manager.mount_device()
    if not isinstance(mount_device, torch.device):
        raise RuntimeError(
            "LTX2 runtime expected memory manager mount_device() -> torch.device, "
            f"got {type(mount_device).__name__}."
        )
    if normalized in {"auto", ""}:
        if mount_device.type in {"cpu", "cuda"}:
            return mount_device
        raise RuntimeError(
            "LTX2 runtime auto device requires a cpu/cuda mount device; "
            f"got {mount_device!s}."
        )
    if normalized == "cpu":
        return torch.device("cpu")
    if normalized == "cuda":
        if not bool(getattr(manager.hardware_probe, "cuda_available", False)):
            raise RuntimeError(
                "LTX2 runtime requested device='cuda', but CUDA is unavailable in the memory-manager probe."
            )
        return torch.device("cuda")
    raise RuntimeError(f"LTX2 runtime device must be one of auto|cpu|cuda, got {device_label!r}.")


def _import_native_ltx2_runtime_symbols() -> dict[str, Any]:
    try:
        from apps.backend.runtime.families.ltx2.native import (
            Ltx2AudioAutoencoder,
            Ltx2TextConnectors,
            Ltx2VideoAutoencoder,
            Ltx2VideoTransformer3DModel,
            Ltx2Vocoder,
            load_ltx2_connectors,
            load_ltx2_vocoder,
            run_ltx2_img2vid_native,
            run_ltx2_txt2vid_native,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "LTX2 native runtime modules are unavailable. "
            "The active LTX2 slice requires local model/scheduler/pipeline execution under "
            "`apps/backend/runtime/families/ltx2/native/**`."
        ) from exc

    try:
        import transformers
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("LTX2 runtime requires `transformers==4.57.3` with Gemma3 support.") from exc

    return {
        "Ltx2AudioAutoencoder": Ltx2AudioAutoencoder,
        "Ltx2TextConnectors": Ltx2TextConnectors,
        "Ltx2VideoAutoencoder": Ltx2VideoAutoencoder,
        "Ltx2VideoTransformer3DModel": Ltx2VideoTransformer3DModel,
        "Ltx2Vocoder": Ltx2Vocoder,
        "load_ltx2_connectors": load_ltx2_connectors,
        "load_ltx2_vocoder": load_ltx2_vocoder,
        "run_ltx2_img2vid_native": run_ltx2_img2vid_native,
        "run_ltx2_txt2vid_native": run_ltx2_txt2vid_native,
        "transformers_version": getattr(transformers, "__version__", "unknown"),
    }


def _read_component_config(repo_dir: Path, component_name: str) -> dict[str, Any]:
    component_dir = repo_dir / component_name
    if not component_dir.is_dir():
        raise RuntimeError(f"LTX2 vendored component directory not found: {component_dir}")
    return read_arbitrary_config(str(component_dir))


def _finalize_loaded_module(
    *,
    module: Any,
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Any:
    try:
        module = module.to(device=device, dtype=torch_dtype)
    except Exception:
        module = module.to(device=device)
    module.eval()
    return module


def _load_native_component_module(
    *,
    label: str,
    module_cls: Any,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Any:
    try:
        module = module_cls.from_config(dict(config))
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 {label} config instantiation failed: {exc}") from exc

    load_strict_state_dict = getattr(module, "load_strict_state_dict", None)
    if not callable(load_strict_state_dict):
        raise RuntimeError(
            f"LTX2 {label} native module {module_cls.__name__} must implement load_strict_state_dict()."
        )

    try:
        load_strict_state_dict(state_dict)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 {label} state load failed: {exc}") from exc

    return _finalize_loaded_module(module=module, device=device, torch_dtype=torch_dtype)


def _load_native_component_via_loader(
    *,
    label: str,
    loader_fn: Any,
    config: Mapping[str, Any],
    state_dict: Mapping[str, Any],
    device: torch.device,
    torch_dtype: torch.dtype,
) -> Any:
    try:
        return loader_fn(
            config=dict(config),
            state_dict=state_dict,
            device=device,
            torch_dtype=torch_dtype,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"LTX2 {label} state load failed: {exc}") from exc


def build_ltx2_native_components(
    *,
    bundle_inputs: Ltx2BundleInputs,
    device: str,
    dtype: str,
    engine_options: Mapping[str, Any] | None = None,
) -> Ltx2NativeComponents:
    from .streaming import Ltx2StreamingConfig, wrap_for_streaming

    streaming_config = Ltx2StreamingConfig.from_options(engine_options)
    symbols = _import_native_ltx2_runtime_symbols()
    resolved_device = _resolve_device_name(device)
    torch_dtype = _as_torch_dtype(dtype)
    dtype_label = str(dtype).strip().lower()
    if resolved_device.type == "cpu" and torch_dtype != torch.float32:
        logger.info(
            "[ltx2] forcing fp32 on CPU (requested dtype=%s, resolved device=%s)",
            dtype_label,
            resolved_device,
        )
        torch_dtype = torch.float32
        dtype_label = "fp32"

    repo_dir = Path(bundle_inputs.vendor_paths.repo_dir)
    text_runtime: Ltx2TextEncoderRuntime = load_ltx2_text_encoder_runtime(
        asset=bundle_inputs.text_encoder,
        vendor_paths=bundle_inputs.vendor_paths,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )

    scheduler_config = _read_component_config(repo_dir, "scheduler")
    connectors = _load_native_component_via_loader(
        label="connectors",
        loader_fn=symbols["load_ltx2_connectors"],
        config=_read_component_config(repo_dir, "connectors"),
        state_dict=bundle_inputs.components.connectors,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    transformer = _load_native_component_module(
        label="transformer",
        module_cls=symbols["Ltx2VideoTransformer3DModel"],
        config=_read_component_config(repo_dir, "transformer"),
        state_dict=bundle_inputs.components.transformer,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    if streaming_config.enabled:
        logger.info(
            "[ltx2] enabling transformer-core streaming (policy=%s blocks_per_segment=%d window_size=%d)",
            streaming_config.policy,
            streaming_config.blocks_per_segment,
            streaming_config.window_size,
        )
        transformer = wrap_for_streaming(
            transformer,
            policy=streaming_config.policy,
            blocks_per_segment=streaming_config.blocks_per_segment,
            window_size=streaming_config.window_size,
        )
    vae = _load_native_component_module(
        label="video_vae",
        module_cls=symbols["Ltx2VideoAutoencoder"],
        config=_read_component_config(repo_dir, "vae"),
        state_dict=bundle_inputs.components.vae,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    audio_vae = _load_native_component_module(
        label="audio_vae",
        module_cls=symbols["Ltx2AudioAutoencoder"],
        config=_read_component_config(repo_dir, "audio_vae"),
        state_dict=bundle_inputs.components.audio_vae,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )
    vocoder_config = bundle_inputs.vocoder_config
    if is_ltx2_wrapped_vocoder_state(bundle_inputs.components.vocoder):
        if vocoder_config is None:
            raise RuntimeError(
                "LTX2 wrapped vocoder assembly requires metadata-carried `vocoder_config`. "
                "The bundle lost the real audio-bundle wrapper config."
            )
    else:
        if vocoder_config is None:
            vocoder_config = _read_component_config(repo_dir, "vocoder")
    vocoder = _load_native_component_via_loader(
        label="vocoder",
        loader_fn=symbols["load_ltx2_vocoder"],
        config=vocoder_config,
        state_dict=bundle_inputs.components.vocoder,
        device=resolved_device,
        torch_dtype=torch_dtype,
    )

    audio_sample_rate_hz = int(getattr(getattr(vocoder, "config", None), "output_sampling_rate", 24000) or 24000)
    return Ltx2NativeComponents(
        text_encoder=text_runtime.model,
        tokenizer=text_runtime.tokenizer,
        connectors=connectors,
        transformer=transformer,
        vae=vae,
        audio_vae=audio_vae,
        vocoder=vocoder,
        scheduler_config=dict(scheduler_config),
        requested_device_label=str(device).strip().lower() or "auto",
        device_label=str(resolved_device),
        torch_dtype=torch_dtype,
        dtype_label=dtype_label,
        audio_sample_rate_hz=audio_sample_rate_hz,
        runtime_impl="native",
        transformers_version=str(symbols["transformers_version"]),
        core_streaming_enabled=streaming_config.enabled,
    )


def _normalize_video_frames(video: Any) -> tuple[Image.Image, ...]:
    array: Any = video
    if isinstance(array, (list, tuple)):
        if len(array) != 1:
            raise RuntimeError(
                "LTX2 runtime expects single-batch video output for canonical use-cases; "
                f"got batch={len(array)!r}."
            )
        array = array[0]
    if isinstance(array, torch.Tensor):
        array = array.detach().cpu().numpy()
    array = np.asarray(array)
    if array.ndim == 5:
        if int(array.shape[0]) != 1:
            raise RuntimeError(
                "LTX2 runtime expects single-batch video tensor output; "
                f"got shape={tuple(int(dim) for dim in array.shape)!r}."
            )
        array = array[0]
    if array.ndim != 4:
        raise RuntimeError(
            "LTX2 video output must be 4D after batch normalization "
            f"(frames,height,width,channels or frames,channels,height,width); got {array.ndim}D."
        )
    if array.shape[-1] in {1, 3, 4}:
        frames_array = array
    elif array.shape[1] in {1, 3, 4}:
        frames_array = np.transpose(array, (0, 2, 3, 1))
    else:
        raise RuntimeError(
            "LTX2 video output channel layout is unsupported; "
            f"got shape={tuple(int(dim) for dim in array.shape)!r}."
        )

    if np.issubdtype(frames_array.dtype, np.floating):
        frames_array = np.clip(frames_array, 0.0, 1.0)
        frames_array = (frames_array * 255.0).round().astype(np.uint8)
    elif frames_array.dtype != np.uint8:
        frames_array = np.clip(frames_array, 0, 255).astype(np.uint8)

    frames: list[Image.Image] = []
    for index, frame in enumerate(frames_array):
        try:
            frames.append(Image.fromarray(frame))
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"LTX2 failed to normalize frame {index} into a PIL image: {exc}") from exc
    if not frames:
        raise RuntimeError("LTX2 runtime produced zero video frames.")
    return tuple(frames)


def _coerce_native_media_outputs(outputs: Any, *, mode_label: str) -> tuple[Any, Any]:
    if isinstance(outputs, tuple):
        if len(outputs) < 2:
            raise RuntimeError(
                f"LTX2 {mode_label} native pipeline must return `(video, audio)` or an object with `video` and `audio`; "
                f"got tuple length={len(outputs)}."
            )
        return outputs[0], outputs[1]

    video = getattr(outputs, "video", None)
    audio = getattr(outputs, "audio", None)
    if video is None and hasattr(outputs, "videos"):
        video = getattr(outputs, "videos")
    if video is None:
        raise RuntimeError(
            f"LTX2 {mode_label} native pipeline output is unsupported: "
            f"expected `(video, audio)` tuple or object with `video`/`audio`, got {type(outputs).__name__}."
        )
    return video, audio


def _build_pipeline_metadata(
    *,
    native: Ltx2NativeComponents,
    pipeline_name: str,
    plan: Any,
    frame_count: int,
    has_audio: bool,
    sampler_requested: str | None,
    scheduler_requested: str | None,
    sampler_effective: str,
    scheduler_effective: str,
) -> dict[str, Any]:
    return {
        "pipeline": pipeline_name,
        "requested_device": native.requested_device_label,
        "effective_device": native.device_label,
        "dtype": native.dtype_label,
        "audio_sample_rate_hz": native.audio_sample_rate_hz,
        "frame_count": int(frame_count),
        "fps": int(getattr(plan, "fps", 0) or 0),
        "steps": int(getattr(plan, "steps", 0) or 0),
        "has_audio": bool(has_audio),
        "sampler_requested": sampler_requested,
        "scheduler_requested": scheduler_requested,
        "sampler": sampler_effective,
        "scheduler": scheduler_effective,
        "sampler_effective": sampler_effective,
        "scheduler_effective": scheduler_effective,
        "runtime_impl": native.runtime_impl,
        "transformers_version": native.transformers_version,
    }


def run_ltx2_txt2vid(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    generated_audio_export_policy: GeneratedAudioExportPolicy,
) -> Ltx2RunResult:
    sampler_requested, scheduler_requested, sampler_effective, scheduler_effective = _resolve_ltx2_sampler_contract(
        request
    )
    outputs = _import_native_ltx2_runtime_symbols()["run_ltx2_txt2vid_native"](
        native=native,
        request=request,
        plan=plan,
    )
    video, audio = _coerce_native_media_outputs(outputs, mode_label="txt2vid")
    frames = _normalize_video_frames(video)
    audio_asset = None
    if generated_audio_export_policy.materialize_audio_asset:
        audio_asset = materialize_ltx2_generated_audio_asset(
            audio,
            sample_rate_hz=native.audio_sample_rate_hz,
        )
    metadata = _build_pipeline_metadata(
        native=native,
        pipeline_name="ltx2_native_txt2vid",
        plan=plan,
        frame_count=len(frames),
        has_audio=audio_asset is not None,
        sampler_requested=sampler_requested,
        scheduler_requested=scheduler_requested,
        sampler_effective=sampler_effective,
        scheduler_effective=scheduler_effective,
    )
    return build_ltx2_run_result(frames=frames, audio_asset=audio_asset, metadata=metadata)


def run_ltx2_img2vid(
    *,
    native: Ltx2NativeComponents,
    request: Any,
    plan: Any,
    generated_audio_export_policy: GeneratedAudioExportPolicy,
) -> Ltx2RunResult:
    init_image = getattr(request, "init_image", None)
    if init_image is None:
        raise RuntimeError("LTX2 img2vid requires `request.init_image`.")

    sampler_requested, scheduler_requested, sampler_effective, scheduler_effective = _resolve_ltx2_sampler_contract(
        request
    )
    outputs = _import_native_ltx2_runtime_symbols()["run_ltx2_img2vid_native"](
        native=native,
        request=request,
        plan=plan,
    )
    video, audio = _coerce_native_media_outputs(outputs, mode_label="img2vid")
    frames = _normalize_video_frames(video)
    audio_asset = None
    if generated_audio_export_policy.materialize_audio_asset:
        audio_asset = materialize_ltx2_generated_audio_asset(
            audio,
            sample_rate_hz=native.audio_sample_rate_hz,
        )
    metadata = _build_pipeline_metadata(
        native=native,
        pipeline_name="ltx2_native_img2vid",
        plan=plan,
        frame_count=len(frames),
        has_audio=audio_asset is not None,
        sampler_requested=sampler_requested,
        scheduler_requested=scheduler_requested,
        sampler_effective=sampler_effective,
        scheduler_effective=scheduler_effective,
    )
    return build_ltx2_run_result(frames=frames, audio_asset=audio_asset, metadata=metadata)


def require_ltx2_bundle_inputs(bundle: object) -> Ltx2BundleInputs:
    family = getattr(bundle, "family", None)
    if family is not ModelFamily.LTX2:
        raise RuntimeError(
            "LTX2 runtime bundle rehydration requires a `ModelFamily.LTX2` bundle; "
            f"got {getattr(family, 'value', family)!r}."
        )

    metadata = getattr(bundle, "metadata", None)
    if not isinstance(metadata, dict):
        raise RuntimeError("LTX2 runtime bundle rehydration requires bundle.metadata to be a dict.")

    components = getattr(bundle, "components", None)
    if not isinstance(components, dict):
        raise RuntimeError("LTX2 runtime bundle rehydration requires bundle.components to be a dict.")

    model_ref = str(getattr(bundle, "model_ref", "") or "").strip()
    if not model_ref:
        raise RuntimeError("LTX2 runtime bundle rehydration requires a non-empty bundle.model_ref.")

    estimated_config = getattr(bundle, "estimated_config", None)
    signature = getattr(bundle, "signature", None)
    if estimated_config is None or signature is None:
        raise RuntimeError("LTX2 runtime bundle rehydration requires estimated_config and signature.")

    text_encoder = Ltx2TextEncoderAsset(
        alias=str(metadata.get("tenc_alias") or "").strip(),
        path=str(metadata.get("tenc_path") or "").strip(),
        kind=str(metadata.get("tenc_kind") or "").strip(),
        tokenizer_dir=str(metadata.get("tokenizer_dir") or "").strip(),
    )
    vendor_paths = Ltx2VendorPaths(
        repo_dir=str(metadata.get("vendor_repo_dir") or "").strip(),
        model_index_path=str(metadata.get("model_index_path") or "").strip(),
        tokenizer_dir=str(metadata.get("tokenizer_dir") or "").strip(),
        connectors_config_path=str(metadata.get("connectors_config_path") or "").strip(),
    )

    if not text_encoder.alias or not text_encoder.path or not text_encoder.kind:
        raise RuntimeError("LTX2 runtime bundle metadata is missing text-encoder planning fields.")
    if not vendor_paths.repo_dir or not vendor_paths.model_index_path or not vendor_paths.connectors_config_path:
        raise RuntimeError("LTX2 runtime bundle metadata is missing vendored LTX2 metadata paths.")

    vocoder_config = metadata.get("vocoder_config")
    if vocoder_config is not None and not isinstance(vocoder_config, Mapping):
        raise RuntimeError(
            "LTX2 runtime bundle metadata field `vocoder_config` must be a mapping when present; "
            f"got {type(vocoder_config).__name__}."
        )

    bundle_inputs = Ltx2BundleInputs(
        model_ref=model_ref,
        signature=signature,
        estimated_config=estimated_config,
        components=Ltx2ComponentStates.from_component_map(components),
        text_encoder=text_encoder,
        vendor_paths=vendor_paths,
        vocoder_config=vocoder_config,
    )
    if is_ltx2_wrapped_vocoder_state(bundle_inputs.components.vocoder) and bundle_inputs.vocoder_config is None:
        raise RuntimeError(
            "LTX2 runtime bundle rehydration requires metadata-carried `vocoder_config` for wrapped 2.3 vocoder states."
        )
    return bundle_inputs
