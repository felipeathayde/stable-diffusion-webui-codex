"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Bounded live diagnostics for the IP-Adapter conditioning seam.
Provides strict diagnostics request parsing plus operator-facing receipts for the
reference-image -> CLIP preprocess -> CLIP encode -> projector/resampler path
without duplicating the public generation pipeline.

Symbols (top-level; keep in sync; no ghosts):
- `IpAdapterProbeInvalidRequest` (exception): Raised for invalid diagnostics payloads.
- `IpAdapterProbeRequest` (dataclass): Normalized bounded request for the IP-Adapter probe route.
- `IpAdapterProbeTensorStats` (dataclass): Numeric summary for one tensor receipt.
- `IpAdapterProbeImageMeta` (dataclass): Metadata summary for the resolved reference image.
- `IpAdapterProbeReport` (dataclass): Structured live diagnostics response for the IP-Adapter probe route.
- `parse_ip_adapter_probe_request` (function): Validates and normalizes the bounded diagnostics request payload.
- `run_ip_adapter_probe` (function): Executes the live IP-Adapter conditioning probe and returns a structured report.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from PIL import Image
import torch

from apps.backend.runtime.adapters.ip_adapter.assets import prepare_ip_adapter_assets_for_paths
from apps.backend.services.media_service import MediaService

_ALLOWED_SOURCE_KINDS: Final[frozenset[str]] = frozenset({"uploaded", "path"})


class IpAdapterProbeInvalidRequest(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class IpAdapterProbeRequest:
    model_path: str
    image_encoder_path: str
    source_kind: str
    reference_image_data: str | None
    reference_image_path: str | None
    crop: bool


@dataclass(frozen=True, slots=True)
class IpAdapterProbeTensorStats:
    shape: tuple[int, ...]
    dtype: str
    device: str
    numel: int
    finite: bool
    minimum: float | None
    maximum: float | None
    mean: float | None
    std: float | None
    l2_norm: float | None


@dataclass(frozen=True, slots=True)
class IpAdapterProbeImageMeta:
    mode: str
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class IpAdapterProbeReport:
    ok: bool
    phase: str
    reason_code: str | None
    reason_detail: str | None
    model_path: str
    image_encoder_path: str
    source_kind: str
    crop: bool
    layout: str | None
    uses_hidden_states: bool | None
    slot_count: int | None
    token_count: int | None
    output_cross_attention_dim: int | None
    internal_cross_attention_dim: int | None
    encoder_variant: str | None
    encoder_hidden_size: int | None
    encoder_projection_dim: int | None
    reference_image: IpAdapterProbeImageMeta | None
    source_pixels: IpAdapterProbeTensorStats | None
    preprocessed_pixels: IpAdapterProbeTensorStats | None
    image_embeds: IpAdapterProbeTensorStats | None
    penultimate_hidden_states: IpAdapterProbeTensorStats | None
    condition_tokens: IpAdapterProbeTensorStats | None
    uncondition_tokens: IpAdapterProbeTensorStats | None
    condition_uncondition_max_abs_diff: float | None
    condition_uncondition_mean_abs_diff: float | None

    def to_payload(self) -> dict[str, Any]:
        return asdict(self)


def parse_ip_adapter_probe_request(payload: Any) -> IpAdapterProbeRequest:
    if not isinstance(payload, dict):
        raise IpAdapterProbeInvalidRequest("payload must be a JSON object")
    allowed_keys = {
        "model_path",
        "image_encoder_path",
        "source_kind",
        "reference_image_data",
        "reference_image_path",
        "crop",
    }
    unknown_keys = sorted(str(key) for key in payload.keys() if key not in allowed_keys)
    if unknown_keys:
        raise IpAdapterProbeInvalidRequest(f"unknown payload keys: {', '.join(unknown_keys)}")
    model_path = _require_non_empty_str(payload.get("model_path"), field_name="model_path")
    image_encoder_path = _require_non_empty_str(payload.get("image_encoder_path"), field_name="image_encoder_path")
    source_kind = _require_non_empty_str(payload.get("source_kind"), field_name="source_kind").lower()
    if source_kind not in _ALLOWED_SOURCE_KINDS:
        allowed = ", ".join(sorted(_ALLOWED_SOURCE_KINDS))
        raise IpAdapterProbeInvalidRequest(f"source_kind must be one of: {allowed}")
    crop = _parse_bool(payload.get("crop", True), field_name="crop")
    reference_image_data = payload.get("reference_image_data")
    reference_image_path = payload.get("reference_image_path")
    if source_kind == "uploaded":
        if not isinstance(reference_image_data, str) or not reference_image_data.strip():
            raise IpAdapterProbeInvalidRequest("reference_image_data is required when source_kind='uploaded'")
        if reference_image_path is not None:
            raise IpAdapterProbeInvalidRequest("reference_image_path is only valid when source_kind='path'")
        normalized_reference_image_data = reference_image_data.strip()
        normalized_reference_image_path = None
    else:
        if not isinstance(reference_image_path, str) or not reference_image_path.strip():
            raise IpAdapterProbeInvalidRequest("reference_image_path is required when source_kind='path'")
        if reference_image_data is not None:
            raise IpAdapterProbeInvalidRequest("reference_image_data is only valid when source_kind='uploaded'")
        normalized_reference_image_data = None
        normalized_reference_image_path = reference_image_path.strip()
    return IpAdapterProbeRequest(
        model_path=model_path,
        image_encoder_path=image_encoder_path,
        source_kind=source_kind,
        reference_image_data=normalized_reference_image_data,
        reference_image_path=normalized_reference_image_path,
        crop=crop,
    )


def run_ip_adapter_probe(request: IpAdapterProbeRequest) -> IpAdapterProbeReport:
    try:
        assets = prepare_ip_adapter_assets_for_paths(
            model_path=request.model_path,
            image_encoder_path=request.image_encoder_path,
        )
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="assets",
            reason_code="E_IP_ADAPTER_PROBE_ASSET_LOAD",
            reason_detail=str(exc),
        )
    try:
        reference_image = _load_reference_image(request)
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="reference_image",
            reason_code="E_IP_ADAPTER_PROBE_REFERENCE_IMAGE",
            reason_detail=str(exc),
            assets=assets,
        )
    try:
        source_pixels = _image_to_bhwc_tensor(reference_image)
        encoder = assets.image_encoder_runtime
        projector = copy.deepcopy(assets.image_projector)
        projector.to(device=encoder.load_device, dtype=encoder.runtime_dtype)
        projector.eval()
        with torch.inference_mode():
            processed = encoder.prepare_pixels(source_pixels, crop=request.crop)
            encoded = encoder.encode_pixels(processed)
            if assets.uses_hidden_states:
                uncondition_encoded = encoder.encode_pixels(torch.zeros_like(processed))
                condition_inputs = encoded.penultimate_hidden_states
                uncondition_inputs = uncondition_encoded.penultimate_hidden_states
            else:
                condition_inputs = encoded.image_embeds
                uncondition_inputs = torch.zeros_like(condition_inputs)
            condition = projector(condition_inputs.to(device=encoder.load_device, dtype=encoder.runtime_dtype))
            uncondition = projector(uncondition_inputs.to(device=encoder.load_device, dtype=encoder.runtime_dtype))
        max_abs_diff, mean_abs_diff = _tensor_difference(condition, uncondition)
        return IpAdapterProbeReport(
            ok=True,
            phase="complete",
            reason_code=None,
            reason_detail=None,
            model_path=request.model_path,
            image_encoder_path=request.image_encoder_path,
            source_kind=request.source_kind,
            crop=bool(request.crop),
            layout=assets.layout.value,
            uses_hidden_states=bool(assets.uses_hidden_states),
            slot_count=int(assets.slot_count),
            token_count=int(assets.token_count),
            output_cross_attention_dim=int(assets.output_cross_attention_dim),
            internal_cross_attention_dim=int(assets.internal_cross_attention_dim),
            encoder_variant=assets.image_encoder_runtime.spec.variant.value,
            encoder_hidden_size=int(assets.image_encoder_runtime.spec.hidden_size),
            encoder_projection_dim=int(assets.image_encoder_runtime.spec.projection_dim),
            reference_image=IpAdapterProbeImageMeta(
                mode=str(reference_image.mode),
                width=int(reference_image.width),
                height=int(reference_image.height),
            ),
            source_pixels=_tensor_stats(source_pixels),
            preprocessed_pixels=_tensor_stats(processed),
            image_embeds=_tensor_stats(encoded.image_embeds),
            penultimate_hidden_states=_tensor_stats(encoded.penultimate_hidden_states),
            condition_tokens=_tensor_stats(condition),
            uncondition_tokens=_tensor_stats(uncondition),
            condition_uncondition_max_abs_diff=max_abs_diff,
            condition_uncondition_mean_abs_diff=mean_abs_diff,
        )
    except Exception as exc:
        return _failure_report(
            request=request,
            phase="conditioning",
            reason_code="E_IP_ADAPTER_PROBE_CONDITIONING",
            reason_detail=str(exc),
            assets=assets,
            reference_image=reference_image,
            source_pixels=locals().get("source_pixels"),
        )


def _load_reference_image(request: IpAdapterProbeRequest) -> Image.Image:
    if request.source_kind == "uploaded":
        return MediaService().decode_image(request.reference_image_data).convert("RGB")
    assert request.reference_image_path is not None
    image_path = Path(request.reference_image_path)
    if not image_path.is_file():
        raise RuntimeError(f"reference image path does not exist: {request.reference_image_path!r}")
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _image_to_bhwc_tensor(image: Image.Image) -> torch.Tensor:
    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise RuntimeError(f"IP-Adapter probe reference image must decode to HWC RGB; got shape={array.shape}.")
    return torch.from_numpy(array / 255.0).unsqueeze(0)


def _tensor_stats(tensor: torch.Tensor | None) -> IpAdapterProbeTensorStats | None:
    if not isinstance(tensor, torch.Tensor):
        return None
    detached = tensor.detach()
    working = detached.float().cpu()
    numel = int(working.numel())
    finite = bool(torch.isfinite(working).all().item()) if numel > 0 else True
    if numel == 0:
        minimum = maximum = mean = std = l2_norm = None
    else:
        minimum = float(working.min().item())
        maximum = float(working.max().item())
        mean = float(working.mean().item())
        std = float(working.std(unbiased=False).item()) if numel > 1 else 0.0
        l2_norm = float(torch.linalg.vector_norm(working).item())
    return IpAdapterProbeTensorStats(
        shape=tuple(int(dim) for dim in detached.shape),
        dtype=str(detached.dtype),
        device=str(detached.device),
        numel=numel,
        finite=finite,
        minimum=minimum,
        maximum=maximum,
        mean=mean,
        std=std,
        l2_norm=l2_norm,
    )


def _tensor_difference(left: torch.Tensor, right: torch.Tensor) -> tuple[float | None, float | None]:
    left_cpu = left.detach().float().cpu()
    right_cpu = right.detach().float().cpu()
    if left_cpu.shape != right_cpu.shape:
        raise RuntimeError(
            "IP-Adapter probe expected condition/uncondition tensors to share the same shape; "
            f"got {tuple(left_cpu.shape)} vs {tuple(right_cpu.shape)}."
        )
    delta = (left_cpu - right_cpu).abs()
    if delta.numel() == 0:
        return None, None
    return float(delta.max().item()), float(delta.mean().item())


def _failure_report(
    *,
    request: IpAdapterProbeRequest,
    phase: str,
    reason_code: str,
    reason_detail: str,
    assets: Any | None = None,
    reference_image: Image.Image | None = None,
    source_pixels: torch.Tensor | None = None,
) -> IpAdapterProbeReport:
    image_meta = None
    if isinstance(reference_image, Image.Image):
        image_meta = IpAdapterProbeImageMeta(
            mode=str(reference_image.mode),
            width=int(reference_image.width),
            height=int(reference_image.height),
        )
    return IpAdapterProbeReport(
        ok=False,
        phase=str(phase),
        reason_code=str(reason_code),
        reason_detail=str(reason_detail),
        model_path=request.model_path,
        image_encoder_path=request.image_encoder_path,
        source_kind=request.source_kind,
        crop=bool(request.crop),
        layout=getattr(getattr(assets, "layout", None), "value", None),
        uses_hidden_states=getattr(assets, "uses_hidden_states", None),
        slot_count=getattr(assets, "slot_count", None),
        token_count=getattr(assets, "token_count", None),
        output_cross_attention_dim=getattr(assets, "output_cross_attention_dim", None),
        internal_cross_attention_dim=getattr(assets, "internal_cross_attention_dim", None),
        encoder_variant=getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "variant", None).value
        if getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "variant", None) is not None
        else None,
        encoder_hidden_size=getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "hidden_size", None),
        encoder_projection_dim=getattr(getattr(getattr(assets, "image_encoder_runtime", None), "spec", None), "projection_dim", None),
        reference_image=image_meta,
        source_pixels=_tensor_stats(source_pixels),
        preprocessed_pixels=None,
        image_embeds=None,
        penultimate_hidden_states=None,
        condition_tokens=None,
        uncondition_tokens=None,
        condition_uncondition_max_abs_diff=None,
        condition_uncondition_mean_abs_diff=None,
    )


def _require_non_empty_str(raw: Any, *, field_name: str) -> str:
    if not isinstance(raw, str) or not raw.strip():
        raise IpAdapterProbeInvalidRequest(f"{field_name} must be a non-empty string")
    return raw.strip()


def _parse_bool(raw: Any, *, field_name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    raise IpAdapterProbeInvalidRequest(f"{field_name} must be a boolean")
