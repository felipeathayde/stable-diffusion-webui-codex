"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF VAE IO helpers (I2V condition encode + decode to frames).
Loads WAN VAE weights via explicit native lanes (`2d_native` or `3d_native`), applies latent normalization, and converts between latents and RGB frames for the WAN22 GGUF runtime.
Includes strict finite checks and explicit dtype/device retry logic (no silent fallbacks).

Symbols (top-level; keep in sync; no ghosts):
- `WAN22VAEContractError` (exception): Deterministic WAN VAE path/config contract failure (non-retryable by dtype fallback loops).
- `_detect_wan_vae_lane` (function): Resolves canonical WAN VAE lane (`2d_native`/`3d_native`) from core convolution weights.
- `load_vae` (function): Loads the WAN VAE component (from directory bundles or single-file weights with sibling/override config dir).
- `_cuda_bf16_supported` (function): Best-effort BF16 support probe for CUDA (used for dtype fallbacks).
- `_vae_dtype_candidates` (function): Ordered dtype candidates for VAE encode/decode attempts (requested dtype first).
- `vae_encode_video_condition` (function): Encodes the Diffusers-style I2V conditioning video into latents (deterministic mode).
- `vae_decode_video` (function): Decodes video latents to frames; can validate the expected output frame count.
- `decode_latents_to_frames` (function): Validates strict WAN latent-channel decode input (no implicit slicing) and returns frames (optional frame-count validation).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, Optional

import torch

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.smart_offload import smart_fallback_enabled
from apps.backend.runtime.checkpoint.io import load_torch_file
from apps.backend.runtime.models.state_dict import safe_load_state_dict
from apps.backend.runtime.common.vae_ldm import AutoencoderKL_LDM, sanitize_ldm_vae_config
from apps.backend.runtime.common.vae_codex3d import (
    AutoencoderCodex3D,
    remap_codex3d_vae_state_dict,
    sanitize_codex3d_vae_config,
)

from .config import RunConfig, as_torch_dtype, resolve_device_name
from .diagnostics import cuda_empty_cache, get_logger, log_numerics_enabled, summarize_numerics, warn_fallback
from .wan_latent_norms import resolve_norm

_SUPPORTED_WAN_VAE_LATENT_CHANNELS = (16, 48)
_SUPPORTED_WAN_VAE_LANES = ("2d_native", "3d_native")


class WAN22VAEContractError(RuntimeError):
    """Deterministic WAN22 VAE path/config contract failure."""


def _detect_wan_vae_lane(state_dict: Mapping[str, Any]) -> str:
    evidence_keys = (
        "encoder.conv_in.weight",
        "decoder.conv_in.weight",
        "encoder.conv1.weight",
        "decoder.conv1.weight",
    )
    observed: set[str] = set()
    seen: list[tuple[str, int, tuple[int, ...]]] = []
    for key in evidence_keys:
        tensor = state_dict.get(key)
        if not torch.is_tensor(tensor):
            continue
        ndim = int(tensor.ndim)
        shape = tuple(int(dim) for dim in tensor.shape)
        seen.append((key, ndim, shape))
        if ndim == 4:
            observed.add("2d_native")
            continue
        if ndim == 5:
            observed.add("3d_native")
            continue
        raise WAN22VAEContractError(
            "WAN22 GGUF: unsupported core VAE kernel rank "
            f"key={key!r} ndim={ndim} shape={shape} (expected 4D or 5D)."
        )
    if not seen:
        fallback = [
            (str(key), tuple(int(dim) for dim in tensor.shape))
            for key, tensor in state_dict.items()
            if torch.is_tensor(tensor)
            and str(key).endswith(".weight")
            and (
                str(key).startswith("encoder.conv")
                or str(key).startswith("decoder.conv")
                or str(key).startswith("encoder.head.2")
                or str(key).startswith("decoder.head.2")
            )
        ]
        if not fallback:
            raise WAN22VAEContractError(
                "WAN22 GGUF: cannot detect VAE lane (missing canonical core convolution weights)."
            )
        for key, shape in fallback:
            if len(shape) == 4:
                observed.add("2d_native")
            elif len(shape) == 5:
                observed.add("3d_native")
            else:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: unsupported fallback VAE kernel rank "
                    f"key={key!r} shape={shape} (expected rank 4 or 5)."
                )
    if len(observed) != 1:
        raise WAN22VAEContractError(
            "WAN22 GGUF: mixed VAE lane evidence in core kernels "
            f"(lanes={sorted(observed)} evidence={seen[:8]})."
        )
    return next(iter(observed))


def _infer_latent_channels_from_state_dict(state_dict: Mapping[str, Any]) -> int | None:
    candidates: tuple[tuple[str, int], ...] = (
        ("post_quant_conv.weight", 0),
        ("quant_conv.weight", 0),
        ("encoder.conv_out.weight", 0),
        ("decoder.conv_in.weight", 1),
        ("conv2.weight", 0),
        ("conv1.weight", 0),
        ("encoder.head.2.weight", 0),
        ("decoder.conv1.weight", 1),
    )
    for key, axis in candidates:
        tensor = state_dict.get(key)
        if not torch.is_tensor(tensor):
            continue
        shape = tuple(int(dim) for dim in tensor.shape)
        if len(shape) < 2 or axis >= len(shape):
            continue
        value = int(shape[axis])
        if key in {"quant_conv.weight", "encoder.conv_out.weight", "conv1.weight", "encoder.head.2.weight"} and value % 2 == 0:
            value = value // 2
        if value > 0:
            return value
    return None


def load_vae(
    vae_path: Optional[str],
    *,
    torch_dtype: torch.dtype,
    enable_tiling: bool = False,
    config_dir_override: Optional[str] = None,
) -> Any:
    if not vae_path:
        raise WAN22VAEContractError(
            "WAN22 GGUF: wan_vae_path is required when running the GGUF runtime "
            "(VAE bundle directory path missing)."
        )

    path = os.path.expanduser(str(vae_path))

    def _instantiate_with_state_dict(state_dict_path: str, config_dir: str) -> Any:
        try:
            raw_state_dict = load_torch_file(state_dict_path, device="cpu")
            if not isinstance(raw_state_dict, Mapping):
                raise WAN22VAEContractError(
                    "WAN22 GGUF: VAE checkpoint loader returned non-mapping state_dict "
                    f"(type={type(raw_state_dict).__name__})."
                )
            lane = _detect_wan_vae_lane(raw_state_dict)
            if lane not in _SUPPORTED_WAN_VAE_LANES:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: unsupported VAE lane "
                    f"{lane!r} (supported={list(_SUPPORTED_WAN_VAE_LANES)})."
                )

            if lane == "2d_native":
                state_dict = dict(raw_state_dict)
                config = AutoencoderKL_LDM.load_config(config_dir)
                native_config = sanitize_ldm_vae_config(config)
                inferred_latent_channels = _infer_latent_channels_from_state_dict(state_dict)
                if inferred_latent_channels is not None:
                    configured_channels = native_config.get("latent_channels")
                    if configured_channels is None:
                        native_config["latent_channels"] = int(inferred_latent_channels)
                    elif int(configured_channels) != int(inferred_latent_channels):
                        raise WAN22VAEContractError(
                            "WAN22 GGUF: VAE config/state_dict latent channel mismatch "
                            f"(lane=2d_native config latent_channels={int(configured_channels)} "
                            f"inferred={int(inferred_latent_channels)})."
                        )
                vae = AutoencoderKL_LDM.from_config(native_config)
                missing, unexpected = safe_load_state_dict(vae, state_dict, log_name="WAN22 VAE (2d_native)")
                if missing or unexpected:
                    raise WAN22VAEContractError(
                        "WAN22 GGUF: native VAE load failed strict validation "
                        f"(lane=2d_native missing={len(missing)} unexpected={len(unexpected)} "
                        f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]})."
                    )
                setattr(vae, "_codex_vae_lane", "2d_native")
                return vae

            remap_style, state_dict = remap_codex3d_vae_state_dict(dict(raw_state_dict))
            config = AutoencoderCodex3D.load_config(config_dir)
            native_config = sanitize_codex3d_vae_config(config)
            inferred_latent_channels = _infer_latent_channels_from_state_dict(state_dict)
            if inferred_latent_channels is not None:
                configured_channels = native_config.get("z_dim")
                if configured_channels is None:
                    native_config["z_dim"] = int(inferred_latent_channels)
                elif int(configured_channels) != int(inferred_latent_channels):
                    raise WAN22VAEContractError(
                        "WAN22 GGUF: VAE config/state_dict latent channel mismatch "
                        f"(lane=3d_native style={remap_style} config z_dim={int(configured_channels)} "
                        f"inferred={int(inferred_latent_channels)})."
                    )
            vae = AutoencoderCodex3D.from_config(native_config)
            missing, unexpected = safe_load_state_dict(vae, state_dict, log_name="WAN22 VAE (3d_native)")
            if missing or unexpected:
                raise WAN22VAEContractError(
                    "WAN22 GGUF: native VAE load failed strict validation "
                    f"(lane=3d_native missing={len(missing)} unexpected={len(unexpected)} "
                    f"missing_sample={missing[:10]} unexpected_sample={unexpected[:10]})."
                )
            setattr(vae, "_codex_vae_lane", "3d_native")
            return vae
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            raise WAN22VAEContractError(
                "WAN22 GGUF: failed to load native VAE lane from checkpoint "
                f"path={state_dict_path!r} config_dir={config_dir!r}: {exc}"
            ) from exc

    if os.path.isdir(path):
        weights_candidates = (
            "diffusion_pytorch_model.safetensors",
            "diffusion_pytorch_model.bin",
            "model.safetensors",
            "model.bin",
            "pytorch_model.bin",
        )
        state_dict_path = None
        for name in weights_candidates:
            candidate = os.path.join(path, name)
            if os.path.isfile(candidate):
                state_dict_path = candidate
                break
        if state_dict_path is None:
            raise WAN22VAEContractError(f"WAN22 GGUF: no VAE weights file found under directory: {path}")
        vae = _instantiate_with_state_dict(state_dict_path, path).to(dtype=torch_dtype)
        if enable_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass
        return vae
    if os.path.isfile(path):
        config_dirs: list[str] = []
        if isinstance(config_dir_override, str) and str(config_dir_override).strip():
            config_dirs.append(os.path.expanduser(str(config_dir_override).strip()))
        config_dirs.append(os.path.dirname(path))
        chosen_config_dir: str | None = None
        for config_dir in config_dirs:
            config_path = os.path.join(config_dir, "config.json")
            if os.path.isfile(config_path):
                chosen_config_dir = config_dir
                break
        if not chosen_config_dir:
            raise WAN22VAEContractError(
                "WAN22 GGUF: single-file VAE load requires config.json at sibling path "
                "or provided metadata config directory. "
                f"VAE file={path} checked_config_dirs={config_dirs}"
            )
        vae = _instantiate_with_state_dict(path, chosen_config_dir).to(dtype=torch_dtype)
        if enable_tiling and hasattr(vae, "enable_tiling"):
            try:
                vae.enable_tiling()
            except Exception:
                pass
        return vae
    raise WAN22VAEContractError(f"WAN22 GGUF: VAE path not found: {path}")


def _retrieve_latents(encoder_output: Any, *, sample_mode: str) -> torch.Tensor:
    dist = getattr(encoder_output, "latent_dist", None)
    if dist is not None:
        mode = str(sample_mode or "").strip().lower()
        if mode in {"mode", "argmax"}:
            return dist.mode()
        if mode in {"sample", ""}:
            return dist.sample()
        raise ValueError(f"Unsupported VAE sample_mode: {sample_mode!r} (expected 'mode' or 'sample')")
    if hasattr(encoder_output, "latents"):
        return encoder_output.latents
    if torch.is_tensor(encoder_output):
        return encoder_output
    if isinstance(encoder_output, (tuple, list)) and encoder_output and torch.is_tensor(encoder_output[0]):
        return encoder_output[0]
    raise AttributeError(
        "VAE encode output has neither latent_dist nor latents "
        f"(type={type(encoder_output).__name__})."
    )


def _maybe_resize_hw(x: torch.Tensor, *, height: int, width: int) -> torch.Tensor:
    if x.ndim != 4:
        return x
    _, _, h, w = x.shape
    if int(h) == int(height) and int(w) == int(width):
        return x
    import torch.nn.functional as F

    return F.interpolate(x, size=(int(height), int(width)), mode="bilinear", align_corners=False)


def _prepare_init_image_tensor(
    init_image: Any,
    *,
    device: str,
    torch_dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:
    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name == "cuda" and torch.cuda.is_available() else "cpu"

    if hasattr(init_image, "to"):
        t = init_image
        if hasattr(t, "ndim") and int(t.ndim) == 5:
            t = t[:, :, 0, ...]
        if hasattr(t, "ndim") and int(t.ndim) == 3:
            t = t.unsqueeze(0)
        if not hasattr(t, "ndim") or int(getattr(t, "ndim", 0)) != 4:
            raise RuntimeError(
                "WAN22 GGUF: init_image tensor must be 4D [B,C,H,W] (or 5D [B,C,T,H,W]); "
                f"got {getattr(t, 'shape', None)}"
            )
        try:
            t = t.to(target).to(torch_dtype)
        except Exception:
            t = t.to(target)
        t = _maybe_resize_hw(t, height=height, width=width)
        return t

    from PIL import Image
    import numpy as np

    if isinstance(init_image, Image.Image):
        img = init_image.convert("RGB")
        img = img.resize((int(width), int(height)), resample=Image.BICUBIC)
        arr = np.array(img).astype("float32") / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        t = t.to(target).to(torch_dtype)
        return t * 2.0 - 1.0

    arr = np.asarray(init_image).astype("float32")
    if arr.ndim == 3 and arr.shape[2] in (1, 3):
        arr = arr / 255.0 if arr.max() > 1.0 else arr
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3):
        t = torch.from_numpy(arr).unsqueeze(0)
    else:
        raise RuntimeError("WAN22 GGUF: unsupported init_image array shape")

    t = t.to(target).to(torch_dtype)
    t = _maybe_resize_hw(t, height=height, width=width)
    return t * 2.0 - 1.0


def _cuda_bf16_supported() -> bool:
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return False
    fn = getattr(torch.cuda, "is_bf16_supported", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def _vae_dtype_candidates(*, device: str, preferred: torch.dtype) -> list[torch.dtype]:
    dev = resolve_device_name(device)
    if dev.startswith("cuda") and torch.cuda.is_available():
        out: list[torch.dtype] = [preferred]
        if preferred == torch.float16 and _cuda_bf16_supported():
            out.append(getattr(torch, "bfloat16", torch.float16))
        if preferred != torch.float32:
            out.append(torch.float32)
        if preferred != torch.float16:
            out.append(torch.float16)
        # Deduplicate while preserving order.
        seen: set[torch.dtype] = set()
        uniq: list[torch.dtype] = []
        for dt in out:
            if dt in seen:
                continue
            seen.add(dt)
            uniq.append(dt)
        return uniq

    # CPU: default to float32 for stability (BF16 is optional and hardware-dependent).
    return [torch.float32]


def _assert_supported_wan_vae_latent_channels(channels: int, *, context: str) -> None:
    if int(channels) in _SUPPORTED_WAN_VAE_LATENT_CHANNELS:
        return
    supported = ", ".join(str(value) for value in _SUPPORTED_WAN_VAE_LATENT_CHANNELS)
    raise RuntimeError(
        f"WAN22 GGUF: {context} supports latent channels [{supported}] only (got C={int(channels)}). "
        "If C includes mask/image channels (e.g., I2V model-input state), pass pure VAE latents to decode."
    )


def _resolve_loaded_vae_lane(vae: Any) -> str:
    # Runtime-loaded WAN VAEs are always stamped with `_codex_vae_lane`.
    # Keep a deterministic default for stubs/fixtures that do not set it.
    lane = str(getattr(vae, "_codex_vae_lane", "3d_native")).strip().lower()
    if lane not in _SUPPORTED_WAN_VAE_LANES:
        raise RuntimeError(
            "WAN22 GGUF: loaded VAE exposes unsupported lane marker "
            f"{lane!r} (supported={list(_SUPPORTED_WAN_VAE_LANES)})."
        )
    return lane


def _to_frame_batch_4d(video_tensor: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Convert `[B,C,T,H,W]` tensor to frame-batched `[B*T,C,H,W]`."""
    if video_tensor.ndim != 5:
        raise RuntimeError(
            f"WAN22 GGUF: expected 5D video tensor [B,C,T,H,W], got shape={tuple(video_tensor.shape)}."
        )
    b, c, t, h, w = video_tensor.shape
    batched = video_tensor.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    return batched, int(b), int(t)


def _from_frame_batch_4d(frame_tensor: torch.Tensor, *, batch: int, frames: int) -> torch.Tensor:
    """Convert frame-batched `[B*T,C,H,W]` tensor back to `[B,C,T,H,W]`."""
    if frame_tensor.ndim != 4:
        raise RuntimeError(
            f"WAN22 GGUF: expected 4D frame-batch tensor [B*T,C,H,W], got shape={tuple(frame_tensor.shape)}."
        )
    bt, c, h, w = frame_tensor.shape
    expected = int(batch) * int(frames)
    if int(bt) != expected:
        raise RuntimeError(
            "WAN22 GGUF: frame-batch reshape mismatch "
            f"(B*T={expected} expected, got {int(bt)})."
        )
    return frame_tensor.view(int(batch), int(frames), int(c), int(h), int(w)).permute(0, 2, 1, 3, 4).contiguous()


def vae_encode_video_condition(
    init_image: Any,
    *,
    num_frames: int,
    height: int,
    width: int,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    vae_config_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
) -> torch.Tensor:
    log = get_logger(logger)

    if int(num_frames) <= 0:
        raise RuntimeError(f"WAN22 GGUF: invalid num_frames={num_frames} for I2V video_condition")

    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name.startswith("cuda") and torch.cuda.is_available() else "cpu"
    preferred = as_torch_dtype(dtype)
    dtypes = _vae_dtype_candidates(device=device, preferred=preferred)
    last_exc: Exception | None = None

    for attempt_idx, torch_dtype in enumerate(dtypes):
        vae = None
        try:
            if attempt_idx > 0:
                warn_fallback(
                    logger,
                    component="VAE encode",
                    detail=f"retrying with dtype={torch_dtype} device={target}",
                    reason=str(last_exc) if last_exc is not None else "previous attempt failed",
                )
            vae = load_vae(
                vae_dir,
                torch_dtype=torch_dtype,
                enable_tiling=bool(memory_management.manager.vae_always_tiled),
                config_dir_override=vae_config_dir,
            )
            vae = vae.to(device=target, dtype=torch_dtype)

            image = _prepare_init_image_tensor(
                init_image,
                device=device,
                torch_dtype=torch_dtype,
                height=height,
                width=width,
            )
            if image.ndim != 4:
                raise RuntimeError(
                    "WAN22 GGUF: expected preprocessed init image to be 4D [B,C,H,W], "
                    f"got {tuple(image.shape)}"
                )

            # Diffusers-style I2V conditioning video: first frame is the init image; remaining frames are 0 (i.e., 0.5 gray).
            image = image.unsqueeze(2)  # [B,C,1,H,W]
            video_condition = torch.cat(
                [image, image.new_zeros((image.shape[0], image.shape[1], int(num_frames) - 1, int(height), int(width)))],
                dim=2,
            )
            lane = _resolve_loaded_vae_lane(vae)
            with torch.no_grad():
                if lane == "2d_native":
                    video_batched, batch_size, frame_count = _to_frame_batch_4d(video_condition)
                    encoded_raw = _retrieve_latents(vae.encode(video_batched), sample_mode="mode")
                    if encoded_raw.ndim == 4:
                        encoded = _from_frame_batch_4d(encoded_raw, batch=batch_size, frames=frame_count)
                    elif encoded_raw.ndim == 5:
                        encoded = encoded_raw
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: VAE encode produced unsupported tensor rank "
                            f"(lane=2d_native shape={tuple(encoded_raw.shape)})."
                        )
                else:
                    encoded_raw = _retrieve_latents(vae.encode(video_condition), sample_mode="mode")
                    if encoded_raw.ndim == 5:
                        encoded = encoded_raw
                    elif encoded_raw.ndim == 4:
                        encoded = encoded_raw.unsqueeze(2)
                    else:
                        raise RuntimeError(
                            "WAN22 GGUF: VAE encode produced unsupported tensor rank "
                            f"(lane=3d_native shape={tuple(encoded_raw.shape)})."
                        )
            latent_channels = int(encoded.shape[1])
            _assert_supported_wan_vae_latent_channels(latent_channels, context="VAE encode")
            norm = resolve_norm(None, channels=latent_channels)
            log.info("[wan22.gguf] VAE latent norm=%s channels=%d", norm.name, norm.channels)
            encoded = norm.process_in(encoded)

            if not torch.isfinite(encoded).all():
                n_bad = int((~torch.isfinite(encoded)).sum().item())
                raise RuntimeError(
                    "WAN22 GGUF: VAE encode produced non-finite latents "
                    f"(bad={n_bad} dtype={torch_dtype} device={target}; {summarize_numerics(encoded, name='encoded')})."
                )

            if log_numerics_enabled():
                log.info(
                    "[wan22.gguf] VAE encode ok: device=%s dtype=%s %s",
                    target,
                    str(torch_dtype),
                    summarize_numerics(encoded, name="encoded"),
                )

            return encoded
        except torch.OutOfMemoryError as exc:
            last_exc = exc
            if target != "cuda" or not smart_fallback_enabled():
                raise
            warn_fallback(
                logger,
                component="VAE encode",
                detail=f"OOM on CUDA at dtype={torch_dtype}; retrying on CPU fp32",
                reason="cuda_oom",
            )
            cuda_empty_cache(logger, label="vae-encode-oom")
            target = "cpu"
            # CPU retry (single attempt).
            dtypes = [torch.float32]
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            last_exc = exc
        finally:
            if offload_after and vae is not None:
                try:
                    vae.to("cpu")
                except Exception:
                    pass
                del vae
                cuda_empty_cache(logger, label="after-vae-encode")

    raise RuntimeError("WAN22 GGUF: VAE encode failed for all dtype fallbacks.") from last_exc


def vae_decode_video(
    video_latents: Any,
    *,
    model_dir: str,
    device: str,
    dtype: str,
    vae_dir: str | None = None,
    vae_config_dir: str | None = None,
    logger: Any = None,
    offload_after: bool = True,
    expected_frames: int | None = None,
) -> list[object]:
    _ = model_dir  # kept for signature symmetry (callers pass stage dir; current VAE loads from explicit path)
    log = get_logger(logger)

    if hasattr(video_latents, "ndim"):
        if video_latents.ndim == 4:
            video_latents = video_latents.unsqueeze(2)
        elif video_latents.ndim != 5:
            raise RuntimeError(
                f"WAN22 VAE decode expects 4D or 5D latents; got shape={tuple(getattr(video_latents,'shape',()))}"
            )

    b, c, t_lat, h, w = video_latents.shape

    _assert_supported_wan_vae_latent_channels(int(c), context="VAE decode")
    norm = resolve_norm(None, channels=int(c))

    from PIL import Image

    frames: list[Image.Image] = []
    dev_name = resolve_device_name(device)
    target = "cuda" if dev_name.startswith("cuda") and torch.cuda.is_available() else "cpu"
    preferred = as_torch_dtype(dtype)
    dtypes = _vae_dtype_candidates(device=device, preferred=preferred)
    last_exc: Exception | None = None

    def _decode_attempt(*, attempt_device: str, torch_dtype: torch.dtype) -> torch.Tensor:
        vae = load_vae(
            vae_dir,
            torch_dtype=torch_dtype,
            enable_tiling=bool(memory_management.manager.vae_always_tiled),
            config_dir_override=vae_config_dir,
        )
        vae = vae.to(device=attempt_device, dtype=torch_dtype)
        lane = _resolve_loaded_vae_lane(vae)
        lat_in = video_latents.to(device=attempt_device, dtype=torch_dtype)
        lat = norm.process_out(lat_in)
        if not torch.isfinite(lat).all():
            n_bad = int((~torch.isfinite(lat)).sum().item())
            raise RuntimeError(
                "WAN22 GGUF: non-finite latents after unnormalize; refusing to decode "
                f"(bad={n_bad} dtype={torch_dtype} device={attempt_device}; {summarize_numerics(lat, name='lat_unnorm')})."
            )
        with torch.no_grad():
            if lane == "2d_native":
                lat_batched, batch_size, frame_count = _to_frame_batch_4d(lat)
                decoded = vae.decode(lat_batched)
            else:
                batch_size, frame_count = int(lat.shape[0]), int(lat.shape[2])
                decoded = vae.decode(lat)
            sample = getattr(decoded, "sample", None)
            if sample is not None:
                img = sample
            elif torch.is_tensor(decoded):
                img = decoded
            elif isinstance(decoded, (tuple, list)) and decoded and torch.is_tensor(decoded[0]):
                img = decoded[0]
            else:
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode output has no tensor sample "
                    f"(type={type(decoded).__name__})."
                )
            if lane == "2d_native" and img.ndim == 4:
                img = _from_frame_batch_4d(img, batch=batch_size, frames=frame_count)
            elif lane == "3d_native" and img.ndim == 4:
                img = img.unsqueeze(2)
            elif img.ndim != 5:
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode produced unsupported tensor rank "
                    f"(lane={lane} shape={tuple(img.shape)})."
                )
        if offload_after:
            try:
                vae.to("cpu")
            except Exception:
                pass
            del vae
            cuda_empty_cache(logger, label="after-vae-decode")
        return img

    img = None
    for attempt_idx, torch_dtype in enumerate(dtypes):
        if attempt_idx > 0:
            warn_fallback(
                logger,
                component="VAE decode",
                detail=f"retrying with dtype={torch_dtype} device={target}",
                reason=str(last_exc) if last_exc is not None else "previous attempt failed",
            )
        try:
            img = _decode_attempt(attempt_device=target, torch_dtype=torch_dtype)
            bad = int((~torch.isfinite(img)).sum().item()) if isinstance(img, torch.Tensor) else -1
            if bad > 0:
                raise RuntimeError(
                    "WAN22 GGUF: VAE decode produced non-finite outputs "
                    f"(bad={bad} dtype={torch_dtype} device={target}; {summarize_numerics(img, name='vae_out')})."
                )
            break
        except torch.OutOfMemoryError as exc:
            last_exc = exc
            if target == "cuda" and smart_fallback_enabled():
                warn_fallback(
                    logger,
                    component="VAE decode",
                    detail=f"OOM on CUDA at dtype={torch_dtype}; will retry other dtypes and then CPU fp32",
                    reason="cuda_oom",
                )
                cuda_empty_cache(logger, label="vae-decode-oom")
                continue
            raise
        except WAN22VAEContractError:
            raise
        except Exception as exc:
            last_exc = exc
            # Continue to next dtype.
            continue

    if img is None or not isinstance(img, torch.Tensor):
        raise RuntimeError("WAN22 GGUF: VAE decode failed (no tensor output).") from last_exc

    # Last-resort: if CUDA path fails, try CPU fp32 (only when smart fallback is enabled).
    if not torch.isfinite(img).all() and target == "cuda" and smart_fallback_enabled():
        warn_fallback(
            logger,
            component="VAE decode",
            detail="all CUDA dtype attempts produced non-finite outputs; retrying on CPU fp32",
            reason="nonfinite_cuda",
        )
        cuda_empty_cache(logger, label="vae-decode-nonfinite")
        img = _decode_attempt(attempt_device="cpu", torch_dtype=torch.float32)

    log.info("[wan22.gguf] VAE decode output shape=%s", tuple(getattr(img, "shape", ())))
    if not hasattr(img, "ndim") or img.ndim != 5:
        raise RuntimeError(
            f"WAN22 GGUF: VAE decode produced unexpected rank: shape={tuple(getattr(img,'shape',()))}; expected [B,C,T,H,W]"
        )
    if int(img.shape[0]) < 1 or int(img.shape[1]) != 3:
        raise RuntimeError(
            f"WAN22 GGUF: VAE decode produced unexpected shape: {tuple(img.shape)}; expected B>=1 and C=3."
        )
    t_out = int(img.shape[2])
    if expected_frames is not None and int(expected_frames) != t_out:
        raise RuntimeError(
            "WAN22 GGUF: VAE decode time dimension mismatch: "
            f"expected T={int(expected_frames)} got T={t_out} (latent_T={int(t_lat)})."
        )

    if not torch.isfinite(img).all():
        n_bad = int((~torch.isfinite(img)).sum().item())
        raise RuntimeError(
            "WAN22 GGUF: VAE decode produced non-finite outputs after fallbacks "
            f"(bad={n_bad}; {summarize_numerics(img, name='vae_out')})."
        )

    if log_numerics_enabled():
        log.info("[wan22.gguf] VAE decode ok: %s", summarize_numerics(img, name="vae_out"))

    for ti in range(t_out):
        x = img[0, :, ti, :, :].detach()
        if x.ndim != 3:
            raise RuntimeError(
                f"WAN22 GGUF: VAE decode produced unexpected frame tensor rank: shape={tuple(x.shape)}; expected [C,H,W]"
            )
        # Diffusers VAEs output [-1, 1]; convert to [0, 1] for image conversion.
        x = (x + 1.0) * 0.5
        x = x.clamp(0, 1)
        arr = (x.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
        frames.append(Image.fromarray(arr))

    return frames


def decode_latents_to_frames(
    *,
    latents: torch.Tensor,
    model_dir: str,
    cfg: RunConfig,
    logger: Any = None,
    debug_preview: bool = False,
    expected_frames: int | None = None,
) -> list[object]:
    log = get_logger(logger)
    x = latents
    log.info("[wan22.gguf] decode latents: shape=%s", tuple(x.shape))
    _ = debug_preview  # debug-only clamp removed (no env-driven behavior)

    c = int(x.shape[1])
    _assert_supported_wan_vae_latent_channels(c, context="decode")

    if not torch.isfinite(x).all():
        n_bad = int((~torch.isfinite(x)).sum().item())
        raise RuntimeError(
            "WAN22 GGUF: decode input latents are non-finite; aborting before VAE decode "
            f"(bad={n_bad}; {summarize_numerics(x, name='latents_in')})."
        )

    if log_numerics_enabled():
        log.info("[wan22.gguf] decode latents (pre-VAE): %s", summarize_numerics(x, name="latents_in"))

    return vae_decode_video(
        x,
        model_dir=model_dir,
        device=resolve_device_name(cfg.device),
        dtype=cfg.dtype,
        vae_dir=cfg.vae_dir,
        vae_config_dir=cfg.vae_config_dir,
        logger=logger,
        expected_frames=expected_frames,
    )
