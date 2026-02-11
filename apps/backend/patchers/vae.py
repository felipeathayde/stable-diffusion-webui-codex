"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: VAE patcher + tiling helpers for encode/decode (diffusers + WAN-aware).
Provides a VAE wrapper that normalizes diffusers outputs (scalar and optional per-channel latent stats) using family-aware policy resolution for scale/shift semantics,
supports tiled decode/encode paths, and integrates memory-management and smart-fallback behavior. Supports separate storage vs compute dtype selection
(compute defaults to fp32 unless overridden).
Tiling helpers are shared with the global upscalers runtime to avoid drift.

Symbols (top-level; keep in sync; no ghosts):
- `_tensor_stats` (function): Logs tensor shape/dtype/device and basic statistics for debugging VAE behavior.
- `_unwrap_decode_output` (function): Normalizes diffusers decode outputs to a plain tensor (`DecoderOutput.sample` or passthrough).
- `_unwrap_encode_output` (function): Normalizes diffusers encode outputs to a latent tensor (handles `latent_dist`, `.sample()`, `.mean`, etc.).
- `_NormalizingFirstStage` (class): Wrapper around a first-stage VAE that applies strict scalar/per-channel latent normalization (including optional shift semantics) and proxies encode/decode APIs.
- `tiled_scale_multidim` (function): Multi-dim tiled scaling helper (used to process large images in overlapping tiles).
- `get_tiled_scale_steps` (function): Computes the number of tile steps given dimensions and overlap.
- `tiled_scale` (function): Convenience wrapper for tiled scaling in 2D (tile_x/tile_y).
- `VAE` (class): ModelPatcher for VAEs; provides encode/decode APIs (optionally tiled), device/dtype placement, and fallback/normalization logic
  (includes nested helpers for memory-management and diffusers/WAN VAE compatibility).
"""

import logging
import math

import torch

try:  # Optional import; diffusers may not be present in minimal environments
    from diffusers.models.autoencoder_kl import AutoencoderKL as DiffusersAutoencoderKL
except Exception:  # noqa: BLE001
    DiffusersAutoencoderKL = None

try:  # Optional; only needed to detect native LDM VAEs explicitly
    from apps.backend.runtime.families.wan22.vae import AutoencoderKL_LDM
except Exception:  # noqa: BLE001
    AutoencoderKL_LDM = None

from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload import smart_fallback_enabled
from apps.backend.runtime.vision.upscalers.tiled_scale import (
    get_tiled_scale_steps,
    tiled_scale,
    tiled_scale_multidim,
)
from .base import ModelPatcher
from .vae_normalization_policy import read_vae_config_field, resolve_vae_normalization_policy

logger = logging.getLogger("backend.patchers.vae")


def _tensor_stats(label: str, tensor: torch.Tensor) -> None:
    if tensor is None:
        logger.info("[vae] %s: <none>", label)
        return
    with torch.no_grad():
        data = tensor.detach()
        stats_tensor = data.float()
        logger.info(
            "[vae] %s: shape=%s dtype=%s device=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
            label,
            tuple(data.shape),
            data.dtype,
            data.device,
            float(stats_tensor.min().item()),
            float(stats_tensor.max().item()),
            float(stats_tensor.mean().item()),
            float(stats_tensor.std(unbiased=False).item()),
        )


def _unwrap_decode_output(output):
    """Extract tensor from diffusers DecoderOutput or passthrough."""
    if hasattr(output, "sample"):
        sample = getattr(output, "sample")
        if torch.is_tensor(sample):
            return sample
    return output


def _unwrap_encode_output(output):
    """Extract latent tensor from diffusers AutoencoderKLOutput or passthrough."""
    # Newer diffusers-style outputs: AutoencoderKLOutput with latent_dist
    if hasattr(output, "latent_dist"):
        dist = getattr(output, "latent_dist")
        if hasattr(dist, "sample"):
            try:
                return dist.sample()
            except Exception:  # noqa: BLE001
                pass
        if hasattr(dist, "mean") and torch.is_tensor(dist.mean):
            return dist.mean
    # Objects that are themselves distributions (e.g., DiagonalGaussianDistribution)
    if hasattr(output, "sample") and callable(getattr(output, "sample", None)):
        try:
            sample = output.sample()
            if torch.is_tensor(sample):
                return sample
        except Exception:  # noqa: BLE001
            pass
    if hasattr(output, "mean") and torch.is_tensor(getattr(output, "mean")):
        return getattr(output, "mean")
    # Some implementations return a plain tensor or an object with `.sample` tensor attribute
    if hasattr(output, "sample") and torch.is_tensor(getattr(output, "sample")):
        return output.sample
    if torch.is_tensor(output):
        return output
    # Legacy/variant encoders may return tuples like (latents, aux) or (AutoencoderKLOutput, aux).
    # Walk the tuple/list and recursively unwrap the first tensor-like item we find.
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if torch.is_tensor(item):
                return item
            try:
                inner = _unwrap_encode_output(item)
                if torch.is_tensor(inner):
                    return inner
            except Exception:
                continue
    # Fallback: surface an explicit error instead of returning an unsupported type.
    raise RuntimeError(f"VAE encode returned unsupported output type: {type(output)!r}")


class _NormalizingFirstStage:
    """Adapter that guarantees process_in/out around a diffusers VAE.

    - scalar-only path:
      - process_in: (x - shift) * scale
      - process_out: (x / scale) + shift
    - per-channel path:
      - process_in: (x - (latents_mean + shift)) * scale / latents_std
      - process_out: x * latents_std / scale + (latents_mean + shift)
    Also proxies encode/decode/to/attributes to the wrapped object.
    """

    def __init__(
        self,
        base,
        *,
        scale: float,
        shift: float | None,
        latents_mean: tuple[float, ...] | None = None,
        latents_std: tuple[float, ...] | None = None,
    ) -> None:
        self._base = base
        self._scale = float(scale)
        self._shift = None if shift is None else float(shift)
        if not math.isfinite(self._scale) or self._scale == 0.0:
            raise RuntimeError(f"Invalid VAE scaling_factor: {self._scale!r} (must be finite and non-zero).")
        if self._shift is not None and not math.isfinite(self._shift):
            raise RuntimeError(f"Invalid VAE shift_factor: {self._shift!r} (must be finite).")

        if (latents_mean is None) != (latents_std is None):
            raise RuntimeError("VAE latent stats must provide both latents_mean and latents_std (or neither).")

        self._latents_mean = None
        self._latents_std = None
        if latents_mean is not None and latents_std is not None:
            mean_values = tuple(float(x) for x in latents_mean)
            std_values = tuple(float(x) for x in latents_std)
            if not mean_values:
                raise RuntimeError("VAE latent stats are empty; expected at least one channel value.")
            if len(mean_values) != len(std_values):
                raise RuntimeError(
                    "VAE latent stats length mismatch: "
                    f"len(latents_mean)={len(mean_values)} len(latents_std)={len(std_values)}."
                )
            if any(not math.isfinite(value) for value in mean_values):
                raise RuntimeError("VAE latents_mean contains non-finite values.")
            if any((not math.isfinite(value)) or value <= 0.0 for value in std_values):
                raise RuntimeError("VAE latents_std must contain finite positive values.")
            self._latents_mean = torch.tensor(mean_values, dtype=torch.float32)
            self._latents_std = torch.tensor(std_values, dtype=torch.float32)

    # Proxy core API used by VAE wrapper
    def encode(self, *args, **kwargs):  # noqa: D401
        return self._base.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):  # noqa: D401
        return self._base.decode(*args, **kwargs)

    def to(self, *args, **kwargs):  # noqa: D401
        return self._base.to(*args, **kwargs)

    # Normalization API expected by engines
    def process_in(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("process_in expects a torch.Tensor")
        stats = self._stats_for(x)
        shift = 0.0 if self._shift is None else self._shift
        if stats is None:
            return (x - shift) * self._scale
        latents_mean, latents_std = stats
        return (x - (latents_mean + shift)) * self._scale / latents_std

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("process_out expects a torch.Tensor")
        stats = self._stats_for(x)
        shift = 0.0 if self._shift is None else self._shift
        if stats is None:
            return (x / self._scale) + shift
        latents_mean, latents_std = stats
        return x * latents_std / self._scale + (latents_mean + shift)

    def _stats_for(self, x: torch.Tensor):
        if self._latents_mean is None or self._latents_std is None:
            return None
        if x.ndim not in (4, 5):
            raise RuntimeError(
                "VAE latent stats only support 4D/5D tensors; "
                f"got shape={tuple(x.shape)}."
            )
        channels = int(x.shape[1]) if x.ndim >= 2 else -1
        expected_channels = int(self._latents_mean.shape[0])
        if channels != expected_channels:
            raise RuntimeError(
                "VAE latent channel mismatch for per-channel normalization: "
                f"tensor_channels={channels} expected_channels={expected_channels} "
                f"shape={tuple(x.shape)}."
            )
        view_shape = (1, expected_channels) + (1,) * (x.ndim - 2)
        latents_mean = self._latents_mean.to(device=x.device, dtype=x.dtype).view(view_shape)
        latents_std = self._latents_std.to(device=x.device, dtype=x.dtype).view(view_shape)
        return latents_mean, latents_std

    def __getattr__(self, name: str):
        # Delegate any other attribute access to the base VAE
        return getattr(self._base, name)

    @staticmethod
    def wrap(base, *, family=None):
        """Wrap a VAE with normalization.
        
        Args:
            base: The base VAE model.
            family: Optional ModelFamily for fallback scaling/shift values.
        
        Returns:
            _NormalizingFirstStage wrapper.
        """
        cfg = getattr(base, "config", None)
        policy = resolve_vae_normalization_policy(config=cfg, family=family)
        _, latents_mean = read_vae_config_field(cfg, "latents_mean")
        _, latents_std = read_vae_config_field(cfg, "latents_std")

        def _coerce_optional_float_tuple(name: str, value):
            if value is None:
                return None
            try:
                return tuple(float(x) for x in value)
            except TypeError as exc:
                raise RuntimeError(f"VAE config field '{name}' must be an iterable of numbers.") from exc
            except ValueError as exc:
                raise RuntimeError(f"VAE config field '{name}' contains non-numeric values.") from exc

        latents_mean_values = _coerce_optional_float_tuple("latents_mean", latents_mean)
        latents_std_values = _coerce_optional_float_tuple("latents_std", latents_std)

        if latents_mean_values is not None and latents_std_values is not None:
            logger.info(
                "[VAE] normalization enabled: scaling_factor=%s shift_factor=%s channels=%d (per-channel stats)",
                policy.scaling_factor,
                policy.shift_factor,
                len(latents_mean_values),
            )
        else:
            logger.info(
                "[VAE] normalization enabled: scaling_factor=%s shift_factor=%s",
                policy.scaling_factor,
                policy.shift_factor,
            )
        return _NormalizingFirstStage(
            base,
            scale=float(policy.scaling_factor),
            shift=policy.shift_factor,
            latents_mean=latents_mean_values,
            latents_std=latents_std_values,
        )

class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False, *, family=None):
        if no_init:
            return

        self.memory_used_encode = (
            lambda shape, dtype: (1767 * shape[2] * shape[3]) * torch.empty((), dtype=dtype).element_size()
        )
        self.memory_used_decode = (
            lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * torch.empty((), dtype=dtype).element_size()
        )
        self.downscale_ratio = int(2 ** (len(model.config.down_block_types) - 1))
        self.latent_channels = int(model.config.latent_channels)

        # Ensure process_in/out are always available via adapter
        self.first_stage_model = _NormalizingFirstStage.wrap(model.eval(), family=family)

        if device is None:
            device = memory_management.manager.get_device(DeviceRole.VAE)

        self.device = device
        offload_device = memory_management.manager.get_offload_device(DeviceRole.VAE)

        if dtype is None:
            native_storage = None
            try:
                native_storage = next(model.parameters()).dtype
            except Exception:  # noqa: BLE001
                native_storage = None
            if native_storage is None:
                native_storage = torch.float32
            dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=native_storage)

        self.vae_dtype: torch.dtype | None = None
        self.vae_compute_dtype: torch.dtype | None = None
        self._pending_dtype = dtype  # Will be applied lazily when VAE is first used
        self.offload_device = offload_device
        self.output_device = memory_management.manager.get_device(DeviceRole.INTERMEDIATE)

        self.patcher = ModelPatcher(
            self.first_stage_model,
            load_device=self.device,
            offload_device=offload_device
        )

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.vae_compute_dtype = self.vae_compute_dtype
        n.output_device = self.output_device
        return n

    def _resolve_dtypes(self) -> tuple[torch.dtype, torch.dtype]:
        native_storage = self.vae_dtype or self._pending_dtype or torch.float32
        storage_dtype = memory_management.manager.dtype_for_role(DeviceRole.VAE, native_dtype=native_storage)
        compute_dtype = memory_management.manager.compute_dtype_for_role(DeviceRole.VAE, storage_dtype=storage_dtype)
        return storage_dtype, compute_dtype

    def _active_forward_dtype(self) -> torch.dtype:
        if self.vae_dtype is not None:
            return self.vae_dtype
        if self.vae_compute_dtype is not None:
            return self.vae_compute_dtype
        return torch.float32

    def _apply_precision(self, dtype: torch.dtype, device: torch.device | str | None = None) -> None:
        if dtype == self.vae_dtype:
            return
        previous = self.vae_dtype
        target_device = device if device is not None else self.device
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        base.to(device=target_device, dtype=dtype)
        self.vae_dtype = dtype
        logger.info(
            "VAE precision updated: %s -> %s on %s",
            "none" if previous is None else str(previous),
            str(dtype),
            target_device,
        )

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        def decode_fn(a: torch.Tensor) -> torch.Tensor:
            forward_dtype = self._active_forward_dtype()
            decoded = self.first_stage_model.decode(a.to(device=self.device, dtype=forward_dtype))
            return (_unwrap_decode_output(decoded) + 1.0).float()

        output = torch.clamp(((tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device))
                              / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        def encode_fn(a: torch.Tensor) -> torch.Tensor:
            forward_dtype = self._active_forward_dtype()
            encoded = self.first_stage_model.encode((2.0 * a - 1.0).to(device=self.device, dtype=forward_dtype))
            return _unwrap_encode_output(encoded).float()

        samples = tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples /= 3.0
        return samples

    def _decode_cpu_fallback(self, samples_in: torch.Tensor) -> torch.Tensor:
        """Best-effort CPU decode path used after CUDA OOM when smart fallback is enabled.

        This bypasses GPU memory heuristics and runs a single full-image decode on CPU,
        restoring the original VAE device afterwards where possible.
        """
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        orig_device: torch.device | None = None
        orig_dtype: torch.dtype | None = None
        try:
            try:
                params = base.parameters()
                first = next(params)
                orig_device = first.device
                orig_dtype = first.dtype
            except Exception:  # noqa: BLE001
                orig_device = None
                orig_dtype = None

            cpu_device = memory_management.manager.cpu_device
            base.to(device=cpu_device, dtype=torch.float32)

            with torch.no_grad():
                samples_cpu = samples_in.to(cpu_device, dtype=torch.float32)
                decoded_raw = self.first_stage_model.decode(samples_cpu)
                decoded = _unwrap_decode_output(decoded_raw).to(self.output_device).float()
                pixel_samples = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)

            return pixel_samples
        finally:
            if orig_device is not None:
                try:
                    base.to(device=orig_device, dtype=orig_dtype or self.vae_dtype or torch.float32)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to restore VAE device after CPU fallback.", exc_info=True)

    def _encode_cpu_fallback(self, pixel_samples_chw: torch.Tensor, regulation) -> torch.Tensor:
        """Best-effort CPU encode path used after CUDA OOM when smart fallback is enabled.

        Mirrors the GPU encode logic but runs entirely on CPU to avoid repeated
        OOM loops on large inputs. Restores the original VAE device afterwards
        when possible.
        """
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        orig_device: torch.device | None = None
        orig_dtype: torch.dtype | None = None
        try:
            try:
                params = base.parameters()
                first = next(params)
                orig_device = first.device
                orig_dtype = first.dtype
            except Exception:  # noqa: BLE001
                orig_device = None
                orig_dtype = None

            cpu_device = memory_management.manager.cpu_device
            base.to(device=cpu_device, dtype=torch.float32)

            with torch.no_grad():
                pixels_cpu = pixel_samples_chw.to(cpu_device, dtype=torch.float32)
                pixels_in = 2.0 * pixels_cpu - 1.0

                if DiffusersAutoencoderKL is not None and isinstance(base, DiffusersAutoencoderKL):
                    encoded_raw = base.encode(pixels_in, return_dict=True)
                elif AutoencoderKL_LDM is not None and isinstance(base, AutoencoderKL_LDM):
                    encoded_raw = base.encode(pixels_in, regulation)
                else:
                    try:
                        encoded_raw = base.encode(pixels_in, regulation)
                    except TypeError:
                        encoded_raw = base.encode(pixels_in)

                if isinstance(encoded_raw, (tuple, list)) and encoded_raw:
                    encoded_raw = encoded_raw[0]
                encoded = _unwrap_encode_output(encoded_raw).to(self.output_device).float()

            return encoded
        finally:
            if orig_device is not None:
                try:
                    base.to(device=orig_device, dtype=orig_dtype or self.vae_dtype or torch.float32)
                except Exception:  # noqa: BLE001
                    logger.warning("Failed to restore VAE device after CPU encode fallback.", exc_info=True)

    def decode_inner(self, samples_in):
        _tensor_stats("decode_inner.latents", samples_in)
        if memory_management.manager.vae_always_tiled:
            return self.decode_tiled(samples_in).to(self.output_device)

        while True:
            desired_storage, desired_compute = self._resolve_dtypes()
            self._apply_precision(desired_storage)
            self.vae_compute_dtype = desired_compute

            try:
                forward_dtype = self._active_forward_dtype()
                memory_used = self.memory_used_decode(samples_in.shape, forward_dtype)
                memory_management.manager.load_models([self.patcher], memory_required=memory_used)
                free_memory = memory_management.manager.get_free_memory(self.device)
                batch_number = max(1, int(free_memory / memory_used))

                pixel_samples = torch.empty(
                    (
                        samples_in.shape[0],
                        3,
                        round(samples_in.shape[2] * self.downscale_ratio),
                        round(samples_in.shape[3] * self.downscale_ratio),
                    ),
                    device=self.output_device,
                )
                for x in range(0, samples_in.shape[0], batch_number):
                    samples = samples_in[x:x + batch_number].to(device=self.device, dtype=forward_dtype)
                    decoded_raw = self.first_stage_model.decode(samples)
                    decoded = _unwrap_decode_output(decoded_raw).to(self.output_device).float()
                    pixel_samples[x:x + batch_number] = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                    _tensor_stats("decode_inner.batch_decoded", decoded)
            except memory_management.manager.oom_exception:
                if smart_fallback_enabled():
                    logger.warning(
                        "VAE decode OOM on %s with tiled=%s; attempting CPU fallback.",
                        self.device,
                        bool(memory_management.manager.vae_always_tiled),
                    )
                    pixel_samples = self._decode_cpu_fallback(samples_in)
                else:
                    print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                    pixel_samples = self.decode_tiled_(samples_in)

            # Return BCHW format in [-1, 1] range directly
            # This is what sampling pipelines expect - no conversion needed in engines
            result = pixel_samples.to(self.output_device)
            result = result * 2.0 - 1.0  # [0,1] → [-1,1]
            _tensor_stats("decode_inner.result", result)
            if torch.isnan(result).any():
                logger.warning(
                    "VAE decode produced NaNs on %s using dtype %s; requesting precision fallback.",
                    self.device,
                    str(self.vae_dtype),
                )
                next_dtype = memory_management.manager.report_precision_failure(
                    DeviceRole.VAE,
                    location="vae.decode",
                    reason="NaN detected in decoded output",
                )
                if next_dtype is None:
                    hint = memory_management.manager.precision_hint(DeviceRole.VAE)
                    raise RuntimeError(
                        f"VAE decode produced NaNs on {self.device} with dtype {self.vae_dtype}. {hint}"
                    )
                del pixel_samples
                del result
                self._apply_precision(next_dtype)
                memory_management.manager.soft_empty_cache(force=True)
                continue

            return result

    def decode(self, samples_in):
        wrapper = self.patcher.model_options.get('model_vae_decode_wrapper', None)
        if wrapper is None:
            return self.decode_inner(samples_in)
        else:
            return wrapper(self.decode_inner, samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        memory_management.manager.load_model(self.patcher)
        desired_storage, desired_compute = self._resolve_dtypes()
        self._apply_precision(desired_storage)
        self.vae_compute_dtype = desired_compute
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        # Return BCHW format in [-1, 1] range like decode_inner
        return output * 2.0 - 1.0

    def encode_inner(self, pixel_samples):
        if memory_management.manager.vae_always_tiled:
            return self.encode_tiled(pixel_samples)

        regulation = self.patcher.model_options.get("model_vae_regulation", None)

        pixel_samples = pixel_samples.movedim(-1, 1)

        while True:
            desired_storage, desired_compute = self._resolve_dtypes()
            self._apply_precision(desired_storage)
            self.vae_compute_dtype = desired_compute

            try:
                forward_dtype = self._active_forward_dtype()
                memory_used = self.memory_used_encode(pixel_samples.shape, forward_dtype)
                memory_management.manager.load_models([self.patcher], memory_required=memory_used)
                free_memory = memory_management.manager.get_free_memory(self.device)
                batch_number = max(1, int(free_memory / memory_used))
                samples = torch.empty(
                    (
                        pixel_samples.shape[0],
                        self.latent_channels,
                        round(pixel_samples.shape[2] // self.downscale_ratio),
                        round(pixel_samples.shape[3] // self.downscale_ratio),
                    ),
                    device=self.output_device,
                )
                for x in range(0, pixel_samples.shape[0], batch_number):
                    pixels_in = (2.0 * pixel_samples[x:x + batch_number] - 1.0).to(
                        device=self.device,
                        dtype=forward_dtype,
                    )
                    base = getattr(self.first_stage_model, "_base", self.first_stage_model)

                    if DiffusersAutoencoderKL is not None and isinstance(base, DiffusersAutoencoderKL):
                        encoded_raw = base.encode(pixels_in, return_dict=True)
                    elif AutoencoderKL_LDM is not None and isinstance(base, AutoencoderKL_LDM):
                        encoded_raw = base.encode(pixels_in, regulation)
                    else:
                        try:
                            encoded_raw = base.encode(pixels_in, regulation)
                        except TypeError:
                            encoded_raw = base.encode(pixels_in)

                    if isinstance(encoded_raw, (tuple, list)) and encoded_raw:
                        encoded_raw = encoded_raw[0]
                    encoded = _unwrap_encode_output(encoded_raw).to(self.output_device).float()
                    samples[x:x + batch_number] = encoded
            except memory_management.manager.oom_exception:
                if smart_fallback_enabled():
                    logger.warning(
                        "VAE encode OOM on %s with tiled=%s; attempting CPU fallback.",
                        self.device,
                        bool(memory_management.manager.vae_always_tiled),
                    )
                    samples = self._encode_cpu_fallback(pixel_samples, regulation)
                else:
                    print("Warning: Ran out of memory when regular VAE encoding, retrying with tiled VAE encoding.")
                    samples = self.encode_tiled_(pixel_samples)

            if torch.isnan(samples).any():
                logger.warning(
                    "VAE encode produced NaNs on %s using dtype %s; requesting precision fallback.",
                    self.device,
                    str(self.vae_dtype),
                )
                next_dtype = memory_management.manager.report_precision_failure(
                    DeviceRole.VAE,
                    location="vae.encode",
                    reason="NaN detected in encoded output",
                )
                if next_dtype is None:
                    hint = memory_management.manager.precision_hint(DeviceRole.VAE)
                    raise RuntimeError(
                        f"VAE encode produced NaNs on {self.device} with dtype {self.vae_dtype}. {hint}"
                    )
                del samples
                self._apply_precision(next_dtype)
                memory_management.manager.soft_empty_cache(force=True)
                continue

            return samples

    def encode(self, pixel_samples):
        wrapper = self.patcher.model_options.get('model_vae_encode_wrapper', None)
        if wrapper is None:
            return self.encode_inner(pixel_samples)
        else:
            return wrapper(self.encode_inner, pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        memory_management.manager.load_model(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        desired_storage, desired_compute = self._resolve_dtypes()
        self._apply_precision(desired_storage)
        self.vae_compute_dtype = desired_compute
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        return samples
