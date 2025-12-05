import torch
import math
import itertools
import logging

try:  # Optional import; diffusers may not be present in minimal environments
    from diffusers.models.autoencoder_kl import AutoencoderKL as DiffusersAutoencoderKL
except Exception:  # noqa: BLE001
    DiffusersAutoencoderKL = None

try:  # Optional; only needed to detect WAN VAEs explicitly
    from apps.backend.runtime.wan22.vae import AutoencoderKLWan
except Exception:  # noqa: BLE001
    AutoencoderKLWan = None

from tqdm import trange
from apps.backend.runtime.memory import memory_management
from apps.backend.runtime.memory.config import DeviceRole
from apps.backend.runtime.memory.smart_offload import smart_fallback_enabled
from .base import ModelPatcher

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
    # Some implementations return a plain tensor or an object with `.sample`
    if hasattr(output, "sample") and torch.is_tensor(getattr(output, "sample")):
        return output.sample
    if torch.is_tensor(output):
        return output
    # Legacy/variant encoders may return tuples like (latents, aux); pick first tensor.
    if isinstance(output, (tuple, list)) and output:
        for item in output:
            if torch.is_tensor(item):
                return item
    # Fallback: caller must handle unexpected shapes/types
    return output


class _NormalizingFirstStage:
    """Adapter that guarantees process_in/out around a diffusers VAE.

    - process_in: (x - shift) * scale
    - process_out: (x / scale) + shift
    Also proxies encode/decode/to/attributes to the wrapped object.
    """

    def __init__(self, base, *, scale: float, shift: float) -> None:
        self._base = base
        self._scale = float(scale)
        self._shift = float(shift)

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
        return (x - self._shift) * self._scale

    def process_out(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("process_out expects a torch.Tensor")
        return (x / self._scale) + self._shift

    def __getattr__(self, name: str):
        # Delegate any other attribute access to the base VAE
        return getattr(self._base, name)

    @staticmethod
    def wrap(base):
        cfg = getattr(base, "config", None)
        if cfg is None:
            raise RuntimeError("VAE model is missing config; cannot determine scaling_factor for normalization")
        # Attempt to fetch values from attribute or mapping-like config
        scale = None
        shift = 0.0
        if hasattr(cfg, "scaling_factor"):
            scale = getattr(cfg, "scaling_factor")
        elif isinstance(cfg, dict):
            scale = cfg.get("scaling_factor")
            shift = cfg.get("shift_factor", 0.0)
        else:
            try:
                scale = cfg.get("scaling_factor")  # type: ignore[attr-defined]
                shift = cfg.get("shift_factor", 0.0)  # type: ignore[attr-defined]
            except Exception:
                scale = None
        if hasattr(cfg, "shift_factor"):
            try:
                shift = float(getattr(cfg, "shift_factor"))
            except Exception:
                shift = 0.0
        if scale is None:
            raise RuntimeError("VAE config missing 'scaling_factor'; engines require normalization semantics")
        logger.info("[VAE] normalization enabled: scaling_factor=%s shift_factor=%s", scale, shift)
        return _NormalizingFirstStage(base, scale=float(scale), shift=float(shift))


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    dims = len(tile)
    output = torch.empty([samples.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), samples.shape[2:])), device=output_device)

    for b in trange(samples.shape[0]):
        s = samples[b:b + 1]
        out = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + list(map(lambda a: round(a * upscale_amount), s.shape[2:])), device=output_device)

        for it in itertools.product(*map(lambda a: range(0, a[0], a[1] - overlap), zip(s.shape[2:], tile))):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap, it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(pos * upscale_amount))
            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)
            feather = round(overlap * upscale_amount)
            for t in range(feather):
                for d in range(2, dims + 2):
                    m = mask.narrow(d, t, 1)
                    m *= ((1.0 / feather) * (t + 1))
                    m = mask.narrow(d, mask.shape[d] - 1 - t, 1)
                    m *= ((1.0 / feather) * (t + 1))

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o += ps * mask
            o_d += mask

        output[b:b + 1] = out / out_div
    return output


def get_tiled_scale_steps(width, height, tile_x, tile_y, overlap):
    return math.ceil((height / (tile_y - overlap))) * math.ceil((width / (tile_x - overlap)))


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap, upscale_amount, out_channels, output_device)


class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False):
        if no_init:
            return

        self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
        self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype)
        self.downscale_ratio = int(2 ** (len(model.config.down_block_types) - 1))
        self.latent_channels = int(model.config.latent_channels)

        # Ensure process_in/out are always available via adapter
        self.first_stage_model = _NormalizingFirstStage.wrap(model.eval())

        if device is None:
            device = memory_management.vae_device()

        self.device = device
        offload_device = memory_management.vae_offload_device()

        if dtype is None:
            dtype = memory_management.vae_dtype(device=device)

        self.vae_dtype: torch.dtype | None = None
        self._apply_precision(dtype)
        self.output_device = memory_management.intermediate_device()

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
        n.output_device = self.output_device
        return n

    def _apply_precision(self, dtype: torch.dtype) -> None:
        if dtype == self.vae_dtype:
            return
        previous = self.vae_dtype
        base = getattr(self.first_stage_model, "_base", self.first_stage_model)
        base.to(device=self.device, dtype=dtype)
        self.vae_dtype = dtype
        logger.info(
            "VAE precision updated: %s -> %s on %s",
            "none" if previous is None else str(previous),
            str(dtype),
            self.device,
        )

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        steps = samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x, tile_y, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += samples.shape[0] * get_tiled_scale_steps(samples.shape[3], samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        decode_fn = lambda a: (_unwrap_decode_output(self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device))) + 1.0).float()
        output = torch.clamp(((tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device) +
                               tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.downscale_ratio, output_device=self.output_device))
                              / 3.0) / 2.0, min=0.0, max=1.0)
        return output

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        steps = pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x, tile_y, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x // 2, tile_y * 2, overlap)
        steps += pixel_samples.shape[0] * get_tiled_scale_steps(pixel_samples.shape[3], pixel_samples.shape[2], tile_x * 2, tile_y // 2, overlap)

        encode_fn = lambda a: _unwrap_encode_output(self.first_stage_model.encode((2. * a - 1.).to(self.vae_dtype).to(self.device))).float()
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

            cpu_device = memory_management.cpu
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

            cpu_device = memory_management.cpu
            base.to(device=cpu_device, dtype=torch.float32)

            with torch.no_grad():
                pixels_cpu = pixel_samples_chw.to(cpu_device, dtype=torch.float32)
                pixels_in = 2.0 * pixels_cpu - 1.0

                if DiffusersAutoencoderKL is not None and isinstance(base, DiffusersAutoencoderKL):
                    encoded_raw = base.encode(pixels_in, return_dict=True)
                elif AutoencoderKLWan is not None and isinstance(base, AutoencoderKLWan):
                    encoded_raw = base.encode(pixels_in, regulation)
                else:
                    try:
                        encoded_raw = base.encode(pixels_in, regulation)
                    except TypeError:
                        encoded_raw = base.encode(pixels_in)

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
        if memory_management.VAE_ALWAYS_TILED:
            return self.decode_tiled(samples_in).to(self.output_device)

        while True:
            desired_dtype = memory_management.vae_dtype(device=self.device)
            self._apply_precision(desired_dtype)

            try:
                memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
                memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
                free_memory = memory_management.get_free_memory(self.device)
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
                    samples = samples_in[x:x + batch_number].to(self.vae_dtype).to(self.device)
                    decoded_raw = self.first_stage_model.decode(samples)
                    decoded = _unwrap_decode_output(decoded_raw).to(self.output_device).float()
                    pixel_samples[x:x + batch_number] = torch.clamp((decoded + 1.0) / 2.0, min=0.0, max=1.0)
                    _tensor_stats("decode_inner.batch_decoded", decoded)
            except memory_management.OOM_EXCEPTION:
                if smart_fallback_enabled():
                    logger.warning(
                        "VAE decode OOM on %s with tiled=%s; attempting CPU fallback.",
                        self.device,
                        bool(memory_management.VAE_ALWAYS_TILED),
                    )
                    pixel_samples = self._decode_cpu_fallback(samples_in)
                else:
                    print("Warning: Ran out of memory when regular VAE decoding, retrying with tiled VAE decoding.")
                    pixel_samples = self.decode_tiled_(samples_in)

            result = pixel_samples.to(self.output_device).movedim(1, -1)
            _tensor_stats("decode_inner.result", result)
            if torch.isnan(result).any():
                logger.warning(
                    "VAE decode produced NaNs on %s using dtype %s; requesting precision fallback.",
                    self.device,
                    str(self.vae_dtype),
                )
                next_dtype = memory_management.report_precision_failure(
                    DeviceRole.VAE,
                    location="vae.decode",
                    reason="NaN detected in decoded output",
                )
                if next_dtype is None:
                    hint = memory_management.precision_hint(DeviceRole.VAE)
                    raise RuntimeError(
                        f"VAE decode produced NaNs on {self.device} with dtype {self.vae_dtype}. {hint}"
                    )
                del pixel_samples
                del result
                self._apply_precision(next_dtype)
                memory_management.soft_empty_cache(force=True)
                continue

            return result

    def decode(self, samples_in):
        wrapper = self.patcher.model_options.get('model_vae_decode_wrapper', None)
        if wrapper is None:
            return self.decode_inner(samples_in)
        else:
            return wrapper(self.decode_inner, samples_in)

    def decode_tiled(self, samples, tile_x=64, tile_y=64, overlap=16):
        memory_management.load_model_gpu(self.patcher)
        output = self.decode_tiled_(samples, tile_x, tile_y, overlap)
        return output.movedim(1, -1)

    def encode_inner(self, pixel_samples):
        if memory_management.VAE_ALWAYS_TILED:
            return self.encode_tiled(pixel_samples)

        regulation = self.patcher.model_options.get("model_vae_regulation", None)

        pixel_samples = pixel_samples.movedim(-1, 1)

        while True:
            desired_dtype = memory_management.vae_dtype(device=self.device)
            self._apply_precision(desired_dtype)

            try:
                memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
                memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
                free_memory = memory_management.get_free_memory(self.device)
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
                    pixels_in = (2.0 * pixel_samples[x:x + batch_number] - 1.0).to(self.vae_dtype).to(self.device)
                    base = getattr(self.first_stage_model, "_base", self.first_stage_model)

                    if DiffusersAutoencoderKL is not None and isinstance(base, DiffusersAutoencoderKL):
                        encoded_raw = base.encode(pixels_in, return_dict=True)
                    elif AutoencoderKLWan is not None and isinstance(base, AutoencoderKLWan):
                        encoded_raw = base.encode(pixels_in, regulation)
                    else:
                        try:
                            encoded_raw = base.encode(pixels_in, regulation)
                        except TypeError:
                            encoded_raw = base.encode(pixels_in)

                    encoded = _unwrap_encode_output(encoded_raw).to(self.output_device).float()
                    samples[x:x + batch_number] = encoded
            except memory_management.OOM_EXCEPTION:
                if smart_fallback_enabled():
                    logger.warning(
                        "VAE encode OOM on %s with tiled=%s; attempting CPU fallback.",
                        self.device,
                        bool(memory_management.VAE_ALWAYS_TILED),
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
                next_dtype = memory_management.report_precision_failure(
                    DeviceRole.VAE,
                    location="vae.encode",
                    reason="NaN detected in encoded output",
                )
                if next_dtype is None:
                    hint = memory_management.precision_hint(DeviceRole.VAE)
                    raise RuntimeError(
                        f"VAE encode produced NaNs on {self.device} with dtype {self.vae_dtype}. {hint}"
                    )
                del samples
                self._apply_precision(next_dtype)
                memory_management.soft_empty_cache(force=True)
                continue

            return samples

    def encode(self, pixel_samples):
        wrapper = self.patcher.model_options.get('model_vae_encode_wrapper', None)
        if wrapper is None:
            return self.encode_inner(pixel_samples)
        else:
            return wrapper(self.encode_inner, pixel_samples)

    def encode_tiled(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        memory_management.load_model_gpu(self.patcher)
        pixel_samples = pixel_samples.movedim(-1, 1)
        samples = self.encode_tiled_(pixel_samples, tile_x=tile_x, tile_y=tile_y, overlap=overlap)
        return samples
