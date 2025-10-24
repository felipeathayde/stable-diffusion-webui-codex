from __future__ import annotations

import os
import logging
from dataclasses import dataclass
from typing import Any, Optional

from apps.server.backend.core.exceptions import EngineLoadError
from apps.server.backend.runtime.utils import _load_gguf_state_dict


@dataclass
class WanComponents:
    # Placeholders for actual model components once integrated
    text_encoder: Any | None = None
    transformer: Any | None = None
    vae: Any | None = None
    pipeline: Any | None = None
    pipeline_high: Any | None = None
    pipeline_low: Any | None = None
    model_dir: str | None = None
    high_dir: str | None = None
    low_dir: str | None = None
    device: str = "cpu"
    dtype: str = "fp16"


class WanLoader:
    """Lightweight loader for WAN 2.2 components.

    For MVP we only verify that a model directory exists. Real component
    construction will be added in the next iteration.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or logging.getLogger(self.__class__.__name__)
        self._components: Optional[WanComponents] = None

    def load(self, model_ref: str, *, device: str = "auto", dtype: str = "fp16", allow_download: bool = False) -> WanComponents:
        if not model_ref or not isinstance(model_ref, str):
            raise EngineLoadError("WAN loader requires a non-empty model_ref (path or repo id)")

        # Resolve local path; do not attempt remote download in MVP
        maybe_path = model_ref
        # If a file path was provided (e.g., HighNoise .gguf), use its directory
        try:
            if os.path.isfile(maybe_path):
                maybe_path = os.path.dirname(maybe_path)
        except Exception:
            pass
        if not os.path.isabs(maybe_path):
            # Allow relative path from repo root
            maybe_path = os.path.abspath(maybe_path)

        if not os.path.isdir(maybe_path):
            # Common local layout hint
            alt = os.path.abspath(os.path.join("models", "Wan", model_ref))
            if os.path.isdir(alt):
                maybe_path = alt
            else:
                raise EngineLoadError(
                    (
                        "WAN 2.2 model path not found. Provided: %s. "
                        "Place weights under 'models/Wan/<dir>' and set Quicksettings Checkpoint to that folder, "
                        "or pass an absolute path as model_ref."
                    )
                    % model_ref
                )

        resolved_device = self._resolve_device(device)
        resolved_dtype = dtype.lower()
        if resolved_dtype not in ("fp16", "bf16", "fp32"):
            resolved_dtype = "fp16"

        # Try loading via Diffusers WanPipeline if available (local files only unless explicitly allowed)
        pipe = None
        vae = None
        try:
            from diffusers import AutoencoderKLWan  # type: ignore
            from diffusers import WanPipeline  # type: ignore
            import torch  # type: ignore

            torch_dtype = {
                "fp16": torch.float16,
                "bf16": getattr(torch, "bfloat16", torch.float16),
                "fp32": torch.float32,
            }[resolved_dtype]

            vae = AutoencoderKLWan.from_pretrained(
                maybe_path,
                subfolder="vae",
                torch_dtype=torch_dtype,
                local_files_only=not allow_download,
            )

            pipe = WanPipeline.from_pretrained(
                maybe_path,
                torch_dtype=torch_dtype,
                vae=vae,
                local_files_only=not allow_download,
            )

            target_device = "cuda" if resolved_device == "cuda" and torch.cuda.is_available() else "cpu"
            pipe = pipe.to(target_device)
            # Apply attention backend preference (torch-sdpa/xformers/sage)
            try:
                from apps.server.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as apply_attn  # type: ignore
                from apps.server.backend.engines.util.accelerator import apply_to_diffusers_pipeline as apply_accel  # type: ignore
                apply_attn(pipe, logger=self._logger)
                apply_accel(pipe, logger=self._logger)
            except Exception:
                pass
            self._logger.info("WAN diffusers pipeline loaded: %s (device=%s, dtype=%s)", maybe_path, target_device, resolved_dtype)
        except Exception as exc:
            # Keep components None; engine will raise actionable error on use
            self._logger.warning(
                "WAN diffusers pipeline not available from %s (local-only=%s): %s",
                maybe_path,
                str(not allow_download),
                exc,
            )

        comp = WanComponents(
            text_encoder=None,
            transformer=None,
            vae=vae,
            pipeline=pipe,
            model_dir=maybe_path,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        # If there is no Diffusers pipeline but the directory contains GGUF
        # files, enable GGUF-only mode using this single folder for both
        # stages. The executor will pick High/Low by filename (e.g., '*HighNoise*').
        try:
            if comp.pipeline is None and self._contains_gguf(maybe_path):
                comp.high_dir = maybe_path
                comp.low_dir = maybe_path
                self._logger.info(
                    "WAN GGUF detected in model_dir=%s; enabling GGUF-only path (single folder)",
                    maybe_path,
                )
        except Exception:
            # Non-fatal: continue; engine will surface actionable errors later
            pass

        self._components = comp
        return self._components

    def load_stages(self, high_dir: str, low_dir: str, *, device: str = "auto", dtype: str = "fp16", allow_download: bool = False, prefer_format: str | None = None) -> WanComponents:
        comp = self._components or WanComponents()
        resolved_device = self._resolve_device(device)
        resolved_dtype = dtype.lower() if dtype.lower() in ("fp16", "bf16", "fp32") else "fp16"
        try:
            from diffusers import WanPipeline  # type: ignore
            from diffusers import AutoencoderKLWan  # type: ignore
            import torch  # type: ignore

            torch_dtype = {
                "fp16": torch.float16,
                "bf16": getattr(torch, "bfloat16", torch.float16),
                "fp32": torch.float32,
            }[resolved_dtype]

            def _normalize_dir(p: str) -> str:
                try:
                    if os.path.isfile(p):
                        p = os.path.dirname(p)
                except Exception:
                    pass
                return p if os.path.isabs(p) else os.path.abspath(p)

            def _load_dir(d: str):
                path = _normalize_dir(d)
                if not os.path.isdir(path):
                    raise EngineLoadError(f"WAN stage path not found: {d}")
                vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch_dtype, local_files_only=not allow_download)
                pipe = WanPipeline.from_pretrained(path, torch_dtype=torch_dtype, vae=vae, local_files_only=not allow_download)
                target_device = "cuda" if resolved_device == "cuda" and torch.cuda.is_available() else "cpu"
                pipe = pipe.to(target_device)
                try:
                    from ..util.attention_backend import apply_to_diffusers_pipeline as apply_attn  # type: ignore
                    from ..util.accelerator import apply_to_diffusers_pipeline as apply_accel  # type: ignore
                    apply_attn(pipe, logger=self._logger)
                    apply_accel(pipe, logger=self._logger)
                except Exception:
                    pass
                return pipe, path

            # Format selection: 'gguf' | 'diffusers' | None (auto)
            # Normalize potential file paths to directories for GGUF checks
            high_dir = _normalize_dir(high_dir)
            low_dir = _normalize_dir(low_dir)

            if (prefer_format == 'gguf') or (prefer_format not in ('diffusers', 'gguf') and (self._contains_gguf(high_dir) or self._contains_gguf(low_dir))):
                comp.pipeline_high = None
                comp.pipeline_low = None
                comp.high_dir = os.path.abspath(high_dir)
                comp.low_dir = os.path.abspath(low_dir)
                comp.device = resolved_device
                comp.dtype = resolved_dtype
                # Load minimal GGUF state dict metadata to confirm readability
                try:
                    hi_gguf = self._first_gguf(comp.high_dir)
                    lo_gguf = self._first_gguf(comp.low_dir)
                    _ = _load_gguf_state_dict(hi_gguf) if hi_gguf else None
                    _ = _load_gguf_state_dict(lo_gguf) if lo_gguf else None
                    self._logger.info("WAN GGUF detected: high=%s low=%s", hi_gguf, lo_gguf)
                except Exception as e:
                    raise EngineLoadError(f"Failed to read GGUF state dict(s): {e}")
                self._components = comp
                return comp
            if prefer_format == 'diffusers':
                self._logger.info("WAN format hint: diffusers (skipping GGUF even if present)")

            pipe_hi, path_hi = _load_dir(high_dir)
            pipe_lo, path_lo = _load_dir(low_dir)
            comp.pipeline_high = pipe_hi
            comp.pipeline_low = pipe_lo
            comp.high_dir = path_hi
            comp.low_dir = path_lo
            comp.device = resolved_device
            comp.dtype = resolved_dtype
            self._components = comp
            self._logger.info("WAN stage pipelines loaded: high=%s low=%s", path_hi, path_lo)
            return comp
        except Exception as exc:
            raise EngineLoadError(f"Failed to load WAN stage pipelines: {exc}")

    def components(self) -> WanComponents:
        if self._components is None:
            raise EngineLoadError("WAN components not loaded; call load(model_ref) first")
        return self._components

    # Best-effort LoRA application for diffusers pipelines
    def apply_lora(self, pipe: Any, lora_path: str, weight: float | int | None = None) -> bool:
        try:
            if pipe is None or not lora_path:
                return False
            # Newer diffusers: load_lora_weights + set_adapters
            if hasattr(pipe, 'load_lora_weights'):
                pipe.load_lora_weights(lora_path)
                if hasattr(pipe, 'set_adapters'):
                    # Use single default adapter name if none provided by loader
                    w = float(weight) if weight is not None else 1.0
                    try:
                        pipe.set_adapters(["default"], weights=[w])
                    except Exception:
                        # Some pipelines expect a mapping or different default name; ignore
                        pass
                self._logger.info("Applied LoRA to pipeline: %s (w=%s)", lora_path, weight)
                return True
        except Exception as exc:
            self._logger.warning("Failed to apply LoRA %s: %s", lora_path, exc)
        return False

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device not in ("auto", "cpu", "cuda"):
            return "cpu"
        if device == "auto":
            try:  # local import to avoid hard dependency at import time
                import torch  # type: ignore

                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        return device

    @staticmethod
    def _contains_gguf(path: str) -> bool:
        try:
            if not path:
                return False
            abspath = path if os.path.isabs(path) else os.path.abspath(path)
            for fn in os.listdir(abspath):
                if fn.lower().endswith('.gguf'):
                    return True
        except Exception:
            return False
        return False

    @staticmethod
    def _first_gguf(path: str) -> Optional[str]:
        try:
            abspath = path if os.path.isabs(path) else os.path.abspath(path)
            cands = [os.path.join(abspath, fn) for fn in os.listdir(abspath) if fn.lower().endswith('.gguf')]
            return cands[0] if cands else None
        except Exception:
            return None
