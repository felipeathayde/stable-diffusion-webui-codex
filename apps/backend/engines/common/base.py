from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import safetensors.torch as sf
import torch

from apps.backend.core.engine_interface import BaseInferenceEngine
from apps.backend.runtime.memory.smart_offload import (
    smart_offload_enabled,
    smart_fallback_enabled,
    smart_cache_enabled,
    record_smart_cache_hit,
    record_smart_cache_miss,
)
from apps.backend.runtime.model_registry.specs import ModelFamily
from apps.backend.runtime.models.loader import (
    DiffusionModelBundle,
    TextEncoderOverrideConfig,
    resolve_diffusion_bundle,
)
from apps.backend.runtime.utils import get_state_dict_after_quant
from apps.backend.runtime.utils import load_torch_file
from apps.backend.runtime.models.state_dict import safe_load_state_dict


logger = logging.getLogger("backend.engines.common.base")


@dataclass(slots=True)
class CodexObjects:
    """Container for core diffusion components attached to an engine.
    
    text_encoders is a flexible dict allowing engines to specify their own
    text encoder types (e.g., {"clip": ...}, {"qwen3": ...}, {"clip": ..., "t5": ...}).
    """

    unet: Any
    vae: Any
    text_encoders: dict[str, Any]  # Flexible text encoders dict
    clipvision: Any | None = None

    def shallow_copy(self) -> "CodexObjects":
        """Return a shallow copy preserving component references."""
        return CodexObjects(
            unet=self.unet,
            vae=self.vae,
            text_encoders=dict(self.text_encoders),  # Shallow copy of dict
            clipvision=self.clipvision,
        )

    def validate(self, context: str, *, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        """Ensure all mandatory components are present.
        
        Args:
            context: Error message context.
            required_text_encoders: Tuple of required text encoder names.
        """
        if self.unet is None:
            raise ValueError(f"{context}: UNet component is required.")
        if self.vae is None:
            raise ValueError(f"{context}: VAE component is required.")
        for te_name in required_text_encoders:
            if te_name not in self.text_encoders or self.text_encoders[te_name] is None:
                raise ValueError(f"{context}: '{te_name}' text encoder is required.")

    def describe(self) -> dict[str, str]:
        """Return human-readable component metadata for logging."""
        def _name(component: Any) -> str:
            return component.__class__.__name__ if component is not None else "None"

        result = {
            "unet": _name(self.unet),
            "vae": _name(self.vae),
            "clipvision": _name(self.clipvision),
        }
        # Add text encoders to description
        for te_name, te_obj in self.text_encoders.items():
            result[f"text_encoder.{te_name}"] = _name(te_obj)
        return result


class _ComponentTracker:
    """Tracks active/original/LoRA component snapshots for an engine."""

    def __init__(self, *, logger: logging.Logger) -> None:
        self._logger = logger
        self._active: CodexObjects | None = None
        self._original: CodexObjects | None = None
        self._after_lora: CodexObjects | None = None

    @staticmethod
    def _ensure_codex_objects(value: Any, context: str) -> CodexObjects:
        if not isinstance(value, CodexObjects):
            raise TypeError(f"{context}: expected CodexObjects, received {type(value).__name__}.")
        return value

    def initialize(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._active = components
        self._original = components.shallow_copy()
        self._after_lora = components.shallow_copy()
        snapshot = components.describe()
        self._logger.debug(
            "Engine components bound (%s): unet=%s vae=%s clipvision=%s text_encoders=%s",
            context,
            snapshot["unet"],
            snapshot["vae"],
            snapshot["clipvision"],
            list(components.text_encoders.keys()),
        )

    def replace_active(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._active = components
        self._logger.debug(
            "Engine components replaced (%s): %s", context, components.describe()
        )

    def snapshot_after_lora(self) -> None:
        active = self.require_active()
        self._after_lora = active.shallow_copy()
        snapshot = self._after_lora.describe()
        self._logger.debug(
            "Stored post-LoRA snapshot: unet=%s vae=%s clipvision=%s text_encoders=%s",
            snapshot["unet"],
            snapshot["vae"],
            snapshot["clipvision"],
            list(active.text_encoders.keys()),
        )

    def set_after_lora(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._after_lora = components
        self._logger.debug(
            "External post-LoRA snapshot registered (%s): %s",
            context,
            components.describe(),
        )

    def set_original(self, components: CodexObjects, *, context: str, required_text_encoders: tuple[str, ...] = ("clip",)) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context, required_text_encoders=required_text_encoders)
        self._original = components
        self._logger.debug(
            "Original component snapshot replaced (%s): %s",
            context,
            components.describe(),
        )

    def require_active(self) -> CodexObjects:
        if self._active is None:
            raise RuntimeError("Diffusion engine components have not been bound.")
        return self._active

    def require_original(self) -> CodexObjects:
        if self._original is None:
            raise RuntimeError("Original engine components are unavailable.")
        return self._original

    def require_after_lora(self) -> CodexObjects:
        if self._after_lora is None:
            raise RuntimeError("LoRA-applied engine components are unavailable.")
        return self._after_lora

    def peek_active(self) -> CodexObjects | None:
        return self._active


class CodexDiffusionEngine(BaseInferenceEngine, ABC):
    """Common foundation for Codex diffusion engines."""

    engine_id = "codex.diffusion"
    matched_guesses: tuple[str, ...] = ()

    _MODEL_FAMILY_FLAGS: dict[str, str] = {
        "sd1": "is_sd1",
        "sd2": "is_sd2",
        "sd3": "is_sd3",
        "sdxl": "is_sdxl",
    }

    def __init__(self) -> None:  # noqa: D401
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.model_config: Any | None = None
        self.is_inpaint: bool = False
        self.current_lora_hash = "[]"
        self._component_tracker = _ComponentTracker(logger=self._logger)
        self._model_families: set[str] = set()
        self._tiling_enabled = False
        self._use_distilled_cfg_scale = False
        self._component_source: Mapping[str, Any] | None = None
        self._current_bundle: DiffusionModelBundle | None = None
        self._current_model_ref: str | None = None
        self._load_options: dict[str, Any] = {}
        self._smart_offload_enabled = smart_offload_enabled()
        self._smart_fallback_enabled = smart_fallback_enabled()
        self._smart_cache_enabled = smart_cache_enabled()
        # Conditioning cache: keyed by (prompt_tuple, is_negative) -> dict of tensors
        # Subclasses can use this for caching CLIP/T5/etc outputs.
        # Tensors are stored on CPU to avoid pinning VRAM between jobs.
        self._cond_cache: dict[tuple, dict[str, Any]] = {}

    # ------------------------------------------------------------------ Components
    @property
    def required_text_encoders(self) -> tuple[str, ...]:
        """Text encoders required by this engine. Override in subclasses.
        
        Default is ("clip",) for SD/SDXL compatibility.
        Other engines can override, e.g., ("qwen3",) for Z Image.
        """
        return ("clip",)

    def bind_components(self, components: CodexObjects, *, label: str | None = None) -> None:
        """Bind engine components and seed original/LoRA snapshots."""
        context = label or self.__class__.__name__
        self._component_tracker.initialize(
            components, context=context, required_text_encoders=self.required_text_encoders
        )

    @property
    def codex_objects(self) -> CodexObjects:
        return self._component_tracker.require_active()

    @codex_objects.setter
    def codex_objects(self, value: CodexObjects) -> None:
        self._component_tracker.replace_active(
            value, context="codex_objects setter", required_text_encoders=self.required_text_encoders
        )

    @property
    def codex_objects_original(self) -> CodexObjects:
        return self._component_tracker.require_original()

    @codex_objects_original.setter
    def codex_objects_original(self, value: CodexObjects) -> None:
        self._component_tracker.set_original(
            value, context="codex_objects_original", required_text_encoders=self.required_text_encoders
        )

    @property
    def codex_objects_after_applying_lora(self) -> CodexObjects:
        return self._component_tracker.require_after_lora()

    @codex_objects_after_applying_lora.setter
    def codex_objects_after_applying_lora(self, value: CodexObjects) -> None:
        self._component_tracker.set_after_lora(
            value, context="codex_objects_after_applying_lora", required_text_encoders=self.required_text_encoders
        )

    @property
    def smart_offload_enabled(self) -> bool:
        return self._smart_offload_enabled

    @property
    def smart_fallback_enabled(self) -> bool:
        return self._smart_fallback_enabled

    @property
    def smart_cache_enabled(self) -> bool:
        return self._smart_cache_enabled

    # ------------------------------------------------------------------ Conditioning Cache
    def _get_cached_cond(self, cache_key: tuple, bucket_name: str) -> Optional[dict[str, Any]]:
        """Retrieve cached conditioning if smart cache is enabled and key exists."""
        if not self._smart_cache_enabled:
            return None
        cached = self._cond_cache.get(cache_key)
        if cached is not None:
            record_smart_cache_hit(bucket_name)
            return cached
        record_smart_cache_miss(bucket_name)
        return None

    def _set_cached_cond(self, cache_key: tuple, cond_dict: dict[str, Any]) -> None:
        """Store conditioning in cache (tensors should be on CPU to avoid pinning VRAM)."""
        if not self._smart_cache_enabled:
            return
        # Clear old entries to keep cache bounded
        self._cond_cache.clear()
        self._cond_cache[cache_key] = cond_dict

    def _clear_cond_cache(self) -> None:
        """Clear conditioning cache (called on model reload)."""
        self._cond_cache.clear()

    def snapshot_after_lora(self) -> None:
        """Capture the current components as the LoRA-applied snapshot."""
        self._component_tracker.snapshot_after_lora()

    # ------------------------------------------------------------------ Lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        if self._is_loaded:
            self.unload()
        raw_options: dict[str, Any] = dict(options)
        te_override_raw = raw_options.pop("text_encoder_override", None)
        bundle_obj = raw_options.pop("_bundle", None)
        te_override_cfg: TextEncoderOverrideConfig | None = None
        if te_override_raw is not None:
            if not isinstance(te_override_raw, dict):
                raise TypeError("text_encoder_override must be a mapping when provided.")
            family_raw = str(te_override_raw.get("family") or "").strip()
            label_raw = str(te_override_raw.get("label") or "").strip()
            if not family_raw or not label_raw:
                raise RuntimeError("text_encoder_override requires non-empty 'family' and 'label' fields.")
            try:
                family = ModelFamily(family_raw)
            except ValueError as exc:
                raise RuntimeError(f"Unsupported text encoder override family='{family_raw}'") from exc
            components_val = te_override_raw.get("components")
            components: tuple[str, ...] | None
            explicit_paths: dict[str, str] | None = None
            if components_val is None:
                components = None
            elif isinstance(components_val, (list, tuple)):
                aliases: list[str] = []
                paths: dict[str, str] = {}
                for raw in components_val:
                    s = str(raw).strip()
                    if not s:
                        continue
                    # Support \"alias=path\" entries for explicit path overrides (e.g., Flux).
                    if "=" in s:
                        alias, path = s.split("=", 1)
                        alias = alias.strip()
                        path = path.strip()
                        if alias and path:
                            aliases.append(alias)
                            paths[alias] = path
                    else:
                        aliases.append(s)
                # De-duplicate aliases while preserving order.
                seen: set[str] = set()
                ordered_aliases: list[str] = []
                for alias in aliases:
                    if alias not in seen:
                        seen.add(alias)
                        ordered_aliases.append(alias)
                components = tuple(ordered_aliases) if ordered_aliases else None
                explicit_paths = paths or None
            else:
                raise RuntimeError("text_encoder_override.components must be an array of strings when provided.")
            te_override_cfg = TextEncoderOverrideConfig(
                family=family,
                root_label=label_raw,
                components=components,
                explicit_paths=explicit_paths,
            )
        if bundle_obj is None:
            bundle = resolve_diffusion_bundle(model_ref, text_encoder_override=te_override_cfg)
        elif isinstance(bundle_obj, DiffusionModelBundle):
            bundle = bundle_obj
        else:
            raise TypeError("_bundle must be a DiffusionModelBundle when provided.")

        try:
            comp_keys = sorted(getattr(bundle, "components", {}).keys())  # type: ignore[arg-type]
        except Exception:
            comp_keys = []

        self._logger.info("[engine] Loading %s (ref=%s, source=%s)", self.engine_id, model_ref, bundle.source)
        self._reset_state()

        self._current_bundle = bundle
        self._current_model_ref = model_ref
        self._component_source = bundle.components
        self._load_options = raw_options

        self.model_config = bundle.estimated_config
        try:
            self.is_inpaint = bool(bundle.estimated_config.inpaint_model())
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Failed to determine inpaint capability.") from exc

        # Map UI setting name to engine option name for streaming
        if "codex_core_streaming" in self._load_options:
            self._load_options["core_streaming_enabled"] = bool(self._load_options.pop("codex_core_streaming"))

        components = self._build_components(bundle, options=self._load_options)

        # Optional VAE override: explicit user path has priority over bundled VAE.
        override_vae_path = self._load_options.get("vae_path")
        if override_vae_path:
            if not os.path.isfile(override_vae_path):
                raise FileNotFoundError(f"vae_path '{override_vae_path}' does not exist.")
            if getattr(components, "vae", None) is None:
                raise RuntimeError(
                    "vae_path was provided, but no VAE component was built from the checkpoint; "
                    "cannot apply override."
                )
            vae_device = getattr(components.vae, "device", None) or getattr(components.vae, "load_device", None)
            state_dict = load_torch_file(override_vae_path, device=vae_device)
            missing, unexpected = safe_load_state_dict(components.vae, state_dict, log_name="VAE override")
            if missing:
                sample = missing[:10]
                raise RuntimeError(
                    f"VAE override is missing {len(missing)} keys; sample={sample}. "
                    "Ensure the override matches the SDXL VAE architecture."
                )
            if unexpected:
                logger.warning("VAE override: unexpected %d keys (sample=%s)", len(unexpected), unexpected[:10])

        # Optional: best-effort probe for internal diagnostics (no console output)
        def _probe_device_dtype(obj):
            try:
                # Prefer nested .model/.diffusion_model when available
                candidate = getattr(obj, 'model', obj)
                candidate = getattr(candidate, 'diffusion_model', candidate)
                params = getattr(candidate, 'parameters', None)
                if callable(params):
                    it = params()
                    t = next(it)
                    return getattr(t, 'device', None), getattr(t, 'dtype', None)
            except Exception:
                return None, None
            return getattr(candidate, 'device', None), getattr(candidate, 'dtype', None)
        _ = _probe_device_dtype(getattr(components, 'unet', None))
        _ = _probe_device_dtype(getattr(components, 'clip', None))
        _ = _probe_device_dtype(getattr(components, 'vae', None))
        self.bind_components(components, label=self.engine_id)
        self.snapshot_after_lora()

        self.mark_loaded()
        self._logger.info(
            "[engine] Loaded %s (families=%s)",
            self.engine_id,
            ",".join(sorted(self._model_families)) or "unknown",
        )

    def unload(self) -> None:  # type: ignore[override]
        if not self._is_loaded:
            return
        self._logger.info("[engine] Unloading %s", self.engine_id)
        try:
            self._unload_bound_models_from_memory_manager()
        except Exception:
            self._logger.debug("Failed to unload bound models from memory manager", exc_info=True)
        try:
            self._on_unload()
        finally:
            self._reset_state()
            self.model_config = None
            self.is_inpaint = False
            self._component_source = None
            self._current_bundle = None
            self._current_model_ref = None
            self._load_options = {}
            self.mark_unloaded()

    def _unload_bound_models_from_memory_manager(self) -> None:
        """Best-effort unload of currently bound heavy components.

        The orchestrator caches engine instances, and engines may be reloaded
        when model refs or load-affecting options change. When that happens we
        must drop references held by the memory manager (loaded-model records),
        otherwise repeated reloads can accumulate duplicate model instances.
        """

        from apps.backend.runtime.memory import memory_management

        components = self._component_tracker.peek_active()
        if components is None:
            return

        targets: list[object] = []
        # These wrappers are passed directly into memory_management.load_model_gpu(...)
        targets.append(getattr(components, "unet", None))
        targets.append(getattr(components, "vae", None))
        targets.append(getattr(components, "clipvision", None))

        # Text encoders vary by engine; prefer `.patcher` when present.
        try:
            for obj in (getattr(components, "text_encoders", {}) or {}).values():
                if obj is None:
                    continue
                targets.append(getattr(obj, "patcher", obj))
        except Exception:
            pass

        for target in targets:
            if target is None:
                continue
            try:
                memory_management.unload_model_clones(target)
            except Exception:
                pass
            try:
                memory_management.unload_model(target)
            except Exception:
                pass

    def _reset_state(self) -> None:
        self._component_tracker = _ComponentTracker(logger=self._logger)
        self._model_families.clear()
        self._tiling_enabled = False
        self._use_distilled_cfg_scale = False
        self.current_lora_hash = "[]"
        self._cond_cache.clear()

    @abstractmethod
    def _build_components(
        self,
        bundle: DiffusionModelBundle,
        *,
        options: Mapping[str, Any],
    ) -> CodexObjects:
        """Construct CodexObjects for the provided bundle."""

    def _on_unload(self) -> None:
        """Subclass hook to release additional state on unload."""
        return None

    @property
    def model_ref(self) -> Optional[str]:
        return self._current_model_ref

    def status(self) -> Mapping[str, Any]:  # type: ignore[override]
        data = dict(super().status())
        if self._current_model_ref is not None:
            data["model_ref"] = self._current_model_ref
        if self._current_bundle is not None:
            data["bundle_source"] = self._current_bundle.source
        if self._model_families:
            data["families"] = tuple(sorted(self._model_families))
        return data

    # ------------------------------------------------------------------ Model families
    def register_model_family(self, family: str) -> None:
        """Tag the engine with a known model family (sd1/sd2/sd3/sdxl)."""
        if family not in self._MODEL_FAMILY_FLAGS:
            raise ValueError(f"Unsupported model family '{family}'.")
        self._model_families.add(family)
        self._logger.debug(
            "Model family registered: %s (now %s)",
            family,
            sorted(self._model_families),
        )

    @property
    def model_families(self) -> Sequence[str]:
        return tuple(sorted(self._model_families))

    @property
    def is_sd1(self) -> bool:
        return "sd1" in self._model_families

    @property
    def is_sd2(self) -> bool:
        return "sd2" in self._model_families

    @property
    def is_sd3(self) -> bool:
        return "sd3" in self._model_families

    @property
    def is_sdxl(self) -> bool:
        return "sdxl" in self._model_families

    def is_webui_legacy_model(self) -> bool:
        legacy = {"sd1", "sd2", "sd3", "sdxl"}
        return any(family in legacy for family in self._model_families)

    # ------------------------------------------------------------------ Engine flags
    @property
    def use_distilled_cfg_scale(self) -> bool:
        return self._use_distilled_cfg_scale

    @use_distilled_cfg_scale.setter
    def use_distilled_cfg_scale(self, enabled: bool) -> None:
        self._use_distilled_cfg_scale = bool(enabled)
        self._logger.debug("Distilled CFG scale toggled to %s", self._use_distilled_cfg_scale)

    # ------------------------------------------------------------------ Abstract hooks
    def set_clip_skip(self, clip_skip: int) -> None:
        raise NotImplementedError(f"{self.__class__.__name__}.set_clip_skip must be implemented.")

    def get_first_stage_encoding(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def get_learned_conditioning(self, prompt: Iterable[str]) -> Any:
        raise NotImplementedError(f"{self.__class__.__name__}.get_learned_conditioning must be implemented.")

    def encode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__}.encode_first_stage must be implemented.")

    def decode_first_stage(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(f"{self.__class__.__name__}.decode_first_stage must be implemented.")

    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        raise NotImplementedError(f"{self.__class__.__name__}.get_prompt_lengths_on_ui must be implemented.")

    # ------------------------------------------------------------------ Tasks
    def txt2img(self, request: Any, **kwargs: Any) -> Iterable[Any]:
        """Generic txt2img implementation using the staged pipeline runner.
        
        This default implementation uses the same pipeline as SDXL.
        Subclasses can override for custom behavior.
        
        Required engine methods:
        - get_learned_conditioning(prompts) -> conditioning dict
        - decode_first_stage(latents) -> decoded tensor
        """
        import json
        import secrets
        import threading
        import time
        
        from apps.backend.core.requests import Txt2ImgRequest, ProgressEvent, ResultEvent
        from apps.backend.core.state import state as backend_state
        from apps.backend.engines.util.adapters import build_txt2img_processing
        from apps.backend.use_cases.txt2img import generate_txt2img as _generate_txt2img
        from apps.backend.runtime.processing.conditioners import decode_latent_batch
        from apps.backend.runtime.workflows.common import latents_to_pil
        from apps.backend.runtime.text_processing import last_extra_generation_params
        
        self.ensure_loaded()

        if not isinstance(request, Txt2ImgRequest):
            raise TypeError(f"{self.__class__.__name__}.txt2img expects Txt2ImgRequest")

        # Build processing descriptor from request
        raw_seed = int(getattr(request, "seed", -1) or -1)
        if raw_seed < 0:
            raw_seed = secrets.randbits(32) & 0x7FFFFFFF

        proc = build_txt2img_processing(request)
        proc.sd_model = self
        proc.seed = raw_seed
        proc.seeds = [raw_seed]
        proc.subseed = -1
        proc.subseeds = [-1]

        # Defer conditioning to the pipeline runner
        prompt_texts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
        prompts = prompt_texts
        seeds = [raw_seed]
        subseeds = [-1]
        subseed_strength = 0.0
        cond = None
        uncond = None

        # Run pipeline on a worker thread while streaming progress
        result: dict[str, Any] = {"latents": None, "error": None}
        sampling_times: dict[str, float | None] = {"start": None, "end": None}
        done = threading.Event()

        def _worker() -> None:
            try:
                sampling_times["start"] = time.perf_counter()
                result["latents"] = _generate_txt2img(
                    processing=proc,
                    conditioning=cond,
                    unconditional_conditioning=uncond,
                    seeds=seeds,
                    subseeds=subseeds,
                    subseed_strength=subseed_strength,
                    prompts=prompts,
                )
            except Exception as _exc:
                result["error"] = _exc
            finally:
                sampling_times["end"] = time.perf_counter()
                done.set()

        threading.Thread(target=_worker, name=f"{self.engine_id}-txt2img-worker", daemon=True).start()

        t0 = time.perf_counter()
        last_step = -1
        while not done.is_set():
            try:
                step = int(getattr(backend_state, "sampling_step", 0) or 0)
                total = int(getattr(backend_state, "sampling_steps", 0) or 0)
            except Exception:
                step, total = 0, 0
            if total > 0 and step != last_step:
                elapsed = time.perf_counter() - t0
                eta = (elapsed * (total - step) / max(step, 1)) if step > 0 else None
                pct = max(5.0, min(99.0, (step / total) * 100.0))
                yield ProgressEvent(stage="sampling", percent=pct, step=step, total_steps=total, eta_seconds=eta)
                last_step = step
            time.sleep(0.12)

        if result["error"] is not None:
            raise result["error"]
        latents = result["latents"]

        if not isinstance(latents, torch.Tensor):
            raise RuntimeError(
                f"txt2img pipeline returned {type(latents).__name__}, expected torch.Tensor (latents)"
            )

        # Decode to RGB and package result
        decode_start = time.perf_counter()
        # Check if bypass already decoded (latents has _already_decoded flag)
        if getattr(latents, "_already_decoded", False):
            decoded = latents  # Skip decode, already RGB tensor
        else:
            decoded = decode_latent_batch(self, latents)
        images = latents_to_pil(decoded)
        decode_end = time.perf_counter()

        # Build result metadata
        try:
            primary_prompt = getattr(proc, "primary_prompt", proc.prompt)
        except Exception:
            primary_prompt = str(getattr(proc, "prompt", ""))

        try:
            primary_negative = getattr(proc, "primary_negative_prompt", proc.negative_prompt)
        except Exception:
            primary_negative = str(getattr(proc, "negative_prompt", ""))

        all_seeds = list(getattr(proc, "all_seeds", []) or [])
        seed_value = None
        if all_seeds:
            try:
                seed_value = int(all_seeds[0])
            except Exception:
                seed_value = None
        else:
            raw_seed = getattr(proc, "seed", None)
            if raw_seed is not None:
                try:
                    seed_value = int(raw_seed)
                except Exception:
                    seed_value = None

        extra_params: dict[str, object] = {}
        try:
            extra_params.update(last_extra_generation_params)
            extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
        except Exception:
            extra_params = getattr(proc, "extra_generation_params", {}) or {}

        info: dict[str, object] = {
            "engine": self.engine_id,
            "task": "txt2img",
            "width": int(proc.width),
            "height": int(proc.height),
            "steps": int(proc.steps),
            "guidance_scale": float(proc.guidance_scale),
            "sampler": str(getattr(proc, "sampler_name", "Automatic") or "Automatic"),
            "scheduler": str(getattr(proc, "scheduler", "Automatic") or "Automatic"),
        }
        if primary_prompt:
            info["prompt"] = str(primary_prompt)
        if primary_negative:
            info["negative_prompt"] = str(primary_negative)
        if seed_value is not None:
            info["seed"] = int(seed_value)
        if all_seeds:
            info["all_seeds"] = [int(s) for s in all_seeds]
        if extra_params:
            info["extra"] = extra_params
        
        timings: dict[str, float] = {}
        try:
            if sampling_times["start"] is not None and sampling_times["end"] is not None:
                timings["sampling_ms"] = max(0.0, (sampling_times["end"] - sampling_times["start"]) * 1000.0)
            timings["decode_ms"] = max(0.0, (decode_end - decode_start) * 1000.0)
            info["timings_ms"] = timings
        except Exception:
            pass

        # Post-job cleanup hook
        self._post_txt2img_cleanup()

        yield ResultEvent(payload={"images": images, "info": json.dumps(info)})

    def img2img(self, request: Any, **kwargs: Any) -> Iterable[Any]:
        """Generic img2img implementation over the native workflow helpers.

        Engines that need a different img2img contract (e.g. Flux Kontext) should
        override this method.
        """
        import json
        import secrets
        import threading
        import time

        from apps.backend.core.requests import Img2ImgRequest, ProgressEvent, ResultEvent
        from apps.backend.core.state import state as backend_state
        from apps.backend.engines.util.adapters import build_img2img_processing
        from apps.backend.use_cases.img2img import generate_img2img as _generate_img2img
        from apps.backend.runtime.processing.conditioners import decode_latent_batch
        from apps.backend.runtime.workflows.common import latents_to_pil
        from apps.backend.runtime.text_processing import last_extra_generation_params

        self.ensure_loaded()

        if not isinstance(request, Img2ImgRequest):
            raise TypeError(f"{self.__class__.__name__}.img2img expects Img2ImgRequest")

        raw_seed = int(getattr(request, "seed", -1) or -1)
        if raw_seed < 0:
            raw_seed = secrets.randbits(32) & 0x7FFFFFFF

        proc = build_img2img_processing(request)
        proc.sd_model = self
        proc.seed = raw_seed
        proc.seeds = [raw_seed]
        proc.subseed = -1
        proc.subseeds = [-1]

        prompt_texts = list(getattr(proc, "prompts", []) or []) or [proc.prompt]
        prompts = prompt_texts

        result: dict[str, Any] = {"latents": None, "error": None}
        sampling_times: dict[str, float | None] = {"start": None, "end": None}
        done = threading.Event()

        def _worker() -> None:
            try:
                sampling_times["start"] = time.perf_counter()
                result["latents"] = _generate_img2img(
                    processing=proc,
                    conditioning=None,
                    unconditional_conditioning=None,
                    prompts=prompts,
                )
            except Exception as _exc:
                result["error"] = _exc
            finally:
                sampling_times["end"] = time.perf_counter()
                done.set()

        threading.Thread(target=_worker, name=f"{self.engine_id}-img2img-worker", daemon=True).start()

        t0 = time.perf_counter()
        last_step = -1
        while not done.is_set():
            try:
                step = int(getattr(backend_state, "sampling_step", 0) or 0)
                total = int(getattr(backend_state, "sampling_steps", 0) or 0)
            except Exception:
                step, total = 0, 0
            if total > 0 and step != last_step:
                elapsed = time.perf_counter() - t0
                eta = (elapsed * (total - step) / max(step, 1)) if step > 0 else None
                pct = max(5.0, min(99.0, (step / total) * 100.0))
                yield ProgressEvent(stage="sampling", percent=pct, step=step, total_steps=total, eta_seconds=eta)
                last_step = step
            time.sleep(0.12)

        if result["error"] is not None:
            raise result["error"]
        latents = result["latents"]

        if not isinstance(latents, torch.Tensor):
            raise RuntimeError(
                f"img2img pipeline returned {type(latents).__name__}, expected torch.Tensor (latents)"
            )

        decode_start = time.perf_counter()
        decoded = decode_latent_batch(self, latents)
        images = latents_to_pil(decoded)
        decode_end = time.perf_counter()

        extra_params: dict[str, object] = {}
        try:
            extra_params.update(last_extra_generation_params)
            extra_params.update(getattr(proc, "extra_generation_params", {}) or {})
        except Exception:
            extra_params = getattr(proc, "extra_generation_params", {}) or {}

        info: dict[str, object] = {
            "engine": self.engine_id,
            "task": "img2img",
            "width": int(proc.width),
            "height": int(proc.height),
            "steps": int(proc.steps),
            "guidance_scale": float(proc.guidance_scale),
            "denoise_strength": float(getattr(proc, "denoising_strength", 0.0) or 0.0),
            "sampler": str(getattr(proc, "sampler_name", "Automatic") or "Automatic"),
            "scheduler": str(getattr(proc, "scheduler", "Automatic") or "Automatic"),
        }
        if getattr(proc, "prompt", None):
            info["prompt"] = str(getattr(proc, "prompt", ""))
        if getattr(proc, "negative_prompt", None):
            info["negative_prompt"] = str(getattr(proc, "negative_prompt", ""))
        info["seed"] = int(raw_seed)
        if extra_params:
            info["extra"] = extra_params

        timings: dict[str, float] = {}
        try:
            if sampling_times["start"] is not None and sampling_times["end"] is not None:
                timings["sampling_ms"] = max(0.0, (sampling_times["end"] - sampling_times["start"]) * 1000.0)
            timings["decode_ms"] = max(0.0, (decode_end - decode_start) * 1000.0)
            info["timings_ms"] = timings
        except Exception:
            pass

        self._post_txt2img_cleanup()

        yield ResultEvent(payload={"images": images, "info": json.dumps(info)})

    def _post_txt2img_cleanup(self) -> None:
        """Post-job cleanup when smart offload is enabled.

        Keeps UNet resident but nudges CUDA to release unused cached memory so the
        next job starts from a clean allocator state without paying reload cost.
        """
        if not self.smart_offload_enabled:
            return
        try:
            from apps.backend.runtime.memory import memory_management
            memory_management.soft_empty_cache(force=True)
        except Exception:  # pragma: no cover - diagnostics only
            self._logger.debug("Post-job cleanup failed", exc_info=True)


    # ------------------------------------------------------------------ Persistence helpers
    def save_unet(self, filename: str) -> str:
        """Persist the current UNet weights to a safetensors file."""
        components = self.codex_objects
        unet = getattr(components.unet, "model", None)
        if unet is None:
            raise RuntimeError("UNet patcher is unavailable; cannot export weights.")
        diffusion = getattr(unet, "diffusion_model", None)
        if diffusion is None:
            raise RuntimeError("UNet diffusion model missing; export aborted.")

        state_dict = get_state_dict_after_quant(diffusion)
        sf.save_file(state_dict, filename)
        self._logger.info("UNet weights saved to %s (%d tensors)", filename, len(state_dict))
        return filename


__all__ = ["CodexObjects", "CodexDiffusionEngine"]
