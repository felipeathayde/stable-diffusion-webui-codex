from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence

import safetensors.torch as sf
import torch

from apps.backend.core.engine_interface import BaseInferenceEngine
from apps.backend.runtime.memory.smart_offload import smart_offload_enabled
from apps.backend.runtime.models.loader import DiffusionModelBundle, resolve_diffusion_bundle
from apps.backend.runtime.utils import get_state_dict_after_quant


logger = logging.getLogger("backend.engines.common.base")


@dataclass(slots=True)
class CodexObjects:
    """Container for core diffusion components attached to an engine."""

    unet: Any
    clip: Any
    vae: Any
    clipvision: Any | None = None

    def shallow_copy(self) -> "CodexObjects":
        """Return a shallow copy preserving component references."""
        return CodexObjects(
            unet=self.unet,
            clip=self.clip,
            vae=self.vae,
            clipvision=self.clipvision,
        )

    def validate(self, context: str) -> None:
        """Ensure all mandatory components are present."""
        if self.unet is None:
            raise ValueError(f"{context}: UNet component is required.")
        if self.clip is None:
            raise ValueError(f"{context}: CLIP component is required.")
        if self.vae is None:
            raise ValueError(f"{context}: VAE component is required.")

    def describe(self) -> dict[str, str]:
        """Return human-readable component metadata for logging."""
        def _name(component: Any) -> str:
            return component.__class__.__name__ if component is not None else "None"

        return {
            "unet": _name(self.unet),
            "clip": _name(self.clip),
            "vae": _name(self.vae),
            "clipvision": _name(self.clipvision),
        }


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

    def initialize(self, components: CodexObjects, *, context: str) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context)
        self._active = components
        self._original = components.shallow_copy()
        self._after_lora = components.shallow_copy()
        snapshot = components.describe()
        self._logger.debug(
            "Engine components bound (%s): unet=%s clip=%s vae=%s clipvision=%s",
            context,
            snapshot["unet"],
            snapshot["clip"],
            snapshot["vae"],
            snapshot["clipvision"],
        )

    def replace_active(self, components: CodexObjects, *, context: str) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context)
        self._active = components
        self._logger.debug(
            "Engine components replaced (%s): %s", context, components.describe()
        )

    def snapshot_after_lora(self) -> None:
        active = self.require_active()
        self._after_lora = active.shallow_copy()
        snapshot = self._after_lora.describe()
        self._logger.debug(
            "Stored post-LoRA snapshot: unet=%s clip=%s vae=%s clipvision=%s",
            snapshot["unet"],
            snapshot["clip"],
            snapshot["vae"],
            snapshot["clipvision"],
        )

    def set_after_lora(self, components: CodexObjects, *, context: str) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context)
        self._after_lora = components
        self._logger.debug(
            "External post-LoRA snapshot registered (%s): %s",
            context,
            components.describe(),
        )

    def set_original(self, components: CodexObjects, *, context: str) -> None:
        components = self._ensure_codex_objects(components, context)
        components.validate(context)
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

    # ------------------------------------------------------------------ Components
    def bind_components(self, components: CodexObjects, *, label: str | None = None) -> None:
        """Bind engine components and seed original/LoRA snapshots."""
        context = label or self.__class__.__name__
        self._component_tracker.initialize(components, context=context)

    @property
    def codex_objects(self) -> CodexObjects:
        return self._component_tracker.require_active()

    @codex_objects.setter
    def codex_objects(self, value: CodexObjects) -> None:
        self._component_tracker.replace_active(value, context="codex_objects setter")

    @property
    def codex_objects_original(self) -> CodexObjects:
        return self._component_tracker.require_original()

    @codex_objects_original.setter
    def codex_objects_original(self, value: CodexObjects) -> None:
        self._component_tracker.set_original(value, context="codex_objects_original")

    @property
    def codex_objects_after_applying_lora(self) -> CodexObjects:
        return self._component_tracker.require_after_lora()

    @codex_objects_after_applying_lora.setter
    def codex_objects_after_applying_lora(self, value: CodexObjects) -> None:
        self._component_tracker.set_after_lora(value, context="codex_objects_after_applying_lora")

    @property
    def smart_offload_enabled(self) -> bool:
        return self._smart_offload_enabled

    def snapshot_after_lora(self) -> None:
        """Capture the current components as the LoRA-applied snapshot."""
        self._component_tracker.snapshot_after_lora()

    # ------------------------------------------------------------------ Lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        raw_options: dict[str, Any] = dict(options)
        bundle_obj = raw_options.pop("_bundle", None)
        if bundle_obj is None:
            bundle = resolve_diffusion_bundle(model_ref)
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

        components = self._build_components(bundle, options=self._load_options)
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

    def _reset_state(self) -> None:
        self._component_tracker = _ComponentTracker(logger=self._logger)
        self._model_families.clear()
        self._tiling_enabled = False
        self._use_distilled_cfg_scale = False
        self.current_lora_hash = "[]"

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

    # ------------------------------------------------------------------ Legacy attributes (compatibility)
    @property
    def tiling_enabled(self) -> bool:
        return self._tiling_enabled

    @tiling_enabled.setter
    def tiling_enabled(self, enabled: bool) -> None:
        self._tiling_enabled = bool(enabled)
        self._logger.debug("Tiling toggled to %s", self._tiling_enabled)

    @property
    def use_distilled_cfg_scale(self) -> bool:
        return self._use_distilled_cfg_scale

    @use_distilled_cfg_scale.setter
    def use_distilled_cfg_scale(self, enabled: bool) -> None:
        self._use_distilled_cfg_scale = bool(enabled)
        self._logger.debug("Distilled CFG scale toggled to %s", self._use_distilled_cfg_scale)

    @property
    def first_stage_model(self) -> Any:
        vae = getattr(self.codex_objects, "vae", None)
        model = getattr(vae, "first_stage_model", None)
        if model is None:
            raise RuntimeError("VAE first_stage_model is unavailable on this engine.")
        return model

    @first_stage_model.setter
    def first_stage_model(self, value: Any) -> None:
        vae = getattr(self.codex_objects, "vae", None)
        if vae is None:
            raise RuntimeError("Cannot set first_stage_model without a VAE component.")
        setattr(vae, "first_stage_model", value)

    @property
    def cond_stage_model(self) -> Any:
        clip = getattr(self.codex_objects, "clip", None)
        model = getattr(clip, "cond_stage_model", None)
        if model is None:
            raise RuntimeError("CLIP cond_stage_model is unavailable on this engine.")
        return model

    @cond_stage_model.setter
    def cond_stage_model(self, value: Any) -> None:
        clip = getattr(self.codex_objects, "clip", None)
        if clip is None:
            raise RuntimeError("Cannot set cond_stage_model without a CLIP component.")
        setattr(clip, "cond_stage_model", value)

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
