"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Backend inference orchestrator (engine routing + caching + event streaming).
Resolves engines from the registry, loads/unloads per request, fingerprints load-affecting options, purges VRAM on model swaps, and yields typed progress events back to API callers.
On load/execution failures, performs a best-effort purge to release VRAM/RAM so the backend can recover without restart.

Symbols (top-level; keep in sync; no ghosts):
- `InferenceOrchestrator` (class): Routes typed requests to engines; caches loaded engines with option fingerprinting, reloads when overrides
  change (incl. `vae_source`/`tenc_source`), and manages VRAM hygiene across generations (contains nested helpers for option freezing and cache purges).
"""

from __future__ import annotations

import contextlib
import gc
import logging
import threading
import time
from typing import Iterator, Mapping, MutableMapping, Optional

from .engine_interface import BaseInferenceEngine, TaskType
from .exceptions import EngineExecutionError, EngineNotFoundError, EngineLoadError, UnsupportedTaskError
from .registry import EngineRegistry, registry as global_registry
from .requests import InferenceEvent, ProgressEvent


logger = logging.getLogger(__name__)


class InferenceOrchestrator:
    """Routes typed requests to registered engines.

    This orchestrator is intentionally thin. It resolves engine instances via the
    registry, ensures they are loaded for the requested model, and streams
    events back to the caller. It does not persist results or mutate UI state.
    """

    def __init__(
        self,
        registry: Optional[EngineRegistry] = None,
        *,
        enable_cache: bool = True,
    ) -> None:
        self._registry = registry or global_registry
        self._enable_cache = enable_cache
        self._engine_cache: MutableMapping[str, BaseInferenceEngine] = {}
        self._engine_options_fingerprint: MutableMapping[str, object] = {}
        self._last_generation_signature: object | None = None
        self._run_lock = threading.Lock()

    @staticmethod
    def _freeze_engine_options(value: object) -> object:
        """Return a comparable, stable structure for engine option fingerprints."""
        if isinstance(value, dict):
            return tuple((str(k), InferenceOrchestrator._freeze_engine_options(v)) for k, v in sorted(value.items()))
        if isinstance(value, (list, tuple)):
            return tuple(InferenceOrchestrator._freeze_engine_options(v) for v in value)
        if isinstance(value, set):
            return tuple(sorted((InferenceOrchestrator._freeze_engine_options(v) for v in value), key=repr))
        return value

    @staticmethod
    def _reload_fingerprint(engine_options: Mapping[str, object]) -> object:
        """Fingerprint options that change loaded weights/runtime wiring.

        The orchestrator caches engine instances, so changing options like
        text encoder overrides or VAE overrides must trigger a reload even when
        model_ref is unchanged.
        """

        te_override = engine_options.get("text_encoder_override")
        vae_path = engine_options.get("vae_path")
        vae_source = engine_options.get("vae_source")
        tenc_path = engine_options.get("tenc_path")
        tenc_source = engine_options.get("tenc_source")

        # Normalize streaming option key to a single boolean or None.
        streaming_val: object | None
        if "codex_core_streaming" in engine_options:
            streaming_val = bool(engine_options.get("codex_core_streaming"))
        else:
            streaming_val = engine_options.get("core_streaming_enabled")
            streaming_val = None if streaming_val is None else bool(streaming_val)

        relevant = {
            "text_encoder_override": te_override,
            "vae_path": vae_path,
            "vae_source": vae_source,
            "tenc_path": tenc_path,
            "tenc_source": tenc_source,
            "core_streaming_enabled": streaming_val,
        }
        return InferenceOrchestrator._freeze_engine_options(relevant)

    def _generation_signature(
        self,
        engine_key: str,
        model_ref: str,
        engine_options: Mapping[str, object],
    ) -> object:
        relevant = {
            "engine_key": engine_key,
            "model_ref": model_ref,
            "tenc_path": engine_options.get("tenc_path"),
            "text_encoder_override": engine_options.get("text_encoder_override"),
        }
        return InferenceOrchestrator._freeze_engine_options(relevant)

    @staticmethod
    def _scrub_exception_tracebacks(exc: BaseException) -> None:
        """Best-effort traceback scrub to avoid holding large tensors in exception frames.

        Python tracebacks keep references to stack frames (and thus locals). On load/inference
        failures this can pin large CPU tensors/state dicts in memory longer than intended.
        Scrubbing the tracebacks allows GC to reclaim memory after we unload models.
        """

        stack: list[BaseException] = [exc]
        visited: set[int] = set()
        while stack:
            current = stack.pop()
            ident = id(current)
            if ident in visited:
                continue
            visited.add(ident)
            with contextlib.suppress(Exception):
                current.__traceback__ = None
            nested = []
            with contextlib.suppress(Exception):
                if current.__cause__ is not None:
                    nested.append(current.__cause__)
            with contextlib.suppress(Exception):
                if current.__context__ is not None:
                    nested.append(current.__context__)
            stack.extend(nested)

    def _purge_vram(self, *, reason: str, clear_engine_cache: bool = False) -> None:
        for cached_engine in list(self._engine_cache.values()):
            if not clear_engine_cache and not getattr(cached_engine, "_is_loaded", False):
                continue
            try:
                cached_engine.unload()
                cached_engine.mark_unloaded()
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to unload cached engine during VRAM purge: %s", exc, exc_info=True)

        if clear_engine_cache:
            self._engine_cache.clear()
            self._engine_options_fingerprint.clear()
            self._last_generation_signature = None

        self._engine_options_fingerprint.clear()

        try:
            from apps.backend.runtime.memory import memory_management as _mem

            _mem.manager.unload_all_models()
            _mem.manager.soft_empty_cache(force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("VRAM purge via memory manager failed: %s", exc, exc_info=True)

        try:
            gc.collect()
        except Exception:  # pragma: no cover
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:  # pragma: no cover
            pass

        logger.info("VRAM purge complete (%s).", reason)

    def _maybe_purge_vram_for_generation(
        self,
        *,
        engine_key: str,
        model_ref: str,
        engine_options: Mapping[str, object],
    ) -> None:
        signature = self._generation_signature(engine_key, model_ref, engine_options)
        with self._run_lock:
            prev = self._last_generation_signature
            if prev is not None and prev != signature:
                logger.info(
                    "Generation signature changed; purging VRAM before load. engine=%s model=%s",
                    engine_key,
                    model_ref,
                )
                self._purge_vram(reason="checkpoint/text-encoder selection changed")
            self._last_generation_signature = signature

    # ------------------------------------------------------------------
    def run(
        self,
        task: TaskType,
        engine_key: str,
        request: object,
        *,
        model_ref: Optional[str] = None,
        engine_options: Optional[Mapping[str, object]] = None,
    ) -> Iterator[InferenceEvent]:
        start = time.perf_counter()
        normalized_key = engine_key.strip().lower()
        engine_opts = engine_options or {}
        if model_ref is not None:
            self._maybe_purge_vram_for_generation(
                engine_key=normalized_key,
                model_ref=str(model_ref),
                engine_options=engine_opts,
            )
        engine = self._resolve_engine(engine_key, engine_opts)

        logger.info(
            "Orchestrator dispatch: task=%s engine=%s model=%s", task.value, engine_key, model_ref or "default"
        )

        capabilities = engine.capabilities()
        if not capabilities.supports(task):
            raise UnsupportedTaskError(
                f"Engine '{engine_key}' does not support task '{task.value}'. Supported: {capabilities.tasks}"
            )

        if model_ref is not None:
            needs_load = False
            device_mismatch = False
            if not engine._is_loaded:  # noqa: SLF001 (intentional internal check)
                needs_load = True
            else:
                try:
                    cur_model = engine.status().get("model_ref")
                    needs_load = cur_model != model_ref
                except Exception:
                    needs_load = True
                # Reload when load-affecting engine options changed.
                try:
                    fp = self._reload_fingerprint(engine_opts)
                    prev = self._engine_options_fingerprint.get(normalized_key)
                    if prev is not None and prev != fp:
                        needs_load = True
                except Exception:
                    # Fingerprinting is best-effort; never block inference due to introspection.
                    pass
                # Reload if the primary device changed since last load
                try:
                    from apps.backend.runtime.memory import memory_management as _mem
                    desired = _mem.get_torch_device()
                    unet = getattr(engine, 'codex_objects', None)
                    dcur = None
                    if unet is not None:
                        u = getattr(unet, 'unet', None)
                        if hasattr(u, 'parameters'):
                            try:
                                p = next(u.parameters())
                                dcur = p.device
                            except Exception:
                                dcur = getattr(u, 'device', None)
                    device_mismatch = (dcur is not None and getattr(dcur, 'type', None) != getattr(desired, 'type', None))
                except Exception:
                    device_mismatch = False

            if needs_load or device_mismatch:
                try:
                    if device_mismatch and engine._is_loaded:
                        try:
                            engine.unload()
                        except Exception:
                            pass
                    engine.load(model_ref, **engine_opts)
                    engine.mark_loaded()
                    try:
                        self._engine_options_fingerprint[normalized_key] = self._reload_fingerprint(engine_opts)
                    except Exception:
                        self._engine_options_fingerprint.pop(normalized_key, None)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Engine '%s' failed during load (model=%s).", engine_key, model_ref)
                    self._scrub_exception_tracebacks(exc)
                    with contextlib.suppress(Exception):
                        engine.unload()
                        engine.mark_unloaded()
                    self._purge_vram(reason="engine load failure", clear_engine_cache=True)
                    raise EngineLoadError(
                        f"Failed to load engine '{engine_key}' for model '{model_ref}': {exc}"
                    ) from exc

        handler = getattr(engine, task.value, None)
        if handler is None:
            raise UnsupportedTaskError(f"Engine '{engine_key}' is missing handler for task '{task.value}'")

        try:
            yield ProgressEvent(stage="start", percent=0.0, message="Starting inference")
            for event in handler(request):
                yield event
        except UnsupportedTaskError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Engine '%s' failed during '%s'", engine_key, task.value)
            self._scrub_exception_tracebacks(exc)
            with contextlib.suppress(Exception):
                engine.unload()
                engine.mark_unloaded()
            self._purge_vram(reason="engine execution failure", clear_engine_cache=True)
            raise EngineExecutionError(str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - start
            yield ProgressEvent(stage="end", percent=100.0, message="Inference complete", data={"elapsed": elapsed})

    # ------------------------------------------------------------------
    def _resolve_engine(self, engine_key: str, engine_options: Mapping[str, object]) -> BaseInferenceEngine:
        normalized_key = engine_key.strip().lower()
        if self._enable_cache and normalized_key in self._engine_cache:
            return self._engine_cache[normalized_key]

        try:
            # Do not pass engine_options to constructor: options are applied on load()
            engine = self._registry.create(normalized_key)
        except EngineNotFoundError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise EngineExecutionError(f"Failed to create engine '{engine_key}': {exc}") from exc

        if self._enable_cache:
            self._engine_cache[normalized_key] = engine

        return engine

    # ------------------------------------------------------------------
    def evict(self, engine_key: str) -> None:
        normalized_key = engine_key.strip().lower()
        engine = self._engine_cache.pop(normalized_key, None)
        self._engine_options_fingerprint.pop(normalized_key, None)
        if engine is None:
            return
        with contextlib.suppress(Exception):
            engine.unload()
            engine.mark_unloaded()
        logger.info("Evicted engine '%s'", normalized_key)

    def clear_cache(self) -> None:
        for key in list(self._engine_cache.keys()):
            self.evict(key)
