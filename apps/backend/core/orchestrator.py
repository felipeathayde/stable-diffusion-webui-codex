"""High-level orchestrator to route requests to engines and stream events."""

from __future__ import annotations

import contextlib
import logging
import time
from typing import Dict, Iterable, Iterator, Mapping, MutableMapping, Optional

from .engine_interface import BaseInferenceEngine, TaskType
from apps.backend.runtime.memory import memory_management as _mem
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
        print(f"[orchestrator] resolve engine='{engine_key}' task='{task.value}' model_ref='{model_ref}'", flush=True)
        logger.info("[orchestrator] DEBUG: enter run task=%s engine=%s model=%s", task.value, engine_key, model_ref)
        start = time.perf_counter()
        engine = self._resolve_engine(engine_key, engine_options or {})
        print(f"[orchestrator] resolved engine instance={engine} (loaded={getattr(engine, '_is_loaded', None)})", flush=True)

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
                # Reload if the primary device changed since last load
                try:
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
                print(f"[orchestrator] loading model_ref='{model_ref}' on engine='{engine_key}'", flush=True)
                try:
                    if device_mismatch and engine._is_loaded:
                        try:
                            engine.unload()
                        except Exception:
                            pass
                    engine.load(model_ref, **(engine_options or {}))
                    engine.mark_loaded()
                    print(f"[orchestrator] load ok model_ref='{model_ref}' engine='{engine_key}'", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[orchestrator] load failed model_ref='{model_ref}' engine='{engine_key}' error={exc}", flush=True)
                    raise EngineLoadError(
                        f"Failed to load engine '{engine_key}' for model '{model_ref}': {exc}"
                    ) from exc

        handler = getattr(engine, task.value, None)
        if handler is None:
            raise UnsupportedTaskError(f"Engine '{engine_key}' is missing handler for task '{task.value}'")

        try:
            print(f"[orchestrator] starting handler task='{task.value}' engine='{engine_key}'", flush=True)
            yield ProgressEvent(stage="start", percent=0.0, message="Starting inference")
            for event in handler(request):
                print(f"[orchestrator] handler yielded event={event}", flush=True)
                yield event
        except UnsupportedTaskError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.exception("Engine '%s' failed during '%s'", engine_key, task.value)
            raise EngineExecutionError(str(exc)) from exc
        finally:
            elapsed = time.perf_counter() - start
            yield ProgressEvent(stage="end", percent=100.0, message="Inference complete", data={"elapsed": elapsed})
            logger.info("[orchestrator] DEBUG: exit run task=%s elapsed=%.3fs", task.value, elapsed)

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
        if engine is None:
            return
        with contextlib.suppress(Exception):
            engine.unload()
            engine.mark_unloaded()
        logger.info("Evicted engine '%s'", normalized_key)

    def clear_cache(self) -> None:
        for key in list(self._engine_cache.keys()):
            self.evict(key)
