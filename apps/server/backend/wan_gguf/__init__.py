from __future__ import annotations

"""WAN 2.2 — GGUF plugin wrapper (Phase 1).

This module is loaded with highest precedence by backend.engines.video.wan.gguf_exec
(`_import_executor` tries `backend_ext.wan_gguf` first). It attempts to delegate
execution to a native core if present; otherwise, it falls back to the in‑core
Python executor (`backend.engines.video.wan.gguf_incore`).

Contract:
- expose `run_txt2vid(cfg, logger)` and `run_img2vid(cfg, logger)` returning a
  list of frame objects (e.g., PIL Images or tensors as upstream expects).

There are NO fake results. If neither native core nor the in‑core executor can
run, we raise `GGUFExecutorUnavailable` with actionable context.
"""

from typing import Any, List


def _get_logger(logger: Any):
    import logging
    if logger is not None:
        return logger
    return logging.getLogger("wan_gguf_plugin")


def _try_native_core():
    """Best-effort import of the native core.

    The native core should expose callable `run_txt2vid(cfg, logger)` and
    `run_img2vid(cfg, logger)` functions with the same contract as this module.
    """
    try:
        # Preferred package path
        import importlib

        for name in ("backend_ext.wan_gguf_core", "wan_gguf_core"):
            try:
                mod = importlib.import_module(name)
                # Sanity check
                if hasattr(mod, "run_txt2vid") and hasattr(mod, "run_img2vid"):
                    # Accept core even if not fully operational; it will raise a descriptive error
                    return mod
            except Exception as ex:
                # Surface import errors to logs for easier diagnosis
                _get_logger(None).info("[wan-gguf-plugin] native core import failed from %s: %s", name, ex)
                continue
        return None
    except Exception:
        return None


def _fallback_incore():
    """Import and return the in-core Python executor module.

    The in-core executor may still raise `GGUFExecutorUnavailable` with a clear
    message until forward/sampler mapping is complete — we deliberately do not
    mask or fabricate outputs here.
    """
    import importlib
    return importlib.import_module("apps.server.backend.engines.video.wan.gguf_incore")


def run_txt2vid(cfg, logger=None) -> List[object]:  # noqa: D401
    log = _get_logger(logger)
    core = _try_native_core()
    if core is not None:
        log.info("[wan-gguf-plugin] Using native core for TXT2VID")
        return list(core.run_txt2vid(cfg, logger=logger))

    log.info("[wan-gguf-plugin] Native core not found; falling back to in‑core executor")
    incore = _fallback_incore()
    return list(incore.run_txt2vid(cfg, logger=logger))


def run_img2vid(cfg, logger=None) -> List[object]:  # noqa: D401
    log = _get_logger(logger)
    core = _try_native_core()
    if core is not None:
        log.info("[wan-gguf-plugin] Using native core for IMG2VID")
        return list(core.run_img2vid(cfg, logger=logger))

    log.info("[wan-gguf-plugin] Native core not found; falling back to in‑core executor")
    incore = _fallback_incore()
    return list(incore.run_img2vid(cfg, logger=logger))


__all__ = [
    "run_txt2vid",
    "run_img2vid",
]
