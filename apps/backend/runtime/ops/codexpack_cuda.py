"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CodexPack CUDA extension bridge (optional runtime path).
Loads the `codexpack_cuda` extension that registers `torch.ops.codexpack.q4k_tilepack_linear(...)`.

Symbols (top-level; keep in sync; no ghosts):
- `_AttemptError` (dataclass): Records an extension load/build attempt failure (stage + message).
- `_set_attempt_error` (function): Records an attempt error and updates `last_error()`.
- `_try_load_ext` (function): Best-effort extension loader (prebuilt, in-place build, optional JIT build).
- `available` (function): Returns True when the extension is available (optional JIT build via env).
- `last_error` (function): Returns the last extension load/build error message, when present.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("backend.runtime.ops.codexpack.cuda")

_ext = None
_last_error: Optional[str] = None


@dataclass(frozen=True, slots=True)
class _AttemptError:
    stage: str
    message: str


_attempt_errors: list[_AttemptError] = []


def _set_attempt_error(stage: str, ex: Exception) -> None:
    global _last_error
    msg = f"{type(ex).__name__}: {ex}"
    _attempt_errors.append(_AttemptError(stage=stage, message=msg))
    _last_error = "\n".join(f"{e.stage}: {e.message}" for e in _attempt_errors)


def _try_load_ext(*, build: bool = False) -> None:
    global _ext
    global _last_error

    if _ext is not None:
        return
    _attempt_errors.clear()
    _last_error = None

    # 1) Prebuilt wheel / global module.
    try:
        import codexpack_cuda as _loaded

        _ext = _loaded
        _attempt_errors.clear()
        _last_error = None
        log.info("loaded codexpack_cuda extension (prebuilt)")
        return
    except Exception as ex:
        _set_attempt_error("prebuilt", ex)
        log.info("codexpack_cuda prebuilt not available: %s", ex)

    # 2) In-place build location (apps/backend/runtime/kernels/codexpack).
    try:
        this_dir = os.path.dirname(__file__)
        ext_dir = os.path.join(os.path.dirname(this_dir), "kernels", "codexpack")
        if os.path.isdir(ext_dir) and ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)

        import codexpack_cuda as _loaded

        _ext = _loaded
        _attempt_errors.clear()
        _last_error = None
        log.info("loaded codexpack_cuda extension from in-place build (%s)", ext_dir)
        return
    except Exception as ex:
        _set_attempt_error("in_place", ex)
        log.info("codexpack_cuda not available in in-place dir: %s", ex)

    if not build:
        return

    # 3) Optional JIT build (requires nvcc toolchain).
    try:
        from torch.utils.cpp_extension import load

        this_dir = os.path.dirname(__file__)
        src_dir = os.path.join(os.path.dirname(this_dir), "kernels", "codexpack")

        def _src(p: str) -> str:
            return os.path.join(src_dir, p)

        sources = [
            _src("codexpack_binding.cpp"),
            _src("q4k_tilepack_linear.cu"),
        ]

        _mod = load(
            name="codexpack_cuda_jit",
            sources=sources,
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
        )
        _ext = _mod
        _attempt_errors.clear()
        _last_error = None
        log.info("built codexpack_cuda extension via JIT")
    except Exception as ex:
        _set_attempt_error("jit", ex)
        log.error("failed to build codexpack_cuda via JIT: %s", ex)
        _ext = None


def available() -> bool:
    build = str(os.environ.get("CODEX_CODEXPACK_JIT", "") or "").strip() in {"1", "true", "yes", "on"}
    _try_load_ext(build=build)
    return _ext is not None


def last_error() -> Optional[str]:
    return _last_error
