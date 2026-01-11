"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 GGUF runtime diagnostics helpers.
Centralizes opt-in debug logging (sigma/timestep parity, CUDA memory snapshots) without scattering ad-hoc logger wrappers.

Symbols (top-level; keep in sync; no ghosts):
- `log_sigmas_enabled` (function): Enables/disables sigma/timestep parity logs via env toggles.
- `summarize_tensor` (function): Debug helper summarizing tensor-ish objects (shape/dtype/range sample).
- `get_logger` (function): Normalizes an optional logger argument into a `logging.Logger`.
- `cuda_empty_cache` (function): Best-effort CUDA cache emptying with optional logging.
- `log_cuda_mem` (function): Logs CUDA memory stats for debugging long video runs.
- `log_t_mapping` (function): Logs a coarse mapping of scheduler index → normalized timestep for parity debugging.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from apps.backend.infra.config.env_flags import env_flag


def log_sigmas_enabled() -> bool:
    return env_flag("WAN_LOG_SIGMAS", default=False) or env_flag("CODEX_LOG_SIGMAS", default=False)


def summarize_tensor(t: object, *, window: int = 6) -> str:
    if not isinstance(t, torch.Tensor):
        return "<not-a-tensor>"
    try:
        flat = t.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
        n = int(flat.numel())
        if n == 0:
            return "<empty>"
        if n <= window * 2:
            values = flat.tolist()
            return ",".join(f"{float(v):.6g}" for v in values)
        head = [float(v) for v in flat[:window].tolist()]
        tail = [float(v) for v in flat[-window:].tolist()]
        return f"{','.join(f'{v:.6g}' for v in head)},...,{','.join(f'{v:.6g}' for v in tail)}"
    except Exception:
        return "<unavailable>"


def get_logger(logger: Any) -> logging.Logger:
    return logger or logging.getLogger("backend.runtime.wan22.gguf")


def cuda_empty_cache(logger: Any, *, label: str) -> None:
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return
    log = get_logger(logger)
    try:
        torch.cuda.synchronize()
        alloc_before = int(torch.cuda.memory_allocated() // (1024 * 1024))
        reserved_before = int(torch.cuda.memory_reserved() // (1024 * 1024))
        torch.cuda.empty_cache()
        alloc_after = int(torch.cuda.memory_allocated() // (1024 * 1024))
        reserved_after = int(torch.cuda.memory_reserved() // (1024 * 1024))
        log.info(
            "[wan22.gguf] cuda.gc(%s): alloc %d→%d MB reserved %d→%d MB",
            label,
            alloc_before,
            alloc_after,
            reserved_before,
            reserved_after,
        )
    except Exception:
        # Diagnostics only; keep permissive.
        return


def log_cuda_mem(logger: Any, *, label: str) -> None:
    log = get_logger(logger)
    if not (getattr(torch, "cuda", None) and torch.cuda.is_available()):
        return
    try:
        alloc = float(torch.cuda.memory_allocated()) / (1024**2)
        reserv = float(torch.cuda.memory_reserved()) / (1024**2)
        total = float(torch.cuda.get_device_properties(0).total_memory) / (1024**2)
        log.info(
            "[wan22.gguf] %s: cuda mem alloc=%.0fMB reserved=%.0fMB total=%.0fMB",
            label,
            alloc,
            reserv,
            total,
        )
    except Exception:
        log.debug("[wan22.gguf] failed to read cuda memory stats", exc_info=True)


def log_t_mapping(scheduler: Any, timesteps: Any, *, label: str, logger: Any) -> None:
    log = get_logger(logger)
    try:
        n = len(timesteps)
        idxs = [0, max(0, n // 2 - 1), n - 1]
        vals: list[float] = []
        sigmas = getattr(scheduler, "sigmas", None)
        for i in idxs:
            sig_ok = bool(sigmas is not None and len(sigmas) in (n, n + 1))
            if sig_ok:
                s = float(sigmas[i])
                s_min = float(sigmas[-1])
                s_max = float(sigmas[0])
                t = max(0.0, min(1.0, (s - s_min) / (s_max - s_min))) if (s_max - s_min) > 0 else 0.0
            else:
                t = 1.0 - (float(i) / float(max(1, n - 1)))
            vals.append(float(t))
        log.info(
            "[wan22.gguf] t-map(%s): t0=%.4f tmid=%.4f tend=%.4f (sigmas=%s)",
            label,
            vals[0],
            vals[1],
            vals[2],
            bool(sigmas is not None and len(sigmas) in (n, n + 1)),
        )
    except Exception:
        log.debug("[wan22.gguf] failed to log timestep mapping", exc_info=True)
