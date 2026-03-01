"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Checkpoint IO helpers for runtime codepaths.
Loads safetensors/GGUF/pickle checkpoints and reads lightweight model configs from directories.

Symbols (top-level; keep in sync; no ghosts):
- `read_arbitrary_config` (function): Reads a best-effort config from a directory (supports JSON/YAML-like inputs where present).
- `load_torch_file` (function): Loads a torch checkpoint with safe-load options and explicit device targeting (prefers safe loaders, falls back to pickle loader when allowed).
- `read_gguf_metadata` (function): Reads GGUF key/value metadata from a `.gguf` file header (scoped here to keep quantization imports out of engines).
- `_load_gguf_state_dict` (function): Loads a GGUF state dict from a `.gguf` file path (used by runtime helpers without importing heavy ops).
- `load_gguf_state_dict` (function): Public GGUF state-dict loader that honors runtime flags for normal GGUFs, auto-detects CodexPack artifacts, and supports target-device tensor exposure.
- `_load_pickled_checkpoint` (function): Loads a pickled checkpoint using the restricted/guarded unpickler (`checkpoint_pickle`).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import torch

from apps.backend.runtime.misc import checkpoint_pickle

from ..state_dict.views import LazySafetensorsDict

_log = logging.getLogger("backend.runtime.utils")


def read_arbitrary_config(directory):
    config_path = os.path.join(directory, "config.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"No config.json file found in the directory: {directory}")

    with open(config_path, "rt", encoding="utf-8") as file:
        config_data = json.load(file)

    return config_data


def load_torch_file(ckpt, safe_load=True, device=None):
    """Load a checkpoint (safetensors/gguf/pickle) honoring an explicit device.

    - When ``device`` is None, use the current core execution device from the
      memory manager to avoid accidental CPU pinning.
    - For safetensors, the returned mapping lazily loads tensors using
      ``safe_open(..., device=<device>)`` so values are produced directly on the
      requested device when possible.
    """

    from apps.backend.runtime.memory import memory_management as _mm  # local import avoids cycles

    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        from apps.backend.runtime.memory.config import DeviceRole

        device = _mm.manager.get_device(DeviceRole.CORE)

    checkpoint_path = str(ckpt)
    suffix = os.path.splitext(checkpoint_path)[1].lower()

    if suffix == ".safetensors":
        return LazySafetensorsDict(checkpoint_path, device=str(device))
    if suffix == ".gguf":
        return _load_gguf_state_dict(checkpoint_path, device=device)

    pl_sd = _load_pickled_checkpoint(checkpoint_path, device, safe_load)

    if "global_step" in pl_sd:
        _log.info("Global Step: %s", pl_sd["global_step"])

    if "state_dict" in pl_sd:
        return pl_sd["state_dict"]
    return pl_sd


def load_gguf_state_dict(
    path: str,
    *,
    dequantize: bool | None = None,
    computation_dtype: torch.dtype = torch.float16,
    device: torch.device | str | None = None,
):
    """Load a GGUF state dict, with optional explicit dequantization policy.

    - When `dequantize` is None, resolves GGUF exec mode with `CODEX_GGUF_EXEC`
      env precedence over `--gguf-exec`. The default/only supported load policy in this path is
      `dequant_forward` (`dequantize=False`).
    - Callers with an explicit policy (e.g. "VAE GGUFs always dequantize") should pass
      `dequantize=True` to make the intent unambiguous and avoid drift.
    """

    from apps.backend.quantization.gguf_loader import load_gguf_state_dict as _load
    from apps.backend.infra.config.args import args as runtime_args
    from apps.backend.infra.config.gguf_exec_mode import GgufExecMode, resolve_gguf_exec_mode

    if dequantize is None:
        try:
            exec_mode = resolve_gguf_exec_mode(runtime_args)
        except ValueError as exc:
            raise RuntimeError(
                "GGUF checkpoint loader received invalid gguf exec mode "
                f"(CODEX_GGUF_EXEC/--gguf-exec): {exc}"
            ) from exc
        if exec_mode == GgufExecMode.DEQUANT_FORWARD:
            dequantize = False
        else:
            raise RuntimeError(
                "GGUF checkpoint loader does not support '--gguf-exec=cuda_pack' in this build. "
                "Use '--gguf-exec=dequant_forward'."
            )
    return _load(path, dequantize=bool(dequantize), computation_dtype=computation_dtype, device=device)


def read_gguf_metadata(path: str) -> dict[str, Any]:
    """Read GGUF metadata (key/value table) from the file header."""

    from apps.backend.quantization.gguf_loader import get_gguf_metadata as _get

    return dict(_get(path))


def _load_gguf_state_dict(path: str, *, device: torch.device | str | None = None):
    # Back-compat internal alias; prefer calling `load_gguf_state_dict(...)` directly.
    return load_gguf_state_dict(path, device=device)


def _load_pickled_checkpoint(path, device, safe_load):
    if safe_load:
        from apps.backend.runtime.models import safety as model_safety

        try:
            return model_safety.safe_torch_load(path, map_location=device)
        except model_safety.UnsafeCheckpointError:
            raise
    return torch.load(path, map_location=device, pickle_module=checkpoint_pickle)


__all__ = [
    "_load_gguf_state_dict",
    "load_gguf_state_dict",
    "read_gguf_metadata",
    "load_torch_file",
    "read_arbitrary_config",
]
