"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Flow-shift resolution helpers for FlowMatch schedulers.
Parses scheduler_config.json files (diffusers) to extract shift parameters in a
single, strict format used by engines and runtime code.

Symbols (top-level; keep in sync; no ghosts):
- `FlowShiftMode` (enum): Fixed vs dynamic shift policy (dynamic uses base/max shift + seq_len).
- `FlowTimeShiftType` (enum): Time-shift transform kind for dynamic shifting (`exponential` vs `linear`).
- `FlowShiftSource` (enum): Origin of the shift (scheduler_config or explicit override).
- `FlowShiftSpec` (dataclass): Normalized shift metadata with a strict resolve() helper.
- `_coerce_float` (function): Coerce a raw config value into a float with strict errors.
- `_coerce_int` (function): Coerce a raw config value into an int with strict errors.
- `_require_key` (function): Fetch a required key from scheduler config or raise a strict error.
- `_read_scheduler_config` (function): Read + validate a scheduler_config.json file as a mapping.
- `_flow_shift_spec_from_config` (function): Internal parser from scheduler config mapping to FlowShiftSpec.
- `_flow_shift_spec_cached` (function): Cached flow shift spec loader keyed by scheduler_config.json path.
- `scheduler_config_path_from_repo_dir` (function): Resolve the canonical scheduler config path under a diffusers repo dir.
- `flow_shift_spec_from_scheduler_config` (function): Parse scheduler_config.json into FlowShiftSpec.
- `flow_shift_spec_from_config` (function): Parse a scheduler config mapping into FlowShiftSpec.
- `flow_shift_spec_from_repo_dir` (function): Resolve + parse the scheduler config under a diffusers repo dir.
- `calculate_shift` (function): Diffusers-compatible dynamic shift formula.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Mapping


class FlowShiftMode(str, Enum):
    FIXED = "fixed"
    DYNAMIC = "dynamic"


class FlowTimeShiftType(str, Enum):
    EXPONENTIAL = "exponential"
    LINEAR = "linear"


class FlowShiftSource(str, Enum):
    SCHEDULER_CONFIG = "scheduler_config"
    OVERRIDE = "override"


def calculate_shift(
    image_seq_len: int,
    *,
    base_seq_len: int,
    max_seq_len: int,
    base_shift: float,
    max_shift: float,
) -> float:
    """Match diffusers `calculate_shift` (Flux pipelines)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


@dataclass(frozen=True)
class FlowShiftSpec:
    mode: FlowShiftMode
    source: FlowShiftSource
    value: float | None = None
    base_shift: float | None = None
    max_shift: float | None = None
    base_seq_len: int | None = None
    max_seq_len: int | None = None
    time_shift_type: FlowTimeShiftType | None = None
    config_path: str | None = None

    def resolve(self, *, seq_len: int | None = None) -> float:
        if self.mode is FlowShiftMode.FIXED:
            if self.value is None:
                raise RuntimeError("FlowShiftSpec FIXED is missing value")
            return float(self.value)
        if seq_len is None:
            raise RuntimeError("FlowShiftSpec DYNAMIC requires seq_len")
        if self.base_shift is None or self.max_shift is None:
            raise RuntimeError("FlowShiftSpec DYNAMIC is missing base_shift/max_shift")
        if self.base_seq_len is None or self.max_seq_len is None:
            raise RuntimeError("FlowShiftSpec DYNAMIC is missing base_seq_len/max_seq_len")
        return float(
            calculate_shift(
                int(seq_len),
                base_seq_len=int(self.base_seq_len),
                max_seq_len=int(self.max_seq_len),
                base_shift=float(self.base_shift),
                max_shift=float(self.max_shift),
            )
        )

    def resolve_effective_shift(self, *, seq_len: int | None = None) -> float:
        """Return the effective shift (alpha) used by the standard shift formula.

        Notes
        - For FIXED configs (`use_dynamic_shifting=false`), `shift` is already the
          effective alpha.
        - For DYNAMIC configs, diffusers' time-shift uses either:
          - `exponential`: alpha = exp(mu)
          - `linear`: alpha = mu
        where `mu` is computed by `calculate_shift(...)`.
        """
        if self.mode is FlowShiftMode.FIXED:
            return self.resolve()

        mu = self.resolve(seq_len=seq_len)
        kind = self.time_shift_type or FlowTimeShiftType.EXPONENTIAL
        if kind is FlowTimeShiftType.LINEAR:
            return float(mu)
        return float(math.exp(float(mu)))


def _coerce_float(value: object, *, field: str, path: str | None) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001 - defensive parsing
        hint = f" at {path}" if path else ""
        raise RuntimeError(f"Invalid {field}{hint}: {value!r}") from exc


def _coerce_int(value: object, *, field: str, path: str | None) -> int:
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001 - defensive parsing
        hint = f" at {path}" if path else ""
        raise RuntimeError(f"Invalid {field}{hint}: {value!r}") from exc


def _require_key(config: Mapping[str, object], key: str, *, path: str | None) -> object:
    if key not in config:
        hint = f" at {path}" if path else ""
        raise RuntimeError(f"Missing '{key}' in scheduler_config{hint}")
    return config[key]


def _read_scheduler_config(path: Path) -> Mapping[str, object]:
    if not path.is_file():
        raise RuntimeError(f"scheduler_config.json not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - strict parse
        raise RuntimeError(f"Invalid scheduler_config.json: {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise RuntimeError(f"scheduler_config.json must be a JSON object: {path}")
    return data


def _flow_shift_spec_from_config(config: Mapping[str, object], *, config_path: str | None) -> FlowShiftSpec:
    use_dynamic = bool(config.get("use_dynamic_shifting") is True)
    if use_dynamic:
        base_shift = _coerce_float(_require_key(config, "base_shift", path=config_path), field="base_shift", path=config_path)
        max_shift = _coerce_float(_require_key(config, "max_shift", path=config_path), field="max_shift", path=config_path)
        base_seq_len = config.get("base_image_seq_len") or config.get("base_seq_len")
        max_seq_len = config.get("max_image_seq_len") or config.get("max_seq_len")
        if base_seq_len is None or max_seq_len is None:
            raise RuntimeError(f"Missing base/max seq_len in scheduler_config at {config_path}")
        raw_kind = config.get("time_shift_type") or "exponential"
        try:
            kind = FlowTimeShiftType(str(raw_kind))
        except Exception as exc:  # noqa: BLE001 - strict parsing
            raise RuntimeError(f"Unsupported time_shift_type={raw_kind!r} in scheduler_config at {config_path}") from exc
        return FlowShiftSpec(
            mode=FlowShiftMode.DYNAMIC,
            source=FlowShiftSource.SCHEDULER_CONFIG,
            base_shift=base_shift,
            max_shift=max_shift,
            base_seq_len=_coerce_int(base_seq_len, field="base_seq_len", path=config_path),
            max_seq_len=_coerce_int(max_seq_len, field="max_seq_len", path=config_path),
            time_shift_type=kind,
            config_path=config_path,
        )
    if "flow_shift" in config:
        value = _coerce_float(config["flow_shift"], field="flow_shift", path=config_path)
        return FlowShiftSpec(
            mode=FlowShiftMode.FIXED,
            source=FlowShiftSource.SCHEDULER_CONFIG,
            value=value,
            config_path=config_path,
        )
    if "shift" in config:
        value = _coerce_float(config["shift"], field="shift", path=config_path)
        return FlowShiftSpec(
            mode=FlowShiftMode.FIXED,
            source=FlowShiftSource.SCHEDULER_CONFIG,
            value=value,
            config_path=config_path,
        )
    raise RuntimeError(f"scheduler_config.json missing flow_shift/shift keys: {config_path}")


@lru_cache(maxsize=None)
def _flow_shift_spec_cached(path_str: str) -> FlowShiftSpec:
    path = Path(path_str)
    config = _read_scheduler_config(path)
    return _flow_shift_spec_from_config(config, config_path=str(path))


def flow_shift_spec_from_scheduler_config(path: str | Path) -> FlowShiftSpec:
    return _flow_shift_spec_cached(str(Path(path)))


def flow_shift_spec_from_config(config: Mapping[str, object], *, config_path: str | None = None) -> FlowShiftSpec:
    """Parse a diffusers scheduler config mapping into a FlowShiftSpec.

    This is useful when a scheduler instance is already loaded and exposes
    a `.config` mapping, avoiding any filesystem path assumptions.
    """
    return _flow_shift_spec_from_config(config, config_path=config_path)


def scheduler_config_path_from_repo_dir(repo_dir: str | Path) -> Path:
    """Resolve the canonical scheduler config path under a diffusers repo dir.

    Expects `repo_dir/scheduler/scheduler_config.json` (preferred) or falls back
    to `repo_dir/scheduler/config.json` when present. Raises on missing files.
    """
    repo = Path(repo_dir)
    if not repo.is_dir():
        raise RuntimeError(f"Diffusers repo dir not found: {repo}")
    scheduler_dir = repo / "scheduler"
    if not scheduler_dir.is_dir():
        raise RuntimeError(f"Diffusers repo missing scheduler/ directory: {repo}")
    for candidate in (scheduler_dir / "scheduler_config.json", scheduler_dir / "config.json"):
        if candidate.is_file():
            return candidate
    raise RuntimeError(f"Diffusers repo missing scheduler config: {repo} (expected scheduler/scheduler_config.json)")


def flow_shift_spec_from_repo_dir(repo_dir: str | Path) -> FlowShiftSpec:
    """Resolve and parse the scheduler config under a diffusers repo dir."""
    return flow_shift_spec_from_scheduler_config(scheduler_config_path_from_repo_dir(repo_dir))


__all__ = [
    "FlowShiftMode",
    "FlowTimeShiftType",
    "FlowShiftSource",
    "FlowShiftSpec",
    "flow_shift_spec_from_scheduler_config",
    "flow_shift_spec_from_config",
    "scheduler_config_path_from_repo_dir",
    "flow_shift_spec_from_repo_dir",
    "calculate_shift",
]
