"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Typed launcher settings and validation helpers.
Provides typed env-backed wrappers (no Tk dependency) so UI/service code avoids stringly-typed lookups and scattered
normalization rules, and so this module is unit-testable.

Symbols (top-level; keep in sync; no ghosts):
- `SettingValidationError` (exception): Raised when a launcher setting value is invalid.
- `ChoiceSetting` (dataclass): Typed view over a string setting constrained to a fixed set of choices.
- `BoolSetting` (dataclass): Typed view over a boolean setting serialized as "1"/"0".
- `IntSetting` (dataclass): Typed view over an integer setting serialized as a string.
- `DEVICE_CHOICES` (constant): Allowed values for `CODEX_*_DEVICE`.
- `GGUF_EXEC_CHOICES` (constant): Allowed values for `CODEX_GGUF_EXEC`.
- `LORA_APPLY_CHOICES` (constant): Allowed values for `CODEX_LORA_APPLY_MODE`.
- `LORA_ONLINE_MATH_CHOICES` (constant): Allowed values for `CODEX_LORA_ONLINE_MATH`.
- `normalize_gguf_lora_env` (function): Normalizes GGUF/LoRA env keys enforcing cross-setting invariants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, MutableMapping, Optional, Sequence


class SettingValidationError(ValueError):
    pass


DEVICE_CHOICES: tuple[str, ...] = ("auto", "cuda", "cpu", "mps", "xpu", "directml")
GGUF_EXEC_CHOICES: tuple[str, ...] = ("dequant_forward", "dequant_upfront")
LORA_APPLY_CHOICES: tuple[str, ...] = ("merge", "online")
LORA_ONLINE_MATH_CHOICES: tuple[str, ...] = ("weight_merge",)


def _normalize_lower(value: str) -> str:
    return str(value).strip().lower()


@dataclass(frozen=True, slots=True)
class ChoiceSetting:
    key: str
    default: str
    choices: tuple[str, ...]
    normalize: Callable[[str], str] = _normalize_lower

    def parse(self, raw: str | None) -> str:
        if raw is None:
            return self.default
        value = self.normalize(str(raw))
        if not value:
            return self.default
        if value not in self.choices:
            allowed = ", ".join(self.choices)
            raise SettingValidationError(f"{self.key} must be one of: {allowed} (got {raw!r}).")
        return value

    def get(self, env: Mapping[str, str]) -> str:
        return self.parse(env.get(self.key))

    def set(self, env: MutableMapping[str, str], value: str) -> None:
        env[self.key] = self.parse(value)


@dataclass(frozen=True, slots=True)
class BoolSetting:
    key: str
    default: bool = False

    def parse(self, raw: str | None) -> bool:
        if raw is None:
            return bool(self.default)
        value = str(raw).strip().lower()
        if not value:
            return bool(self.default)
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        raise SettingValidationError(f"{self.key} must be boolean (got {raw!r}).")

    def get(self, env: Mapping[str, str]) -> bool:
        return self.parse(env.get(self.key))

    def set(self, env: MutableMapping[str, str], value: bool) -> None:
        env[self.key] = "1" if bool(value) else "0"


@dataclass(frozen=True, slots=True)
class IntSetting:
    key: str
    default: int
    minimum: int | None = None
    maximum: int | None = None

    def parse(self, raw: str | None) -> int:
        if raw is None:
            return int(self.default)
        s = str(raw).strip()
        if not s:
            return int(self.default)
        try:
            value = int(s)
        except Exception as exc:
            raise SettingValidationError(f"{self.key} must be an integer (got {raw!r}).") from exc
        if self.minimum is not None and value < self.minimum:
            raise SettingValidationError(f"{self.key} must be >= {self.minimum} (got {value}).")
        if self.maximum is not None and value > self.maximum:
            raise SettingValidationError(f"{self.key} must be <= {self.maximum} (got {value}).")
        return value

    def get(self, env: Mapping[str, str]) -> int:
        return self.parse(env.get(self.key))

    def set(self, env: MutableMapping[str, str], value: int) -> None:
        value = int(value)
        if self.minimum is not None and value < self.minimum:
            raise SettingValidationError(f"{self.key} must be >= {self.minimum} (got {value}).")
        if self.maximum is not None and value > self.maximum:
            raise SettingValidationError(f"{self.key} must be <= {self.maximum} (got {value}).")
        env[self.key] = str(value)


def normalize_gguf_lora_env(env: MutableMapping[str, str]) -> tuple[str, str, str]:
    """Normalize GGUF/LoRA env keys enforcing cross-setting invariants.

    Returns (gguf_exec, lora_apply_mode, lora_online_math) as normalized values.
    """

    # Migration: older launchers persisted reserved/removed options.
    if str(env.get("CODEX_GGUF_EXEC", "") or "").strip().lower() == "cuda_pack":
        env["CODEX_GGUF_EXEC"] = "dequant_forward"
    if str(env.get("CODEX_LORA_ONLINE_MATH", "") or "").strip().lower() == "activation":
        env["CODEX_LORA_ONLINE_MATH"] = "weight_merge"

    gguf = ChoiceSetting("CODEX_GGUF_EXEC", default="dequant_forward", choices=GGUF_EXEC_CHOICES).get(env)
    lora_apply = ChoiceSetting("CODEX_LORA_APPLY_MODE", default="merge", choices=LORA_APPLY_CHOICES).get(env)
    lora_math = ChoiceSetting("CODEX_LORA_ONLINE_MATH", default="weight_merge", choices=LORA_ONLINE_MATH_CHOICES).get(env)

    # math only valid on online mode
    if lora_apply != "online":
        lora_math = "weight_merge"

    env["CODEX_GGUF_EXEC"] = gguf
    env["CODEX_LORA_APPLY_MODE"] = lora_apply
    env["CODEX_LORA_ONLINE_MATH"] = lora_math

    return gguf, lora_apply, lora_math
