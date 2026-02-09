"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: StreamingConfig for enabling Flux core streaming and tuning policy thresholds.

Symbols (top-level; keep in sync; no ghosts):
- `StreamingConfig` (dataclass): Configuration for streaming Flux transformer blocks between CPU/GPU during inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass
class StreamingConfig:
    """Configuration for FluxTransformer2DModel streaming.

    This configuration controls whether and how the transformer core
    streams blocks between CPU and GPU during inference.

    Attributes:
        enabled: Whether streaming is enabled.
        policy: Streaming policy name ("naive", "window", "aggressive").
        blocks_per_segment: Number of blocks to group per segment.
        window_size: For "window" policy, number of segments to keep on GPU.
        auto_enable_threshold_mb: Auto-enable streaming if free VRAM < threshold.
            Set to 0 to disable auto-enable.
    """

    enabled: bool = False
    policy: Literal["naive", "window", "aggressive"] = "naive"
    blocks_per_segment: int = 4
    window_size: int = 2
    auto_enable_threshold_mb: int = 0  # 0 = disabled

    @classmethod
    def from_options(cls, options: dict) -> "StreamingConfig":
        """Create config from engine options dict."""
        if not isinstance(options, dict):
            raise RuntimeError(f"Flux streaming options must be a mapping, got {type(options).__name__}.")
        enabled = cls._parse_bool_option(options, "core_streaming_enabled", default=False)
        policy = cls._parse_policy_option(options, "core_streaming_policy", default="naive")
        blocks_per_segment = cls._parse_int_option(options, "core_streaming_blocks_per_segment", default=4, minimum=1)
        window_size = cls._parse_int_option(options, "core_streaming_window_size", default=2, minimum=1)
        auto_enable_threshold_mb = cls._parse_int_option(options, "core_streaming_auto_threshold_mb", default=0, minimum=0)
        return cls(
            enabled=enabled,
            policy=policy,
            blocks_per_segment=blocks_per_segment,
            window_size=window_size,
            auto_enable_threshold_mb=auto_enable_threshold_mb,
        )

    def should_enable(self, free_vram_mb: int) -> bool:
        """Determine if streaming should be enabled based on free VRAM."""
        if self.enabled:
            return True
        if self.auto_enable_threshold_mb > 0:
            return free_vram_mb < self.auto_enable_threshold_mb
        return False

    @staticmethod
    def _parse_bool_option(options: dict, key: str, *, default: bool) -> bool:
        if key not in options or options.get(key) is None:
            return default
        value = options[key]
        if not isinstance(value, bool):
            raise RuntimeError(f"Flux streaming option '{key}' must be boolean (got {type(value).__name__}).")
        return value

    @staticmethod
    def _parse_int_option(options: dict, key: str, *, default: int, minimum: int) -> int:
        if key not in options or options.get(key) is None:
            return default
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise RuntimeError(f"Flux streaming option '{key}' must be an integer (got {type(value).__name__}).")
        if value < minimum:
            raise RuntimeError(f"Flux streaming option '{key}' must be >= {minimum} (got {value}).")
        return value

    @staticmethod
    def _parse_policy_option(
        options: dict,
        key: str,
        *,
        default: Literal["naive", "window", "aggressive"],
    ) -> Literal["naive", "window", "aggressive"]:
        if key not in options or options.get(key) is None:
            return default
        value = options[key]
        if not isinstance(value, str):
            raise RuntimeError(f"Flux streaming option '{key}' must be a string (got {type(value).__name__}).")
        normalized = value.strip().lower()
        if normalized not in {"naive", "window", "aggressive"}:
            raise RuntimeError(
                f"Flux streaming option '{key}' must be one of ('naive','window','aggressive'); got {value!r}."
            )
        return normalized  # type: ignore[return-value]
