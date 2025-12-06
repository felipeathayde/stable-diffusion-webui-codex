"""Configuration for Flux core streaming."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


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
        return cls(
            enabled=options.get("core_streaming_enabled", False),
            policy=options.get("core_streaming_policy", "naive"),
            blocks_per_segment=options.get("core_streaming_blocks_per_segment", 4),
            window_size=options.get("core_streaming_window_size", 2),
            auto_enable_threshold_mb=options.get("core_streaming_auto_threshold_mb", 0),
        )

    def should_enable(self, free_vram_mb: int) -> bool:
        """Determine if streaming should be enabled based on free VRAM."""
        if self.enabled:
            return True
        if self.auto_enable_threshold_mb > 0:
            return free_vram_mb < self.auto_enable_threshold_mb
        return False

