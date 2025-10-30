from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

import torch


@dataclass(frozen=True, slots=True)
class ControlWeightSchedule:
    """Advanced weighting configuration for ControlNet contributions."""

    positive: Optional[Sequence[float]] = None
    negative: Optional[Sequence[float]] = None
    frame: Optional[Sequence[float]] = None
    sigma: Optional[Callable[[torch.Tensor], torch.Tensor]] = None

    def validate(self, *, block_lengths: dict[str, int], batch_size: int) -> None:
        if self.positive is not None:
            _assert_length(self.positive, block_lengths.get("positive", len(self.positive)), "positive weighting")
        if self.negative is not None:
            _assert_length(self.negative, block_lengths.get("negative", len(self.negative)), "negative weighting")
        if self.frame is not None and len(self.frame) != batch_size:
            raise ValueError(f"frame weighting list length ({len(self.frame)}) must match batch size ({batch_size})")


def _assert_length(values: Sequence[float], expected: int, label: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{label} length mismatch: expected {expected}, got {len(values)}")


@dataclass(frozen=True, slots=True)
class ControlMaskConfig:
    """Mask guidance configuration for ControlNet."""

    mask: Optional[torch.Tensor] = None

    def validate(self) -> None:
        if self.mask is None:
            return
        if self.mask.dim() != 4:
            raise ValueError("Control mask must be a 4D tensor (B, 1, H, W)")
        if self.mask.size(1) != 1:
            raise ValueError("Control mask channel dimension must be 1")
        if self.mask.size(0) <= 0 or self.mask.size(2) <= 0 or self.mask.size(3) <= 0:
            raise ValueError("Control mask must have positive batch/height/width dimensions")


@dataclass(frozen=True, slots=True)
class ControlRequest:
    """Input request for ControlNet execution."""

    image: torch.Tensor
    strength: float
    start_percent: float
    end_percent: float
    weight_schedule: ControlWeightSchedule = field(default_factory=ControlWeightSchedule)
    mask_config: ControlMaskConfig = field(default_factory=ControlMaskConfig)

    def validate(self) -> None:
        if not isinstance(self.image, torch.Tensor):
            raise TypeError("ControlRequest.image must be a torch.Tensor")
        if self.image.dim() != 4:
            raise ValueError("ControlRequest.image must be a 4D tensor (B, C, H, W)")
        if not (0.0 <= self.start_percent <= 1.0 and 0.0 <= self.end_percent <= 1.0):
            raise ValueError("start_percent and end_percent must be between 0 and 1")
        if self.start_percent > self.end_percent:
            raise ValueError("start_percent cannot exceed end_percent")
        self.mask_config.validate()


@dataclass(frozen=True, slots=True)
class ControlNodeConfig:
    """Static configuration for a ControlNet node."""

    name: str
    model_type: str
    supports_online_lora: bool = False


@dataclass
class ControlNodeState:
    """Mutable runtime state for a ControlNet node."""

    device: torch.device
    dtype: torch.dtype
    weight_schedule: ControlWeightSchedule
    mask_config: ControlMaskConfig


@dataclass
class ControlNode:
    """Executable ControlNet node embedded within a graph."""

    config: ControlNodeConfig
    request: ControlRequest
    control: Any
    state: Optional[ControlNodeState] = None

    def prepare(self, model, percent_to_sigma: Callable[[float], torch.Tensor]) -> None:
        self.request.validate()
        self.state = ControlNodeState(
            device=model.device,
            dtype=model.dtype,
            weight_schedule=self.request.weight_schedule,
            mask_config=self.request.mask_config,
        )


@dataclass
class ControlGraph:
    """Ordered collection of ControlNet nodes applied to a UNet."""

    nodes: list[ControlNode] = field(default_factory=list)

    def append(self, node: ControlNode) -> None:
        self.nodes.append(node)

    def validate(self) -> None:
        for node in self.nodes:
            node.request.validate()
