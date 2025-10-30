from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class FluxPositionalConfig:
    """Positional embedding configuration for Flux-like DiT models."""

    patch_size: int
    axes_dim: Sequence[int]
    theta: int = 10000

    def __post_init__(self) -> None:
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if not self.axes_dim:
            raise ValueError("axes_dim must not be empty")
        if any(dim <= 0 for dim in self.axes_dim):
            raise ValueError("axes_dim entries must be positive")

    @property
    def positional_dim(self) -> int:
        return sum(self.axes_dim)


@dataclass(frozen=True, slots=True)
class FluxGuidanceConfig:
    """Optional guidance embedding configuration."""

    enabled: bool = False
    embedding_dim: int = 256

    def __post_init__(self) -> None:
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")


@dataclass(frozen=True, slots=True)
class FluxArchitectureConfig:
    """High-level configuration describing a Flux transformer."""

    in_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    num_heads: int
    mlp_ratio: float
    double_blocks: int
    single_blocks: int
    qkv_bias: bool = False
    positional: FluxPositionalConfig = field(
        default_factory=lambda: FluxPositionalConfig(patch_size=2, axes_dim=(16, 16, 16))
    )
    guidance: FluxGuidanceConfig = field(default_factory=FluxGuidanceConfig)

    def __post_init__(self) -> None:
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if self.vec_in_dim <= 0:
            raise ValueError("vec_in_dim must be positive")
        if self.context_in_dim <= 0:
            raise ValueError("context_in_dim must be positive")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if self.double_blocks < 0 or self.single_blocks < 0:
            raise ValueError("block counts must be non-negative")
        if self.positional.positional_dim != self.hidden_size // self.num_heads:
            raise ValueError(
                "sum(axes_dim) must equal hidden_size // num_heads "
                f"(got {self.positional.positional_dim} vs {self.hidden_size // self.num_heads})"
            )

    @property
    def patch_area(self) -> int:
        return self.positional.patch_size * self.positional.patch_size

    @property
    def latent_channels(self) -> int:
        return self.in_channels * self.patch_area
