from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch

from .errors import ClipVisionInputError


@dataclass
class ClipVisionOutput:
    last_hidden_state: torch.Tensor
    penultimate_hidden_states: torch.Tensor
    image_embeds: torch.Tensor
    all_hidden_states: Optional[torch.Tensor] = None
    mm_projected: Optional[torch.Tensor] = None

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            value = getattr(self, key)
        except AttributeError as exc:
            raise ClipVisionInputError(f"Clip vision output does not expose key '{key}'.") from exc
        if value is None:
            raise ClipVisionInputError(f"Clip vision output slot '{key}' is empty for this encoder.")
        return value

    def to_dict(self) -> Dict[str, torch.Tensor]:
        data: Dict[str, torch.Tensor] = {}
        for key in ("last_hidden_state", "penultimate_hidden_states", "image_embeds", "all_hidden_states", "mm_projected"):
            value = getattr(self, key)
            if value is not None:
                data[key] = value
        return data


__all__ = ["ClipVisionOutput"]
