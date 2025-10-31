
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, Optional

import torch

from apps.backend.runtime.models.safety import safe_torch_load

try:  # pragma: no cover - optional dependency
    from zoedepth.models.zoedepth import ZoeDepth
    from zoedepth.utils.config import get_config
except ImportError:  # pragma: no cover
    ZoeDepth = None
    get_config = None


@dataclass(frozen=True)
class ZoeDepthConfig:
    weights_path: str = "zoedepth/ZoeD_M12_N.pt"
    device: Optional[torch.device] = None
    dtype: torch.dtype = torch.float32


@lru_cache(maxsize=1)
def load_zoe_model(config: ZoeDepthConfig):
    if ZoeDepth is None or get_config is None:
        raise RuntimeError("zoedepth package is required for depth_zoe preprocessor")

    conf = get_config("zoedepth", "infer")
    model = ZoeDepth.build_from_config(conf)
    state = safe_torch_load(config.weights_path, map_location="cpu")
    if isinstance(state, Dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)

    device = config.device or torch.device("cpu")
    model.to(device=device, dtype=config.dtype)
    model.eval()
    return model
