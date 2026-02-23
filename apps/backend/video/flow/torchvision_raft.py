"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Torchvision RAFT optical flow estimation and warping utilities.
Implements a lazy-loaded RAFT flow estimator and a `warp_frame(...)` helper for flow-guided video workflows, raising `FlowGuidanceError` when torch/torchvision are missing.

Symbols (top-level; keep in sync; no ghosts):
- `FlowGuidanceError` (class): Raised when flow guidance dependencies are missing or inputs are invalid.
- `_round_down_to_multiple` (function): Rounds a value down to a given multiple (used for downscaled inference shapes).
- `_default_flow_device_name` (function): Resolves RAFT default device identity from memory-manager mount-device authority.
- `RaftFlowEstimator` (dataclass): Lazy-loaded RAFT estimator that produces backward flow tensors `[1,2,H,W]`.
- `warp_frame` (function): Warps a PIL frame using a backward flow tensor (torch grid sampling).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from apps.backend.runtime.memory import memory_management


class FlowGuidanceError(RuntimeError):
    pass


def _round_down_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    return max(multiple, int(value) - (int(value) % multiple))


def _default_flow_device_name() -> str:
    mount_device = memory_management.manager.mount_device()
    if not hasattr(mount_device, "type"):
        raise FlowGuidanceError(
            "RAFT flow estimator requires memory manager mount_device() to return torch.device."
        )
    return str(mount_device)


@dataclass
class RaftFlowEstimator:
    """Optical flow estimator powered by torchvision RAFT (lazy-loaded).

    This module deliberately keeps imports lazy so environments without torch/torchvision
    can still import the backend and use non-flow modes.
    """

    device: str = field(default_factory=_default_flow_device_name)
    use_large: bool = False
    downscale: int = 2

    _model: Any = None
    _weights: Any = None
    _tfm: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch  # type: ignore
            from torchvision.models.optical_flow import (  # type: ignore
                raft_large,
                raft_small,
                Raft_Large_Weights,
                Raft_Small_Weights,
            )
        except Exception as exc:  # noqa: BLE001
            raise FlowGuidanceError(
                "Optical flow requires torch + torchvision (RAFT). Install torchvision or disable flow guidance."
            ) from exc

        dev = str(self.device).lower().strip() if self.device else _default_flow_device_name().lower().strip()
        if dev != "cpu" and not (hasattr(torch, "cuda") and torch.cuda.is_available()):
            raise FlowGuidanceError(
                f"RAFT requested device={self.device!r}, but CUDA is unavailable. "
                "Set flow device to CPU explicitly or configure a CUDA mount device."
            )
        self.device = dev

        if self.use_large:
            weights = Raft_Large_Weights.DEFAULT
            model = raft_large(weights=weights, progress=False)
        else:
            weights = Raft_Small_Weights.DEFAULT
            model = raft_small(weights=weights, progress=False)

        model = model.to(dev)
        model.eval()
        self._model = model
        self._weights = weights
        self._tfm = weights.transforms()

    def estimate_backward_flow(self, *, target_frame: Any, source_frame: Any):
        """Estimate backward flow (target -> source) as a torch tensor [1,2,H,W]."""
        self._ensure_loaded()
        try:
            import torch  # type: ignore
            import torch.nn.functional as F  # type: ignore
            from PIL import Image  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise FlowGuidanceError(str(exc)) from exc

        if not isinstance(target_frame, Image.Image) or not isinstance(source_frame, Image.Image):
            raise FlowGuidanceError("Flow estimator expects PIL.Image inputs")

        target = target_frame.convert("RGB")
        source = source_frame.convert("RGB")

        w, h = target.size
        if source.size != (w, h):
            source = source.resize((w, h))

        ds = max(1, int(self.downscale or 1))
        if ds > 1:
            w2 = max(64, _round_down_to_multiple(w // ds, 8))
            h2 = max(64, _round_down_to_multiple(h // ds, 8))
            target_small = target.resize((w2, h2))
            source_small = source.resize((w2, h2))
        else:
            w2, h2 = w, h
            target_small, source_small = target, source

        # Backward flow: target -> source (swap order accordingly).
        img1, img2 = self._tfm(target_small, source_small)  # type: ignore[misc]
        img1 = img1.unsqueeze(0).to(self.device)
        img2 = img2.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            flows = self._model(img1, img2)
            flow = flows[-1]

        if ds > 1 and (w2 != w or h2 != h):
            flow = F.interpolate(flow, size=(h, w), mode="bilinear", align_corners=False)
            sx = float(w) / float(w2)
            sy = float(h) / float(h2)
            flow[:, 0] = flow[:, 0] * sx
            flow[:, 1] = flow[:, 1] * sy
        return flow


def warp_frame(frame: Any, *, backward_flow, device: Optional[str] = None) -> Any:
    """Warp a frame using a backward flow field (output coords -> input coords)."""
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
        from PIL import Image  # type: ignore
        from torchvision.transforms.functional import to_pil_image, pil_to_tensor  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise FlowGuidanceError(
            "warp_frame requires torch + torchvision. Install torchvision or disable flow guidance."
        ) from exc

    if not isinstance(frame, Image.Image):
        raise FlowGuidanceError("warp_frame expects a PIL.Image")

    flow = backward_flow
    if not hasattr(flow, "shape") or len(getattr(flow, "shape")) != 4:
        raise FlowGuidanceError("backward_flow must be a tensor shaped [1,2,H,W]")

    flow_device = getattr(flow, "device", None)
    dev = str(device or flow_device or _default_flow_device_name())
    img = frame.convert("RGB")
    t = pil_to_tensor(img).float() / 255.0
    t = t.unsqueeze(0).to(dev)
    flow = flow.to(dev)

    _, _, h, w = t.shape

    # Build base grid in normalized coordinates.
    ys = torch.linspace(-1.0, 1.0, h, device=dev)
    xs = torch.linspace(-1.0, 1.0, w, device=dev)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1,H,W,2]

    # Convert pixel flow to normalized offsets (align_corners=True semantics).
    if w <= 1 or h <= 1:
        return frame
    flow_x = flow[:, 0] * (2.0 / float(w - 1))
    flow_y = flow[:, 1] * (2.0 / float(h - 1))
    flow_norm = torch.stack([flow_x, flow_y], dim=-1)  # [1,H,W,2]

    grid = base + flow_norm
    warped = F.grid_sample(t, grid, mode="bilinear", padding_mode="border", align_corners=True)
    warped = warped.clamp(0.0, 1.0).squeeze(0)
    return to_pil_image(warped)
