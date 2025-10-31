from __future__ import annotations

import torch

from apps.backend.runtime.ops.operations import CodexOperations, main_stream_worker, weights_manual_cast


class ControlLoraOps(CodexOperations):
    """LoRA-aware operations used by ControlNet LoRA modules."""

    class Linear(torch.nn.Module):
        def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None) -> None:
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.weight = None
            self.up = None
            self.down = None
            self.bias = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias, signal = weights_manual_cast(self, input)
            with main_stream_worker(weight, bias, signal):
                if self.up is not None:
                    up_down = torch.mm(
                        self.up.flatten(start_dim=1),
                        self.down.flatten(start_dim=1),
                    ).reshape(self.weight.shape).to(input.dtype)
                    return torch.nn.functional.linear(input, weight + up_down, bias)
                return torch.nn.functional.linear(input, weight, bias)

    class Conv2d(torch.nn.Module):
        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = "zeros",
            device=None,
            dtype=None,
        ) -> None:
            super().__init__()
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.kernel_size = kernel_size
            self.stride = stride
            self.padding = padding
            self.dilation = dilation
            self.groups = groups
            self.bias_flag = bias
            self.padding_mode = padding_mode

            self.weight = None
            self.bias = None
            self.up = None
            self.down = None

        def forward(self, input: torch.Tensor) -> torch.Tensor:
            weight, bias, signal = weights_manual_cast(self, input)
            with main_stream_worker(weight, bias, signal):
                if self.up is not None:
                    up_down = torch.mm(
                        self.up.flatten(start_dim=1),
                        self.down.flatten(start_dim=1),
                    ).reshape(self.weight.shape).to(input.dtype)
                    return torch.nn.functional.conv2d(
                        input,
                        weight + up_down,
                        bias,
                        self.stride,
                        self.padding,
                        self.dilation,
                        self.groups,
                    )
                return torch.nn.functional.conv2d(
                    input,
                    weight,
                    bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                    self.groups,
                )
