from __future__ import annotations

import functools
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import load_state_dict, resolve_weights_file


@dataclass(frozen=True)
class LineartAnimeConfig:
    weights_path: str = "lineart_anime/netG.pth"


class UnetSkipConnectionBlock(nn.Module):
    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        *,
        input_nc: int | None = None,
        submodule: nn.Module | None = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        self.outermost = outermost
        if isinstance(norm_layer, functools.partial):  # pragma: no cover
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            self.model = nn.Sequential(*(down + [submodule] + up))
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            self.model = nn.Sequential(*(down + up))
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            if use_dropout:
                self.model = nn.Sequential(*(down + [submodule] + up + [nn.Dropout(0.5)]))
            else:
                self.model = nn.Sequential(*(down + [submodule] + up))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], dim=1)


class UnetGenerator(nn.Module):
    def __init__(
        self,
        input_nc: int,
        output_nc: int,
        num_downs: int,
        ngf: int = 64,
        norm_layer=nn.BatchNorm2d,
        use_dropout: bool = False,
    ) -> None:
        super().__init__()
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        for _ in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


@lru_cache(maxsize=1)
def _cached_lineart_state(weights_path: str):
    path = resolve_weights_file(weights_path)
    state = load_state_dict(path)
    cleaned = {}
    for key, value in state.items():
        cleaned[key.replace("module.", "")] = value.detach().clone()
    return cleaned


def load_lineart_anime_model(config: LineartAnimeConfig) -> nn.Module:
    norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    model = UnetGenerator(3, 1, 8, 64, norm_layer=norm_layer, use_dropout=False)
    state = _cached_lineart_state(config.weights_path)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"Lineart anime weights missing parameters: {missing}")
    if unexpected:
        raise RuntimeError(f"Lineart anime weights contain unexpected parameters: {unexpected}")
    model.eval()
    return model
