from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import torch
from einops import rearrange, repeat
from torch import nn

from apps.backend.runtime import utils
from .config import FluxArchitectureConfig, FluxGuidanceConfig, FluxPositionalConfig
from .geometry import timestep_embedding
from .embed import EmbedND, MLPEmbedder
from .components import DoubleStreamBlock, LastLayer, SingleStreamBlock

logger = logging.getLogger("backend.runtime.flux")


class FluxTransformer2DModel(nn.Module):
    """Codex-native Flux transformer implementation with explicit validation."""

    def __init__(self, config: FluxArchitectureConfig | None = None, **raw_config) -> None:
        super().__init__()
        if config is None:
            if not raw_config:
                raise ValueError("FluxTransformer2DModel requires configuration parameters")
            
            # Support HuggingFace config names alongside internal names
            # HF uses num_layers/num_single_layers, internal uses depth/depth_single_blocks
            depth = raw_config.pop("depth", None) or raw_config.pop("num_layers", None)
            single_depth = raw_config.pop("depth_single_blocks", None) or raw_config.pop("num_single_layers", None)
            if depth is None or single_depth is None:
                raise ValueError("FluxTransformer requires depth (or num_layers) and depth_single_blocks (or num_single_layers)")
            
            # Handle axes_dim with default for GGUF models that may not have it
            axes_dim = raw_config.pop("axes_dim", [16, 56, 56])
            if isinstance(axes_dim, list):
                axes_dim = tuple(axes_dim)
            
            theta = raw_config.pop("theta", 10000)
            positional = FluxPositionalConfig(patch_size=2, axes_dim=axes_dim, theta=theta)
            guidance_enabled = raw_config.pop("guidance_embed", None) or raw_config.pop("guidance_embeds", False)
            guidance = FluxGuidanceConfig(enabled=guidance_enabled)
            
            # Map HF config names to internal names
            if "num_attention_heads" in raw_config and "num_heads" not in raw_config:
                raw_config["num_heads"] = raw_config.pop("num_attention_heads")
            if "attention_head_dim" in raw_config and "head_dim" not in raw_config:
                raw_config["head_dim"] = raw_config.pop("attention_head_dim")
            if "joint_attention_dim" in raw_config and "context_in_dim" not in raw_config:
                raw_config["context_in_dim"] = raw_config.pop("joint_attention_dim")
            if "pooled_projection_dim" in raw_config and "vec_in_dim" not in raw_config:
                raw_config["vec_in_dim"] = raw_config.pop("pooled_projection_dim")
            
            # Calculate hidden_size from num_heads and head_dim if not provided
            head_dim = raw_config.pop("head_dim", 128)  # Default head_dim for Flux
            if "hidden_size" not in raw_config:
                num_heads = raw_config.get("num_heads", 24)  # Default for Flux
                raw_config["hidden_size"] = num_heads * head_dim
            
            # Add default mlp_ratio if not provided
            if "mlp_ratio" not in raw_config:
                raw_config["mlp_ratio"] = 4.0
            
            # Normalize in_channels from diffusers style to Codex style:
            # - Diffusers: in_channels=64, patch_size=1 (pre-patchified, patchification is external)
            # - Codex: in_channels=16, patch_size=2 (raw VAE channels, internal patchification)
            # When diffusers config has patch_size=1 and in_channels=64, we need to convert
            # to in_channels=16 since our model does patchification internally with patch_size=2
            hf_patch_size = raw_config.get("patch_size", 2)
            if hf_patch_size == 1 and raw_config.get("in_channels", 16) == 64:
                # Diffusers-style: 64 = 16 * 2 * 2 (VAE channels * patch_size^2)
                raw_config["in_channels"] = 16
                logger.debug("Normalized in_channels from 64 (diffusers) to 16 (Codex internal patchification)")
            
            # Remove keys that don't belong to FluxArchitectureConfig
            raw_config.pop("patch_size", None)  # We handle this in positional config
            raw_config.pop("_class_name", None)
            raw_config.pop("_diffusers_version", None)
            raw_config.pop("_name_or_path", None)
            
            config = FluxArchitectureConfig(
                positional=positional,
                guidance=guidance,
                double_blocks=depth,
                single_blocks=single_depth,
                **raw_config,
            )
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads

        self.img_in = nn.Linear(config.latent_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(256, self.hidden_size)
        self.vector_in = MLPEmbedder(config.vec_in_dim, self.hidden_size)

        if config.guidance.enabled:
            self.guidance_in = MLPEmbedder(config.guidance.embedding_dim, self.hidden_size)
        else:
            self.guidance_in = nn.Identity()

        self.txt_in = nn.Linear(config.context_in_dim, self.hidden_size)
        self.pe_embedder = EmbedND(
            dim=self.hidden_size // self.num_heads,
            theta=config.positional.theta,
            axes_dim=tuple(config.positional.axes_dim),
        )

        self.double_blocks = nn.ModuleList(
            DoubleStreamBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=config.mlp_ratio,
                qkv_bias=config.qkv_bias,
            )
            for _ in range(config.double_blocks)
        )
        self.single_blocks = nn.ModuleList(
            SingleStreamBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                mlp_ratio=config.mlp_ratio,
            )
            for _ in range(config.single_blocks)
        )
        self.final_layer = LastLayer(
            hidden_size=self.hidden_size,
            patch_size=config.positional.patch_size,
            out_channels=config.in_channels,
        )

    @property
    def patch_size(self) -> int:
        return self.config.positional.patch_size

    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
        *,
        control=None,
        transformer_options=None,
        **_: object,
    ) -> torch.Tensor:
        self._validate_inputs(x, timestep, context, y, guidance)

        batch, _, height, width = x.shape
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Flux forward start: batch=%d latent=%dx%dx%d context=%d vec=%d guidance=%s",
                batch,
                height,
                width,
                x.size(1),
                context.shape[1],
                y.shape[1],
                guidance is not None,
            )
        patch = self.patch_size
        pad_h = (-height) % patch
        pad_w = (-width) % patch
        if pad_h or pad_w:
            x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="circular")
        img = rearrange(x, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        img_ids = self._build_spatial_ids(batch, height=height + pad_h, width=width + pad_w, device=x.device, dtype=x.dtype)
        txt_ids = torch.zeros((batch, context.shape[1], 3), device=x.device, dtype=x.dtype)

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timestep, 256).to(img.dtype)) + self.vector_in(y)

        if self.config.guidance.enabled:
            if guidance is None:
                logger.error("[flux] guidance.enabled=True but guidance tensor is None!")
                raise ValueError("guidance embedding required but not provided")
            logger.debug("[flux] forward guidance: enabled=True, shape=%s, values=[%.2f]", tuple(guidance.shape), guidance[0].item())
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        else:
            logger.debug("[flux] forward guidance: enabled=False (schnell variant), guidance_arg=%s", guidance is not None)

        txt = self.txt_in(context)
        rotary = self._build_rotary(img_ids, txt_ids)

        for block in self.double_blocks:
            img, txt = block(img=img, txt=txt, vec=vec, rotary_freqs=rotary)

        tokens = torch.cat((txt, img), dim=1)
        for block in self.single_blocks:
            tokens = block(tokens, vec=vec, rotary_freqs=rotary)

        tokens = tokens[:, txt.shape[1]:]
        out = self.final_layer(tokens, vec)

        out = rearrange(out, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch,
                        h=(height + pad_h) // patch, w=(width + pad_w) // patch)
        return out[:, :, :height, :width]

    def _build_spatial_ids(self, batch: int, *, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        h_len = height // self.patch_size
        w_len = width // self.patch_size
        base = torch.zeros((h_len, w_len, 3), device=device, dtype=dtype)
        base[..., 1] = torch.linspace(0, h_len - 1, steps=h_len, device=device, dtype=dtype)[:, None]
        base[..., 2] = torch.linspace(0, w_len - 1, steps=w_len, device=device, dtype=dtype)[None, :]
        return repeat(base, "h w c -> b (h w) c", b=batch)

    def _build_rotary(self, img_ids: torch.Tensor, txt_ids: torch.Tensor) -> torch.Tensor:
        ids = torch.cat((txt_ids, img_ids), dim=1)
        rotary = self.pe_embedder(ids)
        # Broadcast rotary to match attention head layout: (B, 1, H, L, D/heads, 2, 2)
        return rotary

    def _validate_inputs(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        y: torch.Tensor,
        guidance: Optional[torch.Tensor],
    ) -> None:
        if x.ndim != 4:
            raise ValueError(f"expected input latent (B, C, H, W), got {tuple(x.shape)}")
        if x.size(1) != self.config.in_channels:
            raise ValueError(f"expected {self.config.in_channels} channels, got {x.size(1)}")
        if context.ndim != 3:
            raise ValueError("context must be (B, tokens, dim)")
        if context.size(2) != self.config.context_in_dim:
            raise ValueError("context dimension mismatch")
        if y.ndim != 2 or y.size(1) < self.config.vec_in_dim:
            raise ValueError("vector conditioning payload has incorrect shape")
        if timestep.ndim != 1 or timestep.size(0) != x.size(0):
            raise ValueError("timestep must be 1D with batch length")
        if guidance is not None and guidance.size(0) != x.size(0):
            raise ValueError("guidance embedding batch mismatch")
