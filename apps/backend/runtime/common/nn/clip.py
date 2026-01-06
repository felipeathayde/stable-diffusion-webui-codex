"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CLIP wrapper module used to integrate transformer backbones with optional text projection.

Symbols (top-level; keep in sync; no ghosts):
- `IntegratedCLIP` (class): Wraps a CLIP-like transformer and exposes HF-style forward outputs.
"""

import torch


class IntegratedCLIP(torch.nn.Module):
    def __init__(self, cls, config, add_text_projection: bool = False):
        super().__init__()
        self.transformer = cls(config)
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))

        if add_text_projection:
            embed_dim = config.hidden_size
            self.transformer.text_projection = torch.nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_hidden_states: bool = True,
        return_dict: bool = True,
    ):
        # Forward mask/position_ids when available; fall back gracefully otherwise.
        kwargs = {"output_hidden_states": output_hidden_states}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        if position_ids is not None:
            kwargs["position_ids"] = position_ids

        outputs = self.transformer(input_ids, **kwargs)

        if return_dict:
            return outputs
        return (outputs.last_hidden_state, outputs.pooler_output, outputs.hidden_states)
