"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Emphasis implementations applied to text encoder hidden states during prompt encoding.
Provides several emphasis modes that post-process token embeddings using per-token multipliers.

Symbols (top-level; keep in sync; no ghosts):
- `Emphasis` (class): Base emphasis adapter holding token/multiplier state and an `after_transformers` hook.
- `EmphasisNone` (class): Disables emphasis interpretation (treats emphasis markup as literal).
- `EmphasisIgnore` (class): Ignores emphasis markup (treats all words as un-emphasized).
- `EmphasisOriginal` (class): Original emphasis implementation with mean normalization.
- `EmphasisOriginalNoNorm` (class): Emphasis implementation without mean normalization (often better for SDXL).
- `get_current_option` (function): Resolve an emphasis class by name, defaulting to `EmphasisOriginal`.
- `get_options_descriptions` (function): Return a human-readable summary of available emphasis modes.
- `options` (constant): List of supported emphasis classes.
"""

import torch


class Emphasis:
    name: str = "Base"
    description: str = ""
    tokens: list[list[int]]
    multipliers: torch.Tensor
    z: torch.Tensor

    def after_transformers(self):
        pass


class EmphasisNone(Emphasis):
    name = "None"
    description = "disable the mechanism entirely and treat (:.1.1) as literal characters"


class EmphasisIgnore(Emphasis):
    name = "Ignore"
    description = "treat all empasised words as if they have no emphasis"


class EmphasisOriginal(Emphasis):
    name = "Original"
    description = "the original emphasis implementation"

    def after_transformers(self):
        original_mean = self.z.mean()
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)
        new_mean = self.z.mean()
        self.z = self.z * (original_mean / new_mean)


class EmphasisOriginalNoNorm(EmphasisOriginal):
    name = "No norm"
    description = "same as original, but without normalization (seems to work better for SDXL)"

    def after_transformers(self):
        self.z = self.z * self.multipliers.reshape(self.multipliers.shape + (1,)).expand(self.z.shape)


def get_current_option(emphasis_option_name):
    return next(iter([x for x in options if x.name == emphasis_option_name]), EmphasisOriginal)


def get_options_descriptions():
    return ", ".join(f"{x.name}: {x.description}" for x in options)


options = [
    EmphasisNone,
    EmphasisIgnore,
    EmphasisOriginal,
    EmphasisOriginalNoNorm,
]
