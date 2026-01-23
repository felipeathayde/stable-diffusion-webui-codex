"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 transformer key-style detection + remapping (Diffusers/WAN-export/Codex).
Normalizes multiple upstream key layouts into the canonical Codex WAN22 runtime layout and fails loud on unknown/ambiguous inputs.

Symbols (top-level; keep in sync; no ghosts):
- `remap_wan22_transformer_state_dict` (function): Returns (detected_style, remapped_view) for WAN22 transformer keys.
"""

from __future__ import annotations

import re
from collections.abc import MutableMapping, Sequence
from typing import TypeVar

from apps.backend.runtime.state_dict.key_mapping import (
    KeyMappingError,
    KeySentinel,
    KeyStyle,
    KeyStyleDetector,
    KeyStyleSpec,
    SentinelKind,
    remap_state_dict_view,
    strip_repeated_prefixes,
)

_T = TypeVar("_T")

_PREFIXES = (
    "model.model.diffusion_model.",
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
)

_RX_BLOCK_ATTN = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_(?P<proj>[qkv])\.(?P<param>weight|bias)$"
)
_RX_BLOCK_ATTN_OUT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_out\.0\.(?P<param>weight|bias)$"
)
_RX_BLOCK_ATTN_NORM = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.norm_(?P<norm>[qk])\.weight$"
)
_RX_BLOCK_FFN_PROJ = re.compile(
    r"^blocks\.(?P<idx>\d+)\.ffn\.net\.(?P<which>0\.proj|2)\.(?P<param>weight|bias)$"
)
_RX_BLOCK_NORM2 = re.compile(r"^blocks\.(?P<idx>\d+)\.norm2\.(?P<param>weight|bias)$")
_RX_BLOCK_NORM3 = re.compile(r"^blocks\.(?P<idx>\d+)\.norm3\.(?P<param>weight|bias)$")
_RX_BLOCK_SCALE_SHIFT = re.compile(r"^blocks\.(?P<idx>\d+)\.scale_shift_table$")

_DETECTOR = KeyStyleDetector(
    name="wan22_transformer_key_style",
    styles=(
        KeyStyleSpec(
            style=KeyStyle.DIFFUSERS,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "condition_embedder."),
                KeySentinel(SentinelKind.SUBSTRING, ".attn1."),
                KeySentinel(SentinelKind.SUBSTRING, ".attn2."),
                KeySentinel(SentinelKind.SUBSTRING, ".ffn.net."),
                KeySentinel(SentinelKind.PREFIX, "proj_out."),
                KeySentinel(SentinelKind.EXACT, "scale_shift_table"),
                KeySentinel(SentinelKind.SUBSTRING, ".scale_shift_table"),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.WAN_EXPORT,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "patch_embedding."),
                KeySentinel(SentinelKind.PREFIX, "time_embedding."),
                KeySentinel(SentinelKind.PREFIX, "time_projection."),
                KeySentinel(SentinelKind.PREFIX, "text_embedding."),
                KeySentinel(SentinelKind.PREFIX, "head.head."),
                KeySentinel(SentinelKind.EXACT, "head.modulation"),
            ),
            min_sentinel_hits=1,
        ),
        KeyStyleSpec(
            style=KeyStyle.CODEX,
            sentinels=(
                KeySentinel(SentinelKind.PREFIX, "patch_embed."),
                KeySentinel(SentinelKind.PREFIX, "time_embed."),
                KeySentinel(SentinelKind.PREFIX, "time_proj."),
                KeySentinel(SentinelKind.PREFIX, "text_embed."),
                KeySentinel(SentinelKind.EXACT, "head_modulation"),
                # A subset of canonical block keys also counts as Codex/WAN-export layout.
                KeySentinel(SentinelKind.REGEX, r"^blocks\.\d+\.(?:self_attn|cross_attn|ffn)\."),
                KeySentinel(SentinelKind.REGEX, r"^blocks\.\d+\.norm[123]\.(?:weight|bias)$"),
                KeySentinel(SentinelKind.REGEX, r"^blocks\.\d+\.modulation$"),
            ),
            min_sentinel_hits=1,
        ),
    ),
)


def remap_wan22_transformer_state_dict(state_dict: MutableMapping[str, _T]) -> tuple[KeyStyle, MutableMapping[str, _T]]:
    def _normalize(key: str) -> str:
        return strip_repeated_prefixes(str(key), _PREFIXES)

    def _export_to_codex(key: str) -> str:
        if key.startswith("patch_embedding."):
            return "patch_embed." + key[len("patch_embedding.") :]
        if key.startswith("time_embedding."):
            return "time_embed." + key[len("time_embedding.") :]
        if key.startswith("time_projection."):
            return "time_proj." + key[len("time_projection.") :]
        if key.startswith("text_embedding."):
            return "text_embed." + key[len("text_embedding.") :]
        if key.startswith("head.head."):
            return "head." + key[len("head.head.") :]
        if key == "head.modulation":
            return "head_modulation"
        return key

    def _diffusers_to_export(key: str) -> str:
        for before, after in (
            ("condition_embedder.time_embedder.linear_1.", "time_embedding.0."),
            ("condition_embedder.time_embedder.linear_2.", "time_embedding.2."),
            ("condition_embedder.text_embedder.linear_1.", "text_embedding.0."),
            ("condition_embedder.text_embedder.linear_2.", "text_embedding.2."),
            ("condition_embedder.time_proj.", "time_projection.1."),
        ):
            if key.startswith(before):
                return after + key[len(before) :]

        if key.startswith("proj_out."):
            return "head.head." + key[len("proj_out.") :]
        if key == "scale_shift_table":
            return "head.modulation"
        if key.endswith(".scale_shift_table"):
            return key[: -len(".scale_shift_table")] + ".modulation"

        m = _RX_BLOCK_ATTN.match(key)
        if m:
            idx = m.group("idx")
            which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
            proj = m.group("proj")
            param = m.group("param")
            return f"blocks.{idx}.{which}.{proj}.{param}"

        m = _RX_BLOCK_ATTN_OUT.match(key)
        if m:
            idx = m.group("idx")
            which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
            param = m.group("param")
            return f"blocks.{idx}.{which}.o.{param}"

        m = _RX_BLOCK_ATTN_NORM.match(key)
        if m:
            idx = m.group("idx")
            which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
            norm = m.group("norm")
            return f"blocks.{idx}.{which}.norm_{norm}.weight"

        m = _RX_BLOCK_FFN_PROJ.match(key)
        if m:
            idx = m.group("idx")
            which = "0" if m.group("which") == "0.proj" else "2"
            param = m.group("param")
            return f"blocks.{idx}.ffn.{which}.{param}"

        # Diffusers uses norm1/norm2/norm3 (SA/CA/FFN), while WAN exports swap 2↔3.
        m = _RX_BLOCK_NORM2.match(key)
        if m:
            idx = m.group("idx")
            param = m.group("param")
            return f"blocks.{idx}.norm3.{param}"

        m = _RX_BLOCK_NORM3.match(key)
        if m:
            idx = m.group("idx")
            param = m.group("param")
            return f"blocks.{idx}.norm2.{param}"

        m = _RX_BLOCK_SCALE_SHIFT.match(key)
        if m:
            idx = m.group("idx")
            return f"blocks.{idx}.modulation"

        return key

    def _validate_output(keys: Sequence[str]) -> None:
        offenders: list[str] = []

        def _is_forbidden(k: str) -> bool:
            return (
                k.startswith("condition_embedder.")
                or k.startswith("proj_out.")
                or k == "scale_shift_table"
                or ".attn1." in k
                or ".attn2." in k
                or ".ffn.net." in k
                or ".scale_shift_table" in k
                or k.startswith("patch_embedding.")
                or k.startswith("time_embedding.")
                or k.startswith("time_projection.")
                or k.startswith("text_embedding.")
                or k.startswith("head.head.")
                or k == "head.modulation"
            )

        for k in keys:
            if _is_forbidden(k):
                offenders.append(k)

        if offenders:
            sample = sorted(offenders)[:10]
            raise KeyMappingError(
                "WAN22 key remap produced non-canonical keys (mapping incomplete). "
                f"offenders_sample={sample}"
            )

        # When loading a full model state dict, patch_embed is required (LoRA-key remaps may not include it).
        if len(keys) > 64 and not any(k.startswith("patch_embed.") for k in keys):
            preview = ", ".join(sorted(keys)[:10])
            raise KeyMappingError(
                "WAN22 key remap output is missing required patch_embed.* keys. "
                f"sample_keys=[{preview}]"
            )

    mappers = {
        KeyStyle.CODEX: lambda k: k,
        KeyStyle.WAN_EXPORT: _export_to_codex,
        KeyStyle.DIFFUSERS: lambda k: _export_to_codex(_diffusers_to_export(k)),
    }

    return remap_state_dict_view(
        state_dict,
        detector=_DETECTOR,
        normalize=_normalize,
        mappers=mappers,
        output_validator=_validate_output,
    )

