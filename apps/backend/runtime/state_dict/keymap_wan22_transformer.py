"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22 transformer key-style detection + remapping (Diffusers/WAN-export/Codex).
Normalizes multiple upstream key layouts into the canonical Codex WAN22 runtime layout and fails loud on unknown/ambiguous inputs.
Also owns WAN22 request-key allowlists used by generation routers, including img2vid temporal mode/window controls, no-stretch guide controls (`img2vid_resize_mode` + crop offsets), optional `video_upscaling`, and canonical sampler keys (`*_sampler`, no legacy `*_sampling` aliases).

Symbols (top-level; keep in sync; no ghosts):
- `Wan22RequestKeys` (dataclass): Canonical WAN22 request-key allowlists for txt2vid/img2vid and WAN stage controls (including stage prompt/negative fields and optional `video_upscaling` key).
- `WAN22_REQUEST_KEYS` (constant): Singleton request-key map used by WAN22 request validators.
- `remap_wan22_lora_logical_key` (function): Maps WAN22 LoRA logical keys to canonical WAN22 transformer weight keys.
- `remap_wan22_transformer_state_dict` (function): Returns (detected_style, remapped_view) for WAN22 transformer keys.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections.abc import MutableMapping, Sequence
from typing import FrozenSet, TypeVar

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
_RX_LORA_CANONICAL_ATTN_DOT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>self_attn|cross_attn)\.(?P<proj>q|k|v|o)$"
)
_RX_LORA_CANONICAL_FFN_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.ffn\.(?P<which>0|2)$")
_RX_LORA_CANONICAL_ATTN_UNDERSCORE = re.compile(
    r"^blocks_(?P<idx>\d+)_(?P<which>self_attn|cross_attn)_(?P<proj>q|k|v|o)$"
)
_RX_LORA_CANONICAL_FFN_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_ffn_(?P<which>0|2)$")
_RX_LORA_DIFFUSERS_ATTN_DOT = re.compile(
    r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_(?P<proj>q|k|v)$"
)
_RX_LORA_DIFFUSERS_ATTN_UNDERSCORE = re.compile(
    r"^blocks_(?P<idx>\d+)_(?P<which>attn1|attn2)_to_(?P<proj>q|k|v)$"
)
_RX_LORA_DIFFUSERS_OUT_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.(?P<which>attn1|attn2)\.to_out\.0$")
_RX_LORA_DIFFUSERS_OUT_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_(?P<which>attn1|attn2)_to_out_0$")
_RX_LORA_DIFFUSERS_FFN_DOT = re.compile(r"^blocks\.(?P<idx>\d+)\.ffn\.net\.(?P<which>0\.proj|2)$")
_RX_LORA_DIFFUSERS_FFN_UNDERSCORE = re.compile(r"^blocks_(?P<idx>\d+)_ffn_net_(?P<which>0_proj|2)$")
_LORA_LOGICAL_PREFIXES = ("lora_unet_", "lycoris_")

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


@dataclass(frozen=True)
class Wan22RequestKeys:
    """Canonical WAN22 request-key allowlists used by generation routers."""

    DEVICE: FrozenSet[str] = frozenset({"codex_device", "device", "codex_diffusion_device"})
    REVISION: FrozenSet[str] = frozenset({"settings_revision"})
    VIDEO_EXPORT: FrozenSet[str] = frozenset(
        {
            "video_return_frames",
            "video_filename_prefix",
            "video_format",
            "video_pix_fmt",
            "video_crf",
            "video_loop_count",
            "video_pingpong",
            "video_save_metadata",
            "video_save_output",
            "video_trim_to_audio",
        }
    )
    VIDEO_INTERPOLATION: FrozenSet[str] = frozenset({"video_interpolation"})
    VIDEO_UPSCALING: FrozenSet[str] = frozenset({"video_upscaling"})
    WAN_STAGE_CONTAINERS: FrozenSet[str] = frozenset({"wan_high", "wan_low"})
    WAN_STAGE_ALLOWED: FrozenSet[str] = frozenset(
        {
            "model_sha",
            "model_dir",
            "prompt",
            "negative_prompt",
            "sampler",
            "scheduler",
            "steps",
            "cfg_scale",
            "seed",
            "lightning",
            "loras",
            "lora_sha",
            "lora_path",
            "lora_weight",
            "flow_shift",
        }
    )
    WAN_ASSETS: FrozenSet[str] = frozenset(
        {
            "wan_format",
            "wan_metadata_repo",
            "wan_metadata_dir",
            "wan_tokenizer_dir",
            "wan_vae_sha",
            "wan_tenc_sha",
            "wan_vae_path",
            "wan_text_encoder_path",
            "wan_text_encoder_dir",
        }
    )
    GGUF_RUNTIME: FrozenSet[str] = frozenset(
        {
            "gguf_offload",
            "gguf_offload_level",
            "gguf_sdpa_policy",
            "gguf_attention_mode",
            "gguf_attn_chunk",
            "gguf_cache_policy",
            "gguf_cache_limit_mb",
            "gguf_log_mem_interval",
            "gguf_te_device",
        }
    )
    TXT2VID: FrozenSet[str] = frozenset(
        {
            "txt2vid_prompt",
            "txt2vid_neg_prompt",
            "txt2vid_width",
            "txt2vid_height",
            "txt2vid_steps",
            "txt2vid_fps",
            "txt2vid_num_frames",
            "txt2vid_sampler",
            "txt2vid_scheduler",
            "txt2vid_seed",
            "txt2vid_cfg_scale",
            "txt2vid_styles",
        }
    )
    IMG2VID: FrozenSet[str] = frozenset(
        {
            "img2vid_prompt",
            "img2vid_neg_prompt",
            "img2vid_width",
            "img2vid_height",
            "img2vid_steps",
            "img2vid_fps",
            "img2vid_num_frames",
            "img2vid_sampler",
            "img2vid_scheduler",
            "img2vid_seed",
            "img2vid_cfg_scale",
            "img2vid_styles",
            "img2vid_init_image",
            "img2vid_chunk_frames",
            "img2vid_overlap_frames",
            "img2vid_anchor_alpha",
            "img2vid_reset_anchor_to_base",
            "img2vid_chunk_seed_mode",
            "img2vid_chunk_buffer_mode",
            "img2vid_mode",
            "img2vid_window_frames",
            "img2vid_window_stride",
            "img2vid_window_commit_frames",
            "img2vid_resize_mode",
            "img2vid_crop_offset_x",
            "img2vid_crop_offset_y",
        }
    )

    @property
    def COMMON(self) -> FrozenSet[str]:
        return (
            self.DEVICE
            | self.REVISION
            | self.VIDEO_EXPORT
            | self.VIDEO_INTERPOLATION
            | self.VIDEO_UPSCALING
            | self.WAN_STAGE_CONTAINERS
            | self.WAN_ASSETS
            | self.GGUF_RUNTIME
        )

    @property
    def TXT2VID_ALL(self) -> FrozenSet[str]:
        return self.COMMON | self.TXT2VID

    @property
    def IMG2VID_ALL(self) -> FrozenSet[str]:
        return self.COMMON | self.IMG2VID


WAN22_REQUEST_KEYS = Wan22RequestKeys()


def remap_wan22_lora_logical_key(logical_key: str) -> str | None:
    """Map a WAN22 LoRA logical key to a canonical WAN22 transformer `.weight` key.

    Supported logical-key families:
    - Canonical Codex style (`blocks.N.self_attn.q`, `blocks_N_self_attn_q`, `blocks.N.ffn.0`, `blocks_N_ffn_0`)
    - Diffusers style (`blocks.N.attn1.to_q`, `blocks_N_attn1_to_q`, `blocks.N.attn1.to_out.0`, `blocks_N_attn1_to_out_0`,
      `blocks.N.ffn.net.0.proj`, `blocks_N_ffn_net_0_proj`)
    - Optional LoRA wrappers (`lora_unet_`, `lycoris_`)
    """

    key = strip_repeated_prefixes(str(logical_key), _PREFIXES)
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    for prefix in _LORA_LOGICAL_PREFIXES:
        if key.startswith(prefix):
            key = key[len(prefix) :]
            break

    m = _RX_LORA_CANONICAL_ATTN_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.{m.group('which')}.{m.group('proj')}.weight"

    m = _RX_LORA_CANONICAL_FFN_DOT.match(key)
    if m:
        return f"blocks.{m.group('idx')}.ffn.{m.group('which')}.weight"

    m = _RX_LORA_CANONICAL_ATTN_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.{m.group('which')}.{m.group('proj')}.weight"

    m = _RX_LORA_CANONICAL_FFN_UNDERSCORE.match(key)
    if m:
        return f"blocks.{m.group('idx')}.ffn.{m.group('which')}.weight"

    m = _RX_LORA_DIFFUSERS_ATTN_DOT.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.{m.group('proj')}.weight"

    m = _RX_LORA_DIFFUSERS_ATTN_UNDERSCORE.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.{m.group('proj')}.weight"

    m = _RX_LORA_DIFFUSERS_OUT_DOT.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.o.weight"

    m = _RX_LORA_DIFFUSERS_OUT_UNDERSCORE.match(key)
    if m:
        which = "self_attn" if m.group("which") == "attn1" else "cross_attn"
        return f"blocks.{m.group('idx')}.{which}.o.weight"

    m = _RX_LORA_DIFFUSERS_FFN_DOT.match(key)
    if m:
        which = "0" if m.group("which") == "0.proj" else "2"
        return f"blocks.{m.group('idx')}.ffn.{which}.weight"

    m = _RX_LORA_DIFFUSERS_FFN_UNDERSCORE.match(key)
    if m:
        which = "0" if m.group("which") == "0_proj" else "2"
        return f"blocks.{m.group('idx')}.ffn.{which}.weight"

    return None


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


__all__ = [
    "Wan22RequestKeys",
    "WAN22_REQUEST_KEYS",
    "remap_wan22_lora_logical_key",
    "remap_wan22_transformer_state_dict",
]
