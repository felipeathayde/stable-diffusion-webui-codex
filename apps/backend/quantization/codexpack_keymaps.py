"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CodexPack keymap identifier registry for runtime loaders.
Defines stable `keymap_id` strings used by CodexPack GGUF manifests so loaders can validate and fail loud on unknown mappings.

Symbols (top-level; keep in sync; no ghosts):
- `ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1` (constant): Z-Image Base core GGUF identity keymap id (no remap).
- `ZIMAGE_TURBO_CORE_GGUF_IDENTITY_V1` (constant): Z-Image Turbo core GGUF identity keymap id (no remap).
- `SUPPORTED_CODEXPACK_KEYMAP_IDS` (constant): Frozen set of supported CodexPack keymap ids.
- `is_supported_codexpack_keymap_id` (function): Returns True when a keymap id is supported.
"""

from __future__ import annotations


ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1 = "tongyi-mai.zimage_base.core.gguf.identity.v1"
ZIMAGE_TURBO_CORE_GGUF_IDENTITY_V1 = "tongyi-mai.zimage_turbo.core.gguf.identity.v1"


SUPPORTED_CODEXPACK_KEYMAP_IDS = frozenset(
    {
        ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1,
        ZIMAGE_TURBO_CORE_GGUF_IDENTITY_V1,
    }
)


def is_supported_codexpack_keymap_id(keymap_id: str) -> bool:
    return str(keymap_id) in SUPPORTED_CODEXPACK_KEYMAP_IDS


__all__ = [
    "SUPPORTED_CODEXPACK_KEYMAP_IDS",
    "ZIMAGE_BASE_CORE_GGUF_IDENTITY_V1",
    "ZIMAGE_TURBO_CORE_GGUF_IDENTITY_V1",
    "is_supported_codexpack_keymap_id",
]
