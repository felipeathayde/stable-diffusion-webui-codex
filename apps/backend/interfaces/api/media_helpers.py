"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Thin API helpers for media encoding/decoding.
Delegates URL verification and base64 image encode/decode to the shared `MediaService` so API modules don't duplicate codec logic.

Symbols (top-level; keep in sync; no ghosts):
- `_MEDIA` (constant): Singleton `MediaService` used for image encode/decode.
- `verify_url` (function): Delegates URL verification to `MediaService._verify_url`.
- `decode_base64_to_image` (function): Decodes a base64 string into an image object via `MediaService`.
- `encode_pil_to_base64` (function): Encodes an image object into base64 bytes via `MediaService`.
"""

from __future__ import annotations

from typing import Any

from apps.backend.services.media_service import MediaService

_MEDIA = MediaService()


def verify_url(url: str) -> bool:
    return MediaService._verify_url(url)


def decode_base64_to_image(encoding: str) -> Any:
    return _MEDIA.decode_image(encoding)


def encode_pil_to_base64(image: Any) -> bytes:
    encoded = _MEDIA.encode_image(image)
    return encoded.encode("utf-8")


__all__ = ["verify_url", "decode_base64_to_image", "encode_pil_to_base64"]
