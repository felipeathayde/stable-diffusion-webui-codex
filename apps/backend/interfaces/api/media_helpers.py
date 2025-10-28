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
