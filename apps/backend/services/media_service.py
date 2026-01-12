"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image encode/decode helpers for API/service layers.
Decodes base64/data URIs or (optionally) fetches remote URLs under env guards, and encodes images with PNG text metadata or JPEG/WEBP EXIF parameters.

Symbols (top-level; keep in sync; no ghosts):
- `MediaService` (class): Decode/encode helpers used by the API and progress services.
"""

from __future__ import annotations

import base64
from io import BytesIO

import os
import requests
from PIL import Image, PngImagePlugin
import piexif
import piexif.helper

from apps.backend.services import options_store


class MediaService:
    """Image encode/decode helpers for API/service layers."""

    @staticmethod
    def _verify_url(url: str) -> bool:
        """Returns True if the url refers to a global resource."""
        import socket
        import ipaddress
        from urllib.parse import urlparse
        try:
            parsed_url = urlparse(url)
            domain_name = parsed_url.netloc
            host = socket.gethostbyname_ex(domain_name)
            for ip in host[2]:
                ip_addr = ipaddress.ip_address(ip)
                if not ip_addr.is_global:
                    return False
        except Exception:
            return False
        return True

    def decode_image(self, encoding: str):
        """Decode base64 or fetch from URL.

        Uses environment flags to guard requests:
        - CODEX_API_ENABLE_REQUESTS (default: 0)
        - CODEX_API_FORBID_LOCAL (default: 1)
        - CODEX_API_USERAGENT (optional)
        """
        if encoding.startswith("http://") or encoding.startswith("https://"):
            if os.getenv("CODEX_API_ENABLE_REQUESTS", "0") not in ("1", "true", "yes", "on"):
                raise ValueError("Requests not allowed")

            forbid_local = os.getenv("CODEX_API_FORBID_LOCAL", "1") in ("1", "true", "yes", "on")
            if forbid_local and not self._verify_url(encoding):
                raise ValueError("Request to local resource not allowed")

            ua = os.getenv("CODEX_API_USERAGENT", "")
            headers = {'user-agent': ua} if ua else {}
            response = requests.get(encoding, timeout=30, headers=headers)
            return Image.open(BytesIO(response.content)).copy()

        if encoding.startswith("data:image/"):
            encoding = encoding.split(";")[1].split(",")[1]

        return Image.open(BytesIO(base64.b64decode(encoding))).copy()

    def encode_image(self, image) -> str:
        with BytesIO() as output_bytes:
            if isinstance(image, str):
                return image
            fmt = str(options_store.get_value("samples_format", "png") or "png").strip().lower()
            jpeg_quality = int(options_store.get_value("jpeg_quality", 80) or 80)
            webp_lossless = str(options_store.get_value("webp_lossless", False)).strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
            if fmt == 'png':
                use_metadata = False
                metadata = PngImagePlugin.PngInfo()
                for key, value in image.info.items():
                    if isinstance(key, str) and isinstance(value, str):
                        metadata.add_text(key, value)
                        use_metadata = True
                image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=jpeg_quality)
            elif fmt in ("jpg", "jpeg", "webp"):
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                parameters = image.info.get('parameters', None)
                exif_bytes = piexif.dump({
                    "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
                })
                if fmt in ("jpg", "jpeg"):
                    image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=jpeg_quality)
                else:
                    image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=jpeg_quality, lossless=webp_lossless)
            else:
                raise ValueError("Invalid image format")

            bytes_data = output_bytes.getvalue()
            return base64.b64encode(bytes_data).decode('utf-8')
