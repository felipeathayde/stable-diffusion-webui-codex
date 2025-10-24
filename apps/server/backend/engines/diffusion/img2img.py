from __future__ import annotations

"""Image→Image task runtime (skeleton).

Mirrors the organization of txt2img.py while keeping responsibilities local to
the engines that call it. Engines can import helpers from here to avoid large
conditionals.
"""

from typing import Any


def run_img2img(*, engine, processing: Any) -> Any:  # placeholder for parity
    raise NotImplementedError("img2img runtime wiring pending for diffusion engines")

