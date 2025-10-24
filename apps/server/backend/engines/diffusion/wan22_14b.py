from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional

from apps.server.backend.core.engine_interface import EngineCapabilities, TaskType
from apps.server.backend.core.requests import InferenceEvent, Txt2VidRequest, Img2VidRequest
from apps.server.backend.engines.diffusion.base_video import BaseVideoEngine
from apps.server.backend.engines.diffusion.txt2vid import run_txt2vid as _run_t2v
from apps.server.backend.engines.diffusion.img2vid import run_img2vid as _run_i2v
from apps.server.backend.core.exceptions import EngineLoadError

from dataclasses import dataclass
from typing import Any, Optional
import os


from .wan22_common import EngineOpts, WanComponents


class Wan2214BEngine(BaseVideoEngine):
    engine_id = "wan22_14b"

    def __init__(self) -> None:
        super().__init__()
        self._comp: Optional[WanComponents] = None
        self._opts = EngineOpts()

    def capabilities(self) -> EngineCapabilities:  # type: ignore[override]
        return EngineCapabilities(
            engine_id=self.engine_id,
            tasks=(TaskType.TXT2VID, TaskType.IMG2VID),
            model_types=("wan-2.2-14b",),
            precision=("fp16", "bf16", "fp32"),
            extras={"notes": "WAN 2.2 14B via Diffusers or GGUF"},
        )

    # ------------------------------ lifecycle
    def load(self, model_ref: str, **options: Any) -> None:  # type: ignore[override]
        dev = str(options.get("device", "auto"))
        dty = str(options.get("dtype", "fp16"))
        self._opts = EngineOpts(device=dev, dtype=dty)
        comp = WanComponents()
        # Resolve path
        p = model_ref
        try:
            if os.path.isfile(p):
                p = os.path.dirname(p)
        except Exception:
            pass
        if not os.path.isabs(p):
            p = os.path.abspath(p)
        if not os.path.isdir(p):
            alt = os.path.abspath(os.path.join("models", "Wan", model_ref))
            if os.path.isdir(alt):
                p = alt
            else:
                raise EngineLoadError(f"WAN22 14B model path not found: {model_ref}")

        # Try Diffusers pipeline
        pipe = None
        vae = None
        try:
            from diffusers import AutoencoderKLWan  # type: ignore
            from diffusers import WanPipeline  # type: ignore
            import torch  # type: ignore
            torch_dtype = {"fp16": torch.float16, "bf16": getattr(torch, "bfloat16", torch.float16), "fp32": torch.float32}[dty.lower() if dty.lower() in ("fp16", "bf16", "fp32") else "fp16"]
            vae = AutoencoderKLWan.from_pretrained(p, subfolder="vae", torch_dtype=torch_dtype, local_files_only=True)
            pipe = WanPipeline.from_pretrained(p, torch_dtype=torch_dtype, vae=vae, local_files_only=True)
            target_device = "cuda" if (dev in ("auto", "cuda")) and getattr(__import__('torch').cuda, 'is_available', lambda: False)() else "cpu"
            pipe = pipe.to(target_device)
            try:
                from apps.server.backend.engines.util.attention_backend import apply_to_diffusers_pipeline as apply_attn  # type: ignore
                from apps.server.backend.engines.util.accelerator import apply_to_diffusers_pipeline as apply_accel  # type: ignore
                apply_attn(pipe, logger=self._logger)
                apply_accel(pipe, logger=self._logger)
            except Exception:
                pass
            comp.pipeline = pipe
            comp.vae = vae
            comp.device = target_device
            comp.dtype = dty
            self._logger.info("WAN22 14B diffusers pipeline loaded: %s", p)
        except Exception as exc:
            self._logger.warning("WAN22 14B diffusers pipeline not available (%s): %s", p, exc)
            comp.device = "cuda" if dev == "cuda" else "cpu"
            comp.dtype = dty

        # GGUF detection (single-folder); forward mapeado depois
        try:
            if comp.pipeline is None:
                if any(fn.lower().endswith('.gguf') for fn in os.listdir(p)):
                    comp.high_dir = p
                    comp.low_dir = p
                    self._logger.info("WAN22 14B GGUF detected at %s; will use GGUF path when wired", p)
        except Exception:
            pass

        comp.model_dir = p
        self._comp = comp
        self.mark_loaded()

    def unload(self) -> None:  # type: ignore[override]
        self._comp = None
        self.mark_unloaded()

    # ------------------------------ tasks
    def txt2vid(self, request: Txt2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_t2v(engine=self, comp=self._comp, request=request)

    def img2vid(self, request: Img2VidRequest, **kwargs: Any) -> Iterator[InferenceEvent]:  # type: ignore[override]
        self.ensure_loaded()
        assert self._comp is not None
        yield from _run_i2v(engine=self, comp=self._comp, request=request)

    # ------------------------------ helpers
    # No per-stage GGUF builder needed here; runtimes will read request.extras directly when needed
