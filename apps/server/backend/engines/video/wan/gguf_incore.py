from __future__ import annotations

"""In-core WAN 2.2 GGUF executor (High/Low) — phase 1.

This module wires the GGUF execution path without relying on external plugins.
Phase 1 loads/validates GGUF weights for High/Low, logs model metadata, and
prepares the execution configuration. The actual UNet forward + sampler loop is
being implemented incrementally in subsequent phases.

Until the forward is implemented, these functions raise GGUFExecutorUnavailable
with actionable context. This avoids silent fallbacks or fake outputs.
"""

from typing import Any, Dict, List, Optional, Tuple

from apps.server.backend.runtime import memory_management
from apps.server.backend.runtime.ops import using_forge_operations
from apps.server.backend.runtime.utils import _load_gguf_state_dict, read_arbitrary_config

from .gguf_exec import GGUFExecutorUnavailable, WanGGUFRunConfig


def _normalize_win_path(p: str) -> str:
    """Best-effort: map Windows paths like C:\\Users\\... to /mnt/c/Users/... when running on Linux/WSL.
    If mapping is not applicable, returns the original path.
    """
    import os, re
    if os.name == 'nt':
        return p  # native Windows
    m = re.match(r'^[A-Za-z]:\\', p)
    if m:
        drive = p[0].lower()
        rest = p[2:].lstrip('\\/')
        candidate = f"/mnt/{drive}/" + rest.replace('\\\\', '/').replace('\\', '/')
        return candidate
    return p


def _first_gguf_in(dir_path: Optional[str]) -> Optional[str]:
    import os
    if not dir_path:
        return None
    _dir = _normalize_win_path(dir_path)
    abspath = _dir if os.path.isabs(_dir) else os.path.abspath(_dir)
    try:
        for fn in os.listdir(abspath):
            if fn.lower().endswith('.gguf'):
                return os.path.join(abspath, fn)
    except Exception:
        return None
    return None


def _pick_stage_gguf(dir_path: Optional[str], stage: str) -> Optional[str]:
    """Prefer a .gguf whose filename mentions the stage (high/low). Fallback to first .gguf."""
    import os
    p = _first_gguf_in(dir_path)
    if not dir_path:
        return p
    _dir = _normalize_win_path(dir_path)
    abspath = _dir if os.path.isabs(_dir) else os.path.abspath(_dir)
    try:
        stage_lc = stage.lower().strip()
        cands = [fn for fn in os.listdir(abspath) if fn.lower().endswith('.gguf')]
        for fn in cands:
            name = fn.lower()
            if stage_lc == 'high' and ('high' in name or 'highnoise' in name):
                return os.path.join(abspath, fn)
            if stage_lc == 'low' and ('low' in name or 'lownoise' in name):
                return os.path.join(abspath, fn)
    except Exception:
        pass
    return p


def _summarize(path: str) -> str:
    try:
        sd = _load_gguf_state_dict(path)
        # Provide a tiny digest (tensor count + a few keys)
        keys = list(sd.keys())
        return f"{len(keys)} tensors; sample keys: {keys[:3]}"
    except Exception as ex:
        return f"failed to read: {ex}"


class _WanUNetGGUF:
    """Minimal UNet wrapper placeholder that owns a GGUF state dict and metadata.

    This class does not yet implement forward(). It prepares the tensor views
    and bakes GGUF tensors once, so wiring the module graph becomes mechanical
    in the next step.
    """

    def __init__(self, gguf_path: str) -> None:
        self.gguf_path = gguf_path
        self.state: Dict[str, Any] = _load_gguf_state_dict(gguf_path)
        self.meta: Dict[str, Any] = {}
        # Attempt to probe a sibling config.json for latent/channel sizes
        try:
            cfg_dir = gguf_path.rsplit('/', 1)[0]
            self.meta = read_arbitrary_config(cfg_dir)
        except Exception:
            self.meta = {}
        # One-time bake of GGUF tensors to enable faster dequant ops on first use
        class _Dummy:
            def parameters(self):
                for v in self.state.values():
                    if hasattr(v, "gguf_cls"):
                        yield v

        try:
            memory_management.bake_gguf_model(_Dummy())
        except Exception:
            pass

    def summary(self) -> str:
        keys = list(self.state.keys())
        return f"{len(keys)} tensors; e.g., {keys[:3]}"

    # Placeholder forward; real mapping comes next iteration
    def forward(self, x, t, cond, guidance_scale: Optional[float] = None):  # noqa: D401
        raise GGUFExecutorUnavailable(
            "WAN GGUF in-core forward() is being wired: layer mapping pending"
        )


def _prep_latents_from_image(init_image: object, width: int, height: int, device: str, dtype: str):
    """Best-effort VAE encode for init image → latent. Uses diffusers AutoencoderKLWan if available.

    If VAE is not available, this function raises GGUFExecutorUnavailable with
    guidance to use the Diffusers path for now.
    """
    try:
        import torch  # type: ignore
        from diffusers import AutoencoderKLWan  # type: ignore
        # Heuristic: expect init_image as PIL.Image.Image or numpy; delegate to VAE preprocess
        torch_dtype = {
            'fp16': torch.float16,
            'bf16': getattr(torch, 'bfloat16', torch.float16),
            'fp32': torch.float32,
        }.get(dtype, torch.float16)
        vae = AutoencoderKLWan.from_pretrained('.', subfolder='vae', torch_dtype=torch_dtype, local_files_only=True)
        vae = vae.to('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
        # Diffusers pipelines handle preprocessing internally; here we assume a tensor input already
        if hasattr(init_image, 'to'):
            x = init_image
        else:
            raise GGUFExecutorUnavailable("init_image preprocessing not available for GGUF path; use Diffusers path for now")
        with torch.no_grad():
            latents = vae.encode(x).latent_dist.sample() * 0.18215
        return latents
    except Exception as ex:  # noqa: BLE001
        raise GGUFExecutorUnavailable(
            f"VAE encode for GGUF init image not available: {ex}. Use Diffusers path meanwhile."
        ) from ex


def _make_linear_schedule(steps: int) -> List[float]:
    # Placeholder simple schedule (0..1). Real betas/sigmas wired with sampler next.
    steps = max(1, int(steps))
    return [i / steps for i in range(steps)]


def run_txt2vid(cfg: WanGGUFRunConfig, logger) -> List[object]:
    hi = _pick_stage_gguf(cfg.high.model_dir if cfg.high else None, 'high')
    lo = _pick_stage_gguf(cfg.low.model_dir if cfg.low else None, 'low')
    if not hi or not lo:
        raise GGUFExecutorUnavailable(
            f"WAN GGUF (T2V) requires .gguf for both stages (found high={bool(hi)} low={bool(lo)})"
        )
    if logger:
        logger.info("[wan-gguf] high: %s (%s)", hi, _summarize(hi))
        logger.info("[wan-gguf] low:  %s (%s)", lo, _summarize(lo))

    # Load stage UNets (GGUF) and prepare wrappers
    hi_unet = _WanUNetGGUF(hi)
    lo_unet = _WanUNetGGUF(lo)
    if logger:
        logger.info("[wan-gguf] high UNet summary: %s", hi_unet.summary())
        logger.info("[wan-gguf] low  UNet summary: %s", lo_unet.summary())

    # Prepare schedule (respect user steps; Lightning may reduce by recommendation but not enforced)
    h_steps = int((cfg.high.steps if cfg.high else 12) or 12)
    l_steps = int((cfg.low.steps if cfg.low else 12) or 12)
    h_sched = _make_linear_schedule(h_steps)
    l_sched = _make_linear_schedule(l_steps)

    # Not yet wired: UNet forward. Keep explicit error (no fake output)
    raise GGUFExecutorUnavailable("WAN GGUF in-core: forward/sampler mapping pending (txt2vid)")


def run_img2vid(cfg: WanGGUFRunConfig, logger) -> List[object]:
    hi = _pick_stage_gguf(cfg.high.model_dir if cfg.high else None, 'high')
    lo = _pick_stage_gguf(cfg.low.model_dir if cfg.low else None, 'low')
    if not hi or not lo:
        raise GGUFExecutorUnavailable(
            f"WAN GGUF (I2V) requires .gguf for both stages (found high={bool(hi)} low={bool(lo)})"
        )
    if logger:
        logger.info("[wan-gguf] high: %s (%s)", hi, _summarize(hi))
        logger.info("[wan-gguf] low:  %s (%s)", lo, _summarize(lo))

    # Load stage UNets (GGUF) and prepare wrappers
    hi_unet = _WanUNetGGUF(hi)
    lo_unet = _WanUNetGGUF(lo)
    if logger:
        logger.info("[wan-gguf] high UNet summary: %s", hi_unet.summary())
        logger.info("[wan-gguf] low  UNet summary: %s", lo_unet.summary())

    # Prepare schedule (respect user steps)
    h_steps = int((cfg.high.steps if cfg.high else 12) or 12)
    l_steps = int((cfg.low.steps if cfg.low else 12) or 12)
    h_sched = _make_linear_schedule(h_steps)
    l_sched = _make_linear_schedule(l_steps)

    # Not yet wired: VAE encode + UNet forward. Keep explicit error (no fake output)
    raise GGUFExecutorUnavailable("WAN GGUF in-core: VAE encode and forward/sampler mapping pending (img2vid)")
