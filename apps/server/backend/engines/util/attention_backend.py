from __future__ import annotations

from typing import Any, Optional


def _get_selected_backend() -> str:
    try:
        from modules import shared as _shared  # type: ignore
        val = getattr(_shared.opts, 'codex_attention_backend', 'torch-sdpa')
        return str(val or 'torch-sdpa')
    except Exception:
        return 'torch-sdpa'


def apply_to_diffusers_pipeline(pipe: Any, *, backend: Optional[str] = None, logger=None) -> str:
    """Apply the chosen attention backend to a diffusers pipeline (if supported).

    Returns the effective backend string applied or attempted.
    """
    choice = (backend or _get_selected_backend()).lower().strip()
    if choice not in ('torch-sdpa', 'xformers', 'sage'):
        choice = 'torch-sdpa'

    # Torch SDPA (Flash/Math/Mem) — default in PyTorch 2.x
    if choice == 'torch-sdpa':
        try:
            # If xformers was previously enabled, disable it when possible
            if hasattr(pipe, 'disable_xformers_memory_efficient_attention'):
                pipe.disable_xformers_memory_efficient_attention()
        except Exception:
            pass
        try:
            import torch  # type: ignore
            # Enable all SDPA modes; PyTorch will choose the best available
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception:
            pass
        if logger:
            logger.info("attention backend: torch-sdpa")
        return 'torch-sdpa'

    # xFormers memory-efficient attention
    if choice == 'xformers':
        try:
            if hasattr(pipe, 'enable_xformers_memory_efficient_attention'):
                pipe.enable_xformers_memory_efficient_attention()
                if logger:
                    logger.info("attention backend: xformers")
                return 'xformers'
        except Exception as ex:
            if logger:
                logger.warning("xformers enable failed: %s", ex)
        # fallback to torch-sdpa if cannot enable
        return apply_to_diffusers_pipeline(pipe, backend='torch-sdpa', logger=logger)

    # SAGE (pluggable)
    if choice == 'sage':
        try:
            import importlib
            mod = None
            for name in ("backend_ext.sage_attention", "sage_attention"):
                try:
                    mod = importlib.import_module(name)
                    break
                except Exception:
                    mod = None
            if mod and hasattr(mod, 'apply'):
                mod.apply(pipe)  # type: ignore[attr-defined]
                if logger:
                    logger.info("attention backend: sage (via plugin)")
                return 'sage'
            if logger:
                logger.warning("sage attention plugin not found; falling back to torch-sdpa")
        except Exception as ex:
            if logger:
                logger.warning("sage attention apply failed: %s; falling back to torch-sdpa", ex)
        return apply_to_diffusers_pipeline(pipe, backend='torch-sdpa', logger=logger)

    return choice

