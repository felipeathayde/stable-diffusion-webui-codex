# AGENT — apps/backend/patchers/controlnet/architectures/sd
Date: 2025-10-31
Last Review: 2026-02-18
Status: Active

## Purpose
- SD/SDXL-compatible ControlNet implementations (full ControlNet, ControlNet Lite placeholder, Control Lora, T2I Adapter).
- Shared building blocks used by `apply_controlnet_advanced` and `UnetPatcher`.

## Notes
- `control.py` — core ControlNet class (hint resize, manual cast, transformer option handling).
- `control_lite.py` — placeholder raising `NotImplementedError` until Lite variants are ported.
- `lora.py` — Control Lora implementation with SafeTensor materialisation.
- `t2i_adapter.py` — Adapter-based conditioning path.
- 2026-02-18: `t2i_adapter.py` no longer emits direct stdout prints for state-dict mismatch notices; it now logs structured warnings/debug entries via backend logger namespace.
