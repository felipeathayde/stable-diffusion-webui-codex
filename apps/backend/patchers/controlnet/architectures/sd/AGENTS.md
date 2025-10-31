# AGENT — apps/backend/patchers/controlnet/architectures/sd
Date: 2025-10-31  
Owner: Backend Runtime Maintainers  
Last Review: 2025-10-31  
Status: Active

## Purpose
- SD/SDXL-compatible ControlNet implementations (full ControlNet, ControlNet Lite placeholder, Control Lora, T2I Adapter).
- Shared building blocks used by `apply_controlnet_advanced` and `UnetPatcher`.

## Notes
- `control.py` — core ControlNet class (hint resize, manual cast, transformer option handling).
- `control_lite.py` — placeholder raising `NotImplementedError` until Lite variants are ported.
- `lora.py` — Control Lora implementation with SafeTensor materialisation.
- `t2i_adapter.py` — Adapter-based conditioning path.
