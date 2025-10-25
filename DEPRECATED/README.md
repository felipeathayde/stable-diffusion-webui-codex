DEPRECATED TREE
================

This folder contains legacy reference code that MUST NOT be imported by the
active backend. It remains only for historical context and diffs.

Moved here on 2025-10-25:
- `wan_gguf/` — plugin wrapper prototype
- `wan_gguf_core/` — native GGUF core prototype and helpers

Source of truth for WAN 2.2 lives in:
- `apps/server/backend/runtime/nn/wan22.py`
- `apps/server/backend/engines/diffusion/wan22_14b.py`
- `apps/server/backend/engines/diffusion/wan22_5b.py`

If you find any import path referencing these packages, treat as a bug.

