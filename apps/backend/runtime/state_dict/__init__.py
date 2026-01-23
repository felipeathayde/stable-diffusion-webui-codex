"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Runtime state-dict views and helpers.
Provides mapping views and small utilities used during checkpoint loading/normalization.

Symbols (top-level; keep in sync; no ghosts):
- `keymap_llama_gguf` (module): Key remapping helpers for llama.cpp-style GGUF tensor names.
- `keymap_wan22_transformer` (module): WAN22 transformer key-style detection + remapping (Diffusers/WAN-export/Codex).
- `key_mapping` (module): Strict key-style detection + declarative key-remapping helpers.
- `tools` (module): State-dict diagnostics and helper utilities.
- `views` (module): Lightweight mapping views for state_dict handling.
"""

__all__ = [
    "keymap_llama_gguf",
    "keymap_wan22_transformer",
    "key_mapping",
    "tools",
    "views",
]
