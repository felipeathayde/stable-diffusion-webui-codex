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
- `keymap_anima` (module): Anima key-style detection + strict remaps for core transformer, WAN VAE, and Qwen3-0.6B text encoder.
- `keymap_llama_gguf` (module): Key remapping helpers for llama.cpp-style GGUF tensor names.
- `keymap_sdxl_checkpoint` (module): SDXL checkpoint wrapper/prefix key normalization (Comfy/original SDXL layout).
- `keymap_sdxl_clip` (module): SDXL base text-encoder key mapping (CLIP-L/CLIP-G → Codex IntegratedCLIP layout).
- `keymap_sdxl_vae` (module): SDXL/Flow16 VAE key-style detection + remapping (LDM-style → diffusers AutoencoderKL).
- `keymap_wan22_transformer` (module): WAN22 transformer key-style detection + remapping (Diffusers/WAN-export/Codex).
- `key_mapping` (module): Strict key-style detection + declarative key-remapping helpers.
- `tools` (module): State-dict diagnostics and helper utilities.
- `views` (module): Lightweight mapping views for state_dict handling.
"""

__all__ = [
    "keymap_anima",
    "keymap_llama_gguf",
    "keymap_sdxl_checkpoint",
    "keymap_sdxl_clip",
    "keymap_sdxl_vae",
    "keymap_wan22_transformer",
    "key_mapping",
    "tools",
    "views",
]
