# AGENT — Runtime State Dict Helpers

Purpose: Lightweight state-dict mapping views + small state-dict utilities used by loaders and runtime codepaths.

Key files:
- `apps/backend/runtime/state_dict/key_mapping.py`: Strict key-style detection + remapping core (fail loud; collision/ambiguity checks).
- `apps/backend/runtime/state_dict/keymap_llama_gguf.py`: llama.cpp-style GGUF tensor-name remaps for text models (HF key layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_checkpoint.py`: SDXL checkpoint wrapper/prefix key normalization (Comfy/original SDXL layout).
- `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`: WAN22 transformer key-style detector + remap (Diffusers/WAN-export/Codex).
- `apps/backend/runtime/state_dict/tools.py`: Small state-dict utilities and diagnostics helpers.
- `apps/backend/runtime/state_dict/views.py`: Mapping views (prefix/filter/remap/cast) + `LazySafetensorsDict`.

Notes:
- Views should stay lightweight and avoid eagerly materializing large state dicts.
- Helpers should remain generic and not import model-family runtime code.
- Key remaps must be explicit and strict: unknown/ambiguous layouts raise (no silent fallbacks). Use the family-specific keymap modules from loaders.

Last Review: 2026-01-23
