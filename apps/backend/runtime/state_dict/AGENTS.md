# AGENT — Runtime State Dict Helpers

Purpose: Lightweight state-dict mapping views + small state-dict utilities used by loaders and runtime codepaths.

Key files:
- `apps/backend/runtime/state_dict/key_mapping.py`: Strict key-style detection + remapping core (fail loud; collision/ambiguity checks).
- `apps/backend/runtime/state_dict/keymap_anima.py`: Anima strict key-style detectors + canonical remaps for core transformer/Qwen3-0.6B text encoder.
- `apps/backend/runtime/state_dict/keymap_llama_gguf.py`: llama.cpp-style GGUF tensor-name remaps for text models (HF key layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_clip.py`: SDXL base text-encoder key mapping (CLIP-L/CLIP-G → Codex IntegratedCLIP layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_checkpoint.py`: SDXL checkpoint wrapper/prefix key normalization (Comfy/original SDXL layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_vae.py`: SDXL/Flow16 VAE key-style detection + remapping (LDM-style → diffusers AutoencoderKL).
- `apps/backend/runtime/state_dict/keymap_t5_text_encoder.py`: T5 text-encoder key-style detection + remap (HF `encoder.*`/`shared.weight` → IntegratedT5 `transformer.*`).
- `apps/backend/runtime/state_dict/keymap_wan21_vae.py`: WAN2.1 VAE key-style detector + strict canonical remap (wrapper-strip + required-key fail-loud).
- `apps/backend/runtime/state_dict/keymap_wan22_vae.py`: WAN22 VAE key-style detector/remaps for 2D native and 3D diffusers/codex lanes (mixed-style/collision fail-loud).
- `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`: WAN22 transformer key-style detector + remap (Diffusers/WAN-export/Codex).
- `apps/backend/runtime/state_dict/tools.py`: Small state-dict utilities and diagnostics helpers.
- `apps/backend/runtime/state_dict/views.py`: Mapping views (prefix/filter/remap/cast) + `LazySafetensorsDict`.

Notes:
- Views should stay lightweight and avoid eagerly materializing large state dicts.
- 2026-01-25: `LazySafetensorsDict` is now truly lazy on non-Windows (persistent `safe_open` handle) and implements `__contains__` so key checks don’t load tensors; `RemapKeysView` also implements `__contains__` for the same reason.
- Helpers should remain generic and not import model-family runtime code.
- Key remaps must be explicit and strict: unknown/ambiguous layouts raise (no silent fallbacks). Use the family-specific keymap modules from loaders.
- 2026-02-08: Added strict Anima keymaps (`keymap_anima.py`) with explicit style detection, wrapper normalization, required-key validation, and collision/unknown fail-loud behavior.
- 2026-02-10: Added canonical T5 text-encoder keymap (`keymap_t5_text_encoder.py`) so loader paths no longer perform ad-hoc inline prefix normalization.
- 2026-02-10: Expanded SDXL checkpoint keymap to normalize nested UNet label-embedding keys (`label_emb.0.0.*` → `label_emb.0.*`), removing parser-side SDXL UNet normalization converter usage.
- 2026-02-10: Structural conversion seams in keymaps are globally policy-gated by `CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto` fail-loud / `convert` explicit opt-in): SDXL CLIP blocks split↔fused QKV/projection conversion in `auto`, and SDXL VAE blocks 1x1-conv flattening in `auto`.
- 2026-02-11: `keymap_sdxl_clip.py` now exposes generic CLIP layout detection/remap APIs (`detect_clip_layout_metadata`, `remap_clip_state_dict_with_layout`) and SDXL wrappers with cache-hint support (`*_with_layout`) to avoid repeated style detection on warm SHA layout cache hits.
- 2026-02-11: SDXL CLIP projection handling is orientation-aware (`auto|linear|matmul`) instead of hard-coded transpose; AUTO keeps native orientation and only transposes when explicitly requested (and policy allows structural conversion).
- 2026-02-11: `keymap_sdxl_vae.py` now maps mid-attention aliases under `encoder/decoder.mid.block_1.{q,k,v,proj_out,norm}.*`, `mid.block_1.attn_1.*`, and prefixed legacy `mid.attn_1.to_{q,k,v,out}.*` to canonical `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}.*`, preventing SDXL VAE missing mid-attention keys on alias-style exports while preserving resnet-key remaps.
- 2026-02-11: `keymap_sdxl_vae.py` now also canonicalizes DIFFUSERS mid-attention legacy aliases (`*.mid_block.attentions.*.{query,key,value,proj_attn}.*` → `*.mid_block.attentions.*.{to_q,to_k,to_v,to_out.0}.*`) and fail-loud rejects any leftover alias outputs in validation.
- 2026-02-11: `keymap_sdxl_vae.py` now uses explicit projection lanes for SDXL VAE mid-attention weights independent of global structural-conversion policy: canonical 2D linear weights pass through, native 1x1 conv 4D weights pass through unchanged, and any non-canonical shape fails loud with key+shape context.
- 2026-02-11: Supersedes the SDXL VAE portion of the 2026-02-10 structural-policy note: SDXL VAE mid-attention projection lanes are now native (`linear_2d`/`conv1x1_4d`) and no longer use keymap flatten gating by `CODEX_WEIGHT_STRUCTURAL_CONVERSION`.
- 2026-02-15: `views.LazySafetensorsDict` now explicitly documents device-targeted lazy loads (`device` controls produced tensor placement; no CPU-only assumption).
- 2026-02-16: Split WAN VAE ownership into model-specific modules (`keymap_wan21_vae.py`, `keymap_wan22_vae.py`), removed WAN VAE remap ownership from `keymap_anima.py`, and wired WAN22 VAE lanes to model keymap modules only.
- 2026-02-17: `keymap_wan22_transformer.py` WAN22 request allowlists now include `gguf_attention_mode` and img2vid chunk-control keys (`img2vid_chunk_frames`, `img2vid_overlap_frames`, `img2vid_anchor_alpha`, `img2vid_chunk_seed_mode`) for strict router passthrough ownership.
- 2026-02-21: `keymap_wan22_transformer.py` WAN22 img2vid request allowlist now also includes `img2vid_chunk_buffer_mode` for strict passthrough ownership.
- 2026-02-21: `keymap_wan22_transformer.py` WAN22 img2vid request allowlist now includes temporal-mode keys (`img2vid_mode`, `img2vid_window_frames`, `img2vid_window_stride`, `img2vid_window_commit_frames`) so router unknown-key checks remain strict with sliding-window mode.

Last Review: 2026-02-21
