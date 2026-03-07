# AGENT — Runtime State Dict Helpers

Purpose: Lightweight state-dict mapping views + small state-dict utilities used by loaders and runtime codepaths.

Key files:
- `apps/backend/runtime/state_dict/keymap_flux_transformer.py`: Flux transformer key-style resolver (native/source Diffusers or internal fused layout → Flux runtime lookup space).
- `apps/backend/runtime/state_dict/keymap_flux2_transformer.py`: FLUX.2 transformer key-style resolver (legacy fused core layout or native/source Diffusers → Diffusers `Flux2Transformer2DModel` lookup space).
- `apps/backend/runtime/state_dict/key_mapping.py`: Strict key-style detection + keyspace resolver core (fail loud; collision/ambiguity checks).
- `apps/backend/runtime/state_dict/keymap_llama_gguf.py`: llama.cpp-style GGUF tensor-name resolver for text models (HF key layout).
- `apps/backend/runtime/state_dict/keymap_qwen_text_encoder.py`: Qwen text-encoder key-style resolver (HF/wrapped layouts → canonical `model.*` backbone keys; optional aux heads accepted).
- `apps/backend/runtime/state_dict/keymap_sdxl_clip.py`: SDXL base text-encoder key mapping (CLIP-L/CLIP-G → Codex IntegratedCLIP layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_checkpoint.py`: SDXL checkpoint wrapper/prefix key normalization (checkpoint-wrapper / original SDXL layout).
- `apps/backend/runtime/state_dict/keymap_sdxl_vae.py`: SDXL/Flow16 VAE key-style resolver (LDM-style → diffusers AutoencoderKL).
- `apps/backend/runtime/state_dict/keymap_t5_text_encoder.py`: T5 text-encoder key-style resolver (HF `encoder.*`/`shared.weight` → IntegratedT5 `transformer.*`).
- `apps/backend/runtime/state_dict/keymap_zimage_transformer.py`: Z Image transformer key-style resolver (native/source Diffusers or internal fused layout → Z Image runtime lookup space).
- `apps/backend/runtime/state_dict/keymap_wan21_vae.py`: WAN2.1 VAE key-style resolver with strict canonical validation (wrapper-strip + required-key fail-loud).
- `apps/backend/runtime/state_dict/keymap_wan22_vae.py`: WAN22 VAE key-style resolvers for 2D native and 3D diffusers/codex lanes (mixed-style/collision fail-loud).
- `apps/backend/runtime/state_dict/keymap_wan22_transformer.py`: WAN22 transformer key-style resolver (Diffusers/WAN-export/Codex).
- `apps/backend/runtime/state_dict/tools.py`: Small state-dict utilities and diagnostics helpers.
- `apps/backend/runtime/state_dict/views.py`: Mapping views (prefix/filter/keyspace-lookup/computed-keyspace/cast) + `LazySafetensorsDict`.

Notes:
- Views should stay lightweight and avoid eagerly materializing large state dicts.
- 2026-01-25: `LazySafetensorsDict` is now truly lazy on non-Windows (persistent `safe_open` handle) and implements `__contains__` so key checks don’t load tensors; `KeyspaceLookupView` also implements `__contains__` for the same reason.
- Helpers should remain generic and not import model-family runtime code.
- Keyspace resolution must be explicit and strict: unknown/ambiguous layouts raise (no silent fallbacks). Use the family-specific keymap modules from loaders.
- 2026-02-10: Added canonical T5 text-encoder keymap (`keymap_t5_text_encoder.py`) so loader paths no longer perform ad-hoc inline prefix normalization.
- 2026-02-10: Expanded SDXL checkpoint keymap to normalize nested UNet label-embedding keys (`label_emb.0.0.*` → `label_emb.0.*`), removing parser-side SDXL UNet normalization converter usage.
- 2026-02-10: Structural conversion seams in keymaps are globally policy-gated by `CODEX_WEIGHT_STRUCTURAL_CONVERSION` (`auto` fail-loud / `convert` explicit opt-in): SDXL CLIP blocks split↔fused QKV/projection conversion in `auto`, and SDXL VAE blocks 1x1-conv flattening in `auto`.
- 2026-02-11: `keymap_sdxl_clip.py` now exposes generic CLIP layout detection + resolver APIs (`detect_clip_layout_metadata`, `resolve_clip_keyspace_with_layout`) and SDXL wrappers with cache-hint support (`*_with_layout`) to avoid repeated style detection on warm SHA layout cache hits.
- 2026-02-11: SDXL CLIP projection handling is orientation-aware (`auto|linear|matmul`) instead of hard-coded transpose; AUTO keeps native orientation and only transposes when explicitly requested (and policy allows structural conversion).
- 2026-02-11: `keymap_sdxl_vae.py` now resolves mid-attention aliases under `encoder/decoder.mid.block_1.{q,k,v,proj_out,norm}.*`, `mid.block_1.attn_1.*`, and prefixed legacy `mid.attn_1.to_{q,k,v,out}.*` to canonical `mid_block.attentions.0.{to_q,to_k,to_v,to_out.0,group_norm}.*`, preventing SDXL VAE missing mid-attention keys on alias-style exports while preserving resnet-key lookup parity.
- 2026-02-11: `keymap_sdxl_vae.py` now also canonicalizes DIFFUSERS mid-attention legacy aliases (`*.mid_block.attentions.*.{query,key,value,proj_attn}.*` → `*.mid_block.attentions.*.{to_q,to_k,to_v,to_out.0}.*`) and fail-loud rejects any leftover alias outputs in validation.
- 2026-02-11: `keymap_sdxl_vae.py` now uses explicit projection lanes for SDXL VAE mid-attention weights independent of global structural-conversion policy: canonical 2D linear weights pass through, native 1x1 conv 4D weights pass through unchanged, and any non-canonical shape fails loud with key+shape context.
- 2026-02-11: Supersedes the SDXL VAE portion of the 2026-02-10 structural-policy note: SDXL VAE mid-attention projection lanes are now native (`linear_2d`/`conv1x1_4d`) and no longer use keymap flatten gating by `CODEX_WEIGHT_STRUCTURAL_CONVERSION`.
- 2026-02-15: `views.LazySafetensorsDict` now explicitly documents device-targeted lazy loads (`device` controls produced tensor placement; no CPU-only assumption).
- 2026-03-06: `views.py` now also exposes `ComputedKeyspaceView`, which mixes direct lookups with quantization-aware row concat/split/swap transforms for fused/unfused runtime keyspaces without mutating source checkpoints.
- 2026-02-28: `keymap_wan22_transformer.py` now exposes `resolve_wan22_lora_logical_key(...)` as canonical WAN22 LoRA logical-key → transformer-weight mapping authority (supports codex and diffusers-style logical keys, optional `lora_unet_`/`lycoris_` wrappers).
- 2026-03-06: WAN video request allowlists were moved out of `keymap_wan22_transformer.py` into `apps/backend/interfaces/api/wan_video_request_keys.py`; the WAN22 keymap module now owns only transformer keyspace understanding and LoRA logical-key resolution.
- 2026-03-03: Added strict generic Qwen text-encoder keymap (`keymap_qwen_text_encoder.py`) covering wrapped HF layouts and known auxiliary heads (`lm_head.*`, `visual.*`) while failing loud on unknown keyspaces.
- 2026-03-06: Added family-scoped Flux / FLUX.2 / Z Image GGUF keyspace resolvers so native/source checkpoints can be interpreted through lazy lookup views (including fused/unfused tensor conventions) without materializing remapped state dicts.

Last Review: 2026-03-06
