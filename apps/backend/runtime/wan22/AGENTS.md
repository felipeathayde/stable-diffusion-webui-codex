# apps/backend/runtime/wan22 Overview
Date: 2025-10-28
Owner: Runtime WAN Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- WAN 2.2-specific runtime components (GGUF loaders, diffusion stages, scheduler logic) used by WAN engines.

## Notes
- Ensure updates stay synchronized with `apps/backend/engines/wan22/` and GGUF helpers under `runtime/gguf`.
- 2025-12-04: `_vae_encode_init`/`_get_text_context` continue to require explicit `vae_dir`/`text_encoder_dir`; these are now commonly populated via WAN22 defaults in `apps/paths.json` when GGUF engines do not receive explicit extras.

## Invariants & Logging (Fase 5)
- `_get_text_context` (GGUF):
  - Requer tokenizer dir válido; carrega `AutoTokenizer` com `local_files_only=True`.
  - TE:
    - `cuda_fp8` exige kernel disponível; sem fallback.
    - Em HF (`_Enc`), valida `hidden_size` do output com `config.hidden_size`.
  - Logs (INFO): shapes de `input_ids` e `last_hidden_state` (prompt/negative), dtype/device.
- Engine (Diffusers path):
  - Loga device/dtype efetivos e tipos das principais componentes (TE/UNet/VAE).
