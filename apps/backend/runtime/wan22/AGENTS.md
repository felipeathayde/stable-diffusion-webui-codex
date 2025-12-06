# apps/backend/runtime/wan22 Overview
Date: 2025-12-06
Owner: Runtime WAN Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- WAN 2.2-specific runtime components (GGUF loaders, diffusion stages, scheduler logic, optional core streaming) used by WAN engines.

## Notes
- Ensure updates stay synchronized with `apps/backend/engines/wan22/` and GGUF helpers under `runtime/gguf`.
- 2025-12-04: `_vae_encode_init`/`_get_text_context` continuam a requerer `vae_dir`/`text_encoder_dir` explícitos; esses diretórios são normalmente preenchidos via WAN22 defaults em `apps/paths.json` quando engines GGUF não recebem `extras` explícitos.
- 2025-12-06: `streaming/` adiciona infraestrutura de streaming de core para WAN 2.2 14B em GGUF (`WanBlockInfo`/`WanExecutionPlan`, `WanCoreController` com cache opcional de desquantização e o wrapper `StreamedWanDiTGGUF`), operando em nível de tensores GGUF em vez de `nn.Module`.

## Invariants & Logging (Fase 5)
- `_get_text_context` (GGUF):
  - Requer tokenizer dir válido; carrega `AutoTokenizer` com `local_files_only=True`.
  - TE:
    - `cuda_fp8` exige kernel disponível; sem fallback.
    - Em HF (`_Enc`), valida `hidden_size` do output com `config.hidden_size`.
  - Logs (INFO): shapes de `input_ids` e `last_hidden_state` (prompt/negative), dtype/device.
- Engine (Diffusers path):
  - Loga device/dtype efetivos e tipos das principais componentes (TE/UNet/VAE).
