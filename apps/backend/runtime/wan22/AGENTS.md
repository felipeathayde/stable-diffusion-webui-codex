<!-- tags: backend, runtime, wan22, gguf, streaming, transformer -->

# apps/backend/runtime/wan22 Overview
Date: 2025-12-06
Owner: Runtime WAN Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- WAN 2.2-specific runtime components (GGUF loaders, diffusion stages, scheduler logic, optional core streaming) used by WAN engines.
- Native nn.Module implementation of the WAN 2.2 transformer (`WanTransformer2DModel`) and a flow-matching sampler for video latents, preparados para uso por um runtime/spec Codex.

## Key files
- `wan22.py`: caminho GGUF atual (WanDiTGGUF + helpers) usado pelos engines WAN22 na master.
- `streaming/`: infraestrutura de streaming de core para WAN 2.2 14B em GGUF (`WanBlockInfo`/`WanExecutionPlan`, `WanCoreController`, `StreamedWanDiTGGUF`), operando em nível de tensores GGUF.
- `model.py`: `WanArchitectureConfig` + `WanTransformer2DModel` e helper `load_wan_transformer_from_state_dict`, definindo o core WAN22 como nn.Module format-agnóstico (ligado ao engine 14B via spec/runtime).
- `sampler.py`: `WanVideoSampler` e `sample_txt2vid`, sampler de fluxo para latentes 5D `[B, C, T, H, W]` usando `WanTransformer2DModel` + VAE, agora exercitado pelo caminho experimental runtime/spec de `Wan2214BEngine.txt2vid`.
- `vae.py`, `wan_latent_norms.py`, `wan_te_*`: componentes auxiliares (VAE, normalização de latentes, text encoder) compartilhados pelos caminhos GGUF/HF.

## Notes
- Ensure updates stay synchronized with `apps/backend/engines/wan22/` e com os helpers GGUF sob `runtime/gguf`; o caminho GGUF atual continua sendo o código em produção.
- 2025-12-04: `_vae_encode_init`/`_get_text_context` continuam a requerer `vae_dir`/`text_encoder_dir` explícitos; esses diretórios são normalmente preenchidos via WAN22 defaults em `apps/paths.json` quando engines GGUF não recebem `extras` explícitos.
- 2025-12-06: `streaming/` adiciona infraestrutura de streaming de core para WAN 2.2 14B em GGUF (`WanBlockInfo`/`WanExecutionPlan`, `WanCoreController` com cache opcional de desquantização e o wrapper `StreamedWanDiTGGUF`), operando em nível de tensores GGUF em vez de `nn.Module`.
- 2025-12-06: `model.py`/`sampler.py` introduzem o caminho nn.Module (`WanTransformer2DModel` + `WanVideoSampler`/`sample_txt2vid`) espelhando a arquitetura da branch `feature/flux-core-streaming`; o engine 14B (`Wan2214BEngine`) já consome esse caminho para `txt2vid` quando o runtime/spec Codex está ativo (flag + `_bundle`), mantendo GGUF/Diffusers como defaults.

## Invariants & Logging (Fase 5)
- `_get_text_context` (GGUF):
  - Requer tokenizer dir válido; carrega `AutoTokenizer` com `local_files_only=True`.
  - TE:
    - `cuda_fp8` exige kernel disponível; sem fallback.
    - Em HF (`_Enc`), valida `hidden_size` do output com `config.hidden_size`.
  - Logs (INFO): shapes de `input_ids` e `last_hidden_state` (prompt/negative), dtype/device.
- Engine (Diffusers path):
  - Loga device/dtype efetivos e tipos das principais componentes (TE/UNet/VAE).
