<!-- tags: backend, runtime, wan22, gguf, streaming, transformer -->

# apps/backend/runtime/families/wan22 Overview
Date: 2025-12-06
Owner: Runtime WAN Maintainers
Last Review: 2026-01-23
Status: Active

## Purpose
- WAN 2.2-specific runtime components (GGUF loaders, diffusion stages, scheduler logic, optional core streaming) used by WAN engines.
- Native nn.Module implementation of the WAN 2.2 transformer (`WanTransformer2DModel`) and a flow-matching sampler for video latents, preparados para uso por um runtime/spec Codex.

## Key files
- `wan22.py`: facade/entrypoint importado pelos engines WAN22; re-exporta `RunConfig`/`StageConfig` e `run_*`/`stream_*` do runtime GGUF.
- `config.py`: dataclasses/config do runtime GGUF (`RunConfig`, `StageConfig`) + parsing helpers (dtype/device).
- `run.py`: entrypoints `run_txt2vid`/`run_img2vid` + versões streaming; orquestra TE, sampling por stage e VAE IO.
- `sampling.py`: geometria + scheduler + loop de sampling por stage (`sample_stage_latents*`) e adaptação de shape/I2V channels.
- `text_context.py`: tokenizer + text encoder (strict local-files-only) para embeddings de prompt/negative.
- `vae_io.py`: VAE encode/decode + latent norms (I2V conditioning encode + decode frames), com offload.
- `stage_loader.py`: validação de paths `.gguf` + load de weights em `WanTransformer2DModel` via `using_codex_operations(..., bnb_dtype="gguf")` + remap de chaves.
- `stage_lora.py`: aplicação de LoRA por stage (High/Low) no caminho GGUF (LightX2V), com mapping Diffusers→Codex e modo global `CODEX_LORA_APPLY_MODE`.
- `paths.py`: normalização de paths Windows (`C:\\...`) para WSL (`/mnt/c/...`) usada por config + loaders (evita drift).
- `diagnostics.py`: logging/diagnostics (sigmas parity, CUDA mem snapshots, cache empty) para o caminho GGUF.
- `model.py`: `WanArchitectureConfig` + `WanTransformer2DModel` e helpers (`remap_wan22_gguf_state_dict`, `infer_wan_architecture_from_state_dict`, `load_wan_transformer_from_state_dict`) para manter o core WAN22 format-agnóstico.
- `inference.py`: helpers compartilhados de inferência de shapes (patch embedding/head) usados por detector e loader (evita drift PyTorch vs GGUF layouts).
- `sdpa.py`: seleção de backend SDPA + chunking (policy/chunk) usada pelo core WAN22 (para alinhar com env/flags do runtime).
- `streaming/`: infraestrutura de streaming de core para `WanTransformer2DModel` (`WanExecutionPlan`, `WanCoreController`, `StreamedWanTransformer`) com execução por segmentos.
- `sampler.py`: `WanVideoSampler` e `sample_txt2vid`, sampler de fluxo para latentes 5D `[B, C, T, H, W]` usando `WanTransformer2DModel` + VAE, agora exercitado pelo caminho experimental runtime/spec de `Wan2214BEngine.txt2vid`.
- `vae.py`, `wan_latent_norms.py`, `wan_te_*`: componentes auxiliares (VAE, normalização de latentes, text encoder) compartilhados pelos caminhos GGUF/HF.

## Notes
- Ensure updates stay synchronized with `apps/backend/engines/wan22/` and the shared GGUF quantization/ops layer (`apps/backend/quantization/**`, `apps/backend/runtime/ops/operations.py`, `apps/backend/runtime/ops/operations_gguf.py`); the production GGUF runtime remains `wan22.py`.
- 2025-12-04: VAE/TE loading in the GGUF runtime requires explicit `vae_dir`/`text_encoder_dir`-style assets; these are normally provided via request `extras` (sha-only), with UI defaults resolved from `apps/paths.json` where applicable.
- 2025-12-06: `streaming/` adiciona infraestrutura de streaming de core para WAN 2.2 14B, operando em nível de `nn.Module` (wrapper `StreamedWanTransformer`) com offload por segmentos.
- 2025-12-06: `model.py`/`sampler.py` introduzem o caminho nn.Module (`WanTransformer2DModel` + `WanVideoSampler`/`sample_txt2vid`) espelhando a arquitetura da branch `feature/flux-core-streaming`; o engine 14B (`Wan2214BEngine`) já consome esse caminho para `txt2vid` quando o runtime/spec Codex está ativo (flag + `_bundle`), mantendo GGUF/Diffusers como defaults.
- 2025-12-13: GGUF WAN22 deixou de usar o runner WAN-específico (`WanDiTGGUF`) e passou a carregar o stage GGUF direto em `WanTransformer2DModel`, com SDPA policy/chunk centralizados em `sdpa.py` e dependência de `CodexOperationsGGUF` para suportar módulos GGUF em `nn.Module`.
- 2025-12-14: `sampler.py` (caminho experimental runtime/spec) agora escala o timestep passado ao core por `flow_multiplier` (default `1000.0`) e tem teste de regressão; `wan22.py` emite logs opt-in de paridade (sigma ladder + mapeamento de timestep) quando `CODEX_LOG_SIGMAS=1`.
- 2026-01-03: Removed upstream pipeline brand wording from WAN22 runtime comments; keep behaviour described directly and treat Diffusers as the baseline only for the HF asset path.
- 2026-01-04: Centralized WAN22 patch/head shape inference in `inference.py` so detector and runtime loaders agree on GGUF vs torch layout.
- 2026-01-06: `sampling.make_scheduler` now recognizes canonical `uni-pc` and instantiates `UniPCMultistepScheduler` for WAN GGUF stage runs.
- 2026-01-08: Removed the global `WAN_FLOW_SHIFT_DEFAULT`; `StageConfig.flow_shift` is required and defaults are resolved from the vendored diffusers `scheduler_config.json` (variant-correct, fail-fast if missing).
- 2026-01-08: Refreshed `run.py` file header block to include new helpers and removed a duplicate `_require_flow_shift` call (no behavior change).
- 2026-01-16: Updated `model.py` remapping to accept Diffusers `WanTransformer3DModel` key layout (`condition_embedder.*`, `attn1/attn2`, `scale_shift_table`, `proj_out`, `norm2↔norm3`) and aligned modulation parameter shapes/head modulation semantics with upstream exports.
- 2026-01-18: Centralized GGUF state-dict loading in runtime IO (`apps/backend/runtime/checkpoint/io.py:load_gguf_state_dict`) for WAN22 (stage loader + text encoder GGUF path), avoiding private helpers and direct quantization imports.
- 2026-01-20: WAN22 GGUF agora suporta LoRA por stage (High/Low) via `wan_high/wan_low.lora_sha` (sha → `.safetensors`) + `lora_weight` e aplica no load do stage; modo global `CODEX_LORA_APPLY_MODE=merge|online` (default `merge`).
- 2026-01-21: WAN22 GGUF now honors Smart flags in video runs: TE runs on CUDA for embedding generation and offloads to CPU when Smart Offload is enabled; VAE IO honors `vae_always_tiled` (Diffusers tiling) and retries encode/decode on CPU under Smart Fallback.
- 2026-01-22: WAN22 GGUF I2V assembly is now upstream-faithful: seed `lat16` from RNG scaled by `scheduler.init_noise_sigma` (Diffusers parity), build `mask4` with Diffusers temporal layout, and encode `img16` from a deterministic VAE encode of Diffusers-style `video_condition` (frame0=init image, rest=0.5 fill). Legacy “assemble Cin=36 from lat16 alone” fallback is removed (fail loud).
- 2026-01-23: WAN22 GGUF enforces `height/width % 16 == 0` (Diffusers parity) to avoid silent patch-grid cropping when latent H/W are odd (e.g. 360px → 352px).

## Invariants & Logging (Fase 5)
- `_get_text_context` (GGUF):
  - Requer tokenizer dir válido; carrega `AutoTokenizer` com `local_files_only=True`.
  - TE:
    - `cuda_fp8` exige kernel disponível; sem fallback.
    - Em HF (`_Enc`), valida `hidden_size` do output com `config.hidden_size`.
  - Logs (INFO): shapes de `input_ids` e `last_hidden_state` (prompt/negative), dtype/device.
- Engine (Diffusers path):
  - Loga device/dtype efetivos e tipos das principais componentes (TE/UNet/VAE).
- 2026-01-02: Added standardized file header docstrings to WAN22 runtime modules (doc-only change; part of rollout).
