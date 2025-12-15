<!-- tags: backend, engines, wan22, gguf, diffusers, huggingface -->

# apps/backend/engines/wan22 Overview
Date: 2025-12-06
Owner: Engine Maintainers
Last Review: 2025-12-14
Status: Active

## Purpose
- WAN22 engine implementations (txt2vid, img2vid, etc.) that coordinate WAN-specific runtime components and GGUF loaders.
- Experimental Codex-style runtime path for WAN22 (spec/runtime separado em `spec.py` e runtime/wan22), ligado a `txt2vid`/`img2vid` em modo opt-in via `_bundle` + option (`codex_wan22_use_spec_runtime`/`use_codex_runtime`).

## Notes
- Keep WAN engines alinhados com `runtime/wan22` (GGUF + nn.Module) e helpers GGUF para garantir tratamento estrito de assets.
- 2025-11-30: WAN22 engines now resolve vendored Hugging Face metadata under `apps/backend/huggingface` using a repo-root anchor, replacing the old `apps/server/backend/huggingface` path.
- 2025-12-04: GGUF execution path now applies WAN22 defaults from `apps/paths.json` (`wan22_vae`, `text_encoders`) when explicit extras are not provided, so a minimal `models/wan22/**` layout works without per-run overrides.
- 2025-12-06: `spec.py` introduz `WanEngineSpec`/`WanEngineRuntime` + `assemble_wan_runtime`; `Wan2214BEngine` aceita um `_bundle: DiffusionModelBundle` opcional e pode montar um runtime Codex experimental quando `codex_wan22_use_spec_runtime`/`use_codex_runtime` estiver ativo; `txt2vid`/`img2vid` usam esse runtime/spec + sampler (`sample_txt2vid`) como caminho opt-in, mantendo GGUF/Diffusers como defaults.
- 2025-12-13: caminho GGUF do engine 5B foi alinhado ao runtime WAN22 (`apps/backend/runtime/wan22/wan22.py`) e agora carrega o stage GGUF direto em `WanTransformer2DModel` (nn.Module), removendo dependência do runner `WanDiTGGUF`.
- 2025-12-14: engine 5B GGUF agora consome overrides por stage via `extras.wan_high/wan_low` (steps/cfg/sampler/scheduler/model_dir) e valida text encoder `.safetensors` (file ou dir com 1 arquivo).

## Execution Paths
- Diffusers: loads vendor tree and constructs `WanPipeline`; logs device/dtype and component classes (TE/UNet/VAE).
- GGUF: strict assets via `resolve_user_supplied_assets`; text context produced by runtime `wan22.py` without fallbacks.
- Codex runtime (experimental): quando `_runtime_spec` está ativo (`_bundle` + `codex_wan22_use_spec_runtime`/`use_codex_runtime`), `Wan2214BEngine` usa `WanEngineRuntime` (T5 + `WanTransformer2DModel` + VAE Codex) e o sampler `sample_txt2vid` para gerar vídeo em `txt2vid`/`img2vid`, com metadata anotando `runtime="wan22_spec"`; sem essa opção, engines permanecem em Diffusers/GGUF.

## Device/Dtype Policy
- CPU only when explicitly requested; otherwise CUDA is required (error if unavailable).
