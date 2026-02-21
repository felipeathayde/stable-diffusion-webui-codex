<!-- tags: backend, engines, wan22, gguf, diffusers, huggingface -->

# apps/backend/engines/wan22 Overview
Date: 2025-12-06
Last Review: 2026-02-20
Status: Active

## Purpose
- WAN22 engine implementations (txt2vid, img2vid, etc.) that coordinate WAN-specific runtime components and GGUF loaders.
- Experimental Codex-style runtime path for WAN22 (spec/runtime separado em `spec.py` e runtime/families/wan22), ligado a `txt2vid`/`img2vid` em modo opt-in via `_bundle` + option (`codex_wan22_use_spec_runtime`/`use_codex_runtime`).

## Key Files
- `apps/backend/engines/wan22/spec.py` — Codex runtime containers + assembly (`WanEngineSpec`/`WanEngineRuntime`, `assemble_wan_runtime`).
- `apps/backend/engines/wan22/factory.py` — Factory seam returning `(runtime, CodexObjects)` for consistent Codex runtime assembly.
- `apps/backend/engines/wan22/wan22_14b.py` — `Wan2214BEngine` (single canonical 14B lane for txt2vid/img2vid/vid2vid; GGUF-backed runtime without inheriting from `wan22_5b`).
- `apps/backend/engines/wan22/wan22_14b_animate.py` — `Wan22Animate14BEngine` (`vid2vid` Diffusers lane; engine id `wan22_14b_animate`).
- `apps/backend/engines/wan22/wan22_5b.py` — `Wan225BEngine` (GGUF-only wrapper; strict stage overrides; fails loud on Diffusers directory inputs).

## Notes
- Keep WAN engines alinhados com `runtime/families/wan22` (GGUF + nn.Module) e helpers GGUF para garantir tratamento estrito de assets.
- 2025-11-30: WAN22 engines now resolve vendored Hugging Face metadata under `apps/backend/huggingface` using a repo-root anchor, replacing the old `apps/server/backend/huggingface` path.
- 2025-12-04: GGUF execution path now applies WAN22 defaults from `apps/paths.json` (`wan22_vae`, `text_encoders`) when explicit extras are not provided, so a minimal `models/wan22/**` layout works without per-run overrides.
- 2025-12-06: `spec.py` introduz `WanEngineSpec`/`WanEngineRuntime` + `assemble_wan_runtime`; `Wan2214BEngine` aceita um `_bundle: DiffusionModelBundle` opcional e pode montar um runtime Codex experimental quando `codex_wan22_use_spec_runtime`/`use_codex_runtime` estiver ativo; `txt2vid`/`img2vid` usam esse runtime/spec + sampler (`sample_txt2vid`) como caminho opt-in, mantendo GGUF/Diffusers como defaults.
- 2025-12-13: caminho GGUF do engine 5B foi alinhado ao runtime WAN22 (`apps/backend/runtime/families/wan22/wan22.py`) e agora carrega o stage GGUF direto em `WanTransformer2DModel` (nn.Module), removendo dependência do runner `WanDiTGGUF`.
- 2025-12-14: engine 5B GGUF agora consome overrides por stage via `extras.wan_high/wan_low` (steps/cfg/sampler/scheduler/model_dir) e valida text encoder weights (`.safetensors` ou `.gguf`, file-only).
- 2026-01-17: WAN22 5B GGUF execution is orchestrated inside the video use-cases (`apps/backend/use_cases/{txt2vid,img2vid}.py`) so the engine stays thin; GGUF `RunConfig` mapping is owned by the WAN22 runtime config module (`apps/backend/runtime/families/wan22/config.py`).
- 2025-12-16: `wan22_5b` now supports `vid2vid` via `apps/backend/use_cases/vid2vid.py` (optical-flow-guided chunking built on `img2vid` + ffmpeg IO/export). Flow uses torchvision RAFT (lazy-loaded) and will fail fast if torch/torchvision are missing.
- 2025-12-16: Added `wan22_14b_animate` engine as a `vid2vid` strategy (`vid2vid_method="wan_animate"`) using Diffusers `WanAnimatePipeline` (expects preprocessed pose/face videos + reference image; `replace` mode also needs bg/mask). Requires a diffusers version that includes `WanAnimatePipeline` (>=0.36).
- 2025-12-29: WAN22 engines now anchor vendored HF paths under `CODEX_ROOT` (required) so they don’t depend on the backend process CWD.
- 2026-01-02: Added standardized file header docstrings to WAN22 engine modules (doc-only change; part of rollout).
- 2026-01-03: Codex runtime (experimental) now stores the sampling core as `WanEngineRuntime.denoiser` via `DenoiserPatcher` (ControlNet is UNet-only).
- 2026-01-03: `Wan2214BEngine` now assembles via `CodexWan22Factory` (factory-first seam; reduces drift in `_build_components`).
- 2026-01-06: GGUF stage overrides now fall back to canonical sampler/scheduler (`uni-pc`/`simple`) when unset (no `"Automatic"` placeholder).
- 2026-01-08: Flow-match `flow_shift` is sourced from diffusers `scheduler_config.json` (vendored HF mirror); WAN22 GGUF stages require an explicit `flow_shift` value (auto-resolved from the vendor config when not provided in `extras.wan_high/wan_low`).
- 2026-01-21: Stage LoRA selection is sha-only via `extras.wan_high/wan_low.lora_sha` (sha → `.safetensors`) + optional `lora_weight`; stage `lora_path` is rejected.
- 2026-01-22: `wan22_14b` no longer advertises `TaskType.IMG2VID`; `img2vid` raises `NotImplementedError` to avoid “success with empty frames”.
- 2026-01-31: WAN22 14B core streaming enablement now fails loud when explicitly requested (`core_streaming_enabled=True`) and setup fails (no silent fallback to non-streaming).
- 2026-02-01: `wan22_5b` is now GGUF-only (Diffusers directory-loading branch removed). WAN22 GGUF scheduler is Diffusers-free in the runtime; WAN22 VAE IO and `vid2vid.method=\"wan_animate\"` remain Diffusers-based until their dedicated port plans are executed.
- 2026-02-03: `wan22_14b` registration stayed gated while the variant-decoupling work was in progress.
- 2026-02-09: WAN22 prompt conditioning entrypoints now use `torch.no_grad()` (not `torch.inference_mode()`) to avoid caching inference tensors across requests (version-counter faults).
- 2026-02-10: `diffusers_loader.py` no longer swallows attention/accelerator hook or logger emit exceptions; hook import/apply and log emit failures now surface as contextual `RuntimeError`.
- 2026-02-11: `diffusers_loader.py` now wraps native WAN VAE instances with a strict diffusers-compat adapter (`encode/decode` contract), including 5D video batch adaptation and explicit `dtype` exposure, so pipeline calls consume `latents`/`sample` outputs without fallback shims.
- 2026-02-16: `WanEngineSpec.family` now uses explicit WAN22 variant families (`WAN22_14B`/`WAN22_5B`) and runtime VAE wrapper ownership follows `spec.family` (no shared WAN family alias).
- 2026-02-17: `wan22_14b` canonical registration now points to a dedicated GGUF 14B lane with no inheritance from `wan22_5b`; 14B/5B routing remains explicit per engine id.
- 2026-02-20: Removed explicit `wan22_14b_native` lane registration; WAN22 14B ownership is single-key (`wan22_14b`) and class export name is `Wan2214BEngine`.
- 2026-02-20: Renamed animate engine key to `wan22_14b_animate` (no compatibility alias for `wan22_animate_14b`).
- 2026-02-20: 14B canonical module path is `wan22_14b.py`; no `_gguf` lane naming in engine/module IDs.
- 2026-02-18: `wan22_14b` first-stage VAE encode/decode now load/unload using base canonical VAE memory target (`self._vae_memory_target()`, patcher when present) to keep memory-manager identity consistent with shared unload cleanup.
- 2026-02-21: `wan22_common.WanStageOptions` now carries stage prompt/negative fields (in addition to sampler/scheduler/steps/LoRA) with strict type normalization so stage prompt contracts are preserved across router/use-case/runtime boundaries.

## Execution Paths
- Diffusers: loads vendor tree and constructs `WanPipeline`; logs device/dtype and component classes (TE/UNet/VAE).
- Diffusers (Animate): loads `WanAnimatePipeline` from the user weights dir, using vendored metadata under `apps/backend/huggingface/Wan-AI/**` when local configs are missing.
- GGUF: strict assets via `resolve_user_supplied_assets`; text context produced by runtime `wan22.py` without fallbacks.
- Codex runtime (experimental): quando `_runtime_spec` está ativo (`_bundle` + `codex_wan22_use_spec_runtime`/`use_codex_runtime`), `Wan2214BEngine` usa `WanEngineRuntime` (T5 + `WanTransformer2DModel` + VAE Codex) e o sampler `sample_txt2vid` para gerar vídeo em `txt2vid`/`img2vid`, com metadata anotando `runtime="wan22_spec"`; sem essa opção, engines permanecem em Diffusers/GGUF.

## Device/Dtype Policy
- CPU only when explicitly requested; otherwise CUDA is required (error if unavailable).
