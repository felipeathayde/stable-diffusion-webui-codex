<!-- tags: backend, engines, wan22, gguf, diffusers, huggingface -->

# apps/backend/engines/wan22 Overview
Date: 2025-12-06
Last Review: 2026-02-28
Status: Active

## Purpose
- WAN22 engine implementations (txt2vid, img2vid, vid2vid) that coordinate WAN-specific runtime components and GGUF loaders.

## Key Files
- `apps/backend/engines/wan22/wan22_14b.py` — `Wan2214BEngine` (canonical 14B lane for txt2vid/img2vid/vid2vid; GGUF-backed runtime without inheriting from `wan22_5b`).
- `apps/backend/engines/wan22/wan22_14b_animate.py` — `Wan22Animate14BEngine` (`vid2vid` Diffusers lane; engine id `wan22_14b_animate`).
- `apps/backend/engines/wan22/wan22_5b.py` — `Wan225BEngine` (GGUF-only wrapper; strict stage overrides; fails loud on Diffusers directory inputs).
- `apps/backend/engines/wan22/wan22_common.py` — shared WAN route/build/asset normalization helpers consumed by all WAN22 engines.
- `apps/backend/engines/wan22/diffusers_loader.py` — WAN Diffusers pipeline preparation/loading with strict hook-application error surfacing.

## Notes
- Keep WAN engines aligned with `runtime/families/wan22` (GGUF + nn.Module) and GGUF helpers to guarantee strict asset handling.
- 2025-11-30: WAN22 engines resolve vendored Hugging Face metadata under `apps/backend/huggingface` using a repo-root anchor, replacing `apps/server/backend/huggingface`.
- 2025-12-04: GGUF execution path applies WAN22 defaults from `apps/paths.json` (`wan22_vae`, `text_encoders`) when explicit extras are not provided.
- 2025-12-13: GGUF execution path for 5B aligned to runtime WAN22 (`apps/backend/runtime/families/wan22/wan22.py`) with direct `WanTransformer2DModel` stage loading.
- 2025-12-14: 5B GGUF consumes stage overrides via `extras.wan_high/wan_low` and validates text encoder weights (`.safetensors` or `.gguf`, file-only).
- 2025-12-16: `wan22_5b` supports `vid2vid` via `apps/backend/use_cases/vid2vid.py` (optical-flow-guided chunking built on `img2vid` + ffmpeg IO/export).
- 2025-12-16: Added `wan22_14b_animate` engine as a `vid2vid` strategy (`vid2vid_method="wan_animate"`) using Diffusers `WanAnimatePipeline`.
- 2025-12-29: WAN22 engines anchor vendored HF paths under `CODEX_ROOT` so they don’t depend on backend process CWD.
- 2026-01-06: GGUF stage overrides fall back to canonical sampler/scheduler (`uni-pc`/`simple`) when unset (no `"Automatic"` placeholder).
- 2026-01-08: Flow-match `flow_shift` is sourced from diffusers `scheduler_config.json` (vendored HF mirror).
- 2026-01-17: WAN22 GGUF execution is orchestrated inside video use-cases (`apps/backend/use_cases/{txt2vid,img2vid}.py`); RunConfig mapping is owned by `runtime/families/wan22/config.py`.
- 2026-01-21: Stage LoRA selection is sha-only via `extras.wan_high/wan_low.lora_sha` + optional `lora_weight`; stage `lora_path` is rejected.
- 2026-01-31: WAN22 14B core streaming enablement fails loud when explicitly requested (`core_streaming_enabled=True`) and setup fails.
- 2026-02-01: `wan22_5b` is GGUF-only (Diffusers directory-loading branch removed).
- 2026-02-09: WAN22 prompt conditioning entrypoints use `torch.no_grad()` (not `torch.inference_mode()`) to avoid cached inference tensors across requests.
- 2026-02-10: `diffusers_loader.py` no longer swallows attention/accelerator hook or logger emit exceptions.
- 2026-02-11: `diffusers_loader.py` wraps native WAN VAE with a strict diffusers-compat adapter (`encode/decode` contract).
- 2026-02-21: Native VAE adapter avoids unconditional `.contiguous()` on 4D->5D restore, preventing an unnecessary full-tensor copy on WAN video-latent paths.
- 2026-02-17: `wan22_14b` canonical registration points to dedicated GGUF 14B lane with no inheritance from `wan22_5b`.
- 2026-02-20: Removed explicit `wan22_14b_native` lane registration; WAN22 14B ownership is single-key (`wan22_14b`) and class export name is `Wan2214BEngine`.
- 2026-02-20: Renamed animate engine key to `wan22_14b_animate` (no compatibility alias for `wan22_animate_14b`).
- 2026-02-20: 14B canonical module path is `wan22_14b.py`; no `_gguf` lane naming in engine/module IDs.
- 2026-02-18: `wan22_14b` first-stage VAE encode/decode loads/unloads using base canonical VAE memory target (`self._vae_memory_target()`), preserving memory-manager identity for shared unload cleanup.
- 2026-02-21: `wan22_common.WanStageOptions` carries stage prompt/negative fields with strict type normalization.
- 2026-02-23: WAN22 engine device defaults resolve from memory-manager mount device (no local `'cuda'`/`'auto'` fallback literals).
- 2026-02-28: Removed dormant alternate 14B ownership lane artifacts (`factory.py`/`spec.py`) to keep a single canonical runtime/control surface on the active GGUF lane.

## Execution Paths
- Diffusers: loads vendor tree and constructs `WanPipeline`; logs device/dtype and component classes (TE/UNet/VAE).
- Diffusers (Animate): loads `WanAnimatePipeline` from user weights dir, using vendored metadata under `apps/backend/huggingface/Wan-AI/**` when local configs are missing.
- GGUF: strict assets via `resolve_user_supplied_assets`; text context is produced by runtime `wan22.py` without fallbacks.

## Device/Dtype Policy
- CPU only when explicitly requested; otherwise CUDA is required (error if unavailable).
