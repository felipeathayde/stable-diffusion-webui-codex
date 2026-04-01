<!-- tags: backend, runtime, supir, restore, sdxl -->

# apps/backend/runtime/families/supir Overview
Date: 2026-02-02
Last Review: 2026-03-31
Status: In progress

## Purpose
- Host SUPIR-specific runtime code for native SDXL img2img/inpaint SUPIR mode.
- Centralize SUPIR guardrails, typed parameter parsing, asset validation, and the dedicated runtime owner so the canonical img2img path can stay thin and fail-loud.

## Key files
- `weights.py` — SUPIR weights discovery/validation under the `supir_models` roots (`apps/paths.json`).
- `sdxl_guard.py` — SDXL base/refiner detection; enforces “reject SDXL Refiner” (SUPIR base must be SDXL base/finetune).
- `config.py` — Typed nested `img2img_extras.supir` config parsing for the tranche-1 public SUPIR surface.
- `nn/` — SUPIR neural network modules (GLVControl + LightGLVUNet + adapters).
- `samplers/` — SUPIR sampler IDs/specs and registry (Enum + dataclasses; no kwargs leakage).
- `loader.py` — Validates the already-selected SDXL checkpoint record and resolves SUPIR variant weights under `supir_models`.
- `runtime.py` — Canonical SUPIR mode execution owner called from `apps/backend/use_cases/img2img.py`.
- `vae_cache.py` — Small in-process LRU cache for Stage-1 precompute (default OFF; explicit clear).

## Notes
- **No LLaVA** support: prompts are optional; no image-to-text captioning path.
- **No fast VAE encode/decode**: do not implement any “fast VAE” shortcuts.
- Default OFF: VAE cache, VAE tiling, diffusion tiling (guardrails required when enabled).
- Keep router/runtime imports **import-light**: torch-heavy work must be inside functions.
- 2026-03-31: `/api/supir/models` is diagnostics-only; live SUPIR generation is owned by canonical SDXL `img2img.py`, not a standalone `/api/supir/enhance` route/task.
