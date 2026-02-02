<!-- tags: backend, runtime, supir, restore, sdxl -->

# apps/backend/runtime/families/supir Overview
Date: 2026-02-02
Owner: Runtime Maintainers
Last Review: 2026-02-02
Status: In progress

## Purpose
- Host SUPIR-specific runtime code used by the SUPIR enhance mode (`/api/supir/enhance`).
- Centralize SUPIR guardrails and typed parameter parsing to keep the API surface thin and fail-loud.

## Key files
- `weights.py` — SUPIR weights discovery/validation under the `supir_models` roots (`apps/paths.json`).
- `sdxl_guard.py` — SDXL base/refiner detection; enforces “reject SDXL Refiner” (SUPIR base must be SDXL base/finetune).
- `config.py` — Typed request config parsing (full SUPIR parameter surface; defaults and validation).
- `nn/` — SUPIR neural network modules (GLVControl + LightGLVUNet + adapters).
- `samplers/` — SUPIR sampler IDs/specs and registry (Enum + dataclasses; no kwargs leakage).
- `runner.py` — SUPIR enhance runner (optional preprocess + sampling; tiling and VAE cache live here).
- `vae_cache.py` — Small in-process LRU cache for Stage-1 precompute (default OFF; explicit clear).

## Notes
- **No LLaVA** support: prompts are optional; no image-to-text captioning path.
- **No fast VAE encode/decode**: do not implement any “fast VAE” shortcuts.
- Default OFF: VAE cache, VAE tiling, diffusion tiling (guardrails required when enabled).
- Keep router/task imports **import-light**: torch-heavy work must be inside functions.
