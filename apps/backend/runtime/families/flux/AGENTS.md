# apps/backend/runtime/families/flux Overview
Date: 2025-12-06
Owner: Runtime Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- Flux-specific runtime helpers (model configs, embedding utilities, transformer implementation, optional core streaming) used by Flux-family engines.

## Notes
- `config.py` hosts `FluxArchitectureConfig` / `FluxPositionalConfig` / `FluxGuidanceConfig`; validate new checkpoints through these dataclasses before wiring loaders.
- `geometry.py`, `embed.py`, and `components.py` provide shared rotary/attention building blocks; Chroma reuses them via imports (no legacy coupling).
- `model.py` exposes the Codex-native `FluxTransformer2DModel` with device/dtype guards and debugging logs; the thin `flux.py` module only re-exports symbols for convenience imports.
- `streaming/` contains block/segment-based streaming helpers for the Flux core (`StreamingConfig`, `CoreController`, `trace_execution_plan`, `StreamedFluxCore`); these wrap `FluxTransformer2DModel` without changing its public interface.
- `FluxTransformer2DModel.forward` now accepts extra keyword-only arguments (`control`, `transformer_options`, plus ignored `**kwargs`) so it can be driven safely via `KModel.apply_model` without relying on them; do not use these for new behaviour without coordinating with `apps/backend/runtime/k_diffusion/k_model.py`.
- Keep this in sync with `apps/backend/engines/flux/` to avoid drift between runtime and engine expectations; document new invariants in engine AGENTS when behaviour changes.
- 2026-01-02: Added standardized file header docstrings to Flux runtime surface modules (`__init__.py`, `config.py`, `embed.py`, `geometry.py`, `flux.py`, and `streaming/{__init__,config,specs}.py`) (doc-only change; part of rollout).
