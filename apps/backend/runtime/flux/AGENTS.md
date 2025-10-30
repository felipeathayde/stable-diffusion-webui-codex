# apps/backend/runtime/flux Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
Status: Active

## Purpose
- Flux-specific runtime helpers (model configs, embedding utilities, transformer implementation) used by flux engines.

## Notes
- `config.py` hosts `FluxArchitectureConfig` / `FluxPositionalConfig` / `FluxGuidanceConfig`; validate new checkpoints through these dataclasses before wiring loaders.
- `geometry.py`, `embed.py`, and `components.py` provide shared rotary/attention building blocks; Chroma reuses them via imports (no legacy coupling).
- `model.py` exposes the Codex-native `FluxTransformer2DModel` with device/dtype guards and debugging logs; the thin `flux.py` module only re-exports symbols for legacy import paths.
- Keep this in sync with `apps/backend/engines/flux/` to avoid drift between runtime and engine expectations; document new invariants in engine AGENTS when behaviour changes.
