# apps/backend/runtime/chroma Overview
Date: 2025-10-30
Owner: Runtime Maintainers
Last Review: 2025-10-30
Status: Active

## Purpose
- Chroma transformer implementation sharing Flux building blocks (embeddings, attention) while layering distilled-guidance modulation.

## Notes
- `config.py` defines `ChromaArchitectureConfig` / `ChromaGuidanceConfig`; loaders must instantiate these via keyword args mirrored in checkpoint configs.
- `chroma.py` consumes Flux components (`geometry`, `embed`, `components`) and adds the modulation approximator; failures when adding ControlNet should raise explicit `NotImplementedError`.
- Keep this directory aligned with the Chroma engine; relocate shared logic into `runtime/common` when applicable and update the AGENT if we split further modules.
