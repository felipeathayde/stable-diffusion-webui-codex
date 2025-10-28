# Model Registry (Work in Progress)
Date: 2025-10-28
Owner: Backend Maintainers
Status: Draft

## Purpose
- Track structured metadata about supported checkpoints and pipelines.
- Provide detection heuristics and signatures for model loading without relying on `huggingface_guess`.

## TODO
- Implement dataclasses/enums for signatures and families.
- Port detection heuristics into typed detectors.
- Expose CLI/inspect tooling for diagnostics.
