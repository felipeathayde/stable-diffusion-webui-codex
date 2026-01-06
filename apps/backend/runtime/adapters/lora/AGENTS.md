# apps/backend/runtime/adapters/lora Overview
Date: 2025-10-28
Owner: Runtime Adapter Maintainers
Last Review: 2026-01-02
Status: Active

## Purpose
- Implements the Codex-native LoRA pipeline (loading, tensor mapping, application ops, type definitions).

## Key Files
- `loader.py` — Loads LoRA weights safely with metadata checks.
- `mapping.py` — Maps LoRA weights onto target modules.
- `ops.py` — Runtime tensor operations for applying LoRAs.
- `pipeline.py` — High-level orchestration used by engines/patchers.
- `selections.py` — Process-wide LoRA selection state used by API endpoints and workflow builders.
- `types.py` — Dataclasses describing LoRA assets.

## Notes
- Keep this pipeline aligned with `apps/backend/patchers/lora_apply.py` and the options service so selections remain consistent.
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `mapping.py`, `ops.py`, `pipeline.py`, and `types.py` (doc-only change; part of rollout).
