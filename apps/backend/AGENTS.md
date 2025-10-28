# apps/backend Overview
Date: 2025-10-28
Owner: Backend Steward Team
Last Review: 2025-10-28
Status: Active

## Purpose
- Implements the Codex-native backend: engine orchestration, runtime components, services, registries, and vendor assets.
- Provides a clean separation from legacy Forge/A1111 code (no imports from `modules.*`, `backend.*`, or `.legacy/`).

## Subdirectories (Top-Level)
- `core/` — Core orchestration (engine interface, device/state management, RNG stack, request handling, parameter schemas).
- `engines/` — Model-specific engines and shared utilities (diffusion, SD, Flux, Chroma, WAN22).
- `runtime/` — Reusable runtime components (attention, adapters, text processing, memory, sampling, ops, WAN22, etc.).
- `services/` — High-level services (image/media options, progress streaming) that the API consumes.
- `use_cases/` — Task orchestrators (txt2img, img2img, txt2vid, img2vid) wiring inputs to engines and runtimes.
- `patchers/` — Runtime patching utilities (LoRA application, token merging, etc.).
- `infra/` — Cross-cutting infrastructure (accelerator detection, dynamic args/registry helpers).
- `interfaces/` — API schemas and adapters that surface backend capabilities.
- `huggingface/` — Hugging Face asset mirrors and helper code maintained for strict/offline modes (actively modified as needed).
- `gguf/` — GGUF-specific helpers and metadata.
- `video/` — Shared video tooling (e.g., interpolation helpers).
- `inventory/` — Machine-readable registries/inventories used during audits.
- `codex/` — Codex-specific metadata and helpers (e.g., configuration defaults).

## Key Files
- `__init__.py` — Package marker.
- `registration.py` (under `engines/`) — Canonical engine registration entrypoint.
- `state.py`, `devices.py`, `orchestrator.py` (under `core/`) — Core primitives for backend execution.

## Notes
- Prefer adding new functionality under the structured subpackages above; avoid creating new ad-hoc directories.
- Keep Hugging Face helpers up to date—Codex builds on these mirrors rather than relying on upstream defaults.
- When retiring subpackages, relocate historical context to `.sangoi/deprecated/` and update this overview.
