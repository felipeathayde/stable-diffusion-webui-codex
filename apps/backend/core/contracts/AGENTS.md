# apps/backend/core/contracts Overview
<!-- tags: backend, core, contracts, assets, drift -->
Date: 2026-01-18
Last Review: 2026-02-10
Status: Active

## Purpose
- Owns backend “contract as code” modules that describe request invariants shared across UI ↔ API ↔ runtime (e.g. per-engine asset requirements).

## Key Files
- `apps/backend/core/contracts/asset_requirements.py` — Canonical per-engine asset requirements (VAE/text encoder) used by the API and exposed to the UI.
- `apps/backend/core/contracts/text_encoder_slots.py` — Header-only text encoder slot classifier used by the API to map sha-selected encoders into contract slots (order-independent).

## Notes
- Contracts here must be deterministic and fail loudly when an engine key is missing (prevents drift).
- Keep these modules lightweight: no heavy model imports at module import time.
- 2026-02-05: Added Anima engine asset contract (`anima`) and Qwen3-0.6B text encoder slot (`qwen3_06b`) for sha-selected TE resolution.
- 2026-02-10: Added explicit contract-ownership maps (`engine_id -> owner`, `semantic_engine -> owner`) so capability aliases (`flux1_fill`) and optional video semantics (`svd`, `hunyuan_video`) remain fail-loud and contract-complete.
- 2026-02-16: WAN22 video engines now have explicit per-engine owners/contracts (`wan22_5b`, `wan22_14b`, `wan22_animate_14b`) with no owner alias fallback across model variants.
