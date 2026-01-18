# apps/backend/core/contracts Overview
<!-- tags: backend, core, contracts, assets, drift -->
Date: 2026-01-18
Owner: Backend Core Maintainers
Last Review: 2026-01-18
Status: Active

## Purpose
- Owns backend “contract as code” modules that describe request invariants shared across UI ↔ API ↔ runtime (e.g. per-engine asset requirements).

## Key Files
- `apps/backend/core/contracts/asset_requirements.py` — Canonical per-engine asset requirements (VAE/text encoder) used by the API and exposed to the UI.

## Notes
- Contracts here must be deterministic and fail loudly when an engine key is missing (prevents drift).
- Keep these modules lightweight: no heavy model imports at module import time.

