# apps/interface/tools Overview
Date: 2025-10-28
Last Review: 2026-03-29
Status: Active

## Purpose
- Contains scripts/utilities that support frontend development and contract validation.

## Key Files
- `port-guard-dev.mjs` — Dev server launcher wrapper that enforces safe/available UI ports.
- `verify-css-contracts.mjs` — Frontend CSS-contract verifier and owner behind `npm run verify:css-contracts`.
- `style-topology.mjs` — Ordered runtime CSS topology declaration for `apps/interface`.
- `css-contracts.config.json` — CSS-contract budgets and typed exception ownership.

## Notes
- Keep these scripts in sync with repository-shipped frontend docs (`apps/interface/README.md`, `apps/interface/AGENTS.md`, `apps/interface/src/styles/AGENTS.md`) so developers know how to invoke them.
- `npm run verify:css-contracts` is the only direct CSS gate.
- `npm run verify` is wrapper-only.
- Detailed topology, budgets, and typed exceptions live in `style-topology.mjs`, `css-contracts.config.json`, and `.sangoi/reference/ui/frontend-css-contracts.md`; this file stays pointer-only.
