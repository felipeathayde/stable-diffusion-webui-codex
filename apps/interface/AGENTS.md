<!-- tags: frontend, interface, overview -->
# apps/interface Overview
Date: 2025-10-28
Last Review: 2026-02-28
Status: Active

## Purpose
- Hosts the Codex Vue 3 + Vite frontend application that replaces the legacy Gradio UI.

## Subdirectories
- `src/` — TypeScript/Vue source code (components, views, stores, styles, API client).
- `public/` — Static assets served as-is (favicons, manifest, etc.).
- `tools/` — Developer tooling scripts (port guard helpers, lint/typecheck wrappers).

## Key Files
- `package.json` / `tsconfig.json` / `vite.config.ts` — Build and tooling configuration.
- `blocks.json` — Server-driven UI definition synced with backend.
- `presets.json` — UI presets served by the backend `/api/ui/presets` endpoint (source of truth for preset IDs/options).

## Notes
- Run `npm run dev` from this directory for local development; backend expects the build artifacts emitted by Vite.
- Keep source structure consistent with the guidelines in `.sangoi/frontend/guidelines/`.
- 2025-12-29: `vite.config.ts` ignores backend-persisted `tabs.json`/`workflows.json` changes to prevent Vite full-reloads during dev toggles.
- 2025-12-29: `tools/port-guard-dev.mjs` now checks IPv4+IPv6 bind targets (0.0.0.0/127.0.0.1/::/::1) to avoid localhost split-brain; when the base port is busy it probes `/api/version` to warn about an existing Codex instance (WSL/Windows) and writes `.webui-ui-<port>.pid` for debugging.
- 2025-11-03: SDXL view now exposes "Save Profile" backed by store persistence to mirror the Test harness.
- 2025-11-14: API requests are built via `src/api/payloads.ts` (Zod schemas) — payload builders trim prompts and always attach the per-tab engine/model metadata (even for img2img).
- 2025-12-03: Txt2Img prompt schema now rejects empty prompts at the frontend (`PromptSchema`), surfacing a validation error instead of silently sending `prompt=""` to the backend.
- 2026-02-28: Frontend follows root testing policy: manual validation by default; automated/unit tests are not maintained unless explicitly requested by the repo owner.
- 2026-01-01: Updated `apps/interface/README.md` to reflect the repo-local `.venv` (and `run-webui.sh` as the recommended dev entrypoint).
- 2026-01-01: Added a branded `public/favicon.ico` and referenced it from `index.html` so the browser tab icon matches the project branding.
- 2026-01-03: Added standardized file header blocks to WebUI entrypoints/config (`vite.config.ts`, `src/{App,main,router}.ts/.vue`, `src/api/types.ts`) (doc-only change; part of rollout).
- 2026-01-21: Updated `blocks.json` WAN22 stage fields to sha-only (`model_sha`/`lora_sha`) to match backend enforcement.
- 2026-01-23: WAN video dimensions now snap to multiples of 16 (rounded up; Diffusers parity) in the UI and payload builders to avoid backend 400s and silent patch-grid cropping.
- 2026-01-27: `package-lock.json` updated to match npm 11 (used by the repo-local `.nodeenv` installer) to avoid lockfile churn on fresh installs.
- 2026-02-06: Added `vue-tsc` typechecking (`npm run typecheck`) and gated `npm run dev` on typecheck to prevent “build passes, types broken” drift.
- 2026-02-08: SDXL swap-model UI contract now uses explicit pointer semantics (`swapAtStep` in frontend state, serialized as `switch_at_step` in API payloads), replacing refiner step-count wording/behavior.
- 2026-02-21: Added UI consistency scanner (`tools/ui-consistency-report.mjs`) and wired `npm run verify` to run strict style-contract gating (`report:ui-consistency:strict`) before typecheck/build.
