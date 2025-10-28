# apps/interface Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-10-28
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
- `presets.json` — Placeholder for frontend preset definitions (subject to Codex alignment).

## Notes
- Run `npm run dev` from this directory for local development; backend expects the build artifacts emitted by Vite.
- Keep source structure consistent with the guidelines in `.sangoi/frontend/guidelines/`.
