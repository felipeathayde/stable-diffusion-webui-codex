# apps/interface/public Overview
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2026-01-01
Status: Active

## Purpose
- Static assets served directly by Vite (favicons, robots, manifest, help docs). No bundling or transformation occurs here.

## Notes
- Place only static files that need to be served as-is. For processed assets, use the `src/` pipeline.
- Markdown help for the Home view lives under `help/*.md` (for example, `home-overview.md`, `wan22-quickstart.md`, `workflows-basics.md`) and is loaded at runtime by `MarkdownHelp.vue`.
- 2026-01-01: Replaced the placeholder `favicon.ico` with a branded multi-size icon (generated from `logo.png`).
