# gitignore Policy — Stable Diffusion WebUI Codex
Date: 2025-12-04
Owner: Repository Maintainers
Last Review: 2026-01-06
Status: Active

## Purpose
- Document the intent behind `.gitignore` so large artefacts and local caches stay out of the repository while keeping all source and documentation tracked.

## Scope
- `.gitignore` governs which files are ignored by Git.
- This document explains *why* certain patterns are ignored and how to extend the policy safely.

## Principles
- **No models in Git:** model weights, checkpoints, and large binary artefacts (e.g., `models/`, `*.ckpt`, `*.safetensors`) must remain untracked.
- **No caches or build outputs:** bytecode caches (`__pycache__/`), node modules (`node_modules/`), frontend bundles (`apps/interface/dist/`), and temporary directories (repo-root `tmp/`, `.tmp/`, plus `.pytest_cache/`) stay ignored.
- **Docs and configs are tracked:** Markdown docs under `.sangoi/**`, configuration files (`*.json`, `*.toml`, `*.yaml`), and source code are always included — except for runtime state JSON files and explicitly ignored repo-root scratch configs/scripts that are created/overwritten locally (see below).
- **Tests are tracked:** repository test sources under `tests/` are kept in version control; only test caches/outputs are ignored.
- **No binary office docs:** keep office exports like `*.docx` local; prefer Markdown under `.sangoi/**`.

## WebUI runtime outputs (tracked? no)
- `output/` — generated images/video artifacts.
- `logs/` — local runtime logs.
- `.webui-*.pid` — dev helper PID files (e.g. UI port guard).
- `.webui-*.log` — local runtime logs.
- `trace.json` — profiler output filename default.

## Local dev caches (tracked? no)
- `.uv/` — repo-local `uv` installer state + managed CPython installs.
- `.npm-cache/` — repo-local npm cache (installer sets `NPM_CONFIG_CACHE`).
- `.ts-out/` — TypeScript output folder (root `tsconfig.json` `outDir`).
- `.refs/` — local upstream/reference snapshots (read-only; never committed).

## Root scratch (tracked? no)
- Repo-root `/*.png` and `/*.txt` — ad-hoc outputs/notes. If you need to version images, place them under `.sangoi/assets/` (or another tracked docs folder) instead.
- `/logo.png` is explicitly tracked as a repo asset.

## Repo-root local configs/scripts (tracked? no)
- `config.json` — local scratch config; do not commit.
- `ui-config.json` — local scratch UI config; do not commit.
- `script-gemini*.js` — local one-off scripts; do not commit.

## Tooling outputs (tracked? no)
- `.sangoi/reports/tooling/pyright/` — type-check reports.
- `.sangoi/backups/webui_reasoning.jsonl` — Codex session artifacts.
- `.sangoi/backups/webui_reasoning_report.md` — Codex session artifacts.

## Runtime state (tracked? no)
- `apps/interface/tabs.json` — backend-managed persisted tab state (created if missing).
- `apps/interface/workflows.json` — backend-managed persisted workflows state (created if missing).
- `apps/settings_values.json` — backend-managed persisted options snapshot (created/overwritten locally).
- `.sangoi/launcher/` — launcher profile persistence (meta + env areas + per-model env overlays); created/overwritten locally by the TUI/GUI launcher and will drift across machines.

## Extending the ignore set
- Prefer directory-level ignores (e.g., `apps/interface/dist/`) over broad `*` patterns.
- When adding a new ignore rule:
  - Confirm the files are reproducible or environment-local.
  - Update this file with a short note if the pattern is non-obvious.
  - Avoid ignoring generic extensions that might hide source (e.g., `*.py`, `*.ts`).

## Related Notes
- Large artefacts, outputs, caches, and heavy model directories stay untracked per this policy, as referenced in `AGENTS.md`.
- If you are unsure whether something should be ignored, log the question in `.sangoi/task-logs/` and coordinate with maintainers before changing `.gitignore`.
