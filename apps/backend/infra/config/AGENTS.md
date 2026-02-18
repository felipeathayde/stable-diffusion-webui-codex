# apps/backend/infra/config Overview
Date: 2026-02-18
Last Review: 2026-02-18
Status: Active

## Purpose
- Houses backend configuration primitives (CLI parsing, env/bootstrap flags, repo/path resolution, runtime feature toggles).
- Keeps startup/runtime configuration contracts centralized and fail-loud.

## Key files
- `args.py` — Runtime CLI/env parser and `RuntimeMemoryConfig` builder.
- `bootstrap_env.py` — Bootstrap env overlay publication without mutating global `os.environ`.
- `env_flags.py` — Canonical env flag parsers shared by runtime diagnostics/features.
- `paths.py` — Paths.json/settings discovery and model root provisioning helpers.
- `repo_root.py` — Repo root resolution (honors `CODEX_ROOT` launcher contract).

## Notes
- 2026-02-18: Interactive device prompts in `args.py` now route explicit stdout writes through `apps.backend.infra.stdio` to keep primitive stream emission centralized while preserving CLI prompt behavior.
- Keep this folder focused on config/bootstrap contracts; runtime execution logic belongs outside `infra/config`.
