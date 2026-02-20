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
- `lora_merge_mode.py` — Strict enum/parser/reader for LoRA merge math mode (`fast|precise`).
- `lora_refresh_signature.py` — Strict enum/parser/reader for LoRA refresh signature mode (`structural|content_sha256`).
- `paths.py` — Paths.json/settings discovery and model root provisioning helpers.
- `repo_root.py` — Repo root resolution (honors `CODEX_ROOT` launcher contract).

## Notes
- 2026-02-18: Interactive device prompts in `args.py` now route explicit stdout writes through `apps.backend.infra.stdio` to keep primitive stream emission centralized while preserving CLI prompt behavior.
- 2026-02-18: Added LoRA loader runtime toggles `CODEX_LORA_MERGE_MODE` (`fast|precise`) and `CODEX_LORA_REFRESH_SIGNATURE` (`structural|content_sha256`) with strict parsing and CLI wiring (`--lora-merge-mode`, `--lora-refresh-signature`).
- 2026-02-20: `paths.py` now enforces fail-loud config semantics: invalid `apps/paths.json` parse/type errors raise, repo-relative entries are containment-checked against `CODEX_ROOT` (parent/symlink escapes rejected), and `_ensure_model_dirs` no longer swallows directory-provisioning failures.
- Keep this folder focused on config/bootstrap contracts; runtime execution logic belongs outside `infra/config`.
