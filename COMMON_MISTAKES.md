**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** Updated `.git/codex-stamp` after making changes, so no files appeared "newer" than the stamp. Re-run by explicitly adding the touched paths.
**Correct command:** `git add .sangoi/CHANGELOG.md apps/backend/runtime/sampling/context.py apps/backend/runtime/sampling/AGENTS.md .sangoi/task-logs/2025-11-02-sdxl-scheduler-normalization.md`

