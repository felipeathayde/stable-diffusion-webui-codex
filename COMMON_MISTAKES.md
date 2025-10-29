**Wrong command:** `rg -n "huggingface_guess" legacy`
**Cause + fix:** Repository archives legacy code under `.legacy/`, so the command targeted a non-existent `legacy/` directory; point ripgrep at `.legacy/` instead.
**Correct command:** `rg -n "huggingface_guess" .legacy`
**Wrong command:** `cat .sangoi/task-guidelines.md`
**Cause + fix:** Task guidelines file lives under `.sangoi/templates/document-guidelines.md`; referencing the old path triggers a file-not-found.
**Correct command:** `cat .sangoi/templates/document-guidelines.md`
**Wrong command:** `rg -ni "unet" apps/frontend`
**Cause + fix:** The project tree does not contain `apps/frontend`; scope the search to existing frontend paths under `apps/interface` instead of a missing directory.
**Correct command:** `rg -ni "unet" apps/interface`
**Wrong command:** `rg -ni "unet" apps/webui`
**Cause + fix:** `apps/webui` is not part of the repository layout; reuse the `apps/interface` path for frontend scans.
**Correct command:** `rg -ni "unet" apps/interface`
**Wrong command:** `rg -ni "unet" apps/backend/runtime/wan || true`
**Cause + fix:** Model runtimes live under versioned directories like `wan22`; target the specific directory instead of a non-existent `wan`.
**Correct command:** `rg -ni "unet" apps/backend/runtime/wan22 || true`
**Wrong command:** `git checkout -- modules sysinfo.py modules/ui_settings.py modules/settings_v2.py modules/infotext_utils.py modules/shared_options.py modules_forge/shared_options.py modules_forge/diffusers_patcher.py modules_forge/supported_controlnet.py scripts/xyz_grid.py NEWS.md extensions-builtin/extra-options-section/scripts/extra_options_section.py`
**Cause + fix:** `Missing directory separators in multi-level paths; Git expects the exact tracked path.`
**Correct command:** `git checkout -- modules/sysinfo.py modules/ui_settings.py modules/settings_v2.py modules/infotext_utils.py modules/shared_options.py modules_forge/shared_options.py modules_forge/diffusers_patcher.py modules_forge/supported_controlnet.py scripts/xyz_grid.py NEWS.md extensions-builtin/extra-options-section/scripts/extra_options_section.py`

**Wrong command:** `find . -type f -not -path ./.git/* -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `Command tried to add ignored caches; rerun with env var to skip errors or filter out ignored paths.`
**Correct command:** `find . -type f -not -path ./.git/* -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`

**Wrong command:** `git diff --staged --check`
**Cause + fix:** `Staged files contain trailing whitespace and missing EOF newlines; clean lint offenders before re-running.`
**Correct command:** `git diff --staged --check`
