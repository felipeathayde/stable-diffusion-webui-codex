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

**Wrong command:** `python - <<'PY'
import importlib
mod = importlib.import_module('apps.backend.use_cases.txt2img')
print('loaded', hasattr(mod, 'generate_txt2img'))
PY`
**Cause + fix:** `Importing the module pulls optional backend dependencies (e.g., safetensors) that are unavailable in this environment; validate syntax without loading heavy modules.`
**Correct command:** `python - <<'PY'
import ast, pathlib
path = pathlib.Path('apps/backend/use_cases/txt2img.py')
ast.parse(path.read_text())
print('syntax ok')
PY`

**Wrong command:** `python - <<'PY'
import ast, pathlib
files = [
    pathlib.Path('apps/backend/gguf/quants/__init__.py'),
    pathlib.Path('apps/backend/gguf/quants/registry.py'),
    pathlib.Path('apps/backend/gguf/quants/utils.py'),
    pathlib.Path('apps/backend/gguf/quants/kernels/__init__.py'),
    pathlib.Path('apps/backend/gguf/quants/kernels/base/__init__.py'),
]
for path in files:
    ast.parse(path.read_text())
print('syntax ok')
PY`
**Cause + fix:** `Residual patch markers remained in the file after editing, leaving invalid syntax. Remove stray '*** End Patch' lines before re-running the parser.`
**Correct command:** `python - <<'PY'
import ast, pathlib
files = [
    pathlib.Path('apps/backend/gguf/quants/__init__.py'),
    pathlib.Path('apps/backend/gguf/quants/registry.py'),
    pathlib.Path('apps/backend/gguf/quants/utils.py'),
    pathlib.Path('apps/backend/gguf/quants/kernels/__init__.py'),
    pathlib.Path('apps/backend/gguf/quants/kernels/base/__init__.py'),
]
for path in files:
    ast.parse(path.read_text())
print('syntax ok')
PY`

**Wrong command:** `python - <<'PY'
import importlib
mod = importlib.import_module('apps.backend.gguf.quants')
print('kernels:', sorted(name for name in ['Q4_0','Q5_0','Q8_0'] if hasattr(mod, name)))
PY`
**Cause + fix:** `Importing the package pulls apps.backend.__init__, which depends on optional safetensors; avoid runtime imports when lightweight structural checks suffice.`
**Correct command:** `python - <<'PY'
import ast, pathlib
path = pathlib.Path('apps/backend/gguf/quants/__init__.py')
ast.parse(path.read_text())
print('syntax ok')
PY`

**Wrong command:** `python tools/gguf/compare_codex_forge.py --iterations 1 --blocks 2 --log-level DEBUG`
**Cause + fix:** `System interpreter lacks project dependencies (numpy, torch); use the repository's virtualenv when executing tooling.`
**Correct command:** `~/.venv/bin/python tools/gguf/compare_codex_forge.py --iterations 1 --blocks 2 --log-level DEBUG`

**Wrong command:** `~/.venv/bin/python tools/gguf/compare_codex_forge.py --iterations 1 --blocks 2 --log-level DEBUG`
**Cause + fix:** `IQ-family layouts still being ported; limit comparisons to supported K-family until the IQ metadata is in place.`
**Correct command:** `~/.venv/bin/python tools/gguf/compare_codex_forge.py --types Q2_K Q3_K Q4_K Q5_K Q6_K`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** `The blanket find walks into legacy submodules under .legacy/, so git add aborts on nested .git refs; prune the legacy tree (or stage files explicitly).`
**Correct command:** `find . -path './.legacy' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`

**Wrong command:** `find . -path './.legacy' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** `Repository .gitignore filters *_m.py, so the new Forge IQ modules stay ignored and git add aborts; whitelist the directory before rerunning.`
**Correct command:** `printf '!apps/backend/gguf/quants/kernels/iq_family/forge_*.py\n' >> .gitignore && git add .gitignore`

**Wrong command:** `git commit -m "feat(gguf): port iq-family forge kernels"`
**Cause + fix:** `Global git identity isn't configured in this workspace; set user.name and user.email before committing.`
**Correct command:** `git config --global user.name "Lucas Sangoi" && git config --global user.email "lucas@sangoi.dev"`
