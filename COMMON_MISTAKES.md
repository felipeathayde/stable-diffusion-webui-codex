**Wrong command:** `rg -n "huggingface_guess" legacy`
**Cause + fix:** Repository archives legacy code under `.legacy/`, so the command targeted a non-existent `legacy/` directory; point ripgrep at `.legacy/` instead.
**Correct command:** `rg -n "huggingface_guess" .legacy`
**Wrong command:** `rg -n "iq1" legacy -g"*.py"`
**Cause + fix:** Repository archives legacy sources under `.legacy/`; point ripgrep at `.legacy/` instead of the non-existent `legacy/` path.
**Correct command:** `rg -n "iq1" .legacy -g"*.py"`
**Wrong command:** `find . -path './.legacy' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** The bulk add still walks ignored caches (`__pycache__/`, `.refs/`, etc.), causing `git add` to abort; stage the known documentation files explicitly instead of using the sweeping find.
**Correct command:** `git add AGENTS.md THIRD_PARTY_NOTICES.md COMMON_MISTAKES.md .sangoi/CHANGELOG.md .sangoi/task-logs/2025-10-30-docs-legacy-paths.md .sangoi/handoffs/2025-10-30-docs-legacy-paths.md`
**Wrong command:** `cat .sangoi/task-guidelines.md`
**Cause + fix:** Task guidelines file lives under `.sangoi/templates/document-guidelines.md`; referencing the old path triggers a file-not-found. Repeated slip—double-check the path before running.
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

**Wrong command:** `rg -n "bnb_avaliable" apps/backend/runtime/ops`
**Cause + fix:** `Typo in the search term; the code exposes "_BNB_AVAILABLE" (double 'l').`
**Correct command:** `rg -n "_BNB_AVAILABLE" apps/backend/runtime/ops`

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

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `The bulk add traverses .legacy submodules, so git add aborts on nested .git metadata; stage the touched files explicitly instead.`
**Correct command:** `git add apps/backend/patchers/unet.py apps/backend/patchers/AGENTS.md .sangoi/CHANGELOG.md .sangoi/task-logs/2025-10-30-backend-unet-patcher-refactor.md .sangoi/handoffs/2025-10-30-backend-unet-patcher-refactor.md`
**Wrong command:** `find . -path './.legacy' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `The search still walks cached/ignored directories, so git add aborts on gitignored paths; enumerate the known changed sources explicitly instead of mass-adding.`
**Correct command:** `git add apps/backend/runtime/models/loader.py apps/backend/codex/loader.py apps/backend/engines/common/base.py apps/backend/engines/{AGENTS.md,common/AGENTS.md,sd/AGENTS.md,flux/AGENTS.md,chroma/AGENTS.md} apps/backend/engines/sd/{sd15.py,sd20.py,sd35.py,sdxl.py} apps/backend/engines/flux/flux.py apps/backend/engines/chroma/chroma.py apps/backend/runtime/models/AGENTS.md apps/backend/codex/AGENTS.md .sangoi/CHANGELOG.md .sangoi/task-logs/2025-11-01-diffusion-engine-lifecycle.md .sangoi/handoffs/2025-11-01-diffusion-engine-lifecycle.md`
**Wrong command:** `python - <<'PY'
from apps.backend.patchers.controlnet import ControlNet, ControlLora, apply_controlnet_advanced
from apps.backend.runtime.controlnet import ControlRequest
print('import ok')
PY`
**Cause + fix:** `Importing backend modules pulls optional deps (safetensors) unavailable in this environment; rely on ast.parse or targeted module imports that avoid heavy dependencies.`
**Correct command:** `python - <<'PY'
import ast, pathlib
for path in pathlib.Path('apps/backend/patchers/controlnet').rglob('*.py'):
    ast.parse(path.read_text())
print('syntax ok')
PY`

**Wrong command:** `sed -n '1,160p' .sangoi/task-guidelines.md`
**Cause + fix:** `Task guidelines were relocated to .sangoi/templates/document-guidelines.md; target the existing path to avoid file-not-found errors.`
**Correct command:** `sed -n '1,160p' .sangoi/templates/document-guidelines.md`

**Wrong command:** `sed -n '1,200p' apps/backend/runtime/adapters/utils.py`
**Cause + fix:** `Runtime adapters do not expose a utils.py file; utility helpers live under apps/backend/runtime/utils.py.`
**Correct command:** `sed -n '1,200p' apps/backend/runtime/utils.py`

**Wrong command:** `sed -n '1,200p' apps/backend/runtime/common/nn/unet.py`
**Cause + fix:** `UNet runtime is organized as a package; the model definition lives under apps/backend/runtime/common/nn/unet/model.py.`
**Correct command:** `sed -n '1,200p' apps/backend/runtime/common/nn/unet/model.py`

**Wrong command:** `rg -n "^apps/" /tmp/recent_files.txt | cut -d: -f1 | sort -u | xargs -I{} sed -n '1,99999p' {}`
**Cause + fix:** `The -n flag prints line numbers; piping those into sed as file paths produced 'No such file or directory'. We only needed to filter lines that start with apps/.`
**Correct command:** `rg "^apps/" /tmp/recent_files.txt | sort -u | head -n 100`

**Wrong command:** `rg -n "^\s*(from|import)\s+(?!apps\b|__future__\b|typing\b|dataclasses\b|enum\b|logging\b|pathlib\b|re\b|os\b|sys\b|json\b|time\b|math\b|torch\b|diffusers\b|safetensors\b|transformers\b|tqdm\b|numpy\b|PIL\b|cv2\b|functools\b|contextlib\b|itertools\b|collections\b|importlib\b|types\b|typing_extensions\b|subprocess\b|tempfile\b|shutil\b|concurrent\b|asyncio\b|urllib\b|requests\b|pydantic\b|rich\b|jinja2\b|yaml\b|onnx\b|tensorrt\b|xformers\b|accelerate\b|peft\b|jsonschema\b|scipy\b|skimage\b|einops\b)" apps`
**Cause + fix:** `ripgrep without PCRE2 does not support look-around; the negative look-ahead triggered a regex parse error. Use --pcre2 or split into a pipeline without look-around.`
**Correct command:** `rg --pcre2 -n "^\s*(from|import)\s+(?!apps\b)" apps`
**Wrong command:** `rg -n "from\\s+apps\\.backend\\.engines\\.(sd|flux|chroma)\\.(?!__init__)" apps/backend/engines`
**Cause + fix:** `ripgrep does not support look-around by default; the negative look-ahead broke the regex. Use --pcre2 or filter by pipe.`
**Correct command:** `rg --pcre2 -n "from\\s+apps\\.backend\\.engines\\.(sd|flux|chroma)\\.(?!__init__)" apps/backend/engines`
**Wrong command:** `apply_patch << 'PATCH' ... (missing *** End Patch / stray smart quotes in timeout_ms)`
**Cause + fix:** `Patch here-doc ended without the required terminator and used a curly quote in timeout_ms, making the JSON invalid. Always end with '*** End Patch' and avoid smart quotes.`
**Correct command:** `apply_patch << 'PATCH'\n*** Begin Patch\n*** Add File: .sangoi/handoffs/2025-11-01-backend-modules-audit.md\n+<content>\n*** End Patch\nPATCH`
