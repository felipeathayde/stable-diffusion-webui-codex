**Wrong command:** `ls -λα apps/interface/src/stores`
**Cause + fix:** `Non-ASCII characters in the flag (locale/IME slip) made ls parse an invalid option. Re-run with plain ASCII flags.`
**Correct command:** `ls -la apps/interface/src/stores`

**Wrong command:** `rg -n "Unit test that `prepare_txt2vid`" .sangoi/plans/2025-12-14-wan22-ui-backend-alignment.md`
**Cause + fix:** `Backticks are command substitution in bash; the shell tries to execute the text inside them before rg runs. Use single quotes around patterns that contain backticks (or escape them) so the pattern is passed literally to rg.`
**Correct command:** `rg -n --fixed-strings 'Unit test that `prepare_txt2vid`' .sangoi/plans/2025-12-14-wan22-ui-backend-alignment.md`

**Wrong command:** `rg -n "--color-border" apps/interface/src/styles.css apps/interface/src/styles/*.css | head -n 50`
**Cause + fix:** `Patterns that start with "--" look like CLI flags to rg. Use "--" to end option parsing (or pass the pattern with -e) so rg treats it as a literal search term.`
**Correct command:** `rg -n -- "--color-border" apps/interface/src/styles.css apps/interface/src/styles/*.css | head -n 50`

**Wrong command:** `rg -n "<<<<<<<|=======|>>>>>>>" .`
**Cause + fix:** `This repo vendors tokenizer vocab files that include tokens like "========" and ">>>>>>>>", producing huge false-positive output that looks like merge conflicts. Exclude the Hugging Face vocab/tokenizer JSON trees (or search only source globs) when checking for real conflict markers.`
**Correct command:** `rg -n "^<<<<<<< |^=======$|^>>>>>>> " --glob '!apps/backend/huggingface/**' --glob '!apps/interface/dist/**' .`

**Wrong command:** `./.uv/bin/uv python install 3.12.10`
**Cause + fix:** `In sandboxed environments, $HOME/.local (XDG_DATA_HOME default) may be read-only, causing uv to fail creating ~/.local/share/uv/python. Set XDG_DATA_HOME to a writable path (e.g., under $HOME/.cache) when running uv python/lock commands.`
**Correct command:** `XDG_DATA_HOME=$HOME/.cache/uv-data ./.uv/bin/uv python install 3.12.10`

**Wrong command:** `~/.venv/binpython -m pytest -q tests/backend/test_codex_quantization_parametergguf_to.py`
**Cause + fix:** `Typo in the venv interpreter path (missing /bin/python). Use the correct virtualenv Python path when running tests.`
**Correct command:** `~/.venv/bin/python -m pytest -q tests/backend/test_codex_quantization_parametergguf_to.py`

**Wrong command:** `CODEX_ZIMAGE_DEBUG=1 CODEX_ZIMAGE_DEBUG_TENC_TOKENS=1 CODEX_ZIMAGE_DEBUG_TENC_TEXT=1 CODEX_LOG_SIGMAS=1 ~/.venv/bin/python tools/diagnostics/run_backend_diag.py --stdout-only -- --port 7850`
**Cause + fix:** `apps.backend.interfaces.api.run_api expects CODEX_ROOT to be set by the launcher (run-webui/run-tui). When invoking the module directly, export CODEX_ROOT (and in CPU-only sandboxes also force CODEX_DIFFUSION_DEVICE=cpu) before running.`
**Correct command:** `CODEX_ROOT=/home/lucas/work/stable-diffusion-webui-codex CODEX_DIFFUSION_DEVICE=cpu CODEX_ZIMAGE_DEBUG=1 CODEX_ZIMAGE_DEBUG_TENC_TOKENS=1 CODEX_ZIMAGE_DEBUG_TENC_TEXT=1 CODEX_LOG_SIGMAS=1 ~/.venv/bin/python tools/diagnostics/run_backend_diag.py --stdout-only -- --port 7850`

**Wrong command:** `sed -n '1,260p' apps/backend/use_cases/txt2img_pipeline.py`
**Cause + fix:** `Txt2img pipeline is a package directory under apps/backend/use_cases/txt2img_pipeline/; the runner lives at runner.py. List the directory before assuming a flat file path.`
**Correct command:** `ls -la apps/backend/use_cases/txt2img_pipeline && sed -n '1,260p' apps/backend/use_cases/txt2img_pipeline/runner.py`

**Wrong command:** `find apps -name AGENTS.md -maxdepth 4 | sort`
**Cause + fix:** `find global options like -maxdepth must appear before the test expressions; placing it after -name triggers warnings and can change behavior.`
**Correct command:** `find apps -maxdepth 4 -name AGENTS.md | sort`

**Wrong command:** `find apps/backend -name AGENTS.md -maxdepth 4`
**Cause + fix:** `find global options like -maxdepth must appear before the test expressions; placing it after -name triggers warnings and can change behavior.`
**Correct command:** `find apps/backend -maxdepth 4 -name AGENTS.md`

**Wrong command:** `python -m pytest tests/test_backend_import_lightweight.py`
**Cause + fix:** `Pytest is not installed in the current environment; install pytest (preferably in the active venv) before running the test suite.`
**Correct command:** `python -m pip install pytest && python -m pytest tests/test_backend_import_lightweight.py`

**Wrong command:** `sed -n '1,200p' .bottle/handoffs/HANDOFF_GUIDE`
**Cause + fix:** `This repository keeps the handoff guide under .sangoi/handoffs/HANDOFF_GUIDE.md; the .bottle path does not exist. Check the actual guide location before reading.`
**Correct command:** `sed -n '1,200p' .sangoi/handoffs/HANDOFF_GUIDE.md`

**Wrong command:** `rg -n "huggingface_guess" legacy`
**Cause + fix:** Repository uses a local reference snapshot tree for upstream inspection; the command targeted a non-existent `legacy/` directory. Search from repo root instead.
**Correct command:** `rg -n "huggingface_guess" .`
**Wrong command:** `rg -n "iq1" legacy -g"*.py"`
**Cause + fix:** Repository uses a local reference snapshot tree for upstream inspection; the command targeted a non-existent `legacy/` path. Search from repo root instead.
**Correct command:** `rg -n "iq1" . -g"*.py"`
**Wrong command:** `find . -path './.legacy' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** The bulk add still walks ignored caches (`__pycache__/`, etc.), causing `git add` to abort; stage the known documentation files explicitly instead of using the sweeping find.
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
**Wrong command:** `git add apps backend/runtime utils.py tools dev/trace_pipeline_graph.py`
**Cause + fix:** `Missing directory separators in multi-level paths; Git expects the exact tracked path.`
**Correct command:** `git add apps/backend/runtime/utils.py tools/dev/trace_pipeline_graph.py`

**Wrong command:** `find . -type f -not -path ./.git/* -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `Command tried to add ignored caches; rerun with env var to skip errors or filter out ignored paths.`
**Correct command:** `find . -type f -not -path ./.git/* -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`

**Wrong command:** `git diff --staged --check`
**Cause + fix:** `Staged files contain trailing whitespace and missing EOF newlines; clean lint offenders before re-running.`
**Correct command:** `git diff --staged --check`

**Wrong command:** `apply_patch << 'PATCH'` with an `*** Add File:` section but file contents not prefixed with `+`
**Cause + fix:** Patch grammar requires every line of a newly added file to start with `+`. The attempt omitted the `+` prefixes, so the tool reported an invalid hunk header. Re-ran with `+` on each content line.
**Correct command:** `apply_patch << 'PATCH'` … `*** Add File: path` then lines prefixed by `+`.

**Wrong command:** `rg -n "bnb_avaliable" apps/backend/runtime/ops`
**Cause + fix:** `Typo in the search term; the code exposes "_BNB_AVAILABLE" (double 'l').`
**Correct command:** `rg -n "_BNB_AVAILABLE" apps/backend/runtime/ops`

**Wrong command:** `uv sync --locked --frozen --extra cpu`
**Cause + fix:** `uv treats --locked and --frozen as mutually exclusive for sync. Use --locked (preferred for installs: errors if the lock would change) or --frozen (use the lock without updating it).`
**Correct command:** `uv sync --locked --extra cpu`

**Wrong command:** `~/.venv/bin/python -c "from apps.backend.infra.config import args as cfg; import apps.backend.engines.sd.sdxl as mod; print(bool(getattr(cfg.args, 'debug_conditioning', False)))"`
**Cause + fix:** `Backend config auto-initializes with AUTO devices and aborts when CUDA is unavailable; set explicit CPU devices before importing.`
**Correct command:** `CODEX_DIFFUSION_DEVICE=cpu CODEX_TE_DEVICE=cpu CODEX_VAE_DEVICE=cpu ~/.venv/bin/python -c "from apps.backend.infra.config import args as cfg; import apps.backend.engines.sd.sdxl as mod; print(bool(getattr(cfg.args, 'debug_conditioning', False)))"`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `Command tries to add ignored __pycache__ paths; filter them out or use a narrower add list.`
**Correct command:** `find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** `Even with --ignore-errors, git exits non-zero for ignored __pycache__; add an explicit path filter before piping to git add.`
**Correct command:** `find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `Commit checklist requires this command, but it aborts whenever ignored dirs like __pycache__, apps/interface/dist, node_modules, tmp, or logNormal2.txt pop up. Re-run with explicit -not -path guards for each ignored tree before piping to git add.`
**Correct command:** `find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' -not -path './apps/interface/dist/*' -not -path './apps/interface/node_modules/*' -not -path './tmp/*' -not -path './logNormal2.txt' -newer .git/codex-stamp -print0 | xargs -0 -- git add`

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
    pathlib.Path('apps/backend/quantization/dequant.py'),
    pathlib.Path('apps/backend/quantization/kernels/__init__.py'),
    pathlib.Path('apps/backend/quantization/api.py'),
    pathlib.Path('apps/backend/quantization/gguf/__init__.py'),
    pathlib.Path('apps/backend/quantization/gguf/reader.py'),
    pathlib.Path('apps/backend/quantization/gguf/writer.py'),
]
for path in files:
    ast.parse(path.read_text())
print('syntax ok')
PY`
**Cause + fix:** `Residual patch markers remained in the file after editing, leaving invalid syntax. Remove stray '*** End Patch' lines before re-running the parser.`
**Correct command:** `python - <<'PY'
import ast, pathlib
files = [
    pathlib.Path('apps/backend/quantization/dequant.py'),
    pathlib.Path('apps/backend/quantization/kernels/__init__.py'),
    pathlib.Path('apps/backend/quantization/api.py'),
    pathlib.Path('apps/backend/quantization/gguf/__init__.py'),
    pathlib.Path('apps/backend/quantization/gguf/reader.py'),
    pathlib.Path('apps/backend/quantization/gguf/writer.py'),
]
for path in files:
    ast.parse(path.read_text())
print('syntax ok')
PY`

**Wrong command:** `python - <<'PY'
import importlib
mod = importlib.import_module('apps.backend.quantization')
print('loaded', hasattr(mod, 'QuantType'))
PY`
**Cause + fix:** `Importing backend packages can pull heavy deps (torch, safetensors) and trigger registration side effects; use ast.parse when you only need a syntax check.`
**Correct command:** `python - <<'PY'
import ast, pathlib
path = pathlib.Path('apps/backend/quantization/__init__.py')
ast.parse(path.read_text())
print('syntax ok')
PY`

**Wrong command:** `python tools/debug/test_dequant.py`
**Cause + fix:** `System interpreter lacks project dependencies (numpy, torch); use the repository's virtualenv when executing tooling.`
**Correct command:** `~/.venv/bin/python tools/debug/test_dequant.py`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** `The blanket find walks into archived submodules under **, so git add aborts on nested .git refs; prune the reference tree (or stage files explicitly).`
**Correct command:** `find . -path './' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`

**Wrong command:** `git commit -m "feat(gguf): port iq-family GGUF kernels"`
**Cause + fix:** `Global git identity isn't configured in this workspace; set user.name and user.email before committing.`
**Correct command:** `git config --global user.name "Lucas Sangoi" && git config --global user.email "lucas@sangoi.dev"`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `The bulk add traverses ** submodules, so git add aborts on nested .git metadata; stage the touched files explicitly instead.`
**Correct command:** `git add apps/backend/patchers/unet.py apps/backend/patchers/AGENTS.md .sangoi/CHANGELOG.md .sangoi/task-logs/2025-10-30-backend-unet-patcher-refactor.md .sangoi/handoffs/2025-10-30-backend-unet-patcher-refactor.md`
**Wrong command:** `find . -path './' -prune -o -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `The search still walks cached/ignored directories, so git add aborts on gitignored paths; enumerate the known changed sources explicitly instead of mass-adding.`
**Correct command:** `git add apps/backend/runtime/models/loader.py apps/backend/core/engine_loader.py apps/backend/engines/common/base.py apps/backend/engines/{AGENTS.md,common/AGENTS.md,sd/AGENTS.md,flux/AGENTS.md,chroma/AGENTS.md} apps/backend/engines/sd/{sd15.py,sd20.py,sd35.py,sdxl.py} apps/backend/engines/flux/flux.py apps/backend/engines/chroma/chroma.py apps/backend/runtime/models/AGENTS.md apps/backend/core/AGENTS.md .sangoi/CHANGELOG.md .sangoi/task-logs/2025-11-01-diffusion-engine-lifecycle.md .sangoi/handoffs/2025-11-01-diffusion-engine-lifecycle.md`
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

**Wrong command:** `sed -n '1,260p' apps/backend/patchers/controlnet/models/sd/control.py`
**Cause + fix:** `ControlNet models live under apps/backend/patchers/controlnet/architectures/sd/; the older models/ path does not exist in this Codex layout. Target the architectures directory instead of the legacy path.`
**Correct command:** `sed -n '1,260p' apps/backend/patchers/controlnet/architectures/sd/control.py`

**Wrong command:** `sed -n '1,200p' apps/backend/runtime/adapters/utils.py`
**Cause + fix:** `Runtime adapters do not expose a utils.py file; utility helpers live under apps/backend/runtime/utils.py.`
**Correct command:** `sed -n '1,200p' apps/backend/runtime/utils.py`

**Wrong command:** `sed -n '1,200p' apps/backend/runtime/common/nn/unet.py`
**Cause + fix:** `UNet runtime is organized as a package; the model definition lives under apps/backend/runtime/common/nn/unet/model.py.`
**Correct command:** `sed -n '1,200p' apps/backend/runtime/common/nn/unet/model.py`

**Wrong command:** `sed -n '1,200p' .sangoi/design/model-tabs-and-workflows.md`
**Cause + fix:** `The model tabs/workflows spec was moved under .sangoi/design/flows/model-workflows.md; the old path no longer exists. Target the flows/ directory when reading or updating the spec.`
**Correct command:** `sed -n '1,200p' .sangoi/design/flows/model-workflows.md`

**Wrong command:** `python -m ast apps/backend/use_cases/txt2img_pipeline/runner.py apps/backend/engines/sd/sdxl.py`
**Cause + fix:** `The stdlib ast module CLI accepts at most one input file; passing multiple paths is treated as extra arguments and causes a usage error. Run the syntax check separately for each file instead of batching them in a single call.`
**Correct command:** `python -m ast apps/backend/use_cases/txt2img_pipeline/runner.py`

**Wrong command:** `python -m pytest tests/backend/model_parser/test_sdxl_validation.py`
**Cause + fix:** Neither the system Python (pyenv 3.12) nor the repo venv have `pytest` preinstalled; install it into `~/.venv` and run tests from that interpreter.
**Correct command:** `~/.venv/bin/pip install -U pytest && ~/.venv/bin/python -m pytest tests/backend/model_parser/test_sdxl_validation.py`

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
**Wrong command:** `python3 tools/diagnostics/dry_run_pipeline.py`
**Cause + fix:** `The environment lacks torch; importing runtime helpers pulls torch and fails. Use the static dry-run that avoids PyTorch imports.`
**Correct command:** `python3 tools/diagnostics/dry_run_pipeline_static.py`
**Wrong command:** `python - <<'PY'
from apps.backend.runtime.logging import calltrace
PY`
**Cause + fix:** `The calltrace module was removed; pipeline debugging now lives in apps/backend/runtime/pipeline_debug.py.`
**Correct command:** `python - <<'PY'
from apps.backend.runtime.pipeline_debug import set_pipeline_debug
set_pipeline_debug(True)
PY`

**Wrong command:** `rg -n "all in fp32" -g"*"`
**Cause + fix:** `The literal phrase does not exist in the tracked sources; search broadly for fp32 flags or inspect relevant modules directly.`
**Correct command:** `rg -ni "fp32" apps`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `Sweep hits archived submodule refs under .legacy, causing git add to abort; stage the known modified files explicitly instead of scanning the entire tree.`
**Correct command:** `git add .sangoi/CHANGELOG.md COMMON_MISTAKES.md apps/AGENTS.md apps/backend/AGENTS.md apps/backend/services/options_store.py apps/backend/infra/config/args.py apps/backend/interfaces/api/run_api.py apps/backend/runtime/memory/memory_management.py apps/launcher/AGENTS.md apps/launcher/profiles.py apps/launcher/services.py .sangoi/handoffs/2025-11-02-backend-device-bootstrap-hardening.md .sangoi/handoffs/2025-11-02-sdxl-device-env-trace.md .sangoi/task-logs/2025-11-02-backend-device-bootstrap-hardening.md .sangoi/task-logs/2025-11-02-sdxl-device-env-trace.md`
**Wrong command:** `python - <<'PY'
from apps.backend.infra.config import args
from apps.backend.runtime.memory.config import DeviceRole
print('diffusion_device', args.memory_config.component_policy(DeviceRole.CORE).preferred_backend)
print('te_device', args.memory_config.component_policy(DeviceRole.TEXT_ENCODER).preferred_backend)
print('vae_device', args.memory_config.component_policy(DeviceRole.VAE).preferred_backend)
print('primary_backend', args.memory_config.device_backend)
print('gpu_prefer_construct', args.memory_config.gpu_prefer_construct)
PY`
**Cause + fix:** `Importing apps.backend triggered optional deps like safetensors that are absent in this sandbox; inspect config by parsing env/args module without pulling heavy runtime packages.`
**Correct command:** `python - <<'PY'
import importlib
args = importlib.import_module('apps.backend.infra.config.args')
from apps.backend.runtime.memory.config import DeviceRole
print('diffusion_device', args.memory_config.component_policy(DeviceRole.CORE).preferred_backend)
print('te_device', args.memory_config.component_policy(DeviceRole.TEXT_ENCODER).preferred_backend)
print('vae_device', args.memory_config.component_policy(DeviceRole.VAE).preferred_backend)
print('primary_backend', args.memory_config.device_backend)
print('gpu_prefer_construct', args.memory_config.gpu_prefer_construct)
PY`
**Wrong command:** `python - <<'PY'
import torch
print('torch cuda available:', torch.cuda.is_available())
print('torch version:', torch.__version__)
PY`
**Cause + fix:** `Global sandbox lacks torch; querying availability without the package raises ModuleNotFoundError. Use project tooling that stubs torch or run inside the configured environment with torch installed.`
**Correct command:** `python - <<'PY'\nprint('torch not installed in sandbox; run inside ~/.venv with torch available')\nPY`
**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** Command traversed into submodules and ignored files; `git add` failed on pathspecs. Limit to tracked repo and rely on `git add` without piping ignored files.
**Correct command:** `git add -A`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** Updated `.git/codex-stamp` after making changes, so no files appeared "newer" than the stamp. Re-run by explicitly adding the touched paths.
**Correct command:** `git add .sangoi/CHANGELOG.md apps/backend/runtime/sampling/context.py apps/backend/runtime/sampling/AGENTS.md .sangoi/task-logs/2025-11-02-sdxl-scheduler-normalization.md`

**Wrong command:** `printf "... \`find . -newer .git/codex-stamp\` ..." >> COMMON_MISTAKES.md`
**Cause + fix:** Using double quotes with backticks executed the subshell, attempting to run `.git/codex-stamp` and failing with `Permission denied`. Use a single-quoted heredoc or escape backticks when appending literal commands.
**Correct command:** `cat <<'EOF' >> COMMON_MISTAKES.md` (paste content, then `EOF`)

**Wrong command:** `sed -n '1,260p' apps/interface/src/stores/txt2vid.ts`
**Cause + fix:** The video store lives at `apps/interface/src/stores/video.ts`; there is no `txt2vid.ts` file. Use the existing consolidated video store path instead of guessing a legacy filename.
**Correct command:** `sed -n '1,260p' apps/interface/src/stores/video.ts`

**Wrong command:** `npm --prefix apps/interface test -- --runTestsByPath=src/components/QuickSettingsBar.vue src/components/quicksettings/QuickSettingsBase.vue src/components/quicksettings/QuickSettingsPerf.vue`
**Cause + fix:** Vitest in this repo does not support the Jest-style `--runTestsByPath` flag; the CLI aborted with `Unknown option`. Use Vitest's native filters or just run the configured test script without extra flags when spot-checking components.
**Correct command:** `npm --prefix apps/interface test`
**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** The generic `find | git add` sweep attempted to stage `__pycache__` files created by `py_compile`, which are ignored by the repo and cause `git add` to error out. Prefer staging only the tracked files you actually changed or extend the `find` filter to skip ignored directories like `__pycache__`.
**Correct command:** `git add .sangoi/CHANGELOG.md .sangoi/handoffs/HANDOFF_2025-12-05-flux-view-and-callgraph.md .sangoi/handoffs/HANDOFF_LOG.md .sangoi/task-logs/2025-12-05-flux-view-and-callgraph.md apps/backend/runtime/model_parser/AGENTS.md apps/backend/runtime/model_parser/families/flux.py apps/backend/runtime/models/AGENTS.md apps/backend/runtime/models/loader.py apps/interface/src/stores/AGENTS.md apps/interface/src/stores/flux.ts`
**Wrong command:** `~/.venv/bin/python - <<'PY'
from apps.backend.runtime.utils import FilterPrefixView
from collections import OrderedDict
base = OrderedDict()
base['conditioner.embedders.0.weight'] = 1
base['conditioner.embedders.0.bias'] = 2
base['other'] = 3
view = FilterPrefixView(base, 'conditioner.embedders.0.', '')
print(dict(view.items()))
PY`
**Cause + fix:** `Importing apps.backend modules bootstraps the CUDA memory manager; without GPU availability it aborts. Use an isolated snippet that reimplements the tiny view for experiments instead of importing the package.`
**Correct command:** `python - <<'PY'
class FilterPrefixView(dict):
    def __init__(self, base, prefix, new_prefix=''):
        super().__init__({(new_prefix + k[len(prefix):]): v for k, v in base.items() if k.startswith(prefix)})
base = {
    'conditioner.embedders.0.weight': 1,
    'conditioner.embedders.0.bias': 2,
    'other': 3,
}
print(dict(FilterPrefixView(base, 'conditioner.embedders.0.', '')))
PY`
**Wrong command:** `python - <<'PY'
from pathlib import Path
path = Path('COMMON_MISTAKES.md')
text = path.read_text()
old = "**Wrong command:** `~/.venv/bin/python - <<'PY'\nfrom apps.backend.runtime.utils import FilterPrefixView\nfrom collections import OrderedDict\nbase = OrderedDict()\nbase['conditioner.embedders.0.weight'] = 1\nbase['conditioner.embedders.0.bias'] = 2\nbase['other'] = 3\nview = FilterPrefixView(base, 'conditioner.embedders.0.', '')\nprint(dict(view.items()))\nPY`\n**Cause + fix:** `Importing apps.backend modules bootstraps the CUDA memory manager; without GPU availability it aborts. Use an isolated snippet that reimplements the tiny view for experiments instead of importing the package.`\n**Correct command:** ..."
if old not in text:
    raise SystemExit('block not found')
text = text.replace(old, new)
path.write_text(text)
PY`
**Cause + fix:** `The exact markdown block had already been altered, so the literal string lookup failed and aborted. Inspect the live snippet first or use regex replacement.`
**Correct command:** `python - <<'PY'
from pathlib import Path
path = Path('COMMON_MISTAKES.md')
text = path.read_text()
start = text.index("**Wrong command:** `~/.venv/bin/python - <<'PY'")
end = text.index("PY`", start) + 3
segment = text[start:end]
print(segment)
PY`
**Wrong command:** `python - <<'PY'
from pathlib import Path
path = Path('COMMON_MISTAKES.md')
text = path.read_text()
old = ...
new = ...
if old not in text:
    raise SystemExit('block not found')
path.write_text(text.replace(old, new))
PY`
**Cause + fix:** `Attempted to inline a multi-line replacement with ellipses; Python treated the placeholder as literal code and raised a SyntaxError. Write the full string or build it programmatically.`
**Correct command:** `python - <<'PY'
from pathlib import Path
path = Path('COMMON_MISTAKES.md')
text = path.read_text()
print(text.count("FilterPrefixView"))
PY`
**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause + fix:** `Traversal picks up ignored __pycache__ entries, so git add bails. Add --ignore-errors or stage explicit files instead of sweeping.`
**Correct command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add --ignore-errors`
**Cause + fix:** `Even with --ignore-errors, git stops when xargs feeds ignored paths; prune __pycache__/refs before piping to git add.`
**Correct command:** `find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' -not -path './*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`

**Wrong command:** `~/.venv/bin/python tools/dev/validate_sdxl_contract.py`
**Cause + fix:** Python couldn't import the local `apps.*` packages when invoked from the repo root; the script lacked a `sys.path` entry for the repository root. Prepend the repo root to `sys.path` within the script before importing `apps.*`.
**Correct command:** `~/.venv/bin/python tools/dev/validate_sdxl_contract.py` (after adding `sys.path.insert(0, <repo_root>)` in the script)

**Wrong command:** `apply_patch` updating `apps/backend/runtime/modules/AGENTS.md` with unmatched context
**Cause + fix:** The patch assumed anchor text that didn’t exist; read the current file and update using an `Update File` hunk that matches real content, or replace the file body coherently.
**Correct command:** `apply_patch` with a hunk aligned to the existing content (or rewrite the file section explicitly).
**Wrong command:** `rg -n "SDXL" docs/plan`
**Cause + fix:** Repository no longer has a `docs/plan/` tree, so ripgrep fails; target the current `.sangoi/plans/` directory that replaced it.
**Correct command:** `rg -n "SDXL" .sangoi/plans`

**Wrong command:** `ls docs/plan`
**Cause and fix:** `Legacy docs/plan/ tree no longer exists—modern plans live under .sangoi/plans. Target the active directory instead of a removed path.`
**Correct command:** `ls .sangoi/plans`

**Wrong command:** `sed -n '1,200p' .sangoi/backend/overview`
**Cause + fix:** `The target path is a directory, not a file; list its contents first and then run sed/cat on a specific file inside (for example a README.md) instead of trying to read the directory directly.`
**Correct command:** `ls .sangoi/backend/overview`

**Wrong command:** `apply_patch <<'PATCH'\n*** Update File: COMMON_MISTAKES.md\n@@\n **Correct command:** `rg -n "SDXL" .sangoi/plans`\n+\n+**Wrong command:** ...\n*** End Patch\nPATCH`
**Cause and fix:** `Patch failed because the required '*** Begin Patch' header was missing. Include the Begin Patch line (and End Patch) whenever using apply_patch.`
**Correct command:** `apply_patch <<'PATCH'\n*** Begin Patch\n*** Update File: COMMON_MISTAKES.md\n@@\n **Correct command:** `rg -n "SDXL" .sangoi/plans`\n+\n+**Wrong command:** ...\n*** End Patch\nPATCH`

**Wrong command:** `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
**Cause and fix:** `Command walks through ignored paths (node_modules/, __pycache__, logs) so git aborts. Filter ignored directories before piping into git add (or add --ignore-errors).`
**Correct command:** `find . -type f -not -path './.git/*' -not -path './*' -not -path '*/__pycache__/*' -not -path '*/node_modules/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`

**Wrong command:** `git push -u origin HEAD` (with 1s timeout)
**Cause and fix:** `CLI invocation limited the push to ~1s and the remote handshake didn't finish in time, so the helper timed out even though git was fine.`
**Correct command:** `git push -u origin HEAD` (let it run without the artificial timeout)
**Wrong command:** `ls docs/plan`
**Cause + fix:** `This repository does not ship a \`docs/\` tree; documentation lives under \`.sangoi/\` (plans under \`.sangoi/plans\`). List the real doc root first instead of drilling into removed paths.`
**Correct command:** `ls .sangoi`
**Wrong command:** `python - <<'PY'
from apps.backend.interfaces.api.run_api import app
print(type(app))
PY`
**Cause + fix:** `Importing the API module pulls PyTorch, which is absent from the base interpreter; rerun the inspection inside the managed virtualenv that already has torch installed.`
**Correct command:** `~/.venv/bin/python - <<'PY'
from apps.backend.interfaces.api.run_api import app
print(type(app))
PY`

**Wrong command:** `python -m pytest tests/backend/model_registry/test_vae_selection.py`
**Cause + fix:** `O Python global não tem pytest instalado; usar o interpretador gerido em ~/.venv (com pytest instalado) para rodar os testes, conforme diretriz anterior.`
**Correct command:** `~/.venv/bin/python -m pytest tests/backend/model_registry/test_vae_selection.py`

**Wrong command:** `sed -n '1,220p' .sangoi/research/wan22-text-encoder-compat.md`
**Cause + fix:** `O caminho estava errado; o arquivo de pesquisa de compatibilidade do text encoder WAN22 fica sob .sangoi/research/runtime/, não diretamente em .sangoi/research/.`
**Correct command:** `sed -n '1,220p' .sangoi/research/runtime/wan22-text-encoder-compat.md`
**Wrong command:** `python - <<'PY'
import inspect, uvicorn.middleware.asgi2
PY`
**Cause + fix:** `Uvicorn is not installed in the stock interpreter; point \`PYTHONPATH\` at the downloaded wheel (or install uvicorn inside the venv) before importing.`
**Correct command:** `PYTHONPATH=tmp/uvicorn python - <<'PY'
import inspect, uvicorn.middleware.asgi2
PY`
**Wrong command:** `type run-webui.bat`
**Cause + fix:** `Bash's \`type\` builtin reports command lookup instead of file contents; use \`cat\` to print the batch script.`
**Correct command:** `cat run-webui.bat`
**Wrong command:** `pip install uvicorn fastapi`
**Cause + fix:** `Command tried to write into the read-only pyenv site-packages tree; install dependencies inside the writable project virtualenv instead.`
**Correct command:** `~/.venv/bin/pip install uvicorn fastapi`
**Wrong command:** `pip install --user uvicorn fastapi`
**Cause + fix:** `The sandbox blocks writes to ~/.local, so the user-level install also fails; direct pip to a writable target under the workspace.`
**Correct command:** `pip install --target tmp/pip uvicorn fastapi`
**Wrong command:** `rg -n "--factory" -n`
**Cause + fix:** `Without a \`--\` separator, ripgrep treated \`--factory\` as another CLI flag and bailed; add \`--\` before the literal you want to search.`
**Correct command:** `rg -n -- '--factory'`
**Wrong command:** `rg -n --literal "--factory"`
**Cause + fix:** `Ripgrep has no \`--literal\` option; use \`-F\`/\`--fixed-strings\` with a \`--\` separator for literal flag text.`
**Correct command:** `rg -n -F -- '--factory'`
**Wrong command:** `rg -n --fixed-strings "--factory"`
**Cause + fix:** `Supplying the pattern without \`--\` makes ripgrep misinterpret it as another option; insert the separator so \`--factory\` is treated as search text.`
**Correct command:** `rg -n --fixed-strings -- '--factory'`

**Wrong command:** `git pull --rebase --autostash`
**Cause + fix:** `Untracked task log / handoff files block the pull with “would be overwritten”; stash them with --include-untracked (or add them) before running the mandated pull.`
**Correct command:** `git stash push --include-untracked && git pull --rebase --autostash && git stash pop`

**Wrong command:** `ls -la docs/plan`
**Cause + fix:** `The repo has no docs/plan directory; SDXL plans live under .sangoi/plans. List that path instead of the non-existent docs/plan.`
**Correct command:** `ls -la .sangoi/plans`

**Wrong command:** `ls docs/plan`
**Cause + fix:** `docs/plan is not present in this repo; plans are cataloged under .sangoi/plans.`
**Correct command:** `ls .sangoi/plans`

**Wrong command:** `rg -n "uvicorn" run-webui.*`
**Cause + fix:** `Ripgrep exits non-zero when no matches are found; add \`|| true\` or search known entrypoints directly.`
**Correct command:** `rg -n "uvicorn" run-webui.* || true`

**Wrong command:** `python - <<'PY'\nfrom apps.launcher.services import default_services\nservices = default_services()\nprint('API command:', services['API'].spec.command)\nPY`
**Cause + fix:** `Importing launcher.services pulls backend runtime and requires torch; the sandbox lacks torch so the import fails. Inspect the command by reading the file instead of importing.`
**Correct command:** `sed -n '80,150p' apps/launcher/services.py`

**Wrong command:** `cd /home/lucas/work/stable-diffusion-webui-codex && PYTHONPATH=$HOME/.netsuite CODEX_DIFFUSION_DEVICE=cpu CODEX_TE_DEVICE=cpu CODEX_VAE_DEVICE=cpu ~/.venv/bin/python - <<'PY'
import os,json
from fastapi.testclient import TestClient
from apps.backend.interfaces.api.run_api import create_api_app

app = create_api_app(argv=[], env=os.environ)
client = TestClient(app)

paths = [(r.path, sorted(list(getattr(r, 'methods', [])))) for r in app.router.routes if getattr(r, 'path', None) in {'/api/txt2img','/api/img2img','/api/tasks/{task_id}'}]
print('routes', paths)

resp = client.post('/api/txt2img', json={'__strict_version':1})
print('status', resp.status_code)
print('body', resp.json())
PY`
**Cause and fix:** `FastAPI startup + TestClient took longer than the harness default (~10s), so the command timed out at exit code 124. Run the same script with a longer timeout (≈60–120s) when bootstrapping the API in the CPU-only sandbox.`
**Correct command:** `timeout 120s cd /home/lucas/work/stable-diffusion-webui-codex && PYTHONPATH=$HOME/.netsuite CODEX_DIFFUSION_DEVICE=cpu CODEX_TE_DEVICE=cpu CODEX_VAE_DEVICE=cpu ~/.venv/bin/python - <<'PY'
import os,json
from fastapi.testclient import TestClient
from apps.backend.interfaces.api.run_api import create_api_app

app = create_api_app(argv=[], env=os.environ)
client = TestClient(app)

paths = [(r.path, sorted(list(getattr(r, 'methods', [])))) for r in app.router.routes if getattr(r, 'path', None) in {'/api/txt2img','/api/img2img','/api/tasks/{task_id}'}]
print('routes', paths)

resp = client.post('/api/txt2img', json={'__strict_version':1})
print('status', resp.status_code)
print('body', resp.json())
PY`

**Wrong command:** `python -m pytest tests/backend/model_registry/test_vae_selection.py`
**Cause and fix:** `Pytest is not installed in the current environment, so the module import fails. Install pytest (or use the project’s test runner) before invoking the test.`
**Correct command:** `python -m pip install pytest`
**Wrong command:** `ls docs/plan`
**Cause + fix:** The repository has no `docs/` directory; documentation lives under `.sangoi/`, so `docs/plan` errors. List the real doc root first to discover the available subdirectories.
**Correct command:** `ls .sangoi`

**Wrong command:** `PYTHONPATH=$HOME/.netsuite:. ~/.venv/bin/python - <<'PY'\nfrom apps.backend.runtime.models.state_dict import load_state_dict\nfrom pathlib import Path\npath = Path('/mnt/f/stable-diffusion-webui-codex/models/Stable-diffusion/cyberrealisticPony_v140.safetensors')\nload_state_dict(path)\nPY`
**Cause + fix:** `load_state_dict` expects a model plus a state-dict mapping; calling it with only a path raises a missing-argument error. To inspect checkpoint keys, load the safetensors mapping with `load_torch_file` instead.
**Correct command:** `PYTHONPATH=$HOME/.netsuite:. ~/.venv/bin/python - <<'PY'\nfrom pathlib import Path\nfrom apps.backend.runtime.utils import load_torch_file\nsd = load_torch_file(Path('/mnt/f/stable-diffusion-webui-codex/models/Stable-diffusion/cyberrealisticPony_v140.safetensors'))\nprint(len(sd), list(sd.keys())[:5])\nPY`

Wrong command: python - <<'PY'
Cause and fix: Executed with system Python that lacks project deps; reran with ~/.venv/python.
Correct command: ~/.venv/bin/python - <<'PY'

Wrong command: unzip *.whl -d wheel
Cause and fix: unzip not available in the shell; inspected wheel metadata via Python zipfile instead.
Correct command: python - <<'PY'

**Wrong command:** `python -m pytest tests/backend/model_parser/test_sdxl_validation.py tests/backend/test_k_prediction.py`
**Cause and fix:** The default pyenv interpreter lacks pytest; run tests with the project venv (or install pytest) before invoking the suite.
**Correct command:** `PYTHONPATH=$HOME/.netsuite ~/.venv/bin/python -m pytest tests/backend/model_parser/test_sdxl_validation.py tests/backend/test_k_prediction.py`

**Wrong command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls .legacy-functional && find .legacy-functional -maxdepth 3 -type f -iname '*sdxl*' | head -n 40`
**Cause and fix:** The `.legacy-functional` directory was assumed to exist at the repo root, but it was not present, so `ls` failed with ENOENT. First locate the directory (or confirm its absence) with a guarded search instead of hard-coding the path.
**Correct command:** `cd /home/lucas/work/stable-diffusion-webui-codex && find . -maxdepth 5 -type d -name '.legacy-functional'`

**Wrong command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls .sangoi/tasks && sed -n '1,200p' .sangoi/tasks/README.md`
**Cause and fix:** The `.sangoi/tasks` directory exposes task specs as `F*.md` files and has no `README.md`; attempting to read a non-existent README raises ENOENT. Inspect the relevant `F*.md` task files directly instead of assuming a README.
**Correct command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls .sangoi/tasks && sed -n '1,200p' .sangoi/tasks/F6-refinements.md`

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && sed -n '1,260p' apps/backend/runtime/text_processing/classic.py
Cause and fix: The classic text processing engine lives in `apps/backend/runtime/text_processing/classic_engine.py`; there is no `classic.py` module in that package, so sed failed with ENOENT. Point sed at the actual engine file.
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && sed -n '1,260p' apps/backend/runtime/text_processing/classic_engine.py
**Wrong command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls docs && ls docs/plan && ls docs/legacy`
**Cause + fix:** `The repository does not ship a \`docs/\` directory; historical docs paths were consolidated under \`.sangoi/**\`. List the real locations instead of chaining removed subdirectories.`
**Correct command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls .sangoi && ls .sangoi/plans && ls .sangoi/research`

**Wrong command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls docs/legacy`
**Cause + fix:** `There is no \`docs/legacy\` directory; legacy documentation and research now live under \`.sangoi/research\` and related subtrees. Target those folders instead of the removed docs/legacy path.`
**Correct command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls .sangoi/research`

**Wrong command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls docs/notes && head -n 20 docs/notes/2025-10-22-roadmap.md`
**Cause + fix:** `There is no \`docs/notes\` tree in this repo; roadmap/timeline docs live under \`.sangoi/roadmap/\`. Discover available files with \`ls\` and open the actual filenames instead of guessing.`
**Correct command:** `cd /home/lucas/work/stable-diffusion-webui-codex && ls .sangoi/roadmap && head -n 20 .sangoi/roadmap/timeline-2025Q4.md`

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && sed -n '260,620p' apps/backend/runtime/sampling/sampling_function_inner.py
Cause and fix: The sampling inner loop lives in `apps/backend/runtime/sampling/__init__.py`; there is no separate `sampling_function_inner.py` module, so sed failed with ENOENT. Point sed at the package file that defines `sampling_function_inner`.
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && sed -n '260,620p' apps/backend/runtime/sampling/__init__.py

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && python tools/dev/trace_pipeline_graph.py --root apps.backend.use_cases.txt2img_pipeline.runner:Txt2ImgPipelineRunner.run --max-depth 6
Cause and fix: The trace script imports `apps.backend`, which in turn imports `torch` via the runtime memory module; running it against the system Python without the project venv caused `ModuleNotFoundError: No module named 'torch'`. Use the project virtualenv (and required `PYTHONPATH`) so `torch` and internal modules are available.
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && PYTHONPATH=$HOME/.netsuite:. ~/.venv/bin/python tools/dev/trace_pipeline_graph.py --root apps.backend.use_cases.txt2img_pipeline.runner:Txt2ImgPipelineRunner.run --max-depth 6
**Wrong command:** `ls docs/plan`
**Cause + fix:** The repository has no `docs/plan/` directory; list the actual docs root before drilling into paths.
**Correct command:** `ls .sangoi`

**Wrong command:** `ls docs/legacy`
**Cause + fix:** There is no `docs/legacy/` tree; inspect the existing docs layout instead of assuming a legacy folder.
**Correct command:** `ls .sangoi`

**Wrong command:** `ls .legacy`
**Cause + fix:** The repo does not ship a `.legacy` directory; use `find` to confirm what local snapshot/reference directories exist before targeting paths.
**Correct command:** `find . -maxdepth 2 -type d -name '.legacy*'`

**Wrong command:** `npm install --save-dev vitest`
**Cause + fix:** Default npm cache under `~/.npm` is owned by root, so installs fail with EACCES; point npm to a writable user cache.
**Correct command:** `npm install --save-dev vitest --cache /home/lucas/.codextools/npm-cache`

**Wrong command:** `npm install --save-dev vitest --cache $HOME/.npm-cache`
**Cause + fix:** The fallback cache path `~/.npm-cache` was also root-owned; use a fresh cache directory under `.codextools` instead.
**Correct command:** `npm install --save-dev vitest --cache /home/lucas/.codextools/npm-cache`

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && ls -ლა .sangoi/task-logs | sed -n '1,120p'
Cause and fix: The `-l` flag was typed with a non-ASCII character (likely a locale/encoding artifact), so `ls` interpreted it as an invalid option. Use plain ASCII flags (e.g. `-la`) or `ls --help` to confirm supported options.
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && ls -la .sangoi/task-logs | head -n 80

Wrong command: rg -n "2025-12-12|2026-01-01: Native `DPM++ 2M`" apps/backend/runtime/sampling/AGENTS.md
Cause and fix: Backticks in the pattern were interpreted by the shell as command substitution (`DPM++`), causing a `command not found` error. Quote/escape backticks (or avoid them) when running `rg` from a shell.
Correct command: rg -n "2025-12-12|2026-01-01: Native DPM\\+\\+ 2M" apps/backend/runtime/sampling/AGENTS.md

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && pytest -q
Cause and fix: Running pytest without the project venv/PYTHONPATH breaks imports like `apps.*` during test collection (`ModuleNotFoundError: No module named 'apps'`). Use the repo venv and set `PYTHONPATH` (and `CODEX_ROOT` for tests that need it).
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && CODEX_ROOT=$PWD PYTHONPATH=$PWD ~/.venv/bin/pytest -q

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && ls -لا diffusers | sed -n '1,120p'
Cause and fix: The `-l` flag was typed with a non-ASCII character (likely a locale/encoding artifact), so `ls` interpreted it as an invalid option. Use plain ASCII flags (`-la`) or run `ls --help` to confirm supported options.
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && ls -la diffusers | sed -n '1,120p'

Wrong command: cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "<<<<<<</|=======|>>>>>>>" apps .sangoi || true
Cause and fix: The pattern was not anchored, so tokenizer/vocab JSONs matched `========`/`>>>>>>>>` and produced huge false positives/output. Use anchored conflict-marker patterns and exclude vendored Hugging Face assets.
Correct command: cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "^<<<<<<< |^=======$|^>>>>>>> " --glob '!apps/backend/huggingface/**' --glob '!apps/interface/dist/**' .
