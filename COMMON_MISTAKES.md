# Common Mistakes (Codex WebUI repo)

This file is a curated list of “lost-time” mistakes that commonly happen in this repo. Keep it:
- **English-only**
- **portable** (no hardcoded machine paths)
- **deduped** (update an existing entry instead of appending a near-duplicate)

## Conventions used in this doc

- All commands assume **bash**.
- Always work relative to `CODEX_ROOT`:

```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
```

- Use the **workspace venv** at `$CODEX_ROOT/.venv`.
- When running Python directly, set `PYTHONPATH="$CODEX_ROOT"` (or use `./run-webui.sh`, which sets it for you).

## Template (when adding a new entry)

Wrong command:
```bash
<exact command that failed>
```

Cause and fix:
<why it failed and how to repair it>

Correct command:
```bash
<safe command that achieves the goal>
```

---

## Environment & repo paths

### Hardcoding an absolute repo path

Wrong command:
```bash
cd /absolute/path/to/stable-diffusion-webui-codex
```

Cause and fix:
Hardcoded paths break on other machines and after renames/moves. Always derive the repo root dynamically.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
```

### Looking for plans under `docs/`

Wrong command:
```bash
ls docs/
```

Cause and fix:
Plans/docs live under `.sangoi/**` in this repo.

Correct command:
```bash
ls .sangoi/plans
```

### Opening a task-log with the wrong filename slug

Wrong command:
```bash
sed -n '1,260p' .sangoi/task-logs/2026-02-11-wan22-gguf-vae-bundle-contract-img2vid.md
```

Cause and fix:
The file slug was incorrect (`bundle` vs the actual `bund`). Check exact filenames before opening task logs.

Correct command:
```bash
sed -n '1,260p' .sangoi/task-logs/2026-02-11-wan22-gguf-vae-bund-contract-img2vid.md
```

### Searching “legacy” sources in non-existent paths

Wrong command:
```bash
rg -n "some_keyword" legacy
```

Cause and fix:
Upstream snapshots are vendored under `.refs/` (and the project’s own living docs live under `.sangoi/`). Searching the wrong tree wastes time and creates “ghost references” in follow-up docs.

Correct command:
```bash
rg -n "some_keyword" .refs
rg -n "some_keyword" .sangoi
rg -n "some_keyword" apps
```

### Searching for a literal that starts with `--` (ripgrep treats it as a flag)

Wrong command:
```bash
rg -n "--core-dtype" apps/backend/infra/config/args.py
```

Cause and fix:
Arguments that look like flags (starting with `--`) are parsed as ripgrep options. Use `--` to terminate option parsing before your pattern.

Correct command:
```bash
rg -n -- "--core-dtype" apps/backend/infra/config/args.py
```

### Using backticks in a double-quoted ripgrep pattern (bash command substitution)

Wrong command:
```bash
rg -n "use_cases/restore\\.py|`restore\\.py`" .sangoi/plans/2026-02-01-supir-webui-integration-v1.md
rg -n "\\(Optional\\) Tile controls: expose `min_tile`" .sangoi/plans/2026-02-01-upscalers-and-hires-fix-global-modules-v1-plan.md
rg -n "Implementing masking for the Flux Kontext|Phase 2 will implement Kontext masking|fail loud|single `/api/img2img`|no dedicated `/api/inpaint`" .sangoi/plans/2026-01-29-masked-img2img-inpaint-v1.md
rg -n "Using outdated `EngineRegistry`|Assuming file paths that do not exist" COMMON_MISTAKES.md
rg -n "Complements follow the engine asset contract|Task layer owns async lifecycle|vid2vid current state|Anima \(`engine_id: anima`\)|chunks` is accepted" SUBSYSTEM-MAP.md
rg -n "Complements follow the engine asset contract|External-assets-first families|Task layer owns async lifecycle \+ SSE wiring|vid2vid current state|Anima \(`engine_id: anima`\)" SUBSYSTEM-MAP.md
```

Cause and fix:
In bash, backticks (`` `...` ``) trigger command substitution even inside double quotes, so the shell tries to execute `restore.py` as a command.
Use single quotes around the pattern, escape backticks, or just search for the literal without backticks.

Correct command:
```bash
rg -n 'use_cases/restore\\.py|`restore\\.py`' .sangoi/plans/2026-02-01-supir-webui-integration-v1.md
rg -n "restore\\.py" .sangoi/plans/2026-02-01-supir-webui-integration-v1.md
rg -n '\\(Optional\\) Tile controls: expose `min_tile`' .sangoi/plans/2026-02-01-upscalers-and-hires-fix-global-modules-v1-plan.md
rg -n 'Implementing masking for the Flux Kontext|Phase 2 will implement Kontext masking|fail loud|single `/api/img2img`|no dedicated `/api/inpaint`' .sangoi/plans/2026-01-29-masked-img2img-inpaint-v1.md
rg -n 'Complements follow the engine asset contract|Task layer owns async lifecycle|vid2vid current state|Anima \(`engine_id: anima`\)|chunks\` is accepted' SUBSYSTEM-MAP.md
rg -n 'Complements follow the engine asset contract|External-assets-first families|Task layer owns async lifecycle \+ SSE wiring|vid2vid current state|Anima \(`engine_id: anima`\)' SUBSYSTEM-MAP.md
```

### Putting `rg` flags after `--` (ripgrep stops parsing options)

Wrong command:
```bash
rg -n -- "\\bhighres\\b|Highres|highres_|highres\\." -S apps/backend apps/interface/src
```

Cause and fix:
In ripgrep, `--` terminates option parsing. Anything after it is treated as a positional argument (pattern/path). If you put `-S` after `--`, ripgrep treats it as a path and fails.

Correct command:
```bash
rg -n -S -- "\\bhighres\\b|Highres|highres_|highres\\." apps/backend apps/interface/src
rg -n -S -- '\\bhighres\\b|Highres|highres_|highres\\.' apps/backend apps/interface/src
```

### Putting `rg` options (like `--glob`) after the pattern/path

Wrong command:
```bash
rg -n -- "`.tmp/`" . --glob '!.refs/**' --glob '!.sangoi/**'
```

Cause and fix:
- In bash, backticks inside double quotes trigger command substitution, so the shell tries to execute `.tmp/`.
- In ripgrep, options must come **before** the pattern/paths. Once you pass the pattern and at least one path, later tokens like `--glob` are treated as paths and can explode the search.

Correct command:
```bash
rg -n --glob '!.refs/**' --glob '!.sangoi/**' -- '`.tmp/`' .
```

---

## Python & virtualenv

### Using the wrong Python interpreter (system Python / random venv)

Wrong command:
```bash
python -c "import apps; print('ok')"
```

Cause and fix:
This repo is expected to run in the **workspace** venv created by `./install-webui.sh`.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -c "import apps; print('ok')"
```

### Installing deps with `pip install` (or `pip install --user`)

Wrong command:
```bash
pip install --user uvicorn fastapi
```

Cause and fix:
This creates untracked, non-reproducible dependency drift (and can “work on my machine” while breaking everyone else). Use the repo’s installer / locked `uv.lock`.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"

./install-webui.sh
# Or, if `uv` already exists and you only need a locked resync:
"$CODEX_ROOT/.uv/bin/uv" sync --locked
```

### Forgetting `PYTHONPATH` when importing `apps.*`

Wrong command:
```bash
"$CODEX_ROOT/.venv/bin/python" -c "import apps.backend; print('ok')"
```

Cause and fix:
Running from the wrong working directory (or via tooling) can make `apps` imports fail. Set `PYTHONPATH="$CODEX_ROOT"` explicitly when running ad-hoc Python commands.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -c "import apps.backend; print('ok')"
```

### Running `.sangoi` pytest suite without exporting `CODEX_ROOT`

Wrong command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"; PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_anima_*keymap*.py
```

Cause and fix:
`.sangoi/dev/tests/conftest.py` requires `CODEX_ROOT` in the process environment. A shell variable without `export` is not visible to pytest's Python process.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
export CODEX_ROOT
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_anima_*keymap*.py
```

### Importing heavy deps when you only need a syntax check

Wrong command:
```bash
"$CODEX_ROOT/.venv/bin/python" -c "import torch; import diffusers; print(torch.__version__, diffusers.__version__)"
```

Cause and fix:
Heavy imports are slow and can fail for environment reasons unrelated to your change. For “is this file valid Python?” checks, parse the source without importing the package.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
"$CODEX_ROOT/.venv/bin/python" - <<'PY'
import ast
from pathlib import Path

path = Path("apps/backend/interfaces/api/run_api.py")
ast.parse(path.read_text(encoding="utf-8"))
print("OK:", path)
PY
```

### Using `python -m ast` with multiple files

Wrong command:
```bash
"$CODEX_ROOT/.venv/bin/python" -m ast file1.py file2.py
```

Cause and fix:
`python -m ast` accepts **one** input file. For multiple files, loop (or use a small `ast.parse` snippet).

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
for f in file1.py file2.py; do
  "$CODEX_ROOT/.venv/bin/python" -m ast "$f" >/dev/null
done
```

---

## PyTest

### Using `|` in `pytest -k` expressions

Wrong command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend -k "(engine_model_scopes|wan22_sampling)"
```

Cause and fix:
`pytest -k` uses a Python-like expression syntax. `|` is not a valid operator there; use `or` (or run explicit test files).

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend -k "engine_model_scopes or wan22_sampling"
```

---

## Tests

### Running `pytest` without the workspace env

Wrong command:
```bash
pytest -q .sangoi/dev/tests
```

Cause and fix:
This often runs whatever `pytest` is on your PATH (wrong interpreter, wrong deps). Use the workspace venv Python explicitly and set `CODEX_ROOT`/`PYTHONPATH`.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests
```

### Running `pytest` at repo root (collects `.tmp/**` tests/artifacts)

Wrong command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Cause and fix:
Running pytest at the repo root can collect “third party” or local artifact tests under `.tmp/**` (and other non-canonical locations) which may require extra dependencies or a different runtime environment. Scope test runs explicitly (or ignore `.tmp`).

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -m pytest -q tests
```

---

## Git (policy + safety)

### Staging “everything” in a shared repo

Wrong command:
```bash
git add .
```

Cause and fix:
This stages *everything* (including unrelated changes) and makes it easy to accidentally commit other people’s work or local artifacts. Stage explicitly.

Correct command:
```bash
git add -- path/to/file1.py path/to/file2.md
git diff --cached --name-only
```

### Using `find ... | xargs git add`

Wrong command:
```bash
find . -type f -print0 | xargs -0 -- git add
```

Cause and fix:
This is a footgun: it will happily stage caches, vendor blobs, and other garbage unless your pruning is perfect. Stage explicit paths instead.

Correct command:
```bash
git status --porcelain
git add -- path/to/file1 path/to/file2
git diff --cached --name-only
```

### Staging paths without `--` (and accidentally splitting paths)

Wrong command:
```bash
git add apps backend/runtime utils.py apps backend/interfaces/api run_api.py
```

Cause and fix:
Spaces split arguments; missing slashes create paths that do not exist. Use real paths and always include `--` before pathspecs.

Correct command:
```bash
git add -- \
  apps/backend/runtime/utils.py \
  apps/backend/interfaces/api/run_api.py
```

### Searching for merge conflict markers with an unanchored pattern

Wrong command:
```bash
rg -n "<<<<<<<|=======|>>>>>>>" .
```

Cause and fix:
This matches random `=======` / `>>>>>>>` sequences inside non-conflict files (tokenizers, notices, separators), flooding your output. Merge-conflict markers are line-start tokens; anchor the pattern and require the marker to be exact.

Correct command:
```bash
rg -n "^(<<<<<<<|=======|>>>>>>>)( |$)" .
```

---

## ripgrep (`rg`) and shell quoting

### Unquoted backticks / command substitution inside search patterns

Wrong command:
```bash
rg -n "`logger`" apps
```

Cause and fix:
Backticks execute a command in the shell. Quote the pattern with **single quotes** to make it literal.

Correct command:
```bash
rg -n '`logger`' apps
```

### Backticks inside double quotes still execute

Wrong command:
```bash
rg -n "Do not touch `git clean`" -n AGENTS.md
```

Cause and fix:
Backticks execute command substitution even inside double quotes. This can run real commands in your shell (here it attempted to execute `git clean`). Use single quotes for literal patterns (or `--fixed-strings` when appropriate).

Correct command:
```bash
rg -n --fixed-strings 'Do not touch `git clean`' AGENTS.md
```

### Using regex when you meant a literal search

Wrong command:
```bash
rg -n "Symbols (top-level; keep in sync; no ghosts):" .sangoi
```

Cause and fix:
Parentheses and other characters have meaning in regex. If you want a literal match, use fixed-strings mode.

Correct command:
```bash
rg -n --fixed-strings "Symbols (top-level; keep in sync; no ghosts):" .sangoi
```

### Searching for patterns that start with `--` (flag parsing)

Wrong command:
```bash
rg -n "--gguf-exec" -S apps .sangoi
```

Cause and fix:
`rg` interprets tokens starting with `--` as flags. Use `--` to stop option parsing (and optionally `--fixed-strings` when you want a literal match).

Correct command:
```bash
rg -n -- "--gguf-exec" -S apps .sangoi
```

---

## Codex CLI tooling (patch hygiene)

### `apply_patch`: forgetting `+` lines under `*** Add File:`

Wrong command:
```bash
apply_patch << 'PATCH'
*** Begin Patch
*** Add File: hello.txt
Hello
*** End Patch
PATCH
```

Cause and fix:
In `apply_patch`, every line of a newly added file must be prefixed with `+`.

Correct command:
```bash
apply_patch << 'PATCH'
*** Begin Patch
*** Add File: hello.txt
+Hello
*** End Patch
PATCH
```

### Pasting tool-call payloads into bash

Wrong command:
```bash
multi_tool_use.parallel <<'JSON'
{ ... }
JSON
```

Cause and fix:
That’s not a shell command; it’s an agent tool call. In a normal shell session, run the underlying commands directly.

Correct command:
```bash
rg -n "some_pattern" .
```

---

## Frontend / Node

### Running `npm install` in the wrong place (or without nodeenv)

Wrong command:
```bash
npm install
```

Cause and fix:
The WebUI frontend lives under `apps/interface`. The repo also supports a pinned, repo-local Node.js via nodeenv (`.nodeenv`) that is provisioned by `./install-webui.sh`.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
./install-webui.sh

# Or, if nodeenv is already installed and you only need frontend deps:
(cd apps/interface && npm install)
```

---

## Shell / CLI

### Grepping text that contains backticks (bash command substitution)

Wrong command:
```bash
rg -n "Validate `dtype` strings" .sangoi/task-logs/2026-01-30-zimage-cleanup.md
```

Cause and fix:
In bash, backticks inside double quotes still perform command substitution. The shell tries to execute `dtype` and replaces it with the output (or errors). Use single quotes around the pattern, or escape the backticks.

Correct command:
```bash
rg -n 'Validate `dtype` strings' .sangoi/task-logs/2026-01-30-zimage-cleanup.md
```

### Inline env var assignments don’t affect `$VAR` expansion (bash)

Wrong command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)" PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q
```

Cause and fix:
In bash, the `VAR=value cmd ...` form sets environment variables for the command, but **does not** make `$VAR` available for argument expansion on the same line. `$CODEX_ROOT` expands before the assignment takes effect, resulting in paths like `/.venv/bin/python`.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q
```

### Backticks inside `rg` patterns trigger command substitution (bash)

Wrong command:
```bash
rg -n "^### `/api/upscalers/download`" -n .sangoi/reference/models/upscalers-hf-manifest-v1.md
```

Cause and fix:
In bash, backticks (`` `...` ``) run **command substitution**. The shell tries to execute `/api/upscalers/download` as a command before `rg` runs.
Wrap the pattern in single quotes (or escape the backticks) so the backticks are treated literally.

Correct command:
```bash
rg -n '^### `/api/upscalers/download`' .sangoi/reference/models/upscalers-hf-manifest-v1.md
```

### `ls` with accidental non-ASCII option flags (IME / keyboard layout)

Wrong command:
```bash
ls -<CTRL>Q>a .sangoi/plans | sed -n "1,120p"
# or (common when copying/pasting): a non-ASCII `-la` variant
ls -<non-ascii> .sangoi/task-logs | sed -n "1,120p"
```

Cause and fix:
Accidental non-ASCII/control flag characters (e.g., from an IME/keyboard layout) can turn `ls` options into invalid flags. Use plain ASCII `-la`.

Correct command:
```bash
ls -la .sangoi/plans | sed -n "1,120p"
# same pattern
ls -la .sangoi/task-logs | sed -n "1,120p"
```

### `rg` patterns starting with `-` are parsed as flags

Wrong command:
```bash
rg -n "-m pytest -q tests" .sangoi/plans/2026-02-02-p0-task-plumbing-and-upscaler-safety-fixes.md
```

Cause and fix:
`rg` treats the first argument starting with `-` as an option flag. Use `--` to terminate option parsing before a pattern that begins with `-`.

Correct command:
```bash
rg -n -- "-m pytest -q tests" .sangoi/plans/2026-02-02-p0-task-plumbing-and-upscaler-safety-fixes.md
```

### `rg` options placed after `--` are treated as file paths

Wrong command:
```bash
rg -n -- "GGUF_DEQUANT_CACHE" -S apps/backend/infra/config/args.py
```

Cause and fix:
In ripgrep, `--` ends option parsing. Anything after `--` is no longer interpreted as an option flag, so `-S` is treated as a file path and ripgrep errors with “No such file or directory”.
Move flags *before* `--`.

Correct command:
```bash
rg -n -S -- "GGUF_DEQUANT_CACHE" apps/backend/infra/config/args.py
```

### Backticks in double quotes trigger shell command substitution

Wrong command:
```bash
rg -n "Docs: audited `.sangoi/plans`" .sangoi/CHANGELOG.md
```

Cause and fix:
In `bash`, backticks inside double quotes still perform command substitution, so the shell tries to execute `.sangoi/plans` before running `rg`.
Use single quotes for literal backticks, or escape them.

Correct command:
```bash
rg -n 'Docs: audited `.sangoi/plans`' .sangoi/CHANGELOG.md
```

### `VAR=value cmd ...` does not make `$VAR` available for argument expansion on the same line

Wrong command:
```bash
CODEX_ROOT="$(pwd)" PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests
```

Cause and fix:
In bash, `VAR=value cmd ...` sets environment variables for `cmd`, but **does not** update `$VAR` for argument expansion on that same line.
This yields an empty `$CODEX_ROOT` and tries to execute `/.venv/bin/python`.
Assign/export first, then run the command.

Correct command:
```bash
CODEX_ROOT="$(pwd)"
export CODEX_ROOT
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests
```

### `SCHEDULER_OPTIONS` is a list of dicts (use `e["name"]`, not `set(SCHEDULER_OPTIONS)`)

Wrong command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)" && PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -c "from apps.backend.runtime.sampling import SCHEDULER_OPTIONS; names=set(SCHEDULER_OPTIONS)"
```

Cause and fix:
`apps.backend.runtime.sampling.SCHEDULER_OPTIONS` is a list of dict entries (e.g. `{"name": "simple", "supported": True}`), so converting it to a `set(...)` fails with `TypeError: unhashable type: 'dict'`.
Extract the scheduler names first.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)" && PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -c "from apps.backend.runtime.sampling import SCHEDULER_OPTIONS; names={e['name'] for e in SCHEDULER_OPTIONS}; print('simple' in names)"
```

### Running shell scripts with `python3`

Wrong command:
```bash
python3 .sangoi/.tools/link-check.sh .sangoi
```

Cause and fix:
`.sangoi/.tools/link-check.sh` is a Bash script, not a Python module. Running it with `python3` raises a shell-syntax `SyntaxError`.

Correct command:
```bash
bash .sangoi/.tools/link-check.sh .sangoi
```

### Calling backend path helpers without `CODEX_ROOT`

Wrong command:
```bash
PYTHONPATH="$(git rev-parse --show-toplevel)" "$(git rev-parse --show-toplevel)/.venv/bin/python" - <<'PY'
from apps.backend.infra.config.paths import get_paths_for
print('anima_tenc', get_paths_for('anima_tenc'))
print('anima_vae', get_paths_for('anima_vae'))
PY
```

### Searching markdown headings with backticks in double quotes

Wrong command:
```bash
rg -n "Calling backend path helpers without `CODEX_ROOT`|anima_tenc|anima_vae" COMMON_MISTAKES.md
```

Cause and fix:
In `bash`, backticks inside double-quoted strings still run command substitution. The shell tries to execute `CODEX_ROOT` before `rg` starts.
Wrap the pattern in single quotes when matching literal backticks.

Correct command:
```bash
rg -n 'Calling backend path helpers without `CODEX_ROOT`|anima_tenc|anima_vae' COMMON_MISTAKES.md
```

Cause and fix:
`apps.backend.infra.config.paths.get_paths_for()` resolves `apps/paths.json` via `get_repo_root()`, which requires `CODEX_ROOT` to be set. `PYTHONPATH` alone is not enough and raises `OSError: CODEX_ROOT not set`.
Set `CODEX_ROOT` first, then run with the workspace venv.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY'
from apps.backend.infra.config.paths import get_paths_for
print('anima_tenc', get_paths_for('anima_tenc'))
print('anima_vae', get_paths_for('anima_vae'))
PY
```

### Running Vitest with repo-root paths while using `npm --prefix`

Wrong command:
```bash
npm --prefix apps/interface test -- --run apps/interface/src/utils/engine_taxonomy.test.ts apps/interface/src/stores/model_tabs.test.ts
```

Cause and fix:
With `--prefix apps/interface`, Vitest runs from `apps/interface`, so filter paths must be relative to that directory (for example, `src/...`), not repo-root `apps/interface/src/...`.

Correct command:
```bash
npm --prefix apps/interface test -- --run src/utils/engine_taxonomy.test.ts src/stores/model_tabs.test.ts
```

### Grepping literal backticks with double quotes (command substitution trap)

Wrong command:
```bash
rg -n "Running Vitest with repo-root paths while using `npm --prefix`" COMMON_MISTAKES.md
```

Cause and fix:
Backticks inside double quotes still trigger command substitution in `bash`. Use single quotes for literal backticks in regex patterns.

Correct command:
```bash
rg -n 'Running Vitest with repo-root paths while using `npm --prefix`' COMMON_MISTAKES.md
```

### Running `.sangoi` pytest without exporting `CODEX_ROOT`/`PYTHONPATH`

Wrong command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
echo "CODEX_ROOT=$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_anima_offline_tokenizers.py | tail -n 5
```

Cause and fix:
`.sangoi/dev/tests/conftest.py` requires `CODEX_ROOT` in the environment and imports repo modules via `PYTHONPATH`. Setting a shell variable alone is not enough for pytest subprocess/import context.
Export both variables before running the suite.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
export CODEX_ROOT
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_anima_offline_tokenizers.py
```

### Grepping pattern with backticks in double quotes (again)

Wrong command:
```bash
rg -n "Running `.sangoi` pytest without exporting|Wrong command:|Correct command:" COMMON_MISTAKES.md
```

Cause and fix:
Backticks inside double-quoted regex trigger shell command substitution (`.sangoi` tried to execute as a command). Use single quotes for literal backticks.

Correct command:
```bash
rg -n 'Running `.sangoi` pytest without exporting|Wrong command:|Correct command:' COMMON_MISTAKES.md
```

### Assuming Anima has `sampler.py` and `conditioning.py` under `runtime/families/anima`

Wrong command:
```bash
rg -n "def |for |while |torch\\.cat|torch\\.stack|permute\\(|view\\(|reshape\\(|repeat\\(|interpolate\\(|einsum|matmul|bmm|chunk\\(|split\\(" apps/backend/runtime/families/anima/wan_vae.py apps/backend/runtime/families/anima/loader.py apps/backend/runtime/families/anima/sampler.py apps/backend/runtime/families/anima/conditioning.py
```

Cause and fix:
`sampler.py` and `conditioning.py` do not exist in `apps/backend/runtime/families/anima`. Shared sampling logic lives in `apps/backend/runtime/sampling/*`.
List the family directory first, then target real files.

Correct command:
```bash
ls -1 apps/backend/runtime/families/anima
rg -n "def |for |while |torch\\.cat|torch\\.stack|permute\\(|view\\(|reshape\\(|repeat\\(|interpolate\\(|einsum|matmul|bmm|chunk\\(|split\\(" apps/backend/runtime/families/anima/wan_vae.py apps/backend/runtime/families/anima/loader.py apps/backend/runtime/sampling/inner_loop.py apps/backend/runtime/sampling/condition.py
```

### Assuming `runtime/sampling/samplers.py` exists

Wrong command:
```bash
rg -n "def |for .* in |while |torch\\.cat|torch\\.stack|chunk\\(|split\\(" apps/backend/runtime/sampling/driver.py apps/backend/runtime/sampling/sigma_schedules.py apps/backend/runtime/sampling/samplers.py
```

Cause and fix:
This repo does not have `apps/backend/runtime/sampling/samplers.py`; sampler selection is registry/context-driven across other modules.
Use `rg --files apps/backend/runtime/sampling` (or `ls`) before targeting filenames.

Correct command:
```bash
rg --files apps/backend/runtime/sampling
rg -n "def |for .* in |while |torch\\.cat|torch\\.stack|chunk\\(|split\\(" apps/backend/runtime/sampling/driver.py apps/backend/runtime/sampling/sigma_schedules.py apps/backend/runtime/sampling/inner_loop.py apps/backend/runtime/sampling/condition.py
```

### Assuming image API helpers live in `interfaces/api/image_io.py` or `interfaces/api/media.py`

Wrong command:
```bash
rg -n "def encode_images|def save_generated_images|for .* in images|base64|PNG" apps/backend/interfaces/api/tasks/generation_tasks.py apps/backend/interfaces/api/image_io.py apps/backend/interfaces/api/media.py
```

Cause and fix:
In this repo, image task helpers are consolidated in `apps/backend/interfaces/api/tasks/generation_tasks.py`; `image_io.py` and `media.py` are not present at those paths.
Discover existing files with `rg --files` before multi-path grep.

Correct command:
```bash
rg --files apps/backend/interfaces/api | rg 'tasks|generation'
rg -n "def encode_images|def save_generated_images|for .* in images|base64|PNG" apps/backend/interfaces/api/tasks/generation_tasks.py
```

### Grepping headings with backticks using double quotes (yet again)

Wrong command:
```bash
rg -n "Assuming Anima has `sampler.py`|Assuming `runtime/sampling/samplers.py` exists|Assuming image API helpers live" COMMON_MISTAKES.md
```

Cause and fix:
Backticks inside double quotes trigger shell command substitution in `bash` (`sampler.py`/`runtime/sampling/samplers.py` were treated as commands/paths).
Use single-quoted regex for literal backticks.

Correct command:
```bash
rg -n 'Assuming Anima has `sampler.py`|Assuming `runtime/sampling/samplers.py` exists|Assuming image API helpers live' COMMON_MISTAKES.md
```

### Grepping phrases with backticks inside double quotes in `bash`

Wrong command:
```bash
rg -n "Flux1 GGUF T5 loader `model` unbound fix|State: Completed|Behavior matrix|4 passed|OK_HEADER_CHANGED|No broken markdown links" .sangoi/CHANGELOG.md .sangoi/plans/2026-02-08-flux1-gguf-unbound-model-fix.md .sangoi/task-logs/2026-02-08-flux1-gguf-unbound-model-fix.md .sangoi/dev/tests/backend/AGENTS.md
```

Cause and fix:
Backticks inside a double-quoted pattern trigger shell command substitution (`model` was executed as a command). Use single quotes around regex containing literal backticks.

Correct command:
```bash
rg -n 'Flux1 GGUF T5 loader `model` unbound fix|State: Completed|Behavior matrix|4 passed|OK_HEADER_CHANGED|No broken markdown links' .sangoi/CHANGELOG.md .sangoi/plans/2026-02-08-flux1-gguf-unbound-model-fix.md .sangoi/task-logs/2026-02-08-flux1-gguf-unbound-model-fix.md .sangoi/dev/tests/backend/AGENTS.md
```

### Grepping with backticks inside double quotes triggered shell substitution

Wrong command:
```bash
rg -n "test_txt2img_accepts_extras_hires_refiner_key|Repair `extras.hires.refiner` contract mismatch" .sangoi/dev/tests/backend/test_api_hires_naming_cutover.py .sangoi/plans/2026-02-08-refiner-hires-contract-repair.md
```

Cause and fix:
Backticks inside a double-quoted regex triggered `bash` command substitution (`extras.hires.refiner` was executed as a command). Use single quotes around patterns that contain literal backticks.

Correct command:
```bash
rg -n 'test_txt2img_accepts_extras_hires_refiner_key|Repair `extras\.hires\.refiner` contract mismatch' .sangoi/dev/tests/backend/test_api_hires_naming_cutover.py .sangoi/plans/2026-02-08-refiner-hires-contract-repair.md
```

### Conflict-marker grep pattern too broad matched tokenizer separators

Wrong command:
```bash
rg -n "^(<<<<<<<|=======|>>>>>>>)" .
```

Cause and fix:
The pattern matched plain separator lines like `========` in tokenizer assets, producing false positives. Use the strict conflict-marker pattern that requires exact markers and spacing.

Correct command:
```bash
rg -n "^(<<<<<<<|=======|>>>>>>>)( |$)" .
```

### Using outdated `EngineRegistry` import path / positional call for keyword-only API

Wrong command:
```bash
sed -n '1,220p' apps/backend/core/engine_registry.py
CODEX_ROOT=$PWD PYTHONPATH=$PWD .venv/bin/python - <<'PY'
from apps.backend.core.engine_registry import EngineRegistry
from apps.backend.engines import register_default_engines
from apps.backend.core.engine_interface import TaskType

registry = EngineRegistry()
register_default_engines(registry)
for engine_id in sorted(registry.list_ids()):
    engine = registry.get(engine_id)
    cap = engine.capabilities()
    if TaskType.IMG2IMG in cap.tasks:
        print(engine_id)
PY
```

Cause and fix:
The registry module is `apps.backend.core.registry` (not `engine_registry.py`), and `register_default_engines` is keyword-only (`registry=...`).
Use the correct module path and pass the argument by keyword.

Correct command:
```bash
CODEX_ROOT=$PWD PYTHONPATH=$PWD .venv/bin/python - <<'PY'
from apps.backend.core.registry import EngineRegistry
from apps.backend.engines import register_default_engines
from apps.backend.core.engine_interface import TaskType

registry = EngineRegistry()
register_default_engines(registry=registry)
for engine_id in sorted(registry.list()):
    engine = registry.create(engine_id)
    cap = engine.capabilities()
    if TaskType.IMG2IMG in cap.tasks:
        print(engine_id)
PY
```

### Assuming file paths that do not exist (sed/nl on guessed locations)

Wrong command:
```bash
nl -ba apps/interface/src/views/QuickSettingsBar.vue | sed -n '1,260p'
sed -n '1,220p' apps/backend/core/engine_registry.py
sed -n '1,240p' apps/backend/runtime/engine_surface.py
sed -n '1,240p' apps/backend/runtime/semantic_engine.py
nl -ba apps/backend/runtime/pipeline_stages/hires.py | sed -n '1,320p'
nl -ba .sangoi/AGENTS.md | sed -n '1,220p'
nl -ba apps/backend/runtime/state_dict/api.py | sed -n '1,240p'
nl -ba apps/backend/runtime/state_dict/index.py | sed -n '1,260p'
nl -ba apps/backend/runtime/state_dict/renames.py | sed -n '1,240p'
rg -n "img2img|inpaint|mask" apps/interface/README.md apps/backend/README.md README.md .sangoi/CHANGELOG.md -g '*.md'
```

Cause and fix:
Guessed paths were stale/wrong for this repo layout.
Confirm the exact path first with `rg --files` or `find`, then open the file.

Correct command:
```bash
rg --files apps/interface/src | rg 'QuickSettingsBar\\.vue$'
rg --files apps/backend/core | rg 'registry\\.py$'
rg --files apps/backend/runtime | rg 'model_registry/capabilities\\.py$'
rg --files apps/backend/runtime/pipeline_stages | rg 'hires_fix\\.py$'
rg --files .sangoi | rg '/AGENTS\\.md$'
find apps/backend/runtime/state_dict -maxdepth 1 -type f -name '*.py' | sort
rg -n "img2img|inpaint|mask" apps/interface/README.md README.md .sangoi/CHANGELOG.md -g '*.md'
```

### Selecting latest task log with wildcard that also matches `AGENTS.md`

Wrong command:
```bash
ls -1t .sangoi/task-logs/*.md | head -n 1
```

Cause and fix:
The wildcard includes `.sangoi/task-logs/AGENTS.md`, so the result may not be a real task log.
Exclude `AGENTS.md` explicitly and pick the newest markdown file by timestamp.

Correct command:
```bash
ls -1t .sangoi/task-logs/*.md | rg -v '/AGENTS\.md$' | head -n 1
```

### Using double quotes around `rg` patterns that include backticks

Wrong command:
```bash
rg -n "mode\` omitted|Explicit `null`|mode omitted|fail-loud" .sangoi/reference/api/tasks-and-streaming.md
```

Cause and fix:
Backticks inside double quotes trigger shell command substitution (`null` was executed).
Use single quotes when the pattern contains backticks.

Correct command:
```bash
rg -n 'mode` omitted|Explicit `null`|mode omitted|fail-loud' .sangoi/reference/api/tasks-and-streaming.md
```

### Using unescaped `{}` in `rg` regex for literal route paths

Wrong command:
```bash
rg -n "@router.post\(\"/api/tasks/{task_id}/cancel\"\)" apps/backend/interfaces/api/routers/tasks.py
```

Cause and fix:
In regex mode, `{}` are quantifier tokens. Literal braces in paths must be escaped, or use fixed-string search.

Correct command:
```bash
rg -n -F '@router.post("/api/tasks/{task_id}/cancel")' apps/backend/interfaces/api/routers/tasks.py
```

### Using `\n` in `rg` pattern without multiline mode

Wrong command:
```bash
rg -n "def codex_root\(|@pytest\.fixture\s*\ndef .*codex_root|codex_root: Path" .sangoi/dev/tests -g '*.py'
```

Cause and fix:
`rg` does not accept literal newlines in a single-line regex unless multiline mode (`-U`) is enabled.
For this use case, run a simpler single-line search.

Correct command:
```bash
rg -n "codex_root" .sangoi/dev/tests -g '*.py'
```

### Using backticks inside a double-quoted `rg` alternation

Wrong command:
```bash
rg -n "hires sentinel \\+ consecutive-generation inference tensor remediation|hires-scheduler-lora-inference-fix|LoRA apply now clears stale patch state|sampling_execute now gates LoRA apply/reset|Added `test_lora_loader_tensor_mode.py`|Added task log for hires sentinel" .sangoi/CHANGELOG.md .sangoi/plans/2026-02-09-hires-scheduler-lora-inference-fix.md apps/backend/patchers/AGENTS.md apps/backend/runtime/pipeline_stages/AGENTS.md .sangoi/dev/tests/backend/AGENTS.md .sangoi/plans/AGENTS.md .sangoi/task-logs/AGENTS.md .sangoi/task-logs/2026-02-09-hires-scheduler-lora-inference-fix.md | sed -n '1,260p'
```

Cause and fix:
Backticks inside double quotes trigger shell command substitution, so parts of the regex were executed by bash and produced noisy/invalid output.
Use single quotes around patterns that include backticks.

Correct command:
```bash
rg -n 'hires sentinel \+ consecutive-generation inference tensor remediation|hires-scheduler-lora-inference-fix|LoRA apply now clears stale patch state|sampling_execute now gates LoRA apply/reset|Added `test_lora_loader_tensor_mode.py`|Added task log for hires sentinel' .sangoi/CHANGELOG.md .sangoi/plans/2026-02-09-hires-scheduler-lora-inference-fix.md apps/backend/patchers/AGENTS.md apps/backend/runtime/pipeline_stages/AGENTS.md .sangoi/dev/tests/backend/AGENTS.md .sangoi/plans/AGENTS.md .sangoi/task-logs/AGENTS.md .sangoi/task-logs/2026-02-09-hires-scheduler-lora-inference-fix.md
```

### Opening a guessed stage filename that does not exist

Wrong command:
```bash
sed -n '1,360p' apps/backend/runtime/pipeline_stages/sampling.py
```

Cause and fix:
Guessed filename was stale (`sampling.py` was renamed/split). Confirm exact stage file names first, then open the real module.

Correct command:
```bash
ls -1 apps/backend/runtime/pipeline_stages
sed -n '1,360p' apps/backend/runtime/pipeline_stages/sampling_execute.py
```

### Using backticks in a double-quoted `rg` pattern (Batch 4B docs sweep)

Wrong command:
```bash
rg -n "2026-02-10: `config/args.py` strict runtime path|2026-02-10: `diffusers_loader.py` no longer swallows|2026-02-10: Added Batch 4B regressions|Updated the open-items scrutiny execution plan|Added Batch 4B scrutiny log" apps/backend/infra/AGENTS.md apps/backend/engines/wan22/AGENTS.md .sangoi/dev/tests/backend/AGENTS.md .sangoi/plans/AGENTS.md .sangoi/task-logs/AGENTS.md
```

Cause and fix:
Backticks inside double quotes were interpreted by bash as command substitution, producing `/bin/bash: ... command not found`.
Use a pattern without backticks or wrap the whole pattern in single quotes.

Correct command:
```bash
rg -n 'strict runtime path now fails loud on unknown CLI arguments|no longer swallows attention/accelerator hook|Added Batch 4B regressions|Updated the open-items scrutiny execution plan|Added Batch 4B scrutiny log' apps/backend/infra/AGENTS.md apps/backend/engines/wan22/AGENTS.md .sangoi/dev/tests/backend/AGENTS.md .sangoi/plans/AGENTS.md .sangoi/task-logs/AGENTS.md
```

### Reading a non-existent `.sangoi/AGENTS.md` path directly

Wrong command:
```bash
sed -n '1,260p' .sangoi/AGENTS.md
```

Cause and fix:
Assumed a top-level `.sangoi/AGENTS.md` exists. In this repo, `.sangoi` folder guidance is split across scoped AGENTS files (e.g., `.sangoi/plans/AGENTS.md`, `.sangoi/task-logs/AGENTS.md`).

Correct command:
```bash
find .sangoi -name AGENTS.md -maxdepth 3 -print
sed -n '1,260p' .sangoi/task-logs/AGENTS.md
```

### Opening guessed plan filenames copied from task-log slugs

Wrong command:
```bash
for f in .sangoi/plans/2026-02-10-backend-scrutiny-batch4c-typing-pass.md .sangoi/plans/2026-02-10-backend-scrutiny-batch4b-args-wan-fail-loud.md .sangoi/plans/2026-02-10-backend-scrutiny-batch4a-tools-state-typing.md; do echo "===== $f ====="; sed -n '1,220p' "$f"; echo; done
```

Cause and fix:
Mixed task-log filenames with plan filenames and opened non-existent paths. Confirm actual plan slugs from `.sangoi/plans/` before reading.

Correct command:
```bash
ls -1 .sangoi/plans | rg 'backend-scrutiny|batch4'
sed -n '1,260p' .sangoi/plans/2026-02-10-backend-scrutiny-open-items-execution-plan.md
```

### Using backticks in a double-quoted `rg` pattern (COMMON_MISTAKES lookup)

Wrong command:
```bash
rg -n "Reading a non-existent `.sangoi/AGENTS.md` path directly|Opening guessed plan filenames copied from task-log slugs" COMMON_MISTAKES.md
```

Cause and fix:
Backticks inside double quotes triggered bash command substitution (`.sangoi/AGENTS.md` attempted as a command). Use single quotes for regex patterns that contain backticks.

Correct command:
```bash
rg -n 'Reading a non-existent `\.sangoi/AGENTS.md` path directly|Opening guessed plan filenames copied from task-log slugs' COMMON_MISTAKES.md
```

### Forgetting to export `CODEX_ROOT` before importing backend API modules

Wrong command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"; PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY'
import importlib
from fastapi.testclient import TestClient
run_api=importlib.import_module('apps.backend.interfaces.api.run_api')
app=run_api.create_api_app(argv=['--core-device=cpu','--te-device=cpu','--vae-device=cpu'], env={})
with TestClient(app) as client:
    data=client.get('/api/engines/capabilities').json()
print('asset_contracts keys:', sorted(data.get('asset_contracts',{}).keys()))
print('engine map subset:', {k:data['engine_id_to_semantic_engine'][k] for k in ['flux1_chroma','flux1_fill','wan22_5b','wan22_14b','svd','hunyuan_video','sd35'] if k in data['engine_id_to_semantic_engine']})
print('engine keys:', sorted(data.get('engines',{}).keys()))
PY
```

Cause and fix:
Set `CODEX_ROOT` as a shell variable but did not export it, so backend imports that require environment access failed (`OSError: CODEX_ROOT not set`). Export `CODEX_ROOT` (and `PYTHONPATH` as needed) before invoking Python.

Correct command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
export PYTHONPATH="$CODEX_ROOT"
"$CODEX_ROOT/.venv/bin/python" - <<'PY'
import importlib
from fastapi.testclient import TestClient
run_api=importlib.import_module('apps.backend.interfaces.api.run_api')
app=run_api.create_api_app(argv=['--core-device=cpu','--te-device=cpu','--vae-device=cpu'], env={})
with TestClient(app) as client:
    data=client.get('/api/engines/capabilities').json()
print('asset_contracts keys:', sorted(data.get('asset_contracts',{}).keys()))
print('engine map subset:', {k:data['engine_id_to_semantic_engine'][k] for k in ['flux1_chroma','flux1_fill','wan22_5b','wan22_14b','svd','hunyuan_video','sd35'] if k in data['engine_id_to_semantic_engine']})
print('engine keys:', sorted(data.get('engines',{}).keys()))
PY
```

### Grepping guessed engine directories that do not exist

Wrong command:
```bash
rg -n "class .*SVD|class .*Hunyuan|class .*Wan|text_encoder|vae|requires" apps/backend/engines/svd apps/backend/engines/hunyuan apps/backend/engines/wan22 -g '*.py'
```

Cause and fix:
Assumed dedicated `apps/backend/engines/svd` and `apps/backend/engines/hunyuan` folders exist; in this repo those optional engines are declared via import paths in `apps/backend/engines/registration.py` (e.g. `.video.svd.engine`, `.video.hunyuan.engine`) and may be absent in the local tree. Confirm real paths before narrowing with `rg`.

Correct command:
```bash
ls -1 apps/backend/engines
rg -n 'register_svd|register_hunyuan_video|video\\.svd|video\\.hunyuan|wan22_14b|wan22_5b|wan22_animate_14b' apps/backend/engines/registration.py apps/backend/engines/wan22 -g '*.py'
```

### Reading a guessed CLIP vision file path that does not exist

Wrong command:
```bash
sed -n '1,360p' apps/backend/runtime/vision/clip/model.py
```

Cause and fix:
Guessed a `model.py` file in `runtime/vision/clip`, but this package uses `encoder.py`, `state_dict.py`, and related modules instead. List actual files before opening specific paths.

Correct command:
```bash
find apps/backend/runtime/vision/clip -maxdepth 2 -type f -name '*.py' -print
sed -n '1,320p' apps/backend/runtime/vision/clip/encoder.py
```

### Reading a guessed LoRA module file that does not exist

Wrong command:
```bash
sed -n '1,340p' apps/backend/runtime/adapters/lora/apply.py
```

Cause and fix:
Assumed an `apply.py` module exists under LoRA adapters; in this repo LoRA logic is split across `mapping.py`, `loader.py`, `pipeline.py`, and `selections.py`. Enumerate files first, then inspect the right module.

Correct command:
```bash
find apps/backend/runtime/adapters/lora -maxdepth 2 -type f -name '*.py' -print
sed -n '1,340p' apps/backend/runtime/adapters/lora/mapping.py
```

### Running repo imports with system Python instead of repo venv

Wrong command:
```bash
python - <<'PY'
import inspect
from transformers.models.clip.modeling_clip import CLIPAttention
print(inspect.signature(CLIPAttention.forward))
PY
```

Cause and fix:
Used system `python`, which does not include repo dependencies (`transformers`), causing `ModuleNotFoundError`. Use the workspace venv Python with `CODEX_ROOT`/`PYTHONPATH` set.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY'
import inspect
from transformers.models.clip.modeling_clip import CLIPAttention
print(inspect.signature(CLIPAttention.forward))
PY
```

Wrong command: `CODEX_ROOT="$(git rev-parse --show-toplevel)"; PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/model_registry/test_vae_selection.py::test_sdxl_vae_keymap_remap_diffusers_mid_block_attention_aliases`
Cause and fix: `CODEX_ROOT`/`PYTHONPATH` were shell variables, not exported env vars for the pytest subprocess, so `conftest.py` raised `RuntimeError: CODEX_ROOT is required`.
Correct command: `CODEX_ROOT="$(git rev-parse --show-toplevel)" PYTHONPATH="$(git rev-parse --show-toplevel)" "$(git rev-parse --show-toplevel)/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/model_registry/test_vae_selection.py::test_sdxl_vae_keymap_remap_diffusers_mid_block_attention_aliases`

Wrong command: `rg -n "DIFFUSERS legacy aliases|DIFFUSERS mid-attention alias regressions|Backend suite now includes SDXL VAE DIFFUSERS|Added execution plan \.sangoi/plans/2026-02-11-sdxl-vae-diffusers-alias-canonicalization|Added task log \.sangoi/task-logs/2026-02-11-sdxl-vae-diffusers-mid-attention-alias-canonicalization|SDXL VAE DIFFUSERS mid-attention alias canonicalization|CODEX_ROOT is required to run this test suite|Wrong command: `CODEX_ROOT=\"\$\(git rev-parse --show-toplevel\)\"; PYTHONPATH=\"\$CODEX_ROOT\"" apps/backend/runtime/state_dict/AGENTS.md .sangoi/dev/tests/backend/AGENTS.md .sangoi/dev/tests/AGENTS.md .sangoi/plans/AGENTS.md .sangoi/task-logs/AGENTS.md .sangoi/task-logs/2026-02-11-sdxl-vae-diffusers-mid-attention-alias-canonicalization.md .sangoi/CHANGELOG.md COMMON_MISTAKES.md`
Cause and fix: I embedded unmatched backticks inside a double-quoted regex pattern, which broke shell parsing (`unexpected EOF while looking for matching \\``).
Correct command: `rg -n "DIFFUSERS legacy aliases|DIFFUSERS mid-attention alias regressions|Backend suite now includes SDXL VAE DIFFUSERS|Added execution plan \.sangoi/plans/2026-02-11-sdxl-vae-diffusers-alias-canonicalization|Added task log \.sangoi/task-logs/2026-02-11-sdxl-vae-diffusers-mid-attention-alias-canonicalization|SDXL VAE DIFFUSERS mid-attention alias canonicalization" apps/backend/runtime/state_dict/AGENTS.md .sangoi/dev/tests/backend/AGENTS.md .sangoi/dev/tests/AGENTS.md .sangoi/plans/AGENTS.md .sangoi/task-logs/AGENTS.md .sangoi/task-logs/2026-02-11-sdxl-vae-diffusers-mid-attention-alias-canonicalization.md .sangoi/CHANGELOG.md`

Wrong command: `sed -n '1,220p' .sangoi/dev/tests/backend/model_registry/AGENTS.md`
Cause and fix: I assumed an `AGENTS.md` existed under `.sangoi/dev/tests/backend/model_registry/`; it does not. The applicable scope file is `.sangoi/dev/tests/backend/AGENTS.md`.
Correct command: `sed -n '1,220p' .sangoi/dev/tests/backend/AGENTS.md`

Wrong command: `cd /home/lucas/work/stable-diffusion-webui-codex && CODEX_ROOT="$(git rev-parse --show-toplevel)" PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY' ...`
Cause and fix: I used `$CODEX_ROOT` in the same env-assignment expression where it is first defined; shell expansion happened before assignment, so the interpreter path became `/.venv/bin/python`.
Correct command: `cd /home/lucas/work/stable-diffusion-webui-codex && CODEX_ROOT="$(git rev-parse --show-toplevel)"; PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY' ...`

Wrong command: `cd /home/lucas/work/stable-diffusion-webui-codex && CODEX_ROOT="$(git rev-parse --show-toplevel)"; PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY' ...`
Cause and fix: `PYTHONPATH` was set for the subprocess, but `CODEX_ROOT` was not exported into the subprocess environment; repo init failed with `OSError: CODEX_ROOT not set`.
Correct command: `cd /home/lucas/work/stable-diffusion-webui-codex && CODEX_ROOT="$(git rev-parse --show-toplevel)"; CODEX_ROOT="$CODEX_ROOT" PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY' ...`

### Inspecting diffusers source with system Python

Wrong command:
```bash
FILE="$(python - <<'PY'
import inspect, diffusers.models.autoencoders.autoencoder_kl as m
print(inspect.getsourcefile(m))
PY
)"
```

Cause and fix:
`python` resolved to the system interpreter (outside `$CODEX_ROOT/.venv`), so `diffusers` was unavailable (`ModuleNotFoundError`). Use the repo venv Python and set `PYTHONPATH` to repo root.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
FILE="$($CODEX_ROOT/.venv/bin/python - <<'PY'
import inspect, diffusers.models.autoencoders.autoencoder_kl as m
print(inspect.getsourcefile(m))
PY
)"
```

### Expanding `CODEX_ROOT` before assignment in one-liner

Wrong command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)" PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/model_registry/test_vae_selection.py
```

Cause and fix:
Shell expanded `$CODEX_ROOT` in `PYTHONPATH`/python path before the assignment took effect, resolving to `/.venv/bin/python`. Assign `CODEX_ROOT` first (or use `$PWD`) in a separate command segment.

Correct command:
```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)" && PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/model_registry/test_vae_selection.py
```

### Trying to run `apply_patch` through `exec_command`

Wrong command:
```bash
exec_command "... && apply_patch <<'PATCH' ... PATCH"
```

Cause and fix:
`apply_patch` must be invoked with the dedicated `apply_patch` tool, not via a shell command. Running it through `exec_command` triggers harness warnings and can desynchronize edit tracking.

Correct command:
```bash
# Use the dedicated tool call:
functions.apply_patch
```

### Expanding `$CODEX_ROOT` before assignment (again) in env prefix

Wrong command: `CODEX_ROOT="/home/lucas/work/stable-diffusion-webui-codex" PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" - <<'PY' ...`
Cause and fix: `$CODEX_ROOT` in `PYTHONPATH` and interpreter path was expanded before that env assignment took effect, resolving to `/.venv/bin/python`. Set variables after `cd` using `$PWD` (or split assignment into two commands).
Correct command: `cd /home/lucas/work/stable-diffusion-webui-codex && CODEX_ROOT="$PWD" PYTHONPATH="$PWD" "$PWD/.venv/bin/python" - <<'PY' ...`

Wrong command: `CODEX_ROOT="$PWD" ./.venv/bin/python -m pip download --no-deps --dest /tmp ccvfi==0.0.3 ffmpeg-downloader==0.4.2`
Cause and fix: The repo venv intentionally does not ship `pip`; invoking `python -m pip` failed immediately (`No module named pip`). Use `uv` for package resolution/download operations.
Correct command: `./.uv/bin/uv tool run --from ffmpeg-downloader python -m ffmpeg_downloader --help` (or `./.uv/bin/uv lock --python "$PWD/.venv/bin/python"` when updating project deps)

Wrong command: `./.uv/bin/uv pip download --no-deps --dest /tmp ccvfi==0.0.3 ffmpeg-downloader==0.4.2`
Cause and fix: `uv pip` has no `download` subcommand, and `ffmpeg-downloader==0.4.2` does not exist. Use `uv tool run --from <pkg>` for inspection and pin available versions.
Correct command: `UV_TOOL_DIR="$PWD/.uv/tools" UV_TOOL_BIN_DIR="$PWD/.uv/tools/bin" ./.uv/bin/uv tool run --from ffmpeg-downloader ffdl --help`

Wrong command: `cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "gguf-exec|gguf_cache|dequant_forward_cache|te_device|te_impl|te_kernel|required|clip_fp32|text_encoder" apps/backend/infra/config/args.py apps/backend/infra/config/extra_args.py apps/backend/infra/config/gguf_exec_mode.py`
Cause and fix: I assumed `apps/backend/infra/config/extra_args.py` existed; in this repo it does not, so `rg` failed with `No such file or directory`. Confirm file inventory first and target real config modules only.
Correct command: `cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "gguf-exec|gguf_cache|dequant_forward_cache|te_device|te_impl|te_kernel|required|clip_fp32|text_encoder" apps/backend/infra/config/args.py apps/backend/infra/config/gguf_exec_mode.py`

Wrong command: `cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "text_context.py` GGUF TE loading no longer" apps/backend/runtime/families/wan22/AGENTS.md`
Cause and fix: I included an unescaped backtick inside a double-quoted regex, which broke shell parsing (`unexpected EOF while looking for matching ```). Use a pattern without backticks (or escape them) for shell-safe `rg`.
Correct command: `cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "GGUF TE loading no longer hard-codes" apps/backend/runtime/families/wan22/AGENTS.md`

Wrong command: `bash -lc "./update-webui.sh --force >\"$BASE/untracked_force.log\" 2>&1"` (inside functional matrix script)
Cause and fix: `update-webui.sh` is not guaranteed to be executable in every checkout; direct invocation returned exit `126`. Invoke it explicitly with `bash`.
Correct command: `bash -lc "bash ./update-webui.sh --force >\"$BASE/untracked_force.log\" 2>&1"`

Wrong command: repeated full-repo clones into `/tmp` for updater behavior matrix (e.g. `git clone -q "$CODEX_ROOT" "$work"`)
Cause and fix: cloning this large repo repeatedly exhausted disk (`No space left on device`). Use a minimal local harness repo with only updater script + tiny tracked files to validate git-state matrix behavior.
Correct command: initialize a tiny harness repo (`git init` + commit `update-webui.sh` + `README.md`) and clone that harness for each matrix case.

Wrong command: `cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "Wrong command: `bash -lc \\\"\\./update-webui.sh --force|full-repo clones into \\`/tmp\\`" COMMON_MISTAKES.md`
Cause and fix: I embedded backticks inside a double-quoted regex, which broke shell parsing (`unexpected EOF while looking for matching ```). Avoid backticks in shell-quoted regex patterns or search with plain, stable substrings.
Correct command: `cd /home/lucas/work/stable-diffusion-webui-codex && rg -n "untracked_force.log|full-repo clones" COMMON_MISTAKES.md`

Wrong command: `CODEX_ROOT="$(git rev-parse --show-toplevel)"; PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_wan22_strict_load_contract.py .sangoi/dev/tests/backend/test_wan22_text_context_defaults.py .sangoi/dev/tests/backend/test_gguf_dequant_forward_cache.py`
Cause and fix: `CODEX_ROOT`/`PYTHONPATH` were shell variables but not exported, so the test suite conftest could not see `CODEX_ROOT` from the environment and aborted. Export both variables before running pytest.
Correct command: `CODEX_ROOT="$(git rev-parse --show-toplevel)"; export CODEX_ROOT PYTHONPATH="$CODEX_ROOT"; "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_wan22_strict_load_contract.py .sangoi/dev/tests/backend/test_wan22_text_context_defaults.py .sangoi/dev/tests/backend/test_gguf_dequant_forward_cache.py`

Wrong command: `CODEX_ROOT="$(git rev-parse --show-toplevel)"; export CODEX_ROOT PYTHONPATH="$CODEX_ROOT"; "$CODEX_ROOT/.venv/bin/python" - <<'PY' ... monkeypatch wan_text_context.torch.cuda.is_available = lambda: True ... PY`
Cause and fix: Forcing `torch.cuda.is_available()` to `True` in a CPU-only torch build triggered runtime memory-manager CUDA probes (`torch.cuda.current_device()`), causing `AssertionError: Torch not compiled with CUDA enabled`. Do not fake CUDA availability in this stack; keep CUDA checks truthful and test CUDA-only contracts conditionally.
Correct command: `CODEX_ROOT="$(git rev-parse --show-toplevel)"; export CODEX_ROOT PYTHONPATH="$CODEX_ROOT"; "$CODEX_ROOT/.venv/bin/python" -m pytest -q .sangoi/dev/tests/backend/test_wan22_strict_load_contract.py`
