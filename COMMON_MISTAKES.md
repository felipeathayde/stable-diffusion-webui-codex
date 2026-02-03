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
rg -n "use_cases/restore\\.py|`restore\\.py`" .sangoi/planning/2026-02-01-supir-webui-integration-v1.md
```

Cause and fix:
In bash, backticks (`` `...` ``) trigger command substitution even inside double quotes, so the shell tries to execute `restore.py` as a command.
Use single quotes around the pattern, escape backticks, or just search for the literal without backticks.

Correct command:
```bash
rg -n 'use_cases/restore\\.py|`restore\\.py`' .sangoi/planning/2026-02-01-supir-webui-integration-v1.md
rg -n "restore\\.py" .sangoi/planning/2026-02-01-supir-webui-integration-v1.md
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

### Running `pytest` at repo root (collects `tmp/**` tests/artifacts)

Wrong command:
```bash
export CODEX_ROOT="$(git rev-parse --show-toplevel)"
cd "$CODEX_ROOT"
PYTHONPATH=. .venv/bin/python -m pytest -q
```

Cause and fix:
Running pytest at the repo root can collect “third party” or local artifact tests under `tmp/**` (and other non-canonical locations) which may require extra dependencies or a different runtime environment. Scope test runs explicitly (or ignore `tmp`).

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
