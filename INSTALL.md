# Install — Codex WebUI (Backend + Vue UI)

This repo ships:
- **Backend API**: FastAPI (`apps/backend/interfaces/api/run_api.py`)
- **Frontend UI**: Vue 3 + Vite (`apps/interface`)

## Prerequisites
- Git
- Internet access (first install: downloads `uv`, CPython 3.12.10, Node.js (via nodeenv), and wheels)

Video runtime note:
- `install-webui.(bat|sh)` provisions repo-local `ffmpeg` + `ffprobe` binaries and the default RIFE checkpoint automatically (no manual PATH/model setup required).

## Quick install (recommended)

### Windows (PowerShell or CMD)
1) Run the installer (downloads `uv`, installs managed CPython **3.12.10** into `.uv/python`, syncs deps from `uv.lock` into `.venv`, installs Node.js into `.nodeenv` (via nodeenv), runs `npm install`; keeps `uv`/`npm` caches repo-local under `.uv/cache` and `.npm-cache`):
```bat
install-webui.bat
```

On Windows, `install-webui.bat` prompts for **Simple vs Advanced** (CUDA 12.6/12.8/13) by default.
For automation, pass `--no-menu`.

2) Launch the GUI launcher:
```bat
run-webui.bat
```

3) Safe update (fail-closed):
```bat
update-webui.bat
```

### Linux / WSL
1) Run the installer (downloads `uv`, installs managed CPython **3.12.10** into `.uv/python`, syncs deps from `uv.lock` into `.venv`, installs Node.js into `.nodeenv` (via nodeenv), runs `npm install`; keeps `uv`/`npm` caches repo-local under `.uv/cache` and `.npm-cache`):
```bash
bash install-webui.sh
```

2) Start API + UI:
```bash
./run-webui.sh
# Advanced (forwarded to backend):
./run-webui.sh --gguf-exec=dequant_upfront
```

3) Safe update (fail-closed):
```bash
bash update-webui.sh
```

## Safe updater contract (`update-webui.(bat|sh)`)
- Update scope is repo root only (no submodule/extension update automation).
- Dirty worktree check is fail-closed (tracked + untracked paths abort; ignored paths do not).
- Abort diagnostics list explicit cause and offending files/directories when applicable.
- Non-destructive update path only: `git fetch --prune` + `git pull --ff-only`.
- Environment refresh runs only when pull actually changed `HEAD`.
- Frontend refresh uses lock-preserving mode (`npm ci`), so `apps/interface/package-lock.json` is required.

## Node.js (frontend)
The installers provision a repo-local Node.js into `.nodeenv` via `nodeenv` (no system Node required).

Override:
- `CODEX_NODE_VERSION` (Node.js version pin for nodeenv; default: 24.13.0)

## PyTorch
This repo uses `uv.lock` to pin and lock dependency versions (including PyTorch variants). The installers choose **one** PyTorch backend via `uv` extras.

Default behavior:
- `CODEX_TORCH_MODE=auto` (default):
  - If AMD/ROCm is detected (Linux), use ROCm 6.4 wheels (`--extra rocm64`).
  - Else if `nvidia-smi` exists, prefer CUDA 12.8 wheels (`--extra cu128`), with fallback to `cu126` if the driver advertises CUDA 12.6 and `cu130` if CUDA 13 is advertised and the driver is new enough.
  - Otherwise CPU (`--extra cpu`).
- On macOS, the installers always use `cpu`.

Override:
- `CODEX_TORCH_MODE=cpu` (force CPU: `--extra cpu`)
- `CODEX_TORCH_MODE=cuda` (force CUDA: defaults to `--extra cu128`)
- `CODEX_TORCH_MODE=rocm` (Linux only: force ROCm 6.4 wheels: `--extra rocm64`)
- `CODEX_TORCH_MODE=skip` (skip torch/torchvision entirely; the WebUI will not run without PyTorch)
- `CODEX_TORCH_BACKEND=cpu|cu126|cu128|cu130|rocm64` (explicitly pick the PyTorch backend extra)
- `CODEX_CUDA_VARIANT=12.6|12.8|13` (choose CUDA wheels when using `auto`/`cuda`; maps to `cu126|cu128|cu130`)
- `CODEX_INSTALL_TRACE=1` (Linux/WSL installer: enable shell trace for debugging)
- `CODEX_FFMPEG_VERSION=<version>` (pin ffmpeg-downloader runtime build; default: `7.0.2`)

If CUDA install fails, try a different backend:
- Example: `CODEX_TORCH_BACKEND=cu126 bash install-webui.sh`

## Troubleshooting

### Windows: `: was unexpected at this time` (cmd.exe parsing)
This is a **cmd.exe batch parsing** error (not a `uv`/Python error). It typically happens when a `.bat` script uses multi-line `(...)` blocks and cmd gets confused by special characters, causing labels like `:install_uv` to become “unexpected”.

Fix:
- `git pull` (the Windows installer routine was hardened to avoid this class of cmd parsing failure)
- Re-run `install-webui.bat`

If it still happens, see the deep dive runbook:
- Open an issue with the full console output, plus:
  - your Windows version
  - whether you cloned via Git or downloaded a ZIP
  - whether the repo path contains non-ASCII characters

### `ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'`
Your `peft` and `transformers` are out of sync (common when extra packages pull older pins).

Fix: remove the venv and re-install with this repo’s locked dependencies:
- Delete `.venv`
- Re-run `install-webui.(bat|sh)`

### Don’t mix “research deps” into the WebUI venv
Packages like `pyiqa`, `datasets`, `numba`, `opencv-python-headless` often pin conflicting `transformers`/`numpy`.

Keep them in a separate virtualenv.
