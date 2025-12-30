# Install — Codex WebUI (Backend + Vue UI)

This repo ships:
- **Backend API**: FastAPI (`apps/backend/interfaces/api/run_api.py`)
- **Frontend UI**: Vue 3 + Vite (`apps/interface`)

## Prerequisites
- Git
- Node.js 18+ (for the Vue UI)
- Internet access (first install: downloads `uv`, CPython 3.12.10, and wheels)

Optional:
- `ffmpeg` + `ffprobe` on `PATH` (video export / vid2vid workflows)

## Quick install (recommended)

### Windows (PowerShell or CMD)
1) Run the installer (downloads `uv`, installs managed CPython **3.12.10** into `.uv/python`, syncs deps from `uv.lock` into `.venv`, runs `npm install`):
```bat
install-webui.bat
```

2) Launch the GUI launcher:
```bat
run-webui.bat
```

### Linux / WSL
1) Run the installer (downloads `uv`, installs managed CPython **3.12.10** into `.uv/python`, syncs deps from `uv.lock` into `.venv`, runs `npm install`):
```bash
bash install-webui.sh
```

2) Start API + UI:
```bash
./run-webui.sh
```

## PyTorch
This repo uses `uv.lock` to pin and lock dependency versions (including PyTorch variants). The installers choose **one** PyTorch backend via `uv` extras.

Default behavior:
- `CODEX_TORCH_MODE=auto` (default): if `nvidia-smi` exists, install the CUDA 12.6 wheels (`--extra cu126`), otherwise CPU (`--extra cpu`).
- On macOS, the installers always use `cpu`.

Override:
- `CODEX_TORCH_MODE=cpu` (force CPU: `--extra cpu`)
- `CODEX_TORCH_MODE=cuda` (force CUDA: defaults to `--extra cu128`)
- `CODEX_TORCH_MODE=skip` (skip torch/torchvision entirely; the WebUI will not run without PyTorch)
- `CODEX_TORCH_BACKEND=cpu|cu118|cu126|cu128|cu130` (explicitly pick the PyTorch backend extra)
- `CODEX_INSTALL_TRACE=1` (Linux/WSL installer: enable shell trace for debugging)

If CUDA install fails, try a different backend:
- Example: `CODEX_TORCH_BACKEND=cu118 bash install-webui.sh`

## Troubleshooting

### `ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'`
Your `peft` and `transformers` are out of sync (common when extra packages pull older pins).

Fix: remove the venv and re-install with this repo’s locked dependencies:
- Delete `.venv`
- Re-run `install-webui.(bat|sh)`

### Don’t mix “research deps” into the WebUI venv
Packages like `pyiqa`, `datasets`, `numba`, `opencv-python-headless` often pin conflicting `transformers`/`numpy`.

Keep them in a separate virtualenv.
