# Install — Codex WebUI (Backend + Vue UI)

This repo ships:
- **Backend API**: FastAPI (`apps/backend/interfaces/api/run_api.py`)
- **Frontend UI**: Vue 3 + Vite (`apps/interface`)

## Prerequisites
- Git
- Python 3.12+ (recommended)
- Node.js 18+ (for the Vue UI)
- **PyTorch** (the installer will try to auto-install; see “PyTorch” below)

Optional:
- `ffmpeg` + `ffprobe` on `PATH` (video export / vid2vid workflows)

## Quick install (recommended)

### Windows (PowerShell or CMD)
1) Run the installer (creates `.venv`, installs Python deps, runs `npm install`):
```bat
install-webui.bat
```

Note: the Windows installer delegates most logic to `tools/install_webui.py` (prints detailed detection + version info).

2) Launch the GUI launcher:
```bat
run-webui.bat
```

If you prefer an interactive shell with the venv activated:
```bat
activate-venv.bat
```

### Linux / WSL
1) Run the installer (creates `.venv`, installs Python deps, runs `npm install`):
```bash
bash install-webui.sh
```

2) Start API + UI:
```bash
./run-webui.sh
```

## PyTorch
`requirements.txt` intentionally **does not** install `torch` / `torchvision`.

The installers try to auto-install `torch` + `torchvision`:
- Detect NVIDIA via `nvidia-smi` and pick CUDA wheels (fallback chain).
- Otherwise install CPU wheels.

Override:
- `CODEX_TORCH_MODE=cpu` (force CPU)
- `CODEX_TORCH_MODE=cuda` (force CUDA wheel attempt)
- `CODEX_TORCH_MODE=skip` (don’t install torch)
- `CODEX_INSTALL_TRACE=1` (Linux/WSL installer: enable shell trace for debugging)

If auto-install fails, install PyTorch for your platform (CPU/CUDA) using the official PyTorch instructions, then re-run the installer.

## Troubleshooting

### `ImportError: cannot import name 'EncoderDecoderCache' from 'transformers'`
Your `peft` and `transformers` are out of sync (common when extra packages pull older pins).

Fix: remove the venv and re-install with this repo’s pinned requirements:
- Delete `.venv`
- Re-run `install-webui.(bat|sh)`

### Don’t mix “research deps” into the WebUI venv
Packages like `pyiqa`, `datasets`, `numba`, `opencv-python-headless` often pin conflicting `transformers`/`numpy`.

Keep them in a separate virtualenv.
