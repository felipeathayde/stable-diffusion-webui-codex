#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PY_BIN="${VENV_DIR}/bin/python"

TORCH_MODE="${CODEX_TORCH_MODE:-auto}" # auto|cpu|cuda|skip
TRACE="${CODEX_INSTALL_TRACE:-0}"

log() { echo "[install] $*"; }
warn() { echo "[install] Warning: $*" >&2; }
die() { echo "[install] Error: $*" >&2; exit 1; }

if [[ "${TRACE}" == "1" ]]; then
  log "Trace enabled (CODEX_INSTALL_TRACE=1)."
  set -x
fi

log "Repo: ${ROOT_DIR}"
log "Venv: ${VENV_DIR}"
log "Torch mode: ${TORCH_MODE} (override via CODEX_TORCH_MODE=auto|cpu|cuda|skip)"
log "Host: $(uname -a)"

if ! command -v python3 >/dev/null 2>&1 && ! command -v python >/dev/null 2>&1; then
  die "missing python3/python on PATH."
fi

BOOTSTRAP_PY="python"
if command -v python3 >/dev/null 2>&1; then
  BOOTSTRAP_PY="python3"
fi

log "Bootstrap Python: $(command -v "${BOOTSTRAP_PY}")"
"${BOOTSTRAP_PY}" -V

if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating venv at ${VENV_DIR} ..."
  "${BOOTSTRAP_PY}" -m venv "${VENV_DIR}"
else
  log "Venv already exists; reusing."
fi

if [[ ! -x "${PY_BIN}" ]]; then
  die "expected venv python at '${PY_BIN}'."
fi

log "Venv Python: ${PY_BIN}"
"${PY_BIN}" -V
"${PY_BIN}" -m pip --version

log "Upgrading pip tooling ..."
"${PY_BIN}" -m pip install -U pip wheel setuptools
"${PY_BIN}" -m pip --version

install_torch() {
  if [[ "${TORCH_MODE}" == "skip" ]]; then
    log "Skipping torch install (CODEX_TORCH_MODE=skip)."
    return 0
  fi

  if "${PY_BIN}" -c "import torch; print(torch.__version__)" >/dev/null 2>&1; then
    log "torch already installed; skipping."
    return 0
  fi

  local platform
  platform="$(uname -s | tr '[:upper:]' '[:lower:]')"
  local arch
  arch="$(uname -m | tr '[:upper:]' '[:lower:]')"
  log "Torch install: platform=${platform} arch=${arch}"

  if [[ "${platform}" == "darwin" ]]; then
    log "Installing torch/torchvision (macOS default wheels) ..."
    "${PY_BIN}" -m pip install torch torchvision || return 1
    return 0
  fi

  local cuda_ver=""
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "nvidia-smi detected at: $(command -v nvidia-smi)"
    local smi_q=""
    smi_q="$(nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>/dev/null || true)"
    if [[ -n "${smi_q}" ]]; then
      log "nvidia-smi GPUs (name, driver, cuda): ${smi_q//$'\n'/ | }"
    else
      log "nvidia-smi present but GPU query failed (continuing with best-effort parse)."
    fi
    cuda_ver="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \\([0-9][0-9]*\\.[0-9][0-9]*\\).*/\\1/p' | head -n 1 || true)"
  fi

  local candidates=()
  if [[ "${TORCH_MODE}" == "cpu" ]]; then
    candidates=(cpu)
  elif [[ "${TORCH_MODE}" == "cuda" ]]; then
    # Force CUDA wheels (fallback chain), even if nvidia-smi is missing.
    candidates=(cu126 cu124 cu121 cu118)
  elif [[ -n "${cuda_ver}" ]]; then
    local major="${cuda_ver%%.*}"
    local minor="${cuda_ver#*.}"
    minor="${minor%%.*}"
    log "Detected NVIDIA driver CUDA version: ${cuda_ver}"
    if [[ "${major}" -ge 12 ]]; then
      if [[ "${minor}" -ge 6 ]]; then candidates=(cu126 cu124 cu121); fi
      if [[ "${minor}" -ge 4 && ${#candidates[@]} -eq 0 ]]; then candidates=(cu124 cu121); fi
      if [[ "${minor}" -ge 1 && ${#candidates[@]} -eq 0 ]]; then candidates=(cu121); fi
      if [[ ${#candidates[@]} -eq 0 ]]; then candidates=(cu118); fi
    elif [[ "${major}" -eq 11 && "${minor}" -ge 8 ]]; then
      candidates=(cu118)
    else
      candidates=(cpu)
    fi
  else
    candidates=(cpu)
  fi

  log "Torch wheel candidates: ${candidates[*]}"

  local ok=1
  for variant in "${candidates[@]}"; do
    log "Installing torch/torchvision (${variant}) via PyTorch index ..."
    if "${PY_BIN}" -m pip install \
      --index-url "https://download.pytorch.org/whl/${variant}" \
      --extra-index-url "https://pypi.org/simple" \
      torch torchvision; then
      ok=0
      break
    fi
  done

  if [[ "${ok}" -ne 0 ]]; then
    warn "failed to install torch automatically."
    warn "Tip: set CODEX_TORCH_MODE=cpu to force CPU wheels, CODEX_TORCH_MODE=cuda to force CUDA attempts, or CODEX_TORCH_MODE=skip to skip."
    return 1
  fi
}

install_torch

log "Installing Python requirements ..."
"${PY_BIN}" -m pip install -r "${ROOT_DIR}/requirements.txt"

log "Installed versions (sanity):"
"${PY_BIN}" - <<'PY' || true
import platform
import sys
import importlib.metadata as m

def v(dist: str) -> str:
    try:
        return m.version(dist)
    except Exception:
        return "missing"

print("python", sys.version.split()[0], "exe", sys.executable)
print("platform", platform.platform())
for dist in [
    "torch",
    "torchvision",
    "diffusers",
    "transformers",
    "peft",
    "accelerate",
    "huggingface-hub",
    "tokenizers",
    "safetensors",
    "fastapi",
    "uvicorn",
    "pydantic",
    "numpy",
    "pillow",
]:
    print(f"{dist} {v(dist)}")
try:
    import torch
    print("torch.cuda.is_available", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", repr(e))
PY

log "pip check:"
"${PY_BIN}" -m pip check || true

log "Installing frontend dependencies (npm) ..."
if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  warn "missing 'node' and/or 'npm' on PATH; skipping frontend install."
  warn "Install Node.js (>=18), then run: (cd apps/interface && npm install)"
  exit 0
fi

log "node: $(node -v)  npm: $(npm -v)"
(cd "${ROOT_DIR}/apps/interface" && npm install)

echo ""
log "Done."
log "Run: ./run-webui.sh"
