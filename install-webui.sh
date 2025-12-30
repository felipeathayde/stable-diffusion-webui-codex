#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_VERSION="${CODEX_UV_VERSION:-0.9.17}"
UV_DIR="${ROOT_DIR}/.uv/bin"
UV_BIN="${UV_DIR}/uv"

PYTHON_VERSION="${CODEX_PYTHON_VERSION:-3.12.10}"
VENV_DIR="${ROOT_DIR}/.venv"

TORCH_MODE="${CODEX_TORCH_MODE:-auto}" # auto|cpu|cuda|skip
TORCH_BACKEND="${CODEX_TORCH_BACKEND:-}" # cpu|cu118|cu126|cu128 (optional override)
TRACE="${CODEX_INSTALL_TRACE:-0}"

log() { echo "[install] $*"; }
warn() { echo "[install] Warning: $*" >&2; }
die() { echo "[install] Error: $*" >&2; exit 1; }

if [[ "${TRACE}" == "1" ]]; then
  log "Trace enabled (CODEX_INSTALL_TRACE=1)."
  set -x
fi

log "Repo: ${ROOT_DIR}"
log "uv: ${UV_BIN} (version pin: ${UV_VERSION})"
log "Python: ${PYTHON_VERSION} (managed by uv)"
log "Venv: ${VENV_DIR} (created by uv; uses the managed Python)"
log "Torch mode: ${TORCH_MODE} (override via CODEX_TORCH_MODE=auto|cpu|cuda|skip)"
if [[ -n "${TORCH_BACKEND}" ]]; then
  log "Torch backend override: ${TORCH_BACKEND} (CODEX_TORCH_BACKEND)"
fi
log "Host: $(uname -a)"

bootstrap_uv() {
  if [[ -x "${UV_BIN}" ]]; then
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    die "missing 'curl' on PATH; required to download uv."
  fi

  mkdir -p "${UV_DIR}"

  log "Installing uv ${UV_VERSION} into ${UV_DIR} ..."
  curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | \
    env UV_NO_MODIFY_PATH=1 UV_UNMANAGED_INSTALL="${UV_DIR}" sh

  if [[ ! -x "${UV_BIN}" ]]; then
    die "uv install succeeded but '${UV_BIN}' is missing or not executable."
  fi
}

install_python() {
  export UV_PYTHON_INSTALL_DIR="${ROOT_DIR}/.uv/python"
  export UV_PYTHON_INSTALL_BIN=0
  export UV_PYTHON_PREFERENCE="only-managed"
  export UV_PYTHON_DOWNLOADS="manual"

  log "Installing managed Python ${PYTHON_VERSION} ..."
  "${UV_BIN}" python install "${PYTHON_VERSION}"
}

pick_torch_extra() {
  if [[ -n "${TORCH_BACKEND}" ]]; then
    case "${TORCH_BACKEND}" in
      cpu|cu118|cu126|cu128) echo "${TORCH_BACKEND}"; return 0 ;;
      *) die "invalid CODEX_TORCH_BACKEND='${TORCH_BACKEND}' (expected: cpu|cu118|cu126|cu128)";;
    esac
  fi

  if [[ "${TORCH_MODE}" == "skip" ]]; then
    echo ""
    return 0
  fi

  local platform
  platform="$(uname -s | tr '[:upper:]' '[:lower:]')"
  if [[ "${platform}" == "darwin" ]]; then
    echo "cpu"
    return 0
  fi

  if [[ "${TORCH_MODE}" == "cpu" ]]; then
    echo "cpu"
    return 0
  fi

  if [[ "${TORCH_MODE}" == "cuda" ]]; then
    echo "cu126"
    return 0
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "cu126"
    return 0
  fi

  echo "cpu"
}

sync_python_deps() {
  export UV_PROJECT_ENVIRONMENT="${VENV_DIR}"

  local extra
  extra="$(pick_torch_extra)"
  if [[ -z "${extra}" ]]; then
    warn "Skipping torch/torchvision install (CODEX_TORCH_MODE=skip). The WebUI will not run without PyTorch."
    log "Syncing Python dependencies (locked) ..."
    "${UV_BIN}" sync --locked
    return 0
  fi

  log "Syncing Python dependencies (locked) with torch extra: ${extra} ..."
  "${UV_BIN}" sync --locked --extra "${extra}"
}

bootstrap_uv
install_python
sync_python_deps

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
