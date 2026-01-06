#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UV_VERSION="${CODEX_UV_VERSION:-0.9.17}"
UV_DIR="${ROOT_DIR}/.uv/bin"
UV_BIN="${UV_DIR}/uv"

PYTHON_VERSION="${CODEX_PYTHON_VERSION:-3.12.10}"
VENV_DIR="${ROOT_DIR}/.venv"

TORCH_MODE="${CODEX_TORCH_MODE:-auto}" # auto|cpu|cuda|rocm|skip
TORCH_BACKEND="${CODEX_TORCH_BACKEND:-}" # cpu|cu126|cu128|cu130|rocm64 (optional override)
CUDA_VARIANT="${CODEX_CUDA_VARIANT:-}" # 12.6|12.8|13|cu126|cu128|cu130 (optional override)
TRACE="${CODEX_INSTALL_TRACE:-0}"

log() { echo "[install] $*"; }
warn() { echo "[install] Warning: $*" >&2; }
die() { echo "[install] Error: $*" >&2; exit 1; }

if [[ "${TRACE}" == "1" ]]; then
  log "Trace enabled (CODEX_INSTALL_TRACE=1)."
  set -x
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv/cache}"
NPM_CACHE_DIR="${NPM_CONFIG_CACHE:-${ROOT_DIR}/.npm-cache}"
export UV_CACHE_DIR
export NPM_CONFIG_CACHE="${NPM_CACHE_DIR}"
mkdir -p "${UV_CACHE_DIR}" "${NPM_CACHE_DIR}"

log "Repo: ${ROOT_DIR}"
log "uv: ${UV_BIN} (version pin: ${UV_VERSION})"
log "uv cache: ${UV_CACHE_DIR}"
log "Python: ${PYTHON_VERSION} (managed by uv)"
log "Venv: ${VENV_DIR} (created by uv; uses the managed Python)"
log "Torch mode: ${TORCH_MODE} (override via CODEX_TORCH_MODE=auto|cpu|cuda|rocm|skip)"
if [[ -n "${TORCH_BACKEND}" ]]; then
  log "Torch backend override: ${TORCH_BACKEND} (CODEX_TORCH_BACKEND)"
fi
if [[ -n "${CUDA_VARIANT}" ]]; then
  log "CUDA variant override: ${CUDA_VARIANT} (CODEX_CUDA_VARIANT)"
fi
log "npm cache: ${NPM_CACHE_DIR}"
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
      cpu|cu126|cu128|cu130|rocm64) echo "${TORCH_BACKEND}"; return 0 ;;
      *) die "invalid CODEX_TORCH_BACKEND='${TORCH_BACKEND}' (expected: cpu|cu126|cu128|cu130|rocm64)";;
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
    if [[ -n "${CUDA_VARIANT}" ]]; then
      case "${CUDA_VARIANT}" in
        12.6|cu126) echo "cu126"; return 0 ;;
        12.8|cu128) echo "cu128"; return 0 ;;
        13|cu130) echo "cu130"; return 0 ;;
        *) die "invalid CODEX_CUDA_VARIANT='${CUDA_VARIANT}' (expected: 12.6|12.8|13|cu126|cu128|cu130)";;
      esac
    fi
    echo "cu128"
    return 0
  fi

  if [[ "${TORCH_MODE}" == "rocm" ]]; then
    echo "rocm64"
    return 0
  fi

  # auto: try to detect AMD ROCm first (Linux-only).
  if [[ "${platform}" == "linux" ]]; then
    if command -v rocminfo >/dev/null 2>&1 || command -v rocm-smi >/dev/null 2>&1 || [[ -d "/opt/rocm" ]]; then
      echo "rocm64"
      return 0
    fi
    if command -v lspci >/dev/null 2>&1 && lspci 2>/dev/null | rg -qi 'vga|3d|display' && lspci 2>/dev/null | rg -qi 'amd/ati|advanced micro devices'; then
      # If the machine looks like AMD GPU, prefer ROCm wheels when available.
      echo "rocm64"
      return 0
    fi
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    if [[ -n "${CUDA_VARIANT}" ]]; then
      case "${CUDA_VARIANT}" in
        12.6|cu126) echo "cu126"; return 0 ;;
        12.8|cu128) echo "cu128"; return 0 ;;
        13|cu130) echo "cu130"; return 0 ;;
        *) die "invalid CODEX_CUDA_VARIANT='${CUDA_VARIANT}' (expected: 12.6|12.8|13|cu126|cu128|cu130)";;
      esac
    fi

    local smi_line=""
    smi_line="$(nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader 2>/dev/null | head -n 1 || true)"
    if [[ -n "${smi_line}" ]]; then
      local gpu_name="" driver_version="" cuda_version=""
      IFS=',' read -r gpu_name driver_version cuda_version <<<"${smi_line}"
      gpu_name="$(echo "${gpu_name}" | xargs)"
      driver_version="$(echo "${driver_version}" | xargs)"
      cuda_version="$(echo "${cuda_version}" | xargs)"

      local driver_major="${driver_version%%.*}"
      local cuda_major="${cuda_version%%.*}"
      local cuda_minor="0"
      if [[ "${cuda_version}" == *.* ]]; then
        cuda_minor="${cuda_version#*.}"
        cuda_minor="${cuda_minor%%.*}"
      fi

      # Heuristics:
      # - Prefer CUDA 12.8+ wheels by default when NVIDIA is detected.
      # - If the driver advertises CUDA 13.x and the driver major is new enough, prefer cu130.
      # - If the driver is too old for CUDA 12.x (major < 525), fall back to CPU.
      if [[ "${driver_major}" =~ ^[0-9]+$ ]] && (( driver_major < 525 )); then
        echo "cpu"
        return 0
      fi

      if [[ "${cuda_major}" =~ ^[0-9]+$ ]] && [[ "${cuda_minor}" =~ ^[0-9]+$ ]]; then
        if (( cuda_major >= 13 )); then
          if [[ "${driver_major}" =~ ^[0-9]+$ ]] && (( driver_major >= 580 )); then
            echo "cu130"
            return 0
          fi
          # Driver likely too old for CUDA 13 wheels.
          echo "cu128"
          return 0
        fi
        if (( cuda_major == 12 && cuda_minor >= 8 )); then
          echo "cu128"
          return 0
        fi
        if (( cuda_major == 12 && cuda_minor >= 6 )); then
          echo "cu126"
          return 0
        fi
        if (( cuda_major == 12 )); then
          # Still a CUDA 12 driver, but not 12.6+. Prefer cu126 wheels as a safe default.
          echo "cu126"
          return 0
        fi
      fi

      # RTX 50-series is known to benefit from newer CUDA wheels; prefer cu128.
      if echo "${gpu_name}" | rg -qi 'rtx[[:space:]]*50|rtx[[:space:]]*5[0-9]{3}'; then
        echo "cu128"
        return 0
      fi
    fi

    echo "cu128"
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
(cd "${ROOT_DIR}/apps/interface" && npm install --cache "${NPM_CACHE_DIR}")
if [[ ! -f "${ROOT_DIR}/apps/interface/node_modules/vite/package.json" ]]; then
  die "npm install completed, but apps/interface/node_modules/vite/package.json is missing. Run: (cd apps/interface && npm install)"
fi

echo ""
log "Done."
log "Run: ./run-webui.sh"
