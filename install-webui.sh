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

NODE_VERSION="${CODEX_NODE_VERSION:-24.13.0}"
NODEENV_DIR="${ROOT_DIR}/.nodeenv"
NODEENV_BIN_DIR="${NODEENV_DIR}/bin"
NODEENV_NODE="${NODEENV_BIN_DIR}/node"
NODEENV_NPM="${NODEENV_BIN_DIR}/npm"
FFMPEG_VERSION="${CODEX_FFMPEG_VERSION:-7.0.2}"

log() { echo "[install] $*"; }
warn() { echo "[install] Warning: $*" >&2; }
die() { echo "[install] Error: $*" >&2; exit 1; }

if [[ "${TRACE}" == "1" ]]; then
  log "Trace enabled (CODEX_INSTALL_TRACE=1)."
  set -x
fi

UV_CACHE_DIR="${UV_CACHE_DIR:-${ROOT_DIR}/.uv/cache}"
NPM_CACHE_DIR="${NPM_CONFIG_CACHE:-${ROOT_DIR}/.npm-cache}"
XDG_DATA_HOME="${XDG_DATA_HOME:-${ROOT_DIR}/.uv/xdg-data}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${ROOT_DIR}/.uv/xdg-cache}"
CODEX_ROOT="${ROOT_DIR}"
PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
CODEX_FFMPEG_VERSION="${FFMPEG_VERSION}"
export UV_CACHE_DIR
export NPM_CONFIG_CACHE="${NPM_CACHE_DIR}"
export XDG_DATA_HOME XDG_CACHE_HOME CODEX_ROOT PYTHONPATH CODEX_FFMPEG_VERSION
mkdir -p "${UV_CACHE_DIR}" "${NPM_CACHE_DIR}" "${XDG_DATA_HOME}" "${XDG_CACHE_HOME}"

log "Repo: ${ROOT_DIR}"
log "uv: ${UV_BIN} (version pin: ${UV_VERSION})"
log "uv cache: ${UV_CACHE_DIR}"
log "Python: ${PYTHON_VERSION} (managed by uv)"
log "Venv: ${VENV_DIR} (created by uv; uses the managed Python)"
log "Node.js: ${NODE_VERSION} (managed by nodeenv; installed into ${NODEENV_DIR})"
log "FFmpeg runtime version: ${FFMPEG_VERSION} (managed by ffmpeg-downloader)"
log "XDG data: ${XDG_DATA_HOME}"
log "XDG cache: ${XDG_CACHE_HOME}"
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

provision_video_runtime_deps() {
  local py="${VENV_DIR}/bin/python"
  if [[ ! -x "${py}" ]]; then
    die "venv python not found at '${py}' after uv sync."
  fi

  log "Provisioning ffmpeg runtime (ffmpeg-downloader) ..."
  "${py}" - <<'PY'
import os
from apps.backend.video.runtime_dependencies import ensure_ffmpeg_binaries

resolved = ensure_ffmpeg_binaries(
    version=os.environ.get("CODEX_FFMPEG_VERSION") or "7.0.2",
    no_symlinks=True,
)
print(f"[install] ffmpeg: {resolved['ffmpeg']}")
print(f"[install] ffprobe: {resolved['ffprobe']}")
PY

  log "Provisioning default RIFE model checkpoint ..."
  "${py}" - <<'PY'
from apps.backend.video.runtime_dependencies import ensure_rife_model_file

path = ensure_rife_model_file()
print(f"[install] RIFE model: {path}")
PY

  log "Validating video runtime imports (cv2 + ccvfi) ..."
  "${py}" - <<'PY'
import cv2
import ccvfi

print(f"[install] opencv-python: {cv2.__version__}")
print(f"[install] ccvfi: {getattr(ccvfi, '__version__', 'unknown')}")
PY
}

ensure_nodeenv() {
  if [[ -e "${NODEENV_DIR}" && ! -d "${NODEENV_DIR}" ]]; then
    die "expected '${NODEENV_DIR}' to be a directory (nodeenv), but found a non-directory path."
  fi

  if [[ -x "${NODEENV_NODE}" && -x "${NODEENV_NPM}" ]]; then
    local existing
    existing="$("${NODEENV_NODE}" -v | tr -d '\r\n')"
    existing="${existing#v}"
    if [[ "${existing}" != "${NODE_VERSION}" ]]; then
      die "'${NODEENV_DIR}' already contains Node.js ${existing}, but CODEX_NODE_VERSION=${NODE_VERSION}. Delete '${NODEENV_DIR}' or set CODEX_NODE_VERSION=${existing}."
    fi
    return 0
  fi

  if [[ -e "${NODEENV_DIR}" ]]; then
    die "found '${NODEENV_DIR}', but it does not contain an executable node/npm. Delete it and re-run the installer."
  fi

  log "Installing Node.js ${NODE_VERSION} into ${NODEENV_DIR} ..."
  "${UV_BIN}" tool run --from nodeenv nodeenv -n "${NODE_VERSION}" "${NODEENV_DIR}"

  if [[ ! -x "${NODEENV_NODE}" ]]; then
    die "nodeenv completed, but '${NODEENV_NODE}' is missing or not executable."
  fi
  if [[ ! -x "${NODEENV_NPM}" ]]; then
    die "nodeenv completed, but '${NODEENV_NPM}' is missing or not executable."
  fi
}

bootstrap_uv
install_python
sync_python_deps
provision_video_runtime_deps

log "Installing frontend dependencies (npm ci) ..."
ensure_nodeenv
[[ -f "${ROOT_DIR}/apps/interface/package-lock.json" ]] || die "lock-preserving frontend install requires ${ROOT_DIR}/apps/interface/package-lock.json"
log "node: $("${NODEENV_NODE}" -v)  npm: $("${NODEENV_NPM}" -v)"
(cd "${ROOT_DIR}/apps/interface" && "${NODEENV_NPM}" ci --cache "${NPM_CACHE_DIR}" --no-audit --no-fund)
if [[ ! -f "${ROOT_DIR}/apps/interface/node_modules/vite/package.json" ]]; then
  die "npm ci completed, but apps/interface/node_modules/vite/package.json is missing. Run: (cd apps/interface && \"${NODEENV_NPM}\" ci)"
fi

echo ""
log "Done."
log "Run: ./run-webui.sh"
