#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEFAULT_VENV_DIR="${ROOT_DIR}/.venv"
if [[ -d "${DEFAULT_VENV_DIR}" ]]; then
  VENV_DIR="${CODEX_VENV_DIR:-$DEFAULT_VENV_DIR}"
else
  VENV_DIR="${CODEX_VENV_DIR:-$HOME/.venv}"
fi
PY_BIN="${PYTHON:-$VENV_DIR/bin/python}"
API_ENTRYPOINT="${ROOT_DIR}/apps/backend/interfaces/api/run_api.py"
UI_DIR="${ROOT_DIR}/apps/interface"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: ./run-webui.sh

Starts:
  - Backend API (FastAPI) from ${API_ENTRYPOINT}
  - Frontend UI (Vite) from ${UI_DIR}

Environment overrides:
  - CODEX_VENV_DIR   (default: \$HOME/.venv)
  - PYTHON           (default: \$CODEX_VENV_DIR/bin/python)
  - API_PORT_OVERRIDE / API_PORT / WEB_PORT (advanced; ports are auto-paired when unset)
EOF
  exit 0
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "Error: expected Python at '${PY_BIN}'." >&2
  echo "Set PYTHON to an executable or create the venv at '${VENV_DIR}'." >&2
  exit 1
fi

if [[ ! -f "${API_ENTRYPOINT}" ]]; then
  echo "Error: backend entrypoint not found: '${API_ENTRYPOINT}'." >&2
  exit 1
fi

if [[ ! -d "${UI_DIR}" ]]; then
  echo "Error: frontend directory not found: '${UI_DIR}'." >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1 || ! command -v npm >/dev/null 2>&1; then
  echo "Error: missing 'node' and/or 'npm'. Install Node.js (>=18) and npm." >&2
  exit 1
fi

if [[ ! -d "${UI_DIR}/node_modules" ]]; then
  echo "Error: '${UI_DIR}/node_modules' missing." >&2
  echo "Run: (cd '${UI_DIR}' && npm install)" >&2
  exit 1
fi

export CODEX_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export FORCE_COLOR="${FORCE_COLOR:-1}"

is_uint() {
  [[ "${1:-}" =~ ^[0-9]+$ ]]
}

assert_port() {
  local label="$1"
  local value="$2"
  if ! is_uint "${value}"; then
    echo "Error: ${label} must be an integer; got '${value}'." >&2
    exit 1
  fi
  if (( value < 1 || value > 65535 )); then
    echo "Error: ${label} must be in 1..65535; got '${value}'." >&2
    exit 1
  fi
}

port_free() {
  local port="$1"
  "${PY_BIN}" - "$port" <<'PY'
import errno
import socket
import sys

port = int(sys.argv[1])
targets = [
    (socket.AF_INET, ("0.0.0.0", port)),
    (socket.AF_INET, ("127.0.0.1", port)),
    (socket.AF_INET6, ("::", port, 0, 0)),
    (socket.AF_INET6, ("::1", port, 0, 0)),
]
for family, addr in targets:
    try:
        s = socket.socket(family, socket.SOCK_STREAM)
    except OSError:
        continue
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(addr)
    except OSError as exc:
        if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
            continue
        raise SystemExit(1)
    finally:
        try:
            s.close()
        except Exception:
            pass
raise SystemExit(0)
PY
}

pick_ports() {
  local user_api="${API_PORT_OVERRIDE:-${API_PORT:-}}"
  local user_web="${WEB_PORT:-}"

  if [[ -n "${user_api}" && -n "${user_web}" ]]; then
    assert_port "API_PORT_OVERRIDE/API_PORT" "${user_api}"
    assert_port "WEB_PORT" "${user_web}"
    if ! port_free "${user_api}" || ! port_free "${user_web}"; then
      echo "Error: requested ports are busy: api=${user_api} web=${user_web}." >&2
      exit 1
    fi
    API_PORT_OVERRIDE="${user_api}"
    WEB_PORT="${user_web}"
    API_PORT="${user_api}"
    return 0
  fi

  if [[ -n "${user_api}" ]]; then
    assert_port "API_PORT_OVERRIDE/API_PORT" "${user_api}"
    local derived_web=$(( user_api + 10 ))
    assert_port "WEB_PORT (derived)" "${derived_web}"
    if ! port_free "${user_api}" || ! port_free "${derived_web}"; then
      echo "Error: requested ports are busy: api=${user_api} web=${derived_web}." >&2
      exit 1
    fi
    API_PORT_OVERRIDE="${user_api}"
    WEB_PORT="${derived_web}"
    API_PORT="${user_api}"
    return 0
  fi

  if [[ -n "${user_web}" ]]; then
    assert_port "WEB_PORT" "${user_web}"
    local derived_api=$(( user_web - 10 ))
    assert_port "API_PORT (derived)" "${derived_api}"
    if ! port_free "${derived_api}" || ! port_free "${user_web}"; then
      echo "Error: requested ports are busy: api=${derived_api} web=${user_web}." >&2
      exit 1
    fi
    API_PORT_OVERRIDE="${derived_api}"
    WEB_PORT="${user_web}"
    API_PORT="${derived_api}"
    return 0
  fi

  local candidates=(
    "7850 7860"
    "17850 17860"
    "27850 27860"
  )
  local api_port=""
  local web_port=""
  for pair in "${candidates[@]}"; do
    read -r api_port web_port <<<"${pair}"
    if port_free "${api_port}" && port_free "${web_port}"; then
      API_PORT_OVERRIDE="${api_port}"
      WEB_PORT="${web_port}"
      API_PORT="${api_port}"
      return 0
    fi
  done

  echo "Error: no free port pairs for API/UI." >&2
  echo "Tried: 7850/7860, 17850/17860, 27850/27860." >&2
  echo "Override via API_PORT_OVERRIDE and WEB_PORT." >&2
  exit 1
}

api_health_ok() {
  local port="$1"
  "${PY_BIN}" - "$port" <<'PY'
import json
import sys
from urllib.error import URLError
from urllib.request import urlopen

port = int(sys.argv[1])
try:
    with urlopen(f"http://127.0.0.1:{port}/api/health", timeout=1.0) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
except (URLError, TimeoutError, ValueError, OSError):
    raise SystemExit(1)
raise SystemExit(0 if payload.get("ok") is True else 1)
PY
}

wait_for_api() {
  local pid="$1"
  local port="$2"
  local attempts=60

  echo "[webui] Waiting for API /api/health on port ${port}..."
  for _ in $(seq 1 "${attempts}"); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Error: API process exited before becoming healthy." >&2
      wait "${pid}" || true
      exit 1
    fi
    if api_health_ok "${port}"; then
      echo "[webui] API is healthy."
      return 0
    fi
    sleep 1
  done

  echo "Error: API did not become healthy within ${attempts}s." >&2
  return 1
}

pick_ports
export API_PORT_OVERRIDE API_PORT WEB_PORT

echo "[webui] API: http://localhost:${API_PORT_OVERRIDE}"
echo "[webui]  UI: http://localhost:${WEB_PORT}"

api_pid=""
ui_pid=""

cleanup() {
  local code="${1:-0}"
  if [[ -n "${ui_pid}" ]]; then
    kill "${ui_pid}" 2>/dev/null || true
  fi
  if [[ -n "${api_pid}" ]]; then
    kill "${api_pid}" 2>/dev/null || true
  fi
  wait "${ui_pid}" 2>/dev/null || true
  wait "${api_pid}" 2>/dev/null || true
  exit "${code}"
}

trap 'cleanup 130' INT
trap 'cleanup 143' TERM

(
  cd "${ROOT_DIR}"
  "${PY_BIN}" "${API_ENTRYPOINT}"
) &
api_pid="$!"

wait_for_api "${api_pid}" "${API_PORT_OVERRIDE}"

(
  cd "${UI_DIR}"
  npm run dev -- --host
) &
ui_pid="$!"

set +e
wait -n "${api_pid}" "${ui_pid}"
status="$?"
set -e

echo "Error: API or UI exited (status=${status}). Shutting down..." >&2
cleanup "${status}"
