#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${CODEX_VENV_DIR:-${ROOT_DIR}/.venv}"
PY_BIN="${PYTHON:-${VENV_DIR}/bin/python}"
TUI_ENTRYPOINT="${ROOT_DIR}/apps/docker_tui_launcher.py"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<EOF
Usage: ./run-webui-docker.sh [docker-tui args] [run-webui args]

Docker TUI args:
  --tui               Force interactive terminal configuration.
  --no-tui            Skip interactive terminal configuration.
  --non-interactive   Disable prompts (headless-safe).
  --configure-only    Persist configuration and exit.

Any extra arguments are forwarded to run-webui.sh through apps/docker_tui_launcher.py.
EOF
  exit 0
fi

if [[ ! -x "${PY_BIN}" ]]; then
  echo "Error: expected Python at '${PY_BIN}'." >&2
  echo "Run: ./install-webui.sh" >&2
  exit 1
fi

if [[ ! -f "${TUI_ENTRYPOINT}" ]]; then
  echo "Error: Docker TUI launcher not found: '${TUI_ENTRYPOINT}'." >&2
  exit 1
fi

export CODEX_ROOT="${ROOT_DIR}"
export PYTHONPATH="${ROOT_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

exec "${PY_BIN}" "${TUI_ENTRYPOINT}" "$@"
