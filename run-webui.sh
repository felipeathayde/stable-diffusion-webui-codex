#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_BIN="${PYTHON:-python3}"

if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  if command -v python >/dev/null 2>&1; then
    PY_BIN="python"
  else
    echo "Error: unable to locate a Python interpreter. Set PYTHON env var to an executable." >&2
    exit 1
  fi
fi

exec "${PY_BIN}" "${ROOT_DIR}/tools/tui_bios.py" "$@"
