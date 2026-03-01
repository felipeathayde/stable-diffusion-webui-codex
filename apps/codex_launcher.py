"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GUI launcher entrypoint for Codex (Tk).
This file is the stable entrypoint used by `run-webui.bat`. The implementation lives under `apps.launcher.gui_tk`.

Symbols (top-level; keep in sync; no ghosts):
- `_validate_launcher_args` (function): Validates launcher CLI arguments and warns on unsupported extras.
- `main` (function): Starts the Tk GUI launcher.
"""

from __future__ import annotations

import argparse
import os
import sys


def _validate_launcher_args(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog="codex_launcher.py",
        allow_abbrev=False,
        add_help=True,
    )
    _known, unknown = parser.parse_known_args(argv)
    if unknown:
        unknown_args = " ".join(unknown)
        print(f"[launcher] Warning: ignoring unsupported launcher argument(s): {unknown_args}", file=sys.stderr)


def main() -> None:
    _validate_launcher_args(sys.argv[1:])
    if not str(os.environ.get("CODEX_ROOT", "") or "").strip():
        raise EnvironmentError("CODEX_ROOT not set. Launch via run-webui.bat (Windows) or set CODEX_ROOT explicitly.")

    from apps.launcher.gui_tk import main as _run

    _run()


if __name__ == "__main__":
    main()
