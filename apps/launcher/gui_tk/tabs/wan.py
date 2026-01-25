"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN info tab for the Tk launcher.
This tab is intentionally informational: WAN settings are request/payload-driven and configured via the Web UI.

Symbols (top-level; keep in sync; no ghosts):
- `WanTab` (class): Static informational WAN tab.
"""

from __future__ import annotations

from tkinter import ttk


class WanTab:
    def __init__(self) -> None:
        self.frame: ttk.Frame | None = None

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        msg = (
            "WAN runtime settings are payload/options-driven.\n"
            "Use the Web UI (WAN tab) to configure per-run WAN options.\n\n"
            "This launcher only provides global service/process controls."
        )
        ttk.Label(frame, text=msg, justify="left").pack(anchor="w", padx=16, pady=16)
        self.frame = frame
        return frame

    def reload(self) -> None:
        return

