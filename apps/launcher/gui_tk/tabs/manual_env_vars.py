"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Manual API environment overlay tab for the Tk launcher.
Provides a multiline editor for API-only env overlays (`KEY=VALUE` per line) that are applied on next API start/restart when the runtime advanced toggle is enabled.

Symbols (top-level; keep in sync; no ghosts):
- `ManualEnvVarsTab` (class): Tab controller for manual API env overlay editor and reload/save integration.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable

from apps.launcher.profiles import DEFAULT_MANUAL_API_ENV_TEXT

from ..controller import LauncherController
from ..widgets import ScrollableFrame


class ManualEnvVarsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        canvas_bg: str,
        mark_changed: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._canvas_bg = str(canvas_bg)
        self._mark_changed = mark_changed
        self.frame: ttk.Frame | None = None
        self._editor: tk.Text | None = None
        self._editing_programmatically = False

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True, padx=8, pady=8)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        section = ttk.LabelFrame(body, text="  Manual Env Vars (API only)  ", padding=14)
        section.grid(row=0, column=0, sticky="nsew")
        section.columnconfigure(0, weight=1)
        section.rowconfigure(2, weight=1)

        ttk.Label(
            section,
            text=(
                "Format: one env var per line as KEY=VALUE.\n"
                "Blank lines and lines starting with # are ignored.\n"
                "Stored in plain text in .sangoi/launcher/meta.json.\n"
                "Default template:\n"
                f"{DEFAULT_MANUAL_API_ENV_TEXT}"
            ),
            style="Muted.TLabel",
            justify="left",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        editor_frame = ttk.Frame(section)
        editor_frame.grid(row=2, column=0, sticky="nsew")
        editor_frame.columnconfigure(0, weight=1)
        editor_frame.rowconfigure(0, weight=1)

        editor = tk.Text(
            editor_frame,
            width=98,
            height=20,
            wrap="none",
            undo=True,
        )
        editor.grid(row=0, column=0, sticky="nsew")
        editor.bind("<<Modified>>", self._on_editor_modified)
        self._editor = editor

        yscroll = ttk.Scrollbar(editor_frame, orient="vertical", command=editor.yview)
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll = ttk.Scrollbar(editor_frame, orient="horizontal", command=editor.xview)
        xscroll.grid(row=1, column=0, sticky="ew")
        editor.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)

        self.frame = frame
        self.reload()
        return frame

    def reload(self) -> None:
        current = str(getattr(self._controller.store.meta, "manual_api_env_text", DEFAULT_MANUAL_API_ENV_TEXT) or "")
        self._set_editor_text(current)

    def _set_editor_text(self, value: str) -> None:
        if self._editor is None:
            return
        self._editing_programmatically = True
        try:
            self._editor.delete("1.0", tk.END)
            if value:
                self._editor.insert("1.0", str(value))
            self._editor.edit_modified(False)
        finally:
            self._editing_programmatically = False

    def _on_editor_modified(self, _event: tk.Event[tk.Text]) -> None:
        editor = self._editor
        if editor is None:
            return
        if self._editing_programmatically:
            editor.edit_modified(False)
            return
        if not editor.edit_modified():
            return
        editor.edit_modified(False)
        content = editor.get("1.0", "end-1c")
        self._controller.store.meta.manual_api_env_text = str(content)
        self._mark_changed()
