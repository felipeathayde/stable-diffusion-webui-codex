"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small reusable Tk widget helpers for the launcher GUI.
Provides scrollable frames and repeated layout helpers (section headers, help text) used across tabs.

Symbols (top-level; keep in sync; no ghosts):
- `ScrollableFrame` (class): Canvas-based scrollable container with platform-aware mousewheel binding.
- `add_section_header` (function): Adds a section header label and returns next grid row.
- `add_help` (function): Adds a help/description label and returns next grid row.
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import ttk


class ScrollableFrame(ttk.Frame):
    """Canvas-based scrollable container with a single inner ttk.Frame."""

    def __init__(self, parent: tk.Misc, *, canvas_bg: str) -> None:
        super().__init__(parent)
        self._canvas = tk.Canvas(self, bg=canvas_bg, highlightthickness=0, bd=0, relief="flat")
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.inner = ttk.Frame(self._canvas)

        self._canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self.inner.bind("<Configure>", lambda _e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self.inner.bind("<Configure>", lambda _e: self._refresh_mousewheel_bindings(), add="+")
        self._bound_mousewheel_widgets: set[str] = set()

        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")
        self._refresh_mousewheel_bindings()

    def _refresh_mousewheel_bindings(self) -> None:
        for widget in self._iter_widgets():
            widget_key = str(widget)
            if widget_key in self._bound_mousewheel_widgets:
                continue
            if os.name == "nt":
                widget.bind("<MouseWheel>", self._on_mousewheel_windows, add="+")
            elif os.name == "darwin":
                widget.bind("<MouseWheel>", self._on_mousewheel_macos, add="+")
            else:
                widget.bind("<Button-4>", self._on_mousewheel_linux, add="+")
                widget.bind("<Button-5>", self._on_mousewheel_linux, add="+")
            self._bound_mousewheel_widgets.add(widget_key)

    def _iter_widgets(self) -> list[tk.Widget]:
        collected: list[tk.Widget] = []
        stack: list[tk.Widget] = [self, self._canvas, self.inner]
        seen: set[str] = set()
        while stack:
            widget = stack.pop()
            key = str(widget)
            if key in seen:
                continue
            seen.add(key)
            collected.append(widget)
            try:
                children = list(widget.winfo_children())
            except Exception:
                children = []
            stack.extend(children)
        return collected

    def _on_mousewheel_windows(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0) or 0)
        if not delta:
            return
        self._canvas.yview_scroll(int(-1 * (delta / 120)), "units")

    def _on_mousewheel_macos(self, event: tk.Event) -> None:
        delta = int(getattr(event, "delta", 0) or 0)
        if not delta:
            return
        # On macOS delta is already small; treat it as a pixel-ish value.
        self._canvas.yview_scroll(int(-1 * delta), "units")

    def _on_mousewheel_linux(self, event: tk.Event) -> None:
        num = int(getattr(event, "num", 0) or 0)
        if num == 4:
            self._canvas.yview_scroll(-1, "units")
        elif num == 5:
            self._canvas.yview_scroll(1, "units")


def add_section_header(parent: ttk.Frame, row: int, title: str) -> int:
    ttk.Label(parent, text=title, style="TLabelframe.Label").grid(row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(16, 8))
    return row + 1


def add_help(parent: ttk.Frame, row: int, text: str) -> int:
    ttk.Label(parent, text=text, justify="left", style="Muted.TLabel").grid(row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))
    return row + 1
