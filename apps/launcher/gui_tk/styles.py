"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tk/ttk styles used by the Codex launcher GUI.
Centralizes palette values and ttk.Style configuration so the app/tabs don’t hardcode colors everywhere.

Symbols (top-level; keep in sync; no ghosts):
- `Palette` (dataclass): Color palette used by the GUI.
- `apply_style` (function): Applies ttk theme/style configuration to a Tk root window.
"""

from __future__ import annotations

from dataclasses import dataclass
import tkinter as tk
from tkinter import ttk


@dataclass(frozen=True, slots=True)
class Palette:
    bg0: str = "#1e1e2e"
    bg1: str = "#2d2d3d"
    bg2: str = "#11111b"
    fg0: str = "#cdd6f4"
    fg_muted: str = "#a6adc8"
    accent: str = "#89b4fa"
    accent_hover: str = "#b4befe"
    ok: str = "#a6e3a1"
    warn: str = "#f9e2af"
    err: str = "#f38ba8"


def apply_style(root: tk.Tk, palette: Palette) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(bg=palette.bg0)

    style.configure(".", background=palette.bg0, foreground=palette.fg0, font=("Segoe UI", 11))

    style.configure("TFrame", background=palette.bg0)
    style.configure("TLabel", background=palette.bg0, foreground=palette.fg0)
    style.configure("Muted.TLabel", background=palette.bg0, foreground=palette.fg_muted)

    style.configure("TNotebook", background=palette.bg0, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=palette.bg1,
        foreground=palette.fg0,
        padding=[12, 7],
        font=("Segoe UI", 11, "bold"),
    )
    style.map("TNotebook.Tab", background=[("selected", palette.accent)], foreground=[("selected", palette.bg0)])

    style.configure("TLabelframe", background=palette.bg0, foreground=palette.fg0)
    style.configure("TLabelframe.Label", background=palette.bg0, foreground=palette.accent, font=("Segoe UI", 11, "bold"))

    style.configure("TButton", background=palette.bg1, foreground=palette.fg0, padding=[16, 8], font=("Segoe UI", 11))
    style.map(
        "TButton",
        background=[("active", palette.accent_hover), ("pressed", palette.accent)],
        foreground=[("active", palette.bg0), ("pressed", palette.bg0)],
    )

    style.configure("Filter.TButton", background=palette.bg1, foreground=palette.fg0, padding=[10, 4], font=("Segoe UI", 10, "bold"))
    style.map("Filter.TButton", background=[("active", palette.accent_hover)], foreground=[("active", palette.bg0)])
    style.configure(
        "Filter.Selected.TButton",
        background=palette.accent,
        foreground=palette.bg0,
        padding=[10, 4],
        font=("Segoe UI", 10, "bold"),
    )
    style.map("Filter.Selected.TButton", background=[("active", palette.accent_hover)], foreground=[("active", palette.bg0)])

    style.configure("TCheckbutton", background=palette.bg0, foreground=palette.fg0)
    style.map("TCheckbutton", background=[("active", palette.bg0)])

    style.configure(
        "TEntry",
        fieldbackground=palette.bg1,
        background=palette.bg1,
        foreground=palette.fg0,
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", palette.bg0)],
        foreground=[("disabled", palette.fg_muted)],
    )

    style.configure(
        "TCombobox",
        fieldbackground=palette.bg1,
        background=palette.bg1,
        foreground=palette.fg0,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.bg1), ("disabled", palette.bg0)],
        foreground=[("readonly", palette.fg0), ("disabled", palette.fg_muted)],
    )
    root.option_add("*TCombobox*Listbox.background", palette.bg1)
    root.option_add("*TCombobox*Listbox.foreground", palette.fg0)
    root.option_add("*TCombobox*Listbox.selectBackground", palette.accent)
    root.option_add("*TCombobox*Listbox.selectForeground", palette.bg0)

    style.configure(
        "Treeview",
        background=palette.bg1,
        fieldbackground=palette.bg1,
        foreground=palette.fg0,
        bordercolor=palette.bg0,
        lightcolor=palette.bg0,
        darkcolor=palette.bg0,
        font=("Segoe UI", 10),
    )
    style.configure(
        "Treeview.Heading",
        background=palette.bg0,
        foreground=palette.accent,
        font=("Segoe UI", 10, "bold"),
    )
    style.map("Treeview", background=[("selected", palette.accent)], foreground=[("selected", palette.bg0)])

    style.configure("Status.Running.TLabel", background=palette.bg0, foreground=palette.ok, font=("Segoe UI", 12, "bold"))
    style.configure("Status.Stopped.TLabel", background=palette.bg0, foreground=palette.warn, font=("Segoe UI", 12, "bold"))
    style.configure("Status.Error.TLabel", background=palette.bg0, foreground=palette.err, font=("Segoe UI", 12, "bold"))
