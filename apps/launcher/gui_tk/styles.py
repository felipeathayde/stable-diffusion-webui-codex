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
    bg0: str = "#0d1118"
    bg1: str = "#141b27"
    bg2: str = "#1b2534"
    fg0: str = "#e8eef8"
    fg_muted: str = "#9bb0c9"
    accent: str = "#57a6ff"
    accent_hover: str = "#7ebcff"
    accent_active: str = "#2f8df4"
    line: str = "#2a3a52"
    ok: str = "#50d2a0"
    warn: str = "#f0c66c"
    err: str = "#ff6f91"


def apply_style(root: tk.Tk, palette: Palette) -> None:
    style = ttk.Style(root)
    style.theme_use("clam")

    root.configure(bg=palette.bg0)

    style.configure(".", background=palette.bg0, foreground=palette.fg0, font=("Segoe UI", 10))

    style.configure("TFrame", background=palette.bg0)
    style.configure("Section.Toolbar.TFrame", background=palette.bg0)
    style.configure("TLabel", background=palette.bg0, foreground=palette.fg0)
    style.configure("Muted.TLabel", background=palette.bg0, foreground=palette.fg_muted)
    style.configure("Section.Header.TLabel", background=palette.bg0, foreground=palette.accent, font=("Segoe UI Semibold", 10))

    style.configure(
        "TNotebook",
        background=palette.bg0,
        borderwidth=1,
        relief="solid",
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.configure(
        "TNotebook.Tab",
        background=palette.bg1,
        foreground=palette.fg_muted,
        padding=[14, 8],
        borderwidth=1,
        relief="flat",
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", palette.bg2), ("active", palette.bg2)],
        foreground=[("selected", palette.fg0), ("active", palette.fg0)],
    )

    style.configure(
        "TLabelframe",
        background=palette.bg1,
        bordercolor=palette.line,
        relief="solid",
        borderwidth=1,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.configure(
        "TLabelframe.Label",
        background=palette.bg1,
        foreground=palette.accent,
        font=("Segoe UI Semibold", 9),
    )
    style.configure(
        "Service.Card.TLabelframe",
        background=palette.bg1,
        bordercolor=palette.line,
        relief="solid",
        borderwidth=1,
    )
    style.configure(
        "Service.Card.TLabelframe.Label",
        background=palette.bg1,
        foreground=palette.fg0,
        font=("Segoe UI Semibold", 10),
    )

    style.configure(
        "TButton",
        background=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        relief="solid",
        padding=[14, 7],
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "TButton",
        background=[("active", palette.accent_hover), ("pressed", palette.accent_active), ("disabled", palette.bg1)],
        foreground=[("active", palette.bg0), ("pressed", palette.bg0), ("disabled", palette.fg_muted)],
    )

    style.configure("Filter.TButton", background=palette.bg1, foreground=palette.fg0, padding=[10, 4], font=("Segoe UI", 9))
    style.map(
        "Filter.TButton",
        background=[("active", palette.bg2), ("disabled", palette.bg1)],
        foreground=[("active", palette.fg0), ("disabled", palette.fg_muted)],
    )
    style.configure(
        "Filter.Selected.TButton",
        background=palette.accent,
        foreground=palette.bg0,
        padding=[10, 4],
        font=("Segoe UI Semibold", 9),
    )
    style.map(
        "Filter.Selected.TButton",
        background=[("active", palette.accent_hover), ("disabled", palette.bg1)],
        foreground=[("active", palette.bg0), ("disabled", palette.fg_muted)],
    )

    style.configure("TCheckbutton", background=palette.bg0, foreground=palette.fg0)
    style.map(
        "TCheckbutton",
        background=[("active", palette.bg0), ("disabled", palette.bg0)],
        foreground=[("disabled", palette.fg_muted)],
    )
    style.configure("Toggle.TCheckbutton", background=palette.bg0, foreground=palette.fg0, font=("Segoe UI", 9))
    style.map(
        "Toggle.TCheckbutton",
        background=[("active", palette.bg0), ("disabled", palette.bg0)],
        foreground=[("disabled", palette.fg_muted)],
    )

    style.configure(
        "TEntry",
        fieldbackground=palette.bg2,
        background=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        insertcolor=palette.fg0,
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", palette.bg1)],
        foreground=[("disabled", palette.fg_muted)],
    )

    style.configure(
        "TCombobox",
        fieldbackground=palette.bg2,
        background=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        arrowcolor=palette.fg0,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", palette.bg2), ("disabled", palette.bg1)],
        foreground=[("readonly", palette.fg0), ("disabled", palette.fg_muted)],
    )
    root.option_add("*TCombobox*Listbox.background", palette.bg2)
    root.option_add("*TCombobox*Listbox.foreground", palette.fg0)
    root.option_add("*TCombobox*Listbox.selectBackground", palette.accent)
    root.option_add("*TCombobox*Listbox.selectForeground", palette.bg0)

    style.configure(
        "Treeview",
        background=palette.bg2,
        fieldbackground=palette.bg2,
        foreground=palette.fg0,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        font=("Segoe UI", 9),
    )
    style.configure(
        "Treeview.Heading",
        background=palette.bg1,
        foreground=palette.accent,
        bordercolor=palette.line,
        lightcolor=palette.line,
        darkcolor=palette.line,
        relief="solid",
        font=("Segoe UI Semibold", 9),
    )
    style.map("Treeview", background=[("selected", palette.accent)], foreground=[("selected", palette.bg0)])

    style.configure(
        "Vertical.TScrollbar",
        background=palette.bg2,
        troughcolor=palette.bg1,
        bordercolor=palette.line,
        arrowcolor=palette.fg_muted,
        lightcolor=palette.line,
        darkcolor=palette.line,
    )
    style.map(
        "Vertical.TScrollbar",
        background=[("active", palette.bg2), ("pressed", palette.accent_active)],
        arrowcolor=[("active", palette.fg0), ("pressed", palette.bg0)],
    )

    style.configure("Service.Info.TLabel", background=palette.bg1, foreground=palette.fg_muted, font=("Segoe UI", 9))
    style.configure("Service.Endpoint.TLabel", background=palette.bg1, foreground=palette.fg0, font=("Consolas", 9))

    style.configure("Status.Running.TLabel", background=palette.bg1, foreground=palette.ok, font=("Segoe UI Semibold", 10))
    style.configure("Status.Stopped.TLabel", background=palette.bg1, foreground=palette.warn, font=("Segoe UI Semibold", 10))
    style.configure("Status.Error.TLabel", background=palette.bg1, foreground=palette.err, font=("Segoe UI Semibold", 10))

    style.configure("Health.Ok.TLabel", background=palette.bg1, foreground=palette.ok, font=("Segoe UI", 9))
    style.configure("Health.Stopped.TLabel", background=palette.bg1, foreground=palette.fg_muted, font=("Segoe UI", 9))
    style.configure("Health.Error.TLabel", background=palette.bg1, foreground=palette.err, font=("Segoe UI", 9))
