"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Services tab for the Tk launcher.
Shows API/UI status and provides controls for start/restart/stop/kill, backed by `CodexServiceHandle`.

Symbols (top-level; keep in sync; no ghosts):
- `ServicesTab` (class): Services tab view/controller for the launcher GUI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict

from apps.launcher.services import ServiceStatus

from ..controller import LauncherController


@dataclass(slots=True)
class _ServiceCard:
    status_var: tk.StringVar
    info_var: tk.StringVar
    status_label: ttk.Label
    start_btn: ttk.Button
    restart_btn: ttk.Button
    stop_btn: ttk.Button
    kill_btn: ttk.Button


class ServicesTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        run_in_thread: Callable[[str, Callable[[], None]], None],
        set_status: Callable[[str], None],
    ) -> None:
        self._controller = controller
        self._run_in_thread = run_in_thread
        self._set_status = set_status

        self.frame: ttk.Frame | None = None
        self._external_terminal_var = tk.BooleanVar(value=bool(controller.store.meta.external_terminal))
        self._cards: Dict[str, _ServiceCard] = {}

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(0, weight=1)

        header = ttk.LabelFrame(frame, text="  Services  ", padding=14)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        header.columnconfigure(1, weight=1)

        if self._controller.external_terminal_supported:
            ttk.Checkbutton(
                header,
                text="Launch in external terminal (Windows)",
                variable=self._external_terminal_var,
                command=self._on_toggle_external_terminal,
            ).grid(row=0, column=0, sticky="w")
        else:
            ttk.Label(header, text="External terminal: Windows only", style="Muted.TLabel").grid(row=0, column=0, sticky="w")
            self._external_terminal_var.set(False)

        btns = ttk.Frame(header)
        btns.grid(row=0, column=1, sticky="e")
        ttk.Button(btns, text="▶ Start All", command=self._start_all).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="⏹ Stop All", command=self._stop_all).pack(side="left")

        row = 1
        for svc_name in ("API", "UI"):
            card_frame = ttk.LabelFrame(frame, text=f"  {svc_name}  ", padding=16)
            card_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 10))
            card_frame.columnconfigure(1, weight=1)

            status_var = tk.StringVar(value="STOPPED")
            info_var = tk.StringVar(value="")

            ttk.Label(card_frame, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 10))
            status_lbl = ttk.Label(card_frame, textvariable=status_var, style="Status.Stopped.TLabel")
            status_lbl.grid(row=0, column=1, sticky="w")
            ttk.Label(card_frame, textvariable=info_var, style="Muted.TLabel").grid(row=0, column=2, sticky="e")

            btn_row = ttk.Frame(card_frame)
            btn_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))

            start_btn = ttk.Button(btn_row, text="▶ Start", width=12, command=lambda n=svc_name: self._start(n))
            restart_btn = ttk.Button(btn_row, text="🔄 Restart", width=12, command=lambda n=svc_name: self._restart(n))
            stop_btn = ttk.Button(btn_row, text="⏹ Stop", width=12, command=lambda n=svc_name: self._stop(n))
            kill_btn = ttk.Button(btn_row, text="❗ Kill", width=12, command=lambda n=svc_name: self._kill(n))

            start_btn.pack(side="left", padx=(0, 8))
            restart_btn.pack(side="left", padx=(0, 8))
            stop_btn.pack(side="left", padx=(0, 8))
            kill_btn.pack(side="left")

            self._cards[svc_name] = _ServiceCard(
                status_var=status_var,
                info_var=info_var,
                status_label=status_lbl,
                start_btn=start_btn,
                restart_btn=restart_btn,
                stop_btn=stop_btn,
                kill_btn=kill_btn,
            )
            row += 1

        self.frame = frame
        return frame

    def reload(self) -> None:
        self._external_terminal_var.set(bool(self._controller.store.meta.external_terminal))

    def refresh(self) -> None:
        now = time.time()
        for svc_name, card in self._cards.items():
            svc = self._controller.services[svc_name]
            status = svc.status
            pid = svc.pid or 0
            uptime = "-"
            if svc.started_at and status == ServiceStatus.RUNNING:
                uptime = f"{int(now - svc.started_at)}s"
            last_exit = svc.last_exit_code

            card.status_var.set(status.value.upper())
            info_bits = []
            if pid:
                info_bits.append(f"PID {pid}")
            if uptime != "-":
                info_bits.append(f"Uptime {uptime}")
            if last_exit is not None and status != ServiceStatus.RUNNING:
                info_bits.append(f"Last exit {last_exit}")
            card.info_var.set(" | ".join(info_bits))

            if status == ServiceStatus.RUNNING:
                card.status_label.configure(style="Status.Running.TLabel")
            elif status == ServiceStatus.ERROR:
                card.status_label.configure(style="Status.Error.TLabel")
            else:
                card.status_label.configure(style="Status.Stopped.TLabel")

            card.start_btn.configure(state=("disabled" if status == ServiceStatus.RUNNING else "normal"))
            card.restart_btn.configure(state=("normal" if status == ServiceStatus.RUNNING else "disabled"))
            card.stop_btn.configure(state=("normal" if status == ServiceStatus.RUNNING else "disabled"))
            card.kill_btn.configure(state=("normal" if status == ServiceStatus.RUNNING else "disabled"))

    def _external_terminal(self) -> bool:
        return bool(self._external_terminal_var.get()) and self._controller.external_terminal_supported

    def _on_toggle_external_terminal(self) -> None:
        enabled = bool(self._external_terminal_var.get())
        try:
            self._controller.persist_external_terminal(enabled)
        except Exception as exc:
            messagebox.showerror("Launcher Error", f"Failed to persist external terminal setting:\n\n{exc}")
            self._external_terminal_var.set(bool(self._controller.store.meta.external_terminal))

    def _start_all(self) -> None:
        self._set_status("Starting services…")
        external = self._external_terminal()
        self._run_in_thread("Start All", lambda: self._controller.start_all(external_terminal=external))

    def _stop_all(self) -> None:
        self._set_status("Stopping services…")
        self._run_in_thread("Stop All", lambda: self._controller.stop_all(wait=5.0))

    def _start(self, name: str) -> None:
        self._set_status(f"{name} starting…")
        external = self._external_terminal()
        self._run_in_thread(f"Start {name}", lambda: self._controller.start_service(name, external_terminal=external))

    def _restart(self, name: str) -> None:
        self._set_status(f"{name} restarting…")
        external = self._external_terminal()
        self._run_in_thread(f"Restart {name}", lambda: self._controller.restart_service(name, external_terminal=external))

    def _stop(self, name: str) -> None:
        self._set_status(f"{name} stopping…")
        self._run_in_thread(f"Stop {name}", lambda: self._controller.stop_service(name, wait=5.0))

    def _kill(self, name: str) -> None:
        if not messagebox.askyesno("Confirm kill", f"Force kill {name}?"):
            return
        self._set_status(f"{name} killing…")
        self._run_in_thread(f"Kill {name}", lambda: self._controller.kill_service(name, wait=5.0))
