"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Diagnostics tab for the Tk launcher.
Shows environment preflight checks and exposes debug/logging env flags used by the backend.

Symbols (top-level; keep in sync; no ghosts):
- `DiagnosticsTab` (class): Diagnostics tab (checks + debug flags + log levels).
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Iterable

from apps.launcher.checks import CodexLaunchCheck
from apps.launcher.settings import BoolSetting, IntSetting, SettingValidationError

from ..controller import LauncherController


class DiagnosticsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        mark_changed: Callable[[], None],
        run_checks_async: Callable[[], None],
    ) -> None:
        self._controller = controller
        self._mark_changed = mark_changed
        self._run_checks_async = run_checks_async

        self.frame: ttk.Frame | None = None

        self._checks_tree: ttk.Treeview | None = None

        self._debug_flags: Dict[str, tk.BooleanVar] = {}
        self._log_levels: Dict[str, tk.BooleanVar] = {}

        self._var_cfg_delta_n = tk.StringVar()
        self._var_trace_max = tk.StringVar()
        self._var_dump_path = tk.StringVar()
        self._var_log_file = tk.BooleanVar()

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        frame.columnconfigure(0, weight=1)

        checks_box = ttk.LabelFrame(frame, text="  Environment Checks  ", padding=14)
        checks_box.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        checks_box.columnconfigure(0, weight=1)

        tree = ttk.Treeview(checks_box, columns=("name", "ok", "detail"), show="headings", height=5)
        tree.heading("name", text="Check")
        tree.heading("ok", text="OK")
        tree.heading("detail", text="Detail")
        tree.column("name", width=160, anchor="w")
        tree.column("ok", width=60, anchor="center")
        tree.column("detail", width=600, anchor="w")
        tree.grid(row=0, column=0, sticky="ew")
        ttk.Button(checks_box, text="↻ Re-run checks", command=self._run_checks_async).grid(row=1, column=0, sticky="e", pady=(10, 0))
        self._checks_tree = tree

        diag = ttk.LabelFrame(frame, text="  Debug + Logging  ", padding=14)
        diag.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        diag.columnconfigure(0, weight=1)
        diag.columnconfigure(1, weight=1)

        dbg_col = ttk.Frame(diag)
        dbg_col.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        ttk.Label(dbg_col, text="Debug Flags", style="TLabelframe.Label").grid(row=0, column=0, sticky="w", pady=(0, 6))

        debug_flags = [
            ("CODEX_DEBUG_COND", "Conditioning Debug"),
            ("CODEX_LOG_SAMPLER", "Sampler Verbose Logs"),
            ("CODEX_LOG_CFG_DELTA", "CFG Delta Logs (requires Sampler Verbose Logs)"),
            ("CODEX_LOG_SIGMAS", "Sigma Ladder Logs"),
            ("CODEX_TRACE_DEBUG", "Trace Debug (very verbose)"),
            ("CODEX_PIPELINE_DEBUG", "Pipeline Debug"),
            ("CODEX_DUMP_LATENTS", "Dump Latents"),
            ("CODEX_TIMELINE", "Timeline Tracer (TVA-style execution timeline)"),
            ("CODEX_ZIMAGE_DIFFUSERS_BYPASS", "Z Image: Use Diffusers Pipeline (bypasses Codex sampler)"),
        ]
        r = 1
        for key, label in debug_flags:
            var = tk.BooleanVar(value=BoolSetting(key, default=False).get(self._controller.store.env))
            self._debug_flags[key] = var
            ttk.Checkbutton(dbg_col, text=label, variable=var, command=lambda k=key: self._set_bool(k, self._debug_flags[k].get())).grid(
                row=r, column=0, sticky="w", pady=2
            )
            r += 1

        # Debug numeric/text
        self._var_cfg_delta_n.set(str(self._controller.store.env.get("CODEX_LOG_CFG_DELTA_N", "2") or "2"))
        self._var_trace_max.set(str(self._controller.store.env.get("CODEX_TRACE_DEBUG_MAX_PER_FUNC", "50") or "50"))
        self._var_dump_path.set(str(self._controller.store.env.get("CODEX_DUMP_LATENTS_PATH", "") or ""))

        r = self._add_entry(
            dbg_col,
            r + 1,
            label="CFG Delta Steps (N):",
            var=self._var_cfg_delta_n,
            width=10,
            on_change=lambda: self._set_text("CODEX_LOG_CFG_DELTA_N", self._var_cfg_delta_n.get()),
        )
        r = self._add_entry(
            dbg_col,
            r,
            label="Trace max / func:",
            var=self._var_trace_max,
            width=10,
            on_change=lambda: self._set_text("CODEX_TRACE_DEBUG_MAX_PER_FUNC", self._var_trace_max.get()),
        )
        _ = self._add_entry(
            dbg_col,
            r,
            label="Dump latents path:",
            var=self._var_dump_path,
            width=36,
            on_change=lambda: self._set_text("CODEX_DUMP_LATENTS_PATH", self._var_dump_path.get()),
        )

        log_col = ttk.Frame(diag)
        log_col.grid(row=0, column=1, sticky="nsew")
        ttk.Label(log_col, text="Log Levels", style="TLabelframe.Label").grid(row=0, column=0, sticky="w", pady=(0, 6))

        log_defaults = {
            "CODEX_LOG_DEBUG": False,
            "CODEX_LOG_INFO": True,
            "CODEX_LOG_WARNING": True,
            "CODEX_LOG_ERROR": True,
        }
        r = 1
        for key, label, default in (
            ("CODEX_LOG_DEBUG", "DEBUG (verbose)", log_defaults["CODEX_LOG_DEBUG"]),
            ("CODEX_LOG_INFO", "INFO", log_defaults["CODEX_LOG_INFO"]),
            ("CODEX_LOG_WARNING", "WARNING", log_defaults["CODEX_LOG_WARNING"]),
            ("CODEX_LOG_ERROR", "ERROR", log_defaults["CODEX_LOG_ERROR"]),
        ):
            var = tk.BooleanVar(value=BoolSetting(key, default=bool(default)).get(self._controller.store.env))
            self._log_levels[key] = var
            ttk.Checkbutton(log_col, text=label, variable=var, command=lambda k=key: self._set_bool(k, self._log_levels[k].get())).grid(
                row=r, column=0, sticky="w", pady=2
            )
            r += 1

        self._var_log_file.set(bool(str(self._controller.store.env.get("CODEX_LOG_FILE", "") or "").strip()))
        ttk.Checkbutton(
            log_col,
            text="Write to log file (logs/codex-*.log)",
            variable=self._var_log_file,
            command=self._toggle_log_file,
        ).grid(row=r + 1, column=0, sticky="w", pady=(10, 2))

        self._install_cfg_delta_guard()
        self.frame = frame
        return frame

    def reload(self) -> None:
        env = self._controller.store.env
        for key, var in self._debug_flags.items():
            var.set(BoolSetting(key, default=False).get(env))
        log_defaults = {
            "CODEX_LOG_DEBUG": False,
            "CODEX_LOG_INFO": True,
            "CODEX_LOG_WARNING": True,
            "CODEX_LOG_ERROR": True,
        }
        for key, var in self._log_levels.items():
            var.set(BoolSetting(key, default=bool(log_defaults.get(key, True))).get(env))

        self._var_cfg_delta_n.set(str(env.get("CODEX_LOG_CFG_DELTA_N", "2") or "2"))
        self._var_trace_max.set(str(env.get("CODEX_TRACE_DEBUG_MAX_PER_FUNC", "50") or "50"))
        self._var_dump_path.set(str(env.get("CODEX_DUMP_LATENTS_PATH", "") or ""))
        self._var_log_file.set(bool(str(env.get("CODEX_LOG_FILE", "") or "").strip()))

        self._install_cfg_delta_guard()

    def render_checks(self, checks: Iterable[CodexLaunchCheck]) -> None:
        tree = self._checks_tree
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        for chk in checks:
            tree.insert("", "end", values=(chk.name, "yes" if chk.ok else "no", chk.detail))

    # ------------------------------------------------------------------ env setters

    def _set_bool(self, key: str, enabled: bool) -> None:
        BoolSetting(key, default=False).set(self._controller.store.env, enabled)
        self._mark_changed()

    def _set_text(self, key: str, value: str) -> None:
        self._controller.store.env[key] = str(value).strip()
        self._mark_changed()

    def _add_entry(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        width: int,
        on_change: Callable[[], None],
    ) -> int:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=8)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=8)
        entry.bind("<KeyRelease>", lambda _e: on_change())
        return row + 1

    # ------------------------------------------------------------------ dependency guards

    def _install_cfg_delta_guard(self) -> None:
        if "CODEX_LOG_SAMPLER" not in self._debug_flags or "CODEX_LOG_CFG_DELTA" not in self._debug_flags:
            return
        sampler_var = self._debug_flags["CODEX_LOG_SAMPLER"]
        delta_var = self._debug_flags["CODEX_LOG_CFG_DELTA"]
        guard = {"active": False}

        def _on_delta_changed(*_args: object) -> None:
            if guard["active"] or (not delta_var.get()):
                return
            if sampler_var.get():
                return
            guard["active"] = True
            try:
                sampler_var.set(True)
                BoolSetting("CODEX_LOG_SAMPLER", default=False).set(self._controller.store.env, True)
                self._mark_changed()
            finally:
                guard["active"] = False

        def _on_sampler_changed(*_args: object) -> None:
            if guard["active"] or sampler_var.get():
                return
            if not delta_var.get():
                return
            guard["active"] = True
            try:
                delta_var.set(False)
                BoolSetting("CODEX_LOG_CFG_DELTA", default=False).set(self._controller.store.env, False)
                self._mark_changed()
            finally:
                guard["active"] = False

        delta_var.trace_add("write", _on_delta_changed)
        sampler_var.trace_add("write", _on_sampler_changed)
        _on_delta_changed()

    def _toggle_log_file(self) -> None:
        enabled = bool(self._var_log_file.get())
        env = self._controller.store.env
        if enabled:
            if not str(env.get("CODEX_LOG_FILE", "") or "").strip():
                logs_dir = self._controller.codex_root / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                env["CODEX_LOG_FILE"] = str(logs_dir / f"codex-{stamp}.log")
                self._mark_changed()
        else:
            try:
                del env["CODEX_LOG_FILE"]
            except KeyError:
                pass
            self._mark_changed()

    # ------------------------------------------------------------------ validation hooks (used by app save)

    def validate_int_settings(self) -> None:
        env = self._controller.store.env
        try:
            IntSetting("CODEX_LOG_CFG_DELTA_N", default=2, minimum=1).get(env)
            IntSetting("CODEX_TRACE_DEBUG_MAX_PER_FUNC", default=50, minimum=1).get(env)
        except SettingValidationError as exc:
            raise RuntimeError(str(exc)) from exc

