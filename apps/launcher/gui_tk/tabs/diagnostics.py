"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Diagnostics tab for the Tk launcher.
Shows environment preflight checks and exposes debug/logging/profiling env flags used by the backend.
Includes advanced performance toggles (e.g. CFG batch mode) intended for profiling and troubleshooting.
Includes contract-trace/profiler launcher toggles (`CODEX_TRACE_CONTRACT`, `CODEX_TRACE_PROFILER`) for runtime diagnostics bootstrap.

Symbols (top-level; keep in sync; no ghosts):
- `DiagnosticsTab` (class): Diagnostics tab (checks + debug flags + log levels).
"""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Iterable, List, Tuple

from apps.launcher.checks import CodexLaunchCheck
from apps.launcher.settings import BoolSetting, ChoiceSetting, CFG_BATCH_MODE_CHOICES, IntSetting, SettingValidationError

from ..controller import LauncherController
from ..widgets import ScrollableFrame


TRACE_DEBUG_DEFAULT = "10"


class DiagnosticsTab:
    def __init__(
        self,
        controller: LauncherController,
        *,
        mark_changed: Callable[[], None],
        run_checks_async: Callable[[], None],
        canvas_bg: str,
    ) -> None:
        self._controller = controller
        self._mark_changed = mark_changed
        self._run_checks_async = run_checks_async
        self._canvas_bg = str(canvas_bg)

        self.frame: ttk.Frame | None = None

        self._checks_tree: ttk.Treeview | None = None

        self._debug_flags: Dict[str, tk.BooleanVar] = {}
        self._log_levels: Dict[str, tk.BooleanVar] = {}

        self._var_cfg_delta_n = tk.StringVar()
        self._var_cfg_batch_mode = tk.StringVar()
        self._var_trace_max = tk.StringVar()
        self._var_dump_path = tk.StringVar()
        self._var_profile_top_n = tk.StringVar()
        self._var_profile_max_steps = tk.StringVar()
        self._var_log_file = tk.BooleanVar()
        self._advanced_visible = False
        self._advanced_widgets: List[tk.Widget] = []
        self._cfg_delta_trace_ids: List[Tuple[tk.Variable, str]] = []

    def build(self, notebook: ttk.Notebook) -> ttk.Frame:
        frame = ttk.Frame(notebook)
        scroll = ScrollableFrame(frame, canvas_bg=self._canvas_bg)
        scroll.pack(fill="both", expand=True)
        body = scroll.inner
        body.columnconfigure(0, weight=1)

        checks_box = ttk.LabelFrame(body, text="  Environment Checks  ", padding=14)
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

        diag = ttk.LabelFrame(body, text="  Debug + Logging  ", padding=14)
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
            ("CODEX_TRACE_INFERENCE_DEBUG", "Trace Debug: Inference"),
            ("CODEX_TRACE_LOAD_PATCH_DEBUG", "Trace Debug: Load/Patch"),
            ("CODEX_TRACE_CALL_DEBUG", "Trace Debug: Call Trace"),
            ("CODEX_PIPELINE_DEBUG", "Pipeline Debug"),
            ("CODEX_DUMP_LATENTS", "Dump Latents"),
            ("CODEX_TIMELINE", "Timeline Tracer (TVA-style execution timeline)"),
            ("CODEX_TRACE_CONTRACT", "Contract Trace (JSONL in logs/contract-trace)"),
            ("CODEX_TRACE_PROFILER", "Contract Profiler Toggle (maps to --trace-profiler)"),
            ("CODEX_PROFILE", "Global Profiler (torch.profiler)"),
            ("CODEX_PROFILE_TRACE", "Profiler: export Perfetto trace"),
            ("CODEX_PROFILE_RECORD_SHAPES", "Profiler: record shapes"),
            ("CODEX_PROFILE_PROFILE_MEMORY", "Profiler: profile memory"),
            ("CODEX_PROFILE_WITH_STACK", "Profiler: include stacks (very heavy)"),
        ]
        advanced_debug_keys = {
            "CODEX_TRACE_INFERENCE_DEBUG",
            "CODEX_TRACE_LOAD_PATCH_DEBUG",
            "CODEX_TRACE_CALL_DEBUG",
            "CODEX_PIPELINE_DEBUG",
            "CODEX_DUMP_LATENTS",
            "CODEX_TIMELINE",
            "CODEX_TRACE_CONTRACT",
            "CODEX_TRACE_PROFILER",
            "CODEX_PROFILE",
            "CODEX_PROFILE_TRACE",
            "CODEX_PROFILE_RECORD_SHAPES",
            "CODEX_PROFILE_PROFILE_MEMORY",
            "CODEX_PROFILE_WITH_STACK",
        }
        r = 1
        for key, label in debug_flags:
            var = tk.BooleanVar(value=BoolSetting(key, default=False).get(self._controller.store.env))
            self._debug_flags[key] = var
            checkbox = ttk.Checkbutton(
                dbg_col,
                text=label,
                variable=var,
                command=lambda k=key: self._set_bool(k, self._debug_flags[k].get()),
                style="Toggle.TCheckbutton",
            )
            checkbox.grid(row=r, column=0, sticky="w", pady=2)
            if key in advanced_debug_keys:
                self._register_advanced(checkbox)
            r += 1

        # Debug numeric/text
        self._var_cfg_delta_n.set(str(self._controller.store.env.get("CODEX_LOG_CFG_DELTA_N", "2") or "2"))
        self._var_cfg_batch_mode.set(
            ChoiceSetting("CODEX_CFG_BATCH_MODE", default="fused", choices=CFG_BATCH_MODE_CHOICES).get(self._controller.store.env)
        )
        self._var_trace_max.set(
            str(self._controller.store.env.get("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", TRACE_DEBUG_DEFAULT) or TRACE_DEBUG_DEFAULT)
        )
        self._var_dump_path.set(str(self._controller.store.env.get("CODEX_DUMP_LATENTS_PATH", "") or ""))
        self._var_profile_top_n.set(str(self._controller.store.env.get("CODEX_PROFILE_TOP_N", "25") or "25"))
        self._var_profile_max_steps.set(str(self._controller.store.env.get("CODEX_PROFILE_MAX_STEPS", "0") or "0"))

        r = self._add_entry(
            dbg_col,
            r + 1,
            label="CFG Delta Steps (N):",
            var=self._var_cfg_delta_n,
            width=10,
            on_change=lambda: self._set_text("CODEX_LOG_CFG_DELTA_N", self._var_cfg_delta_n.get()),
        )
        r = self._add_choice(
            dbg_col,
            r,
            label="CFG Cond+Uncond Batch Mode:",
            var=self._var_cfg_batch_mode,
            choices=CFG_BATCH_MODE_CHOICES,
            on_change=lambda: self._set_text("CODEX_CFG_BATCH_MODE", self._var_cfg_batch_mode.get()),
            advanced=True,
        )
        r = self._add_entry(
            dbg_col,
            r,
            label="Call trace max / func (0=unlimited):",
            var=self._var_trace_max,
            width=10,
            on_change=lambda: self._set_text("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", self._var_trace_max.get()),
            advanced=True,
        )
        r = self._add_entry(
            dbg_col,
            r,
            label="Dump latents path:",
            var=self._var_dump_path,
            width=36,
            on_change=lambda: self._set_text("CODEX_DUMP_LATENTS_PATH", self._var_dump_path.get()),
            advanced=True,
        )
        r = self._add_entry(
            dbg_col,
            r,
            label="Profiler top ops (N):",
            var=self._var_profile_top_n,
            width=10,
            on_change=lambda: self._set_text("CODEX_PROFILE_TOP_N", self._var_profile_top_n.get()),
            advanced=True,
        )
        _ = self._add_entry(
            dbg_col,
            r,
            label="Profiler max steps (0=all):",
            var=self._var_profile_max_steps,
            width=10,
            on_change=lambda: self._set_text("CODEX_PROFILE_MAX_STEPS", self._var_profile_max_steps.get()),
            advanced=True,
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
            checkbox = ttk.Checkbutton(
                log_col,
                text=label,
                variable=var,
                command=lambda k=key: self._set_bool(k, self._log_levels[k].get()),
                style="Toggle.TCheckbutton",
            )
            checkbox.grid(row=r, column=0, sticky="w", pady=2)
            r += 1

        self._var_log_file.set(bool(str(self._controller.store.env.get("CODEX_LOG_FILE", "") or "").strip()))
        log_file_toggle = ttk.Checkbutton(
            log_col,
            text="Write to log file (logs/codex-*.log)",
            variable=self._var_log_file,
            command=self._toggle_log_file,
            style="Toggle.TCheckbutton",
        )
        log_file_toggle.grid(row=r + 1, column=0, sticky="w", pady=(10, 2))

        self._install_cfg_delta_guard()
        self._apply_advanced_visibility()
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
        self._var_cfg_batch_mode.set(ChoiceSetting("CODEX_CFG_BATCH_MODE", default="fused", choices=CFG_BATCH_MODE_CHOICES).get(env))
        self._var_trace_max.set(str(env.get("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", TRACE_DEBUG_DEFAULT) or TRACE_DEBUG_DEFAULT))
        self._var_dump_path.set(str(env.get("CODEX_DUMP_LATENTS_PATH", "") or ""))
        self._var_profile_top_n.set(str(env.get("CODEX_PROFILE_TOP_N", "25") or "25"))
        self._var_profile_max_steps.set(str(env.get("CODEX_PROFILE_MAX_STEPS", "0") or "0"))
        self._var_log_file.set(bool(str(env.get("CODEX_LOG_FILE", "") or "").strip()))

        self._install_cfg_delta_guard()
        self._apply_advanced_visibility()

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
        advanced: bool = False,
    ) -> int:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=8)
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=8)
        entry.bind("<KeyRelease>", lambda _e: on_change())
        if advanced:
            self._register_advanced(label_widget, entry)
        return row + 1

    def _add_choice(
        self,
        parent: ttk.Frame,
        row: int,
        *,
        label: str,
        var: tk.StringVar,
        choices: tuple[str, ...],
        on_change: Callable[[], None],
        advanced: bool = False,
    ) -> int:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", pady=8)
        combo = ttk.Combobox(parent, textvariable=var, width=18, state="readonly")
        combo["values"] = list(choices)
        combo.grid(row=row, column=1, sticky="w", padx=(8, 0), pady=8)
        combo.bind("<<ComboboxSelected>>", lambda _e: on_change())
        if advanced:
            self._register_advanced(label_widget, combo)
        return row + 1

    # ------------------------------------------------------------------ dependency guards

    def _register_advanced(self, *widgets: tk.Widget) -> None:
        self._advanced_widgets.extend(widgets)

    def set_advanced_visible(self, visible: bool) -> None:
        self._advanced_visible = bool(visible)
        self._apply_advanced_visibility()

    def _apply_advanced_visibility(self) -> None:
        visible = bool(self._advanced_visible)
        for widget in self._advanced_widgets:
            if visible:
                widget.grid()
            else:
                widget.grid_remove()

    def _clear_cfg_delta_guard(self) -> None:
        for variable, trace_id in self._cfg_delta_trace_ids:
            try:
                variable.trace_remove("write", trace_id)
            except Exception:
                continue
        self._cfg_delta_trace_ids.clear()

    def _install_cfg_delta_guard(self) -> None:
        if "CODEX_LOG_SAMPLER" not in self._debug_flags or "CODEX_LOG_CFG_DELTA" not in self._debug_flags:
            return
        self._clear_cfg_delta_guard()
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

        delta_trace_id = delta_var.trace_add("write", _on_delta_changed)
        sampler_trace_id = sampler_var.trace_add("write", _on_sampler_changed)
        self._cfg_delta_trace_ids.append((delta_var, delta_trace_id))
        self._cfg_delta_trace_ids.append((sampler_var, sampler_trace_id))
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
            IntSetting("CODEX_TRACE_CALL_DEBUG_MAX_PER_FUNC", default=int(TRACE_DEBUG_DEFAULT), minimum=0).get(env)
            IntSetting("CODEX_PROFILE_TOP_N", default=25, minimum=1, maximum=500).get(env)
            IntSetting("CODEX_PROFILE_MAX_STEPS", default=0, minimum=0, maximum=10_000).get(env)
        except SettingValidationError as exc:
            raise RuntimeError(str(exc)) from exc
