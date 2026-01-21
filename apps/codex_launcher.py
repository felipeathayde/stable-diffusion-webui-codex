"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: GUI launcher for Codex services (tkinter).
Provides a touch-friendly UI to manage API/UI service processes and runtime configuration.

Symbols (top-level; keep in sync; no ghosts):
- `CodexGUILauncher` (class): Main Tk app; builds widgets, polls service status, starts/stops services, edits env/profile values, and shows logs.
- `main` (function): CLI entrypoint that instantiates `CodexGUILauncher` and runs the tkinter main loop.
"""
from __future__ import annotations

import os
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional, Tuple

# Root: CODEX_ROOT must be set by .bat launcher
_codex_root = os.environ.get("CODEX_ROOT")
if not _codex_root:
    raise EnvironmentError("CODEX_ROOT not set. Use run-webui.bat to launch.")
_root_path = Path(_codex_root)
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

from apps.backend.infra.config.repo_root import get_repo_root

CODEX_ROOT = get_repo_root()
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

from apps.launcher import (
    CodexLogBuffer,
    CodexServiceHandle,
    DEFAULT_PYTORCH_CUDA_ALLOC_CONF,
    LauncherProfileStore,
    ServiceStatus,
    default_services,
)


class CodexGUILauncher(tk.Tk):
    """Main GUI application for Codex launcher."""

    POLL_INTERVAL_MS = 500

    def __init__(self) -> None:
        super().__init__()

        self.title("Codex Launcher")
        self.geometry("800x600")
        self.minsize(600, 400)

        # Data
        self.log_buffer = CodexLogBuffer(capacity=2000)
        self.services: Dict[str, CodexServiceHandle] = default_services(log_buffer=self.log_buffer)
        self.store = LauncherProfileStore.load()
        self.meta = self.store.meta
        self.env = self.store.env
        self._unsaved_changes = False

        # Styling
        self._setup_style()

        # Build UI
        self._build_notebook()
        self._build_status_bar()

        # Start polling
        self._poll_services()
        self._poll_logs()

        # Handle window close
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_style(self) -> None:
        """Configure ttk styles for dark theme and larger fonts."""
        style = ttk.Style(self)
        style.theme_use("clam")

        # Dark theme colors
        bg_dark = "#1e1e2e"
        bg_medium = "#2d2d3d"
        fg_light = "#cdd6f4"
        accent = "#89b4fa"
        accent_hover = "#b4befe"

        self.configure(bg=bg_dark)

        # Configure base styles
        style.configure(".", background=bg_dark, foreground=fg_light, font=("Segoe UI", 11))
        style.configure("TNotebook", background=bg_dark, borderwidth=0)
        style.configure("TNotebook.Tab", background=bg_medium, foreground=fg_light,
                        padding=[12, 6], font=("Segoe UI", 11, "bold"))
        style.map("TNotebook.Tab",
                  background=[("selected", accent)],
                  foreground=[("selected", bg_dark)])

        style.configure("TFrame", background=bg_dark)
        style.configure("TLabel", background=bg_dark, foreground=fg_light, font=("Segoe UI", 11))
        style.configure("TLabelframe", background=bg_dark, foreground=fg_light)
        style.configure("TLabelframe.Label", background=bg_dark, foreground=accent, font=("Segoe UI", 11, "bold"))

        style.configure("TButton", background=bg_medium, foreground=fg_light,
                        padding=[16, 8], font=("Segoe UI", 11))
        style.map("TButton",
                  background=[("active", accent_hover), ("pressed", accent)],
                  foreground=[("active", bg_dark), ("pressed", bg_dark)])

        style.configure("TCheckbutton", background=bg_dark, foreground=fg_light, font=("Segoe UI", 11))
        style.map("TCheckbutton", background=[("active", bg_dark)])

        style.configure("TCombobox", font=("Segoe UI", 11))
        style.configure("TEntry", font=("Segoe UI", 11))

        # Status-specific styles
        style.configure("Running.TLabel", foreground="#a6e3a1")  # green
        style.configure("Stopped.TLabel", foreground="#f38ba8")  # red
        style.configure("Error.TLabel", foreground="#fab387")    # orange

        # Action button styles
        style.configure("Start.TButton", background="#a6e3a1")
        style.configure("Kill.TButton", background="#f38ba8")
        style.configure("Save.TButton", background="#89b4fa")

        self._bg_dark = bg_dark
        self._fg_light = fg_light

    def _build_notebook(self) -> None:
        """Build the main tabbed interface."""
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=8)

        # Create tabs
        self._tab_main = self._build_tab_main()
        self._tab_runtime = self._build_tab_runtime()
        self._tab_wan = self._build_tab_wan()
        self._tab_debug = self._build_tab_debug()
        self._tab_logging = self._build_tab_logging()
        self._tab_logs = self._build_tab_logs()

        self.notebook.add(self._tab_main, text=" Main ")
        self.notebook.add(self._tab_runtime, text=" Runtime ")
        self.notebook.add(self._tab_wan, text=" WAN ")
        self.notebook.add(self._tab_debug, text=" DEBUG ")
        self.notebook.add(self._tab_logging, text=" Logging ")
        self.notebook.add(self._tab_logs, text=" Logs ")

    def _build_status_bar(self) -> None:
        """Build bottom status bar with save/exit buttons."""
        bar = ttk.Frame(self)
        bar.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Button(bar, text="💾 Save Settings", command=self._save_and_notify).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="❌ Exit Without Saving", command=self._exit_no_save).pack(side="left")

        self._status_label = ttk.Label(bar, text="Ready")
        self._status_label.pack(side="right")

    # ======================== Tab: Main ========================

    def _build_tab_main(self) -> ttk.Frame:
        """Build the Main tab with service status and controls."""
        frame = ttk.Frame(self)
        frame.columnconfigure(1, weight=1)

        row = 0
        for svc_name in ("API", "UI"):
            # Service frame
            svc_frame = ttk.LabelFrame(frame, text=f"  {svc_name} Service  ", padding=16)
            svc_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=8, padx=8)
            svc_frame.columnconfigure(1, weight=1)

            # Status label
            status_var = tk.StringVar(value="stopped")
            setattr(self, f"_status_{svc_name.lower()}", status_var)
            ttk.Label(svc_frame, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 8))
            status_lbl = ttk.Label(svc_frame, textvariable=status_var, font=("Segoe UI", 12, "bold"))
            status_lbl.grid(row=0, column=1, sticky="w")
            setattr(self, f"_status_lbl_{svc_name.lower()}", status_lbl)

            # PID/Uptime
            info_var = tk.StringVar(value="")
            setattr(self, f"_info_{svc_name.lower()}", info_var)
            ttk.Label(svc_frame, textvariable=info_var).grid(row=0, column=2, sticky="e", padx=(16, 0))

            # Buttons
            btn_frame = ttk.Frame(svc_frame)
            btn_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 0))

            ttk.Button(btn_frame, text="▶ Start", width=12,
                       command=lambda n=svc_name: self._start_service(n)).pack(side="left", padx=(0, 8))
            ttk.Button(btn_frame, text="🔄 Restart", width=12,
                       command=lambda n=svc_name: self._restart_service(n)).pack(side="left", padx=(0, 8))
            ttk.Button(btn_frame, text="⏹ Kill", width=12,
                       command=lambda n=svc_name: self._kill_service(n)).pack(side="left")

            row += 1

        return frame

    def _start_service(self, name: str) -> None:
        """Start a service."""
        svc = self.services[name]
        env = self.store.build_env()
        try:
            svc.start(env, external_terminal=self.meta.external_terminal)
            self._set_status(f"{name} starting...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start {name}: {e}")

    def _restart_service(self, name: str) -> None:
        """Restart a service."""
        svc = self.services[name]
        env = self.store.build_env()
        try:
            svc.restart(env, external_terminal=self.meta.external_terminal)
            self._set_status(f"{name} restarting...")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to restart {name}: {e}")

    def _kill_service(self, name: str) -> None:
        """Kill a service."""
        svc = self.services[name]
        svc.stop()
        time.sleep(0.1)
        svc.kill()
        self._set_status(f"{name} killed")

    # ======================== Tab: Runtime ========================

    def _build_tab_runtime(self) -> ttk.Frame:
        """Build Runtime configuration tab."""
        frame = ttk.Frame(self)
        canvas = tk.Canvas(frame, bg=self._bg_dark, highlightthickness=0)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Bind mousewheel
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        row = 0

        # External terminal
        self._var_ext_term = tk.BooleanVar(value=self.meta.external_terminal)
        ttk.Checkbutton(scrollable, text="Launch services in external terminal (Windows)",
                        variable=self._var_ext_term, command=self._mark_changed).grid(
                            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=8)
        row += 1

        # LoRA apply mode (global; backend restart required)
        raw_lora_mode = str(self.env.get("CODEX_LORA_APPLY_MODE", "merge") or "").strip().lower()
        if raw_lora_mode not in {"merge", "online"}:
            raw_lora_mode = "merge"
        self._var_lora_apply_mode = tk.StringVar(value=raw_lora_mode)
        ttk.Label(scrollable, text="LoRA apply mode (requires API restart):").grid(
            row=row, column=0, sticky="w", padx=16, pady=8
        )
        lora_mode_combo = ttk.Combobox(
            scrollable,
            textvariable=self._var_lora_apply_mode,
            values=["merge", "online"],
            state="readonly",
            width=12,
        )
        lora_mode_combo.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=8)
        lora_mode_combo.bind("<<ComboboxSelected>>", lambda _e: self._mark_changed())
        row += 1
        ttk.Label(
            scrollable,
            text=(
                "merge: rewrites weights once at apply-time (default).\n"
                "online: applies LoRA patches on-the-fly during forward."
            ),
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))
        row += 1

        # PyTorch CUDA allocator tuning (global; backend restart required)
        raw_alloc_conf = str(self.env.get("PYTORCH_CUDA_ALLOC_CONF") or DEFAULT_PYTORCH_CUDA_ALLOC_CONF).strip()
        self._var_pytorch_cuda_alloc_conf = tk.StringVar(value=raw_alloc_conf)
        ttk.Label(scrollable, text="PyTorch CUDA alloc conf (requires API restart):").grid(
            row=row, column=0, sticky="w", padx=16, pady=8
        )
        alloc_entry = ttk.Entry(scrollable, textvariable=self._var_pytorch_cuda_alloc_conf, width=48)
        alloc_entry.grid(row=row, column=1, sticky="w", padx=(0, 16), pady=8)
        alloc_entry.bind("<KeyRelease>", lambda _e: self._mark_changed())
        row += 1
        ttk.Label(
            scrollable,
            text=(
                "Env var: PYTORCH_CUDA_ALLOC_CONF\n"
                f"Default: {DEFAULT_PYTORCH_CUDA_ALLOC_CONF}"
            ),
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 8))
        row += 1

        ttk.Label(
            scrollable,
            text=(
                "Runtime settings (device/dtype/attention/cache/offload) are configured via the Web UI.\n"
                "This launcher no longer applies CODEX_* device/dtype/attention settings via environment variables."
            ),
            justify="left",
        ).grid(row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 0))

        return frame

    # ======================== Tab: WAN ========================

    def _build_tab_wan(self) -> ttk.Frame:
        """Build WAN22 configuration tab."""
        frame = ttk.Frame(self)
        ttk.Label(
            frame,
            text="WAN runtime settings are now payload/options-driven.\n"
                 "Use the Web UI (WAN tab) to configure per-run WAN options.",
        ).grid(row=0, column=0, sticky="w", padx=16, pady=16)

        return frame

    # ======================== Tab: DEBUG ========================

    def _build_tab_debug(self) -> ttk.Frame:
        """Build DEBUG configuration tab."""
        frame = ttk.Frame(self)
        row = 0

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

        self._debug_flags: Dict[str, tk.BooleanVar] = {}
        for key, label in debug_flags:
            val = self.env.get(key, "0").lower() in {"1", "true", "yes", "on"}
            var = tk.BooleanVar(value=val)
            self._debug_flags[key] = var
            ttk.Checkbutton(frame, text=label, variable=var,
                            command=self._mark_changed).grid(
                                row=row, column=0, columnspan=2, sticky="w", padx=16, pady=4)
            row += 1

        # Keep CFG delta logs consistent with sampler logging (CFG delta requires sampler logs).
        self._cfg_delta_guard = False
        if "CODEX_LOG_SAMPLER" in self._debug_flags and "CODEX_LOG_CFG_DELTA" in self._debug_flags:
            sampler_var = self._debug_flags["CODEX_LOG_SAMPLER"]
            delta_var = self._debug_flags["CODEX_LOG_CFG_DELTA"]

            def _on_delta_changed(*_args: object) -> None:
                if self._cfg_delta_guard:
                    return
                if not delta_var.get():
                    return
                if sampler_var.get():
                    return
                self._cfg_delta_guard = True
                try:
                    sampler_var.set(True)
                finally:
                    self._cfg_delta_guard = False
                self._mark_changed()

            def _on_sampler_changed(*_args: object) -> None:
                if self._cfg_delta_guard:
                    return
                if sampler_var.get():
                    return
                if not delta_var.get():
                    return
                self._cfg_delta_guard = True
                try:
                    delta_var.set(False)
                finally:
                    self._cfg_delta_guard = False
                self._mark_changed()

            delta_var.trace_add("write", _on_delta_changed)
            sampler_var.trace_add("write", _on_sampler_changed)
            _on_delta_changed()

        # CFG Delta Steps (N)
        row = self._add_entry_row(
            frame,
            row,
            "CFG Delta Steps (N)",
            self.env.get("CODEX_LOG_CFG_DELTA_N", "2"),
            "_var_cfg_delta_n",
        )

        # Trace Max Per Func
        row = self._add_entry_row(frame, row, "Trace Max Per Func",
                                  self.env.get("CODEX_TRACE_DEBUG_MAX_PER_FUNC", "50"),
                                  "_var_trace_max")

        # Dump Latents Path
        row = self._add_entry_row(frame, row, "Dump Latents Path",
                                  self.env.get("CODEX_DUMP_LATENTS_PATH", ""),
                                  "_var_dump_path")

        return frame

    # ======================== Tab: Logging ========================

    def _build_tab_logging(self) -> ttk.Frame:
        """Build Logging configuration tab."""
        frame = ttk.Frame(self)
        row = 0

        # Section: Codex Log Levels (checkboxes)
        ttk.Label(frame, text="Codex Log Levels", style="TLabelframe.Label").grid(
            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 4))
        row += 1

        # Individual checkboxes for each log level
        codex_log_levels = [
            ("CODEX_LOG_DEBUG", "DEBUG (verbose)", "0"),
            ("CODEX_LOG_INFO", "INFO", "1"),
            ("CODEX_LOG_WARNING", "WARNING", "1"),
            ("CODEX_LOG_ERROR", "ERROR", "1"),
        ]

        self._codex_log_levels: Dict[str, tk.BooleanVar] = {}
        for key, label, default in codex_log_levels:
            val = self.env.get(key, default).lower() in {"1", "true", "yes", "on"}
            var = tk.BooleanVar(value=val)
            self._codex_log_levels[key] = var
            ttk.Checkbutton(frame, text=label, variable=var,
                            command=self._mark_changed).grid(
                                row=row, column=0, columnspan=2, sticky="w", padx=32, pady=2)
            row += 1

        # Log File
        log_file = self.env.get("CODEX_LOG_FILE", "")
        self._var_log_file = tk.BooleanVar(value=bool(log_file))
        ttk.Checkbutton(frame, text="Write to log file (logs/codex-*.log)",
                        variable=self._var_log_file, command=self._mark_changed).grid(
                            row=row, column=0, columnspan=2, sticky="w", padx=16, pady=(8, 4))
        row += 1

        return frame

    # ======================== Tab: Logs ========================

    def _build_tab_logs(self) -> ttk.Frame:
        """Build Logs viewer tab."""
        frame = ttk.Frame(self)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Log text area
        self._log_text = ScrolledText(frame, wrap="word", state="disabled",
                                      bg="#11111b", fg="#cdd6f4",
                                      font=("Consolas", 10),
                                      insertbackground="#cdd6f4")
        self._log_text.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Buttons
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))

        ttk.Button(btn_frame, text="🗑 Clear Logs", command=self._clear_logs).pack(side="left")

        self._var_autoscroll = tk.BooleanVar(value=True)
        ttk.Checkbutton(btn_frame, text="Auto-scroll", variable=self._var_autoscroll).pack(side="right")

        return frame

    def _clear_logs(self) -> None:
        """Clear the log buffer and display."""
        self.log_buffer.clear()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    # ======================== Helpers ========================

    def _add_combo_row(self, parent: ttk.Frame, row: int, label: str,
                       values: List[str], current: str, var_name: str) -> int:
        """Add a labeled combobox row."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=16, pady=8)
        var = tk.StringVar(value=current)
        setattr(self, var_name, var)
        combo = ttk.Combobox(parent, textvariable=var, values=values, state="readonly", width=20)
        combo.grid(row=row, column=1, sticky="w", padx=8, pady=8)
        combo.bind("<<ComboboxSelected>>", lambda e: self._mark_changed())
        return row + 1

    def _add_entry_row(self, parent: ttk.Frame, row: int, label: str,
                       current: str, var_name: str) -> int:
        """Add a labeled entry row."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=16, pady=8)
        var = tk.StringVar(value=current)
        setattr(self, var_name, var)
        entry = ttk.Entry(parent, textvariable=var, width=24)
        entry.grid(row=row, column=1, sticky="w", padx=8, pady=8)
        entry.bind("<KeyRelease>", lambda e: self._mark_changed())
        return row + 1

    def _device_to_label(self, raw: str) -> str:
        """Convert device string to display label."""
        raw = raw.strip().lower()
        if raw in ("cuda", "gpu"):
            return "GPU"
        if raw == "cpu":
            return "CPU"
        return "Auto"

    def _label_to_device(self, label: str) -> str:
        """Convert display label to device string."""
        if label == "GPU":
            return "cuda"
        if label == "CPU":
            return "cpu"
        return ""

    def _mark_changed(self) -> None:
        """Mark that there are unsaved changes."""
        self._unsaved_changes = True
        self._set_status("Unsaved changes")

    def _set_status(self, msg: str) -> None:
        """Update status bar message."""
        self._status_label.configure(text=msg)

    # ======================== Polling ========================

    def _poll_services(self) -> None:
        """Poll service status periodically."""
        for name in ("API", "UI"):
            svc = self.services[name]
            status = svc.status.value
            pid = svc.pid or 0
            uptime = "-"
            if svc.started_at and svc.status == ServiceStatus.RUNNING:
                uptime = f"{int(time.time() - svc.started_at)}s"

            status_var: tk.StringVar = getattr(self, f"_status_{name.lower()}")
            status_var.set(status.upper())

            info_var: tk.StringVar = getattr(self, f"_info_{name.lower()}")
            info_var.set(f"PID: {pid}  |  Uptime: {uptime}" if pid else "")

            # Update label style
            lbl: ttk.Label = getattr(self, f"_status_lbl_{name.lower()}")
            if status == ServiceStatus.RUNNING.value:
                lbl.configure(style="Running.TLabel")
            elif status == ServiceStatus.ERROR.value:
                lbl.configure(style="Error.TLabel")
            else:
                lbl.configure(style="Stopped.TLabel")

        self.after(self.POLL_INTERVAL_MS, self._poll_services)

    def _poll_logs(self) -> None:
        """Poll log buffer and update display."""
        logs = self.log_buffer.snapshot()
        current_text = self._log_text.get("1.0", "end-1c")
        new_text = "\n".join(logs)

        if new_text != current_text:
            self._log_text.configure(state="normal")
            self._log_text.delete("1.0", "end")
            self._log_text.insert("1.0", new_text)
            self._log_text.configure(state="disabled")

            if self._var_autoscroll.get():
                self._log_text.see("end")

        self.after(self.POLL_INTERVAL_MS, self._poll_logs)

    # ======================== Save/Exit ========================

    def _collect_settings(self) -> None:
        """Collect all settings from UI into store."""
        env = self.env

        # Meta
        self.meta.external_terminal = self._var_ext_term.get()
        env["CODEX_LORA_APPLY_MODE"] = str(self._var_lora_apply_mode.get()).strip().lower() or "merge"
        alloc_conf = str(self._var_pytorch_cuda_alloc_conf.get() or "").strip()
        if alloc_conf:
            env["PYTORCH_CUDA_ALLOC_CONF"] = alloc_conf
        else:
            env.pop("PYTORCH_CUDA_ALLOC_CONF", None)

        # Debug
        for key, var in self._debug_flags.items():
            env[key] = "1" if var.get() else "0"
        cfg_delta_n = self._var_cfg_delta_n.get().strip()
        if cfg_delta_n:
            env["CODEX_LOG_CFG_DELTA_N"] = cfg_delta_n
        else:
            env.pop("CODEX_LOG_CFG_DELTA_N", None)
        env["CODEX_TRACE_DEBUG_MAX_PER_FUNC"] = self._var_trace_max.get()
        dump_path = self._var_dump_path.get().strip()
        if dump_path:
            env["CODEX_DUMP_LATENTS_PATH"] = dump_path

        # Logging - individual level checkboxes
        for key, var in self._codex_log_levels.items():
            env[key] = "1" if var.get() else "0"
        
        if self._var_log_file.get():
            if not env.get("CODEX_LOG_FILE"):
                logs_dir = CODEX_ROOT / "logs"
                logs_dir.mkdir(parents=True, exist_ok=True)
                stamp = time.strftime("%Y%m%d-%H%M%S")
                env["CODEX_LOG_FILE"] = str(logs_dir / f"codex-{stamp}.log")
        else:
            env.pop("CODEX_LOG_FILE", None)

    def _save_and_notify(self) -> None:
        """Save settings and notify user."""
        self._collect_settings()
        self.store.save()
        self._unsaved_changes = False
        self._set_status("Settings saved!")

    def _exit_no_save(self) -> None:
        """Exit without saving."""
        self.destroy()

    def _on_close(self) -> None:
        """Handle window close event."""
        if self._unsaved_changes:
            if messagebox.askyesno("Unsaved Changes",
                                   "You have unsaved changes. Save before exiting?"):
                self._save_and_notify()
        self.destroy()


def main() -> None:
    app = CodexGUILauncher()
    app.mainloop()


if __name__ == "__main__":
    main()
