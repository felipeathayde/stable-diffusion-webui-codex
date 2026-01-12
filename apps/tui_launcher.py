"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: BIOS-style curses TUI (stdlib-only) for managing Codex services and launcher options.
Settings persist under `.sangoi/launcher/` and the UI controls API/UI service processes plus runtime flags.

Symbols (top-level; keep in sync; no ghosts):
- `BIOSApp` (class): Curses app implementing the tabbed launcher UI (render loop + key handling); includes nested helpers for panes,
  item lists per tab (`_items_main/_items_runtime/_items_debug/_items_logging`), popup selection, and applying changes to the profile/env.
- `main` (function): Curses entrypoint (used with `curses.wrapper`); creates `BIOSApp` and runs the UI loop.
"""
from __future__ import annotations

import curses
import locale
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Root: CODEX_ROOT must be set by .bat launcher
_codex_root = os.environ.get("CODEX_ROOT")
if not _codex_root:
    raise EnvironmentError("CODEX_ROOT not set. Use run-tui.bat to launch.")
_root_path = Path(_codex_root)
if str(_root_path) not in sys.path:
    sys.path.insert(0, str(_root_path))

from apps.backend.infra.config.repo_root import get_repo_root

CODEX_ROOT = get_repo_root()
if str(CODEX_ROOT) not in sys.path:
    sys.path.insert(0, str(CODEX_ROOT))

from apps.backend.infra.config.args import TRACE_DEBUG_DEFAULT
from apps.launcher import (
    CodexLaunchCheck,
    CodexLogBuffer,
    CodexServiceHandle,
    LauncherProfileStore,
    ServiceStatus,
    default_services,
    run_launch_checks,
)
class BIOSApp:
    TABS = ["Main", "Runtime", "DEBUG", "Logging", "Logs", "Exit"]

    def __init__(self, stdscr) -> None:
        self.stdscr = stdscr
        self.log_buffer = CodexLogBuffer(capacity=4000)
        self.services: Dict[str, CodexServiceHandle] = default_services(log_buffer=self.log_buffer)
        self.store = LauncherProfileStore.load()
        self.meta = self.store.meta
        self.env = self.store.env
        self.tab_index: int = max(0, min(len(self.TABS) - 1, self.meta.tab_index))
        self.sel_index: int = 0
        self.log_scroll: int = 0
        self.message: str = ""
        self.popup_active: bool = False
        self.popup_selection: Optional[Tuple[str, List[str], int, str]] = None  # (title, options, index, action)
        self.launch_checks: List[CodexLaunchCheck] = run_launch_checks()

        # colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_CYAN, curses.COLOR_BLUE)    # top/bottom bar
        curses.init_pair(2, curses.COLOR_BLACK, curses.COLOR_CYAN)    # active tab
        curses.init_pair(3, curses.COLOR_WHITE, -1)                   # normal text
        curses.init_pair(4, curses.COLOR_BLACK, curses.COLOR_WHITE)   # selection
        curses.init_pair(5, curses.COLOR_YELLOW, -1)                  # hint
        curses.init_pair(6, curses.COLOR_CYAN, -1)                    # label
        self.C_BAR = curses.color_pair(1)
        self.C_TAB = curses.color_pair(2)
        self.C_SEL = curses.color_pair(4)
        self.C_NRM = curses.color_pair(3)
        self.C_LBL = curses.color_pair(6)

    # --------------- rendering helpers ---------------------------------
    def _prompt(self, prompt: str) -> Optional[str]:
        try:
            h, w = self.stdscr.getmaxyx()
            y = h - 2
            msg = (prompt + " " * w)[:w-1]
            curses.echo()
            try:
                curses.curs_set(1)
            except Exception:
                pass
            self.stdscr.addstr(y, 1, msg)
            self.stdscr.refresh()
            win = curses.newwin(1, max(10, w - len(prompt) - 2), y, len(prompt) + 2)
            val = win.getstr().decode(errors='ignore')
            return val.strip()
        except Exception:
            return None
        finally:
            try:
                curses.noecho()
                curses.curs_set(0)
            except Exception:
                pass
    def _draw_bars(self, h: int, w: int) -> None:
        title = " PhoenixBIOS-like Setup Utility "
        try:
            self.stdscr.attron(self.C_BAR)
            # Top bar
            self.stdscr.addstr(0, 0, (" " * w)[:w])
            self.stdscr.addstr(0, max(0, (w - len(title)) // 2), title[:max(0, w)])
            self.stdscr.attroff(self.C_BAR)
        except Exception:
            pass

        # tabs line
        x = 0
        for i, name in enumerate(self.TABS):
            label = f" {name} "
            attr = self.C_TAB if i == self.tab_index else self.C_NRM
            self.stdscr.attron(attr)
            self.stdscr.addstr(1, x, label)
            self.stdscr.attroff(attr)
            x += len(label) + 1

        # bottom bar
        keys = "F1 Help   Esc Exit   Enter Select   +/- Change   F2/F3 API Start/Kill   F4/F5 UI Start/Kill   F10 Save and Exit"
        try:
            self.stdscr.attron(self.C_BAR)
            y = max(0, h - 1)
            self.stdscr.addstr(y, 0, (keys + " " * w)[:w])
            self.stdscr.attroff(self.C_BAR)
        except Exception:
            pass

    def _pane_geometry(self) -> Tuple[int, int, int, int]:
        h, w = self.stdscr.getmaxyx()
        top = 2
        left_w = max(52, w * 2 // 3)
        right_w = max(20, w - left_w - 1)
        height = max(3, h - top - 2)
        return top, left_w, right_w, height

    def _draw_help(self, top: int, left_w: int, right_w: int, height: int, help_lines: List[str]) -> None:
        x = left_w + 1
        try:
            self.stdscr.attron(self.C_LBL)
            self.stdscr.addstr(top, x, "Item Specific Help")
            self.stdscr.attroff(self.C_LBL)
        except Exception:
            pass
        for i in range(1, height):
            line = help_lines[i - 1] if i - 1 < len(help_lines) else ""
            try:
                self.stdscr.addstr(top + i, x, (line + " " * right_w)[:right_w])
            except Exception:
                pass

    def _persist(self) -> None:
        self.meta.tab_index = max(0, self.tab_index)
        self.store.save()

    # --------------- content per tab -----------------------------------
    def _ensure_log_file(self) -> str:
        logs_dir = CODEX_ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        log_path = logs_dir / f"codex-{stamp}.log"
        return str(log_path)


    def _items_main(self) -> List[Tuple[str, str, str]]:
        out: List[Tuple[str, str, str]] = []
        for name in ("API", "UI"):
            svc = self.services[name]
            status = svc.status.value
            pid = svc.pid or 0
            up = "-"
            if svc.started_at and svc.status == ServiceStatus.RUNNING:
                up = f"{int(time.time() - svc.started_at)}s"
            out.append((f"{name} Status", f"{status} (pid={pid}, up={up})", "status"))
            out.append((f"Start {name}", "", "start"))
            out.append((f"Restart {name}", "", "restart"))
            out.append((f"Kill {name}", "", "kill"))
        return out

    def _items_runtime(self) -> List[Tuple[str, str, str]]:
        ext = "Enabled" if self.meta.external_terminal else "Disabled"
        return [
            ("Services in new terminal", f"[{ext}]", "toggle_extterm"),
            ("Runtime settings", "[Configured in Web UI]", "status"),
        ]

    def _items_debug(self) -> List[Tuple[str, str, str]]:
        env = self.env

        def _enabled(key: str) -> bool:
            return env.get(key, "0").strip().lower() in {"1", "true", "yes", "on"}

        cond = _enabled("CODEX_DEBUG_COND")
        sampler = _enabled("CODEX_LOG_SAMPLER")
        cfg_delta = _enabled("CODEX_LOG_CFG_DELTA")
        sigmas = _enabled("CODEX_LOG_SIGMAS")
        trace = _enabled("CODEX_TRACE_DEBUG")
        pipeline = _enabled("CODEX_PIPELINE_DEBUG")
        dump_latents = _enabled("CODEX_DUMP_LATENTS")
        dump_path = env.get("CODEX_DUMP_LATENTS_PATH", "").strip()
        if not dump_path:
            dump_path = "<logs/diagnostics>"
        cfg_delta_n = env.get("CODEX_LOG_CFG_DELTA_N", "2").strip() or "2"
        trace_max = env.get("CODEX_TRACE_DEBUG_MAX_PER_FUNC", str(TRACE_DEBUG_DEFAULT))
        return [
            ("Conditioning Debug", f"[{'Enabled' if cond else 'Disabled'}]", "toggle_cond_debug"),
            ("Sampler Verbose Logs", f"[{'Enabled' if sampler else 'Disabled'}]", "toggle_sampler_logs"),
            ("CFG Delta Logs", f"[{'Enabled' if cfg_delta else 'Disabled'}]", "toggle_cfg_delta_logs"),
            ("CFG Delta Steps (N)", f"[{cfg_delta_n}]", "edit_cfg_delta_n"),
            ("Sigma Ladder Logs", f"[{'Enabled' if sigmas else 'Disabled'}]", "toggle_sigma_logs"),
            ("Trace Debug", f"[{'ON' if trace else 'OFF'}]", "toggle_trace_debug"),
            ("Trace Max Per Func", f"[{trace_max}]", "edit_trace_max"),
            ("Pipeline Debug", f"[{'ON' if pipeline else 'OFF'}]", "toggle_pipeline_debug"),
            ("Dump Latents", f"[{'Enabled' if dump_latents else 'Disabled'}]", "toggle_dump_latents"),
            ("Dump Latents Path", f"[{dump_path}]", "edit_dump_latents_path"),
        ]

    def _items_logging(self) -> List[Tuple[str, str, str]]:
        env = self.env
        log_file = env.get("CODEX_LOG_FILE", "")
        if log_file:
            file_label = f"Enabled ({Path(log_file).name})"
        else:
            file_label = "Disabled"
        
        # Helper to check if a log level flag is enabled
        def _flag_label(key: str, default: str = "1") -> str:
            val = env.get(key, default).strip().lower()
            return "ON" if val in ("1", "true", "yes", "on") else "OFF"
        
        return [
            # Codex Log Levels (individual toggles)
            ("Codex DEBUG", f"[{_flag_label('CODEX_LOG_DEBUG', '0')}]", "toggle_codex_debug"),
            ("Codex INFO", f"[{_flag_label('CODEX_LOG_INFO', '1')}]", "toggle_codex_info"),
            ("Codex WARNING", f"[{_flag_label('CODEX_LOG_WARNING', '1')}]", "toggle_codex_warning"),
            ("Codex ERROR", f"[{_flag_label('CODEX_LOG_ERROR', '1')}]", "toggle_codex_error"),
            ("Write Codex Log File", f"[{file_label}]", "toggle_log_file"),
        ]

    def _help_for(self, tab: str, key: str) -> List[str]:
        d: Dict[str, List[str]] = {
            "Services in new terminal": [
                "Launch API/UI in a separate Windows console window.",
                "On Linux/macOS this is ignored (attached mode only).",
            ],
            "Runtime settings": [
                "Runtime settings (device/dtype/attention/cache/offload) are configured via the Web UI.",
                "This launcher no longer applies CODEX_* runtime settings via environment variables.",
            ],
            "Codex DEBUG": [
                "Enable/disable DEBUG level logs from Codex backend.",
                "Very verbose; includes internal tracing and diagnostics.",
                "Applied via CODEX_LOG_DEBUG.",
            ],
            "Codex INFO": [
                "Enable/disable INFO level logs from Codex backend.",
                "Standard operational messages.",
                "Applied via CODEX_LOG_INFO.",
            ],
            "Codex WARNING": [
                "Enable/disable WARNING level logs from Codex backend.",
                "Potential issues that don't prevent operation.",
                "Applied via CODEX_LOG_WARNING.",
            ],
            "Codex ERROR": [
                "Enable/disable ERROR level logs from Codex backend.",
                "Errors that may affect operation.",
                "Applied via CODEX_LOG_ERROR.",
            ],
            "Write Codex Log File": [
                "Toggle writing backend logs to logs/codex-<timestamp>.log.",
                "Enabled state sets CODEX_LOG_FILE automatically; disable to stop logging to file.",
            ],
            "Pipeline Debug": [
                "Toggle SDXL/txt2img pipeline trace logs (entrou/saiu).",
                "Applies via CODEX_PIPELINE_DEBUG=1 before starting the API.",
            ],
            "Trace Debug": [
                "Enable global function-call tracing (very verbose).",
                "Applies via CODEX_TRACE_DEBUG=1 / --trace-debug; restart API after toggling.",
            ],
            "Trace Max Per Func": [
                "Limit how many entries per function the call tracer records.",
                f"Set to 0 for unlimited; default is {TRACE_DEBUG_DEFAULT}.",
                "Applies via CODEX_TRACE_DEBUG_MAX_PER_FUNC.",
            ],
            "Conditioning Debug": [
                "Dump CLIP conditioning tensor norms during SDXL runs.",
                "Intended for diagnostics; adds extra logging noise.",
                "Applies via CODEX_DEBUG_COND or --debug-conditioning.",
            ],
            "Sampler Verbose Logs": [
                "Emit per-step sampler diagnostics (sigma schedule, timings).",
                "Applies via CODEX_LOG_SAMPLER.",
            ],
            "CFG Delta Logs": [
                "Log the cond/uncond delta inside CFG for the first N steps.",
                "Requires CODEX_LOG_SAMPLER=1; enabling it also enables sampler logs.",
                "Applies via CODEX_LOG_CFG_DELTA and CODEX_LOG_CFG_DELTA_N.",
            ],
            "CFG Delta Steps (N)": [
                "How many initial steps to log cond/uncond delta for (0 disables).",
                "Applies via CODEX_LOG_CFG_DELTA_N (default 2).",
            ],
            "Sigma Ladder Logs": [
                "Dump full sigma ladder (first/last and compact summary).",
                "Applies via CODEX_LOG_SIGMAS.",
            ],
            "Dump Latents": [
                "Save final latent tensor after each sampling run.",
                "Applies via CODEX_DUMP_LATENTS; files stored under logs/diagnostics by default.",
            ],
            "Dump Latents Path": [
                "Optional file or directory to receive latent dumps.",
                "Directory → auto filenames; file path → overwrite each run.",
                "Applies via CODEX_DUMP_LATENTS_PATH.",
            ],
            "Start API": ["Start backend API server."],
            "Restart API": ["Restart backend API server (stop + start)."],
            "Kill API": ["Force kill backend API server."],
            "Start UI": ["Start UI dev server (npm dev)."],
            "Restart UI": ["Restart UI dev server."],
            "Kill UI": ["Force kill UI dev server."],
            "Logs": [
                "Combined logs. Up/Down scroll, g/G bottom/top, c clear.",
            ],
            "Save and Exit": ["Write settings to .sangoi/launcher/ and quit."],
            "Exit Without Saving": ["Quit without writing settings."],
        }
        return d.get(key, ["<Tab>, <Shift-Tab>, or <Enter> selects field."])

    # --------------- event handling ------------------------------------
    def _act_main(self, idx: int) -> None:
        items = self._items_main()
        if not (0 <= idx < len(items)):
            return
        label, _val, action = items[idx]
        name = "API" if "API" in label else ("UI" if "UI" in label else None)
        if not name:
            return
        svc = self.services[name]
        env = self.store.build_env()
        if action == "start":
            svc.start(env, external_terminal=self.meta.external_terminal)
        elif action == "restart":
            svc.restart(env, external_terminal=self.meta.external_terminal)
        elif action == "kill":
            svc.stop()  # graceful first
            time.sleep(0.2)
            svc.kill() if hasattr(svc, 'kill') else None

    def _act_runtime(self, idx: int) -> None:
        items = self._items_runtime()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        if action == "toggle_extterm":
            self.meta.external_terminal = not self.meta.external_terminal
        else:
            return

    def _act_logging(self, idx: int) -> None:
        items = self._items_logging()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        env = self.env
        
        # Helper to toggle a boolean flag
        def _toggle_flag(key: str, default: str = "1") -> None:
            cur = env.get(key, default).strip().lower()
            env[key] = "0" if cur in ("1", "true", "yes", "on") else "1"
        
        # Codex log level toggles
        if action == "toggle_codex_debug":
            _toggle_flag("CODEX_LOG_DEBUG", "0")
        elif action == "toggle_codex_info":
            _toggle_flag("CODEX_LOG_INFO", "1")
        elif action == "toggle_codex_warning":
            _toggle_flag("CODEX_LOG_WARNING", "1")
        elif action == "toggle_codex_error":
            _toggle_flag("CODEX_LOG_ERROR", "1")
        elif action == "toggle_log_file":
            current = env.get("CODEX_LOG_FILE", "").strip()
            if current:
                env.pop("CODEX_LOG_FILE", None)
                self.message = "File logging disabled"
            else:
                log_path = self._ensure_log_file()
                env["CODEX_LOG_FILE"] = log_path
                self.message = f"Logging to {Path(log_path).name}"

    def _act_debug(self, idx: int) -> None:
        items = self._items_debug()
        if not (0 <= idx < len(items)):
            return
        _key, _val, action = items[idx]
        env = self.env

        def _toggle_flag(key: str, *, message_on: str | None = None, message_off: str | None = None) -> None:
            cur = env.get(key, "0").strip().lower()
            if cur in {"1", "true", "yes", "on"}:
                env.pop(key, None)
                if message_off:
                    self.message = message_off
            else:
                env[key] = "1"
                if message_on:
                    self.message = message_on

        if action == "toggle_cond_debug":
            _toggle_flag("CODEX_DEBUG_COND")
        elif action == "toggle_sampler_logs":
            _toggle_flag(
                "CODEX_LOG_SAMPLER",
                message_on="Sampler logs enabled (restart API to apply).",
                message_off="Sampler logs disabled.",
            )
            # CFG delta logs require sampler logs; if sampler logs are disabled, also disable delta logs.
            if env.get("CODEX_LOG_SAMPLER", "0").strip().lower() not in {"1", "true", "yes", "on"}:
                env.pop("CODEX_LOG_CFG_DELTA", None)
        elif action == "toggle_cfg_delta_logs":
            cur = env.get("CODEX_LOG_CFG_DELTA", "0").strip().lower()
            if cur in {"1", "true", "yes", "on"}:
                env.pop("CODEX_LOG_CFG_DELTA", None)
                self.message = "CFG delta logs disabled."
            else:
                env["CODEX_LOG_CFG_DELTA"] = "1"
                # Requires sampler logs to be enabled; turn them on automatically.
                env["CODEX_LOG_SAMPLER"] = "1"
                self.message = "CFG delta logs enabled (sampler logs enabled too). Restart API to apply."
        elif action == "edit_cfg_delta_n":
            val = self._prompt("CFG delta steps N (>=0): ")
            if val is None:
                return
            val = val.strip()
            if not val:
                env.pop("CODEX_LOG_CFG_DELTA_N", None)
                self.message = "CFG delta steps reset to default (2)."
                return
            try:
                numeric = max(0, int(val))
            except ValueError:
                self.message = "Enter a valid integer."
                return
            env["CODEX_LOG_CFG_DELTA_N"] = str(numeric)
            self.message = f"CFG delta steps set to {numeric}."
        elif action == "toggle_sigma_logs":
            _toggle_flag(
                "CODEX_LOG_SIGMAS",
                message_on="Sigma ladder logs enabled (restart API to apply).",
                message_off="Sigma ladder logs disabled.",
            )
        elif action == "toggle_trace_debug":
            _toggle_flag(
                "CODEX_TRACE_DEBUG",
                message_on="Trace Debug enabled. Restart API to attach call tracing.",
                message_off="Trace Debug disabled. Restart API to stop call tracing.",
            )
        elif action == "toggle_pipeline_debug":
            _toggle_flag("CODEX_PIPELINE_DEBUG")
        elif action == "toggle_dump_latents":
            _toggle_flag(
                "CODEX_DUMP_LATENTS",
                message_on="Latent dumps enabled. Results will be written after sampling.",
                message_off="Latent dumps disabled.",
            )
        elif action == "edit_dump_latents_path":
            val = self._prompt("Dump path (file or directory): ")
            if val is None:
                return
            val = val.strip()
            if val:
                env["CODEX_DUMP_LATENTS_PATH"] = val
                self.message = f"Latent dump path set to {val}"
            else:
                env.pop("CODEX_DUMP_LATENTS_PATH", None)
                self.message = "Latent dump path reset to default."
        elif action == "edit_trace_max":
            val = self._prompt("Trace max per func (>=0): ")
            if val is None:
                return
            try:
                numeric = max(0, int(val))
            except ValueError:
                self.message = "Enter a valid integer."
                return
            env["CODEX_TRACE_DEBUG_MAX_PER_FUNC"] = str(numeric)
            self.message = f"Trace limit set to {numeric}."


    def _log_lines(self) -> List[str]:
        return self.log_buffer.snapshot()

    # --------------- main render ---------------------------------------
    def render(self) -> None:
        self.stdscr.erase()
        h, w = self.stdscr.getmaxyx()
        self._draw_bars(h, w)
        top, left_w, right_w, height = self._pane_geometry()

        # Left content box
        if self.TABS[self.tab_index] == "Logs":
            # logs view uses full left area
            logs = self._log_lines()
            total = len(logs)
            start = max(0, total - height - self.log_scroll)
            lines = logs[start:start + height]
            for i in range(height):
                ln = lines[i] if i < len(lines) else ""
                try:
                    self.stdscr.addstr(top + i, 1, (ln + " " * (left_w - 2))[:left_w - 2])
                except Exception:
                    pass
            help_lines = self._help_for("Logs", "Logs")
            self._draw_help(top, left_w, right_w, height, help_lines)
        elif self.TABS[self.tab_index] == "Exit":
            options = ["Save and Exit", "Exit Without Saving"]
            for i, opt in enumerate(options):
                attr = self.C_SEL if i == self.sel_index else self.C_NRM
                self.stdscr.attron(attr)
                self.stdscr.addstr(top + i, 1, f"{opt}" + " " * max(0, left_w - 2 - len(opt)))
                self.stdscr.attroff(attr)
            self._draw_help(top, left_w, right_w, height, self._help_for("Exit", options[self.sel_index]))
        else:
            tab_name = self.TABS[self.tab_index]
            if tab_name == "Main":
                items = self._items_main()
            elif tab_name == "Runtime":
                items = self._items_runtime()
            elif tab_name == "DEBUG":
                items = self._items_debug()
            elif tab_name == "Logging":
                items = self._items_logging()
            else:
                items = []
            if self.sel_index >= len(items):
                self.sel_index = max(0, len(items) - 1)
            # draw items (name left, value right)
            label_w = 28  # widened label column
            if tab_name == "Main":
                # Two-row layout: first the statuses (even indices 0 and 4), then actions
                status_lines = [items[0], items[4]] if len(items) >= 8 else items[:2]
                action_lines = [it for it in items if it[2] in ("start", "restart", "kill")]
                y = top
                # Status row (bold labels)
                for key, val, _ in status_lines:
                    try:
                        self.stdscr.addstr(y, 1, f" {key:<{label_w}}  ", self.C_LBL | curses.A_BOLD)
                        self.stdscr.addstr(y, 2 + label_w, f"{val}"[:left_w - 4 - label_w], self.C_NRM | curses.A_BOLD)
                    except Exception:
                        pass
                    y += 1
                # Separator
                try:
                    self.stdscr.hline(y, 1, ord(' '), left_w - 2)
                except Exception:
                    pass
                y += 1
                # Actions grid
                for idx, (key, val, _action) in enumerate(action_lines):
                    sel_idx = len(status_lines) + 1 + idx
                    attr = self.C_SEL if sel_idx == self.sel_index else self.C_NRM
                    try:
                        self.stdscr.attron(attr)
                        self.stdscr.addstr(y, 1, f" {key:<{label_w}}  {val}"[:left_w - 2])
                        self.stdscr.attroff(attr)
                    except Exception:
                        pass
                    y += 1
            else:
                for i, (key, val, _action) in enumerate(items):
                    attr = self.C_SEL if i == self.sel_index else self.C_NRM
                    self.stdscr.attron(attr)
                    line = f" {key:<{label_w}}  {val:>}"[:left_w - 2]
                    self.stdscr.addstr(top + i, 1, line)
                    self.stdscr.attroff(attr)
            # help
            key = items[self.sel_index][0] if items else ""
            help_lines = self._help_for(tab_name, key)
            if tab_name == "Runtime" and self.launch_checks:
                help_lines = help_lines + ["", "Launch Checks:"]
                for chk in self.launch_checks:
                    status = "OK" if chk.ok else "FAIL"
                    help_lines.append(f"- {chk.name}: {status}")
                    help_lines.append(f"  {chk.detail}")
            self._draw_help(top, left_w, right_w, height, help_lines)

        # transient message (below tabs)
        if self.message:
            try:
                self.stdscr.addstr(top - 1, 1, (self.message + " " * (w - 2))[:w - 2], self.C_LBL)
            except Exception:
                pass
        self.stdscr.refresh()

        # Keep popup painted on top each frame to avoid flicker/overdraw
        if self.popup_active and self.popup_selection is not None:
            try:
                title, options, idx, _action = self.popup_selection
                self._draw_select_popup(title, options, idx)
            except Exception:
                pass

    def handle_key(self, ch: int) -> None:
        tab = self.TABS[self.tab_index]
        # If a popup is active, handle navigation/confirm/cancel here
        if self.popup_active and self.popup_selection is not None:
            title, options, idx, action = self.popup_selection
            # Navigate within popup
            if ch in (curses.KEY_UP, ord('k')):
                idx = (idx - 1) % len(options)
                self.popup_selection = (title, options, idx, action)
                self._draw_select_popup(title, options, idx)
                return
            if ch in (curses.KEY_DOWN, ord('j')):
                idx = (idx + 1) % len(options)
                self.popup_selection = (title, options, idx, action)
                self._draw_select_popup(title, options, idx)
                return
            if ch in (curses.KEY_LEFT, curses.KEY_RIGHT):
                return
            if ch in (curses.KEY_ENTER, ord('\n'), ord('\r'), 10, 13, 343):
                self._apply_popup_selection(action, options[idx])
                self.popup_active = False
                self.popup_selection = None
                return
            if ch == 27:  # ESC cancels popup
                self.popup_active = False
                self.popup_selection = None
                return
        if ch in (curses.KEY_LEFT, ord('\t') - 256):
            self.tab_index = (self.tab_index - 1) % len(self.TABS)
            self.sel_index = 0
            self.meta.tab_index = self.tab_index
            if self.TABS[self.tab_index] == "Runtime":
                self.launch_checks = run_launch_checks()
            return
        if ch in (curses.KEY_RIGHT,):
            self.tab_index = (self.tab_index + 1) % len(self.TABS)
            self.sel_index = 0
            self.meta.tab_index = self.tab_index
            if self.TABS[self.tab_index] == "Runtime":
                self.launch_checks = run_launch_checks()
            return
        if ch in (curses.KEY_UP, ord('k')):
            self.sel_index = max(0, self.sel_index - 1)
            return
        if ch in (curses.KEY_DOWN, ord('j')):
            self.sel_index += 1
            return
        if ch in (curses.KEY_ENTER, ord('\n'), ord('\r'), 10, 13, 343):
            if tab == "Main":
                self._act_main(self.sel_index)
            elif tab == "Runtime":
                # Open BIOS-style popup for multi-choice fields
                self._act_runtime_popup_or_apply(self.sel_index)
            elif tab == "DEBUG":
                self._act_debug(self.sel_index)
            elif tab == "Logging":
                self._act_logging(self.sel_index)
            elif tab == "Exit":
                if self.sel_index % 2 == 0:
                    self._persist()
                raise SystemExit
            return
        if ch in (ord('+'), ord('-')):
            if tab == "Runtime":
                self._act_runtime(self.sel_index)
            elif tab == "DEBUG":
                self._act_debug(self.sel_index)
            elif tab == "Logging":
                self._act_logging(self.sel_index)
            return
        if tab == "Logs":
            logs = self._log_lines()
            if ch in (ord('g'),):
                self.log_scroll = 0
                return
            if ch in (ord('G'),):
                self.log_scroll = max(0, len(logs) - 1)
                return
            if ch in (ord('c'), ord('C')):
                self.log_buffer.clear()
                self.log_scroll = 0
                return
        # Function keys shortcuts
        if ch == curses.KEY_F2:
            env = self.store.build_env()
            self.services["API"].start(env, external_terminal=self.meta.external_terminal)
            return
        if ch == curses.KEY_F3:
            # fast kill API
            self.services["API"].kill()
            return
        if ch == curses.KEY_F4:
            env = self.store.build_env()
            self.services["UI"].start(env, external_terminal=self.meta.external_terminal)
            return
        if ch == curses.KEY_F5:
            self.services["UI"].kill()
            return
        if ch == curses.KEY_F1:
            self._show_help_popup()
            return
        if ch == 27:  # ESC
            if self.popup_active:
                self.popup_active = False
                self.popup_selection = None
                return
            raise SystemExit
        if ch == curses.KEY_F10:
            self._persist()
            raise SystemExit

    # ---------------- popup helpers ------------------------------------
    def _show_help_popup(self) -> None:
        lines = [
            "Keyboard Shortcuts:",
            "- Left/Right: Change Tab",
            "- Up/Down: Move Selection",
            "- Enter: Select / Open popup",
            "- +/-: Change value inline",
            "- F2 / F3: Start / Kill API",
            "- F4 / F5: Start / Kill UI",
            "- F10: Save and Exit",
            "- Esc: Exit (or close popup)",
        ]
        self._draw_popup("Help", lines)

    def _draw_popup(self, title: str, lines: List[str], *, selectable: bool = False, options: Optional[List[str]] = None, index: int = 0) -> None:
        h, w = self.stdscr.getmaxyx()
        ph = min(len(lines) + 4, max(8, len(lines) + 4))
        pw = min(max(len(title) + 8, max((len(l) for l in lines), default=30) + 4), w - 4)
        y0 = max(2, (h - ph) // 2)
        x0 = max(2, (w - pw) // 2)
        win = curses.newwin(ph, pw, y0, x0)
        win.bkgd(' ', self.C_NRM)
        win.box()
        try:
            win.addstr(0, 2, f" {title} ", self.C_TAB)
        except Exception:
            pass
        for i, ln in enumerate(lines):
            try:
                win.addstr(2 + i, 2, ln[: pw - 4])
            except Exception:
                pass
        win.refresh()
        self.popup_active = True
        # Wait for any key to close if not selectable
        if not selectable:
            self.stdscr.nodelay(False)
            win.getch()
            self.stdscr.nodelay(True)
            self.popup_active = False
            return

    def _act_runtime_popup_or_apply(self, idx: int) -> None:
        return self._act_runtime(idx)
    def _draw_select_popup(self, title: str, options: List[str], index: int) -> None:
        self.popup_active = True
        self.stdscr.nodelay(True)
        h, w = self.stdscr.getmaxyx()
        ph = min(len(options) + 4, max(8, len(options) + 4))
        pw = min(max(len(title) + 8, max((len(o) for o in options), default=20) + 6), w - 4)
        y0 = max(2, (h - ph) // 2)
        x0 = max(2, (w - pw) // 2)
        win = curses.newwin(ph, pw, y0, x0)
        win.bkgd(' ', self.C_NRM)
        win.box()
        try:
            win.addstr(0, 2, f" {title} ", self.C_TAB)
        except Exception:
            pass
        for i, opt in enumerate(options):
            attr = self.C_SEL if i == index else self.C_NRM
            try:
                win.attron(attr)
                win.addstr(2 + i, 2, opt.ljust(pw - 4)[: pw - 4])
                win.attroff(attr)
            except Exception:
                pass
        win.refresh()
        # update current popup selection tuple if exists (keep action and options)
        if self.popup_selection is not None:
            title0, opts0, _idx0, action0 = self.popup_selection
            # if options list changed, keep as is; otherwise set new index
            if opts0 == options and title0 == title:
                self.popup_selection = (title, options, index, action0)

    def _apply_popup_selection(self, action: str, value: str) -> None:
        _ = (action, value)
        self.message = "Popup setting selection removed; configure runtime settings in the Web UI."



def main(stdscr) -> None:
    locale.setlocale(locale.LC_ALL, '')
    try:
        curses.curs_set(0)
    except Exception:
        pass
    stdscr.nodelay(True)
    stdscr.timeout(100)
    app = BIOSApp(stdscr)
    while True:
        app.render()
        try:
            ch = stdscr.getch()
        except KeyboardInterrupt:
            break
        if ch == -1:
            continue
        try:
            app.handle_key(ch)
        except SystemExit:
            break
        except Exception as exc:
            app.message = f"Error: {exc}"


if __name__ == "__main__":
    curses.wrapper(main)
