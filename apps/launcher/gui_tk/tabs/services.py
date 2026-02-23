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

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import threading
import time
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Dict, Tuple
from urllib import error as url_error
from urllib import request as url_request
import webbrowser

from apps.backend.infra.config.repo_root import get_repo_root
from apps.launcher.services import ServiceStatus

from ..controller import LauncherController

LOGGER = logging.getLogger("codex.launcher.gui_tk.services_tab")


@dataclass(slots=True)
class _ServiceCard:
    status_var: tk.StringVar
    info_var: tk.StringVar
    endpoint_var: tk.StringVar
    health_var: tk.StringVar
    status_label: ttk.Label
    endpoint_label: ttk.Label
    health_label: ttk.Label
    start_btn: ttk.Button
    restart_btn: ttk.Button
    stop_btn: ttk.Button
    kill_btn: ttk.Button
    open_btn: ttk.Button
    open_docs_btn: ttk.Button | None


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
        self._health_lock = threading.Lock()
        self._health_cache: Dict[str, Dict[str, object]] = {}
        self._health_targets: Dict[str, str] = {}
        self._health_stop_event = threading.Event()
        self._health_thread: threading.Thread | None = None

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
        ttk.Button(btns, text="Start All", command=self._start_all).pack(side="left", padx=(0, 8))
        ttk.Button(btns, text="Stop All", command=self._stop_all).pack(side="left")

        row = 1
        for svc_name in ("API", "UI"):
            card_frame = ttk.LabelFrame(frame, text=f"  {svc_name}  ", padding=16, style="Service.Card.TLabelframe")
            card_frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 10))
            card_frame.columnconfigure(1, weight=1)

            status_var = tk.StringVar(value="STOPPED")
            info_var = tk.StringVar(value="")
            endpoint_var = tk.StringVar(value="-")
            health_var = tk.StringVar(value="Health: stopped")

            ttk.Label(card_frame, text="Status:").grid(row=0, column=0, sticky="w", padx=(0, 10))
            status_lbl = ttk.Label(card_frame, textvariable=status_var, style="Status.Stopped.TLabel")
            status_lbl.grid(row=0, column=1, sticky="w")
            ttk.Label(card_frame, textvariable=info_var, style="Service.Info.TLabel").grid(row=0, column=2, sticky="e")

            ttk.Label(card_frame, text="Endpoint:").grid(row=1, column=0, sticky="w", padx=(0, 10), pady=(6, 0))
            endpoint_lbl = ttk.Label(card_frame, textvariable=endpoint_var, style="Service.Endpoint.TLabel")
            endpoint_lbl.grid(row=1, column=1, sticky="w", pady=(6, 0))
            health_lbl = ttk.Label(card_frame, textvariable=health_var, style="Health.Stopped.TLabel")
            health_lbl.grid(row=1, column=2, sticky="e", pady=(6, 0))

            btn_row = ttk.Frame(card_frame)
            btn_row.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(12, 0))

            start_btn = ttk.Button(btn_row, text="Start", width=12, command=lambda n=svc_name: self._start(n))
            restart_btn = ttk.Button(btn_row, text="Restart", width=12, command=lambda n=svc_name: self._restart(n))
            stop_btn = ttk.Button(btn_row, text="Stop", width=12, command=lambda n=svc_name: self._stop(n))
            kill_btn = ttk.Button(btn_row, text="Kill", width=12, command=lambda n=svc_name: self._kill(n))
            open_btn = ttk.Button(btn_row, text="Open", width=10, command=lambda n=svc_name: self._open_service(n, target="root"))
            open_docs_btn: ttk.Button | None = None
            if svc_name == "API":
                open_docs_btn = ttk.Button(
                    btn_row,
                    text="Docs",
                    width=10,
                    command=lambda n=svc_name: self._open_service(n, target="docs"),
                )

            start_btn.pack(side="left", padx=(0, 8))
            restart_btn.pack(side="left", padx=(0, 8))
            stop_btn.pack(side="left", padx=(0, 8))
            kill_btn.pack(side="left")
            open_btn.pack(side="right")
            if open_docs_btn is not None:
                open_docs_btn.pack(side="right", padx=(0, 8))

            self._cards[svc_name] = _ServiceCard(
                status_var=status_var,
                info_var=info_var,
                endpoint_var=endpoint_var,
                health_var=health_var,
                status_label=status_lbl,
                endpoint_label=endpoint_lbl,
                health_label=health_lbl,
                start_btn=start_btn,
                restart_btn=restart_btn,
                stop_btn=stop_btn,
                kill_btn=kill_btn,
                open_btn=open_btn,
                open_docs_btn=open_docs_btn,
            )
            row += 1

        self._start_health_worker()
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
            root_url, _docs_url, health_url = self._resolve_service_urls(svc_name)
            card.endpoint_var.set(root_url)
            self._set_health_target(svc_name, health_url)

            if status == ServiceStatus.RUNNING:
                card.status_label.configure(style="Status.Running.TLabel")
            elif status == ServiceStatus.ERROR:
                card.status_label.configure(style="Status.Error.TLabel")
            else:
                card.status_label.configure(style="Status.Stopped.TLabel")

            self._set_widget_enabled(card.start_btn, enabled=(status != ServiceStatus.RUNNING))
            self._set_widget_enabled(card.restart_btn, enabled=(status == ServiceStatus.RUNNING))
            self._set_widget_enabled(card.stop_btn, enabled=(status == ServiceStatus.RUNNING))
            self._set_widget_enabled(card.kill_btn, enabled=(status == ServiceStatus.RUNNING))
            if card.open_docs_btn is not None:
                self._set_widget_enabled(card.open_docs_btn, enabled=(status == ServiceStatus.RUNNING))
            self._set_widget_enabled(card.open_btn, enabled=(status == ServiceStatus.RUNNING))

            health_state = self._health_snapshot(svc_name)
            health_label = str(health_state.get("label", "Health: stopped"))
            health_style = str(health_state.get("style", "Health.Stopped.TLabel"))
            if status != ServiceStatus.RUNNING:
                health_label = "Health: stopped"
                health_style = "Health.Stopped.TLabel"
            card.health_var.set(health_label)
            card.health_label.configure(style=health_style)

    def _external_terminal(self) -> bool:
        return bool(self._external_terminal_var.get()) and self._controller.external_terminal_supported

    @staticmethod
    def _set_widget_enabled(widget: ttk.Widget, *, enabled: bool) -> None:
        target_state = "normal" if enabled else "disabled"
        current_state = str(widget.cget("state"))
        if current_state == target_state:
            return
        widget.configure(state=target_state)

    def _service_runtime_env(self, service_name: str) -> Dict[str, str]:
        service = self._controller.services[service_name]
        env: Dict[str, str] = dict(os.environ)
        env.update(service.spec.base_env)
        env.update(self._controller.build_env())
        return env

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    @staticmethod
    def _parse_port(raw_value: object, *, default: int) -> int:
        try:
            parsed = int(str(raw_value or "").strip())
        except (TypeError, ValueError):
            return int(default)
        if parsed < 1 or parsed > 65535:
            return int(default)
        return int(parsed)

    @staticmethod
    def _parse_int(raw_value: object, *, default: int) -> int:
        try:
            return int(str(raw_value or "").strip())
        except (TypeError, ValueError):
            return int(default)

    @staticmethod
    def _resolve_ui_effective_port(*, base_port: int, interface_cwd: Path) -> int:
        repo_root = get_repo_root()
        expected_cwd = str(interface_cwd.resolve())
        for candidate in (int(base_port), int(base_port) + 10000, int(base_port) + 20000):
            pid_file = repo_root / f".webui-ui-{candidate}.pid"
            if not pid_file.is_file():
                continue
            try:
                payload = json.loads(pid_file.read_text(encoding="utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if str(payload.get("service", "")).strip().lower() != "ui":
                continue
            if ServicesTab._parse_port(payload.get("port"), default=-1) != candidate:
                continue
            raw_cwd = str(payload.get("cwd", "")).strip()
            if raw_cwd:
                try:
                    if str(Path(raw_cwd).resolve()) != expected_cwd:
                        continue
                except (TypeError, ValueError, OSError):
                    continue
            pid = ServicesTab._parse_int(payload.get("pid"), default=-1)
            if not ServicesTab._pid_is_alive(pid):
                continue
            return int(candidate)
        return int(base_port)

    def _resolve_service_urls(self, service_name: str) -> Tuple[str, str | None, str]:
        def _browser_host(raw_host: str) -> str:
            host = str(raw_host or "localhost").strip() or "localhost"
            if host in {"0.0.0.0", "::", "[::]"}:
                return "localhost"
            return host

        env = self._service_runtime_env(service_name)
        service = self._controller.services[service_name]
        if service_name == "API":
            host = _browser_host(str(env.get("API_HOST", "localhost")))
            api_port = self._parse_port(
                env.get("API_PORT_OVERRIDE", service.spec.base_env.get("API_PORT_OVERRIDE", "7850")),
                default=7850,
            )
            port = str(api_port)
            root_url = f"http://{host}:{port}"
            return root_url, f"{root_url}/docs", f"{root_url}/api/health"

        host = _browser_host(str(env.get("SERVER_HOST", "localhost")))
        base_port = self._parse_port(env.get("WEB_PORT", service.spec.base_env.get("WEB_PORT", "7860")), default=7860)
        effective_port = self._resolve_ui_effective_port(base_port=base_port, interface_cwd=Path(service.spec.cwd))
        port = str(effective_port)
        root_url = f"http://{host}:{port}"
        return root_url, None, root_url

    def _health_snapshot(self, service_name: str) -> Dict[str, object]:
        with self._health_lock:
            return dict(self._health_cache.get(service_name, {"label": "Health: unknown", "style": "Health.Stopped.TLabel"}))

    def _set_health_snapshot(self, service_name: str, payload: Dict[str, object]) -> None:
        with self._health_lock:
            self._health_cache[service_name] = dict(payload)

    def _set_health_target(self, service_name: str, health_url: str) -> None:
        with self._health_lock:
            self._health_targets[service_name] = str(health_url)

    def _health_target(self, service_name: str) -> str:
        with self._health_lock:
            return str(self._health_targets.get(service_name, ""))

    def _probe_health(self, service_name: str, health_url: str, *, timeout_s: float = 0.75) -> Dict[str, object]:
        start = time.perf_counter()
        try:
            request = url_request.Request(health_url, method="GET")
            with url_request.urlopen(request, timeout=timeout_s) as response:
                payload = response.read()
                elapsed_ms = int((time.perf_counter() - start) * 1000)
                if service_name == "API":
                    parsed = json.loads(payload.decode("utf-8", errors="replace"))
                    if not bool(parsed.get("ok")):
                        return {"label": "Health: fail", "style": "Health.Error.TLabel", "checked_at": time.time()}
                return {"label": f"Health: ok ({elapsed_ms} ms)", "style": "Health.Ok.TLabel", "checked_at": time.time()}
        except (url_error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            return {"label": "Health: unreachable", "style": "Health.Error.TLabel", "checked_at": time.time()}

    def _health_worker(self) -> None:
        poll_s = 2.0
        while not self._health_stop_event.is_set():
            for service_name in self._cards:
                try:
                    service = self._controller.services[service_name]
                    if service.status != ServiceStatus.RUNNING:
                        self._set_health_snapshot(
                            service_name,
                            {"label": "Health: stopped", "style": "Health.Stopped.TLabel", "checked_at": time.time()},
                        )
                        continue
                    health_url = self._health_target(service_name)
                    if not health_url:
                        self._set_health_snapshot(
                            service_name,
                            {"label": "Health: unknown", "style": "Health.Stopped.TLabel", "checked_at": time.time()},
                        )
                        continue
                    result = self._probe_health(service_name, health_url)
                    self._set_health_snapshot(service_name, result)
                except Exception:
                    LOGGER.exception("Services health worker failed while polling %s", service_name)
                    self._set_health_snapshot(
                        service_name,
                        {"label": "Health: error", "style": "Health.Error.TLabel", "checked_at": time.time()},
                    )
            if self._health_stop_event.wait(poll_s):
                break

    def _start_health_worker(self) -> None:
        if self._health_thread and self._health_thread.is_alive():
            return
        self._health_stop_event.clear()
        worker = threading.Thread(target=self._health_worker, daemon=True, name="launcher-services-health")
        worker.start()
        self._health_thread = worker

    def dispose(self) -> None:
        self._health_stop_event.set()
        thread = self._health_thread
        if thread and thread.is_alive():
            thread.join(timeout=1.0)
        self._health_thread = None

    def _open_service(self, service_name: str, *, target: str) -> None:
        root_url, docs_url, _health_url = self._resolve_service_urls(service_name)
        target_url = root_url
        if target == "docs":
            if docs_url is None:
                messagebox.showerror("Launcher Error", f"{service_name} does not expose a docs URL.")
                return
            target_url = docs_url
        try:
            opened = webbrowser.open(target_url)
        except Exception as exc:
            messagebox.showerror("Launcher Error", f"Failed to open {target_url}:\n\n{exc}")
            return
        if not opened:
            messagebox.showerror("Launcher Error", f"Browser did not open {target_url}.")
            return
        self._set_status(f"Opened {target_url}")

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
