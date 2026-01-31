"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Launcher service specs and process supervision (API + UI).
Defines service specs/handles, spawns subprocesses with environment overrides, streams logs into a shared buffer, and performs strict port
availability checks (IPv4/IPv6) before starting the API.

Symbols (top-level; keep in sync; no ghosts):
- `ServiceStatus` (enum): Launcher service lifecycle status.
- `CodexServiceSpec` (dataclass): Static service definition (command/cwd/env + external-terminal policy).
- `CodexServiceHandle` (dataclass): Runtime service handle; spawns/monitors subprocess and forwards stdout/stderr to a log buffer.
- `_codex_root` (function): Resolves the repo root used for service working directories.
- `default_services` (function): Builds default API+UI service handles with ports/env derived from the environment.
- `_api_backend_args_from_env` (function): Builds backend CLI args for the API service from launcher env settings.
- `_extract_cli_port` (function): Extracts a `--port` value from a command list.
- `_port_free_everywhere` (function): Validates a port is bindable on common IPv4/IPv6 local hosts.
- `_windows_no_activate` (function): Windows startupinfo helper to open consoles without stealing focus.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
import errno
import socket
from contextlib import closing
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterator, List, Mapping, Optional

from .log_buffer import CodexLogBuffer
from apps.backend.infra.config.repo_root import get_repo_root

LOGGER = logging.getLogger("codex.launcher.services")


class ServiceStatus(StrEnum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass(slots=True)
class CodexServiceSpec:
    """Static definition of a service the launcher can run."""

    name: str
    command: List[str]
    cwd: Path
    base_env: Mapping[str, str] = field(default_factory=dict)
    allow_external_terminal: bool = False


@dataclass
class CodexServiceHandle:
    spec: CodexServiceSpec
    log_buffer: Optional[CodexLogBuffer] = None
    process: subprocess.Popen[str] | None = None
    status: ServiceStatus = ServiceStatus.STOPPED
    pid: Optional[int] = None
    started_at: Optional[float] = None
    last_exit_code: int | None = None
    process_group_id: int | None = None
    _stop_requested: bool = False
    _stop_reason: str | None = None
    _stdout_thread: Optional[threading.Thread] = None
    _stderr_thread: Optional[threading.Thread] = None
    _queue: Queue[str] = field(default_factory=Queue, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)

    def start(self, overrides: Mapping[str, str], *, external_terminal: bool = False) -> None:
        with self._lock:
            proc = self.process
            if proc and proc.poll() is None:
                LOGGER.info("Service %s already running (pid=%s)", self.spec.name, proc.pid)
                self.status = ServiceStatus.RUNNING
                self.pid = proc.pid
                return

        overrides_map = dict(overrides)

        env = os.environ.copy()
        env.update(self.spec.base_env)
        env.update(overrides_map)

        command = list(self.spec.command)
        if self.spec.name.upper() == "API":
            command.extend(_api_backend_args_from_env(env))
            port = _extract_cli_port(command)
            if port is None:
                raw_env_port = env.get("API_PORT_OVERRIDE") or env.get("API_PORT")
                if raw_env_port is not None:
                    try:
                        port = int(str(raw_env_port).strip())
                    except Exception:
                        port = None
            if port is not None:
                ok, blocked = _port_free_everywhere(port)
                if not ok:
                    hint = (
                        f"API port {port} is busy ({blocked}). "
                        "You may already have Codex running (WSL/Windows) or another service bound on IPv4/IPv6 localhost. "
                        "Stop the other instance or set API_PORT_OVERRIDE/WEB_PORT to a free pair."
                    )
                    if self.log_buffer:
                        self.log_buffer.log("launcher", hint)
                    raise RuntimeError(hint)
        flags = 0
        startupinfo = None
        use_external = external_terminal and self.spec.allow_external_terminal
        if use_external and os.name == "nt":
            flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            startupinfo = _windows_no_activate()
            command = ["cmd.exe", "/K", subprocess.list2cmdline(command)]
        elif os.name == "nt":
            # Needed for CTRL_BREAK_EVENT to be deliverable (best-effort graceful stop).
            flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
        stdin = subprocess.DEVNULL
        if use_external:
            stdout = None
            stderr = None
            stdin = None

        with self._lock:
            self.status = ServiceStatus.STARTING
            self.started_at = time.time()
            self._stop_requested = False
            self._stop_reason = None
            self.last_exit_code = None
            self.process_group_id = None
        try:
            popen_kwargs: dict[str, object] = {}
            if os.name != "nt" and not use_external:
                popen_kwargs["start_new_session"] = True
            proc = subprocess.Popen(
                command,
                cwd=str(self.spec.cwd),
                env=env,
                stdin=stdin,
                stdout=stdout,
                stderr=stderr,
                text=True if stdout is not None else False,
                bufsize=1,
                creationflags=flags,
                startupinfo=startupinfo,
                **popen_kwargs,
            )
        except Exception as exc:
            with self._lock:
                self.status = ServiceStatus.ERROR
                self.pid = None
                self.process = None
            LOGGER.error("Failed to start %s: %s", self.spec.name, exc)
            raise

        with self._lock:
            self.process = proc
            self.pid = proc.pid
        if os.name != "nt" and not use_external:
            # When `start_new_session=True`, the child becomes a new process group leader.
            with self._lock:
                self.process_group_id = proc.pid
        with self._lock:
            self.status = ServiceStatus.RUNNING
        if stdout is subprocess.PIPE and stderr is subprocess.PIPE:
            self._stdout_thread = threading.Thread(target=self._capture_output, args=("stdout",), daemon=True)
            self._stdout_thread.start()
            self._stderr_thread = threading.Thread(target=self._capture_output, args=("stderr",), daemon=True)
            self._stderr_thread.start()
        threading.Thread(target=self._wait_for_exit, args=(proc,), daemon=True).start()

    def stop(self, *, wait: float = 10.0) -> None:
        with self._lock:
            proc = self.process
            if not proc or proc.poll() is not None:
                self.status = ServiceStatus.STOPPED
                self.pid = None
                self.process = None
                self.process_group_id = None
                self.started_at = None
                return
            LOGGER.info("Stopping service %s", self.spec.name)
            self._stop_requested = True
            if not self._stop_reason:
                self._stop_reason = "stopped"
            reason = str(self._stop_reason or "stopped")
            pgid = self.process_group_id
        exit_code: int | None = None
        try:
            if os.name == "nt":
                try:
                    proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))  # type: ignore[attr-defined]
                except Exception:
                    proc.terminate()
            else:
                if pgid:
                    os.killpg(pgid, signal.SIGTERM)
                else:
                    proc.terminate()
            exit_code = proc.wait(timeout=wait)
            with self._lock:
                self.last_exit_code = exit_code
        except Exception:
            LOGGER.warning("Terminate failed, killing %s", self.spec.name)
            try:
                if os.name != "nt" and pgid:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except Exception:
                pass
        finally:
            with self._lock:
                if self.process is proc:
                    self.status = ServiceStatus.STOPPED
                    self.pid = None
                    self.process = None
                    self.process_group_id = None
                    self.started_at = None
            if self.log_buffer and exit_code is not None:
                self.log_buffer.log(self.spec.name, f"{reason} (code {exit_code})", stream="event")

    def kill(self, *, wait: float = 10.0) -> None:
        with self._lock:
            proc = self.process
            if not proc or proc.poll() is not None:
                self.status = ServiceStatus.STOPPED
                self.pid = None
                self.process = None
                self.process_group_id = None
                self.started_at = None
                return
            LOGGER.warning("Killing service %s", self.spec.name)
            self._stop_requested = True
            if not self._stop_reason:
                self._stop_reason = "killed"
            reason = str(self._stop_reason or "killed")
            pgid = self.process_group_id
        exit_code: int | None = None
        try:
            if os.name != "nt" and pgid:
                os.killpg(pgid, signal.SIGKILL)
            else:
                proc.kill()
            try:
                exit_code = proc.wait(timeout=wait)
                with self._lock:
                    self.last_exit_code = exit_code
            except Exception:
                pass
        finally:
            with self._lock:
                if self.process is proc:
                    self.status = ServiceStatus.STOPPED
                    self.pid = None
                    self.process = None
                    self.process_group_id = None
                    self.started_at = None
            if self.log_buffer and exit_code is not None:
                self.log_buffer.log(self.spec.name, f"{reason} (code {exit_code})", stream="event")

    def restart(self, overrides: Mapping[str, str], *, external_terminal: bool = False) -> None:
        with self._lock:
            self._stop_requested = True
            self._stop_reason = "restarting"
        self.stop(wait=10.0)
        time.sleep(0.2)
        self.start(overrides, external_terminal=external_terminal)

    def iterate_live_output(self) -> Iterator[str]:
        while self.process and self.process.poll() is None:
            try:
                yield self._queue.get(timeout=0.1)
            except Empty:
                continue

    def _capture_output(self, stream_name: str) -> None:
        proc = self.process
        if not proc:
            return
        stream = getattr(proc, stream_name, None)
        if stream is None:
            return
        try:
            for line in stream:
                cleaned = (line.rstrip("\n") or " ")
                if self.log_buffer:
                    self.log_buffer.log(self.spec.name, cleaned, stream=stream_name)
                self._queue.put(cleaned)
        except Exception:
            pass
        finally:
            try:
                stream.close()
            except Exception:
                pass

    def _wait_for_exit(self, proc: subprocess.Popen) -> None:
        code = proc.wait()
        with self._lock:
            if self.process is not proc:
                return
            self.last_exit_code = code
            if self._stop_requested:
                status = ServiceStatus.STOPPED
                reason = str(self._stop_reason or "stopped")
                message = f"{reason} (code {code})"
            else:
                if code == 0:
                    status = ServiceStatus.STOPPED
                    message = "exited cleanly"
                else:
                    status = ServiceStatus.ERROR
                    message = f"exited with code {code}"
            self.status = status
            self.pid = None
            self.process = None
            self.process_group_id = None
            self.started_at = None
        if self.log_buffer and not self._stop_requested:
            self.log_buffer.log(self.spec.name, message, stream="event")


def _codex_root() -> Path:
    return get_repo_root()


def default_services(log_buffer: CodexLogBuffer | None = None) -> Dict[str, CodexServiceHandle]:
    root = _codex_root()
    py_exe = Path(sys.executable)
    api_port = os.getenv("API_PORT_OVERRIDE", "7850")
    web_port = os.getenv("WEB_PORT", "7860")
    api_spec = CodexServiceSpec(
        name="API",
        command=[
            str(py_exe),
            str(root / "apps" / "backend" / "interfaces" / "api" / "run_api.py"),
        ],
        cwd=root,
        base_env={"PYTHONUNBUFFERED": "1", "API_PORT_OVERRIDE": str(api_port)},
        allow_external_terminal=True,
    )
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    ui_spec = CodexServiceSpec(
        name="UI",
        command=[npm_cmd, "run", "dev", "--", "--host"],
        cwd=root / "apps" / "interface",
        base_env={
            "FORCE_COLOR": "1",
            "API_HOST": "localhost",
            "API_PORT": str(api_port),
            "WEB_PORT": str(web_port),
            "SERVER_HOST": "localhost",
        },
        allow_external_terminal=os.name == "nt",
    )
    return {
        "API": CodexServiceHandle(api_spec, log_buffer=log_buffer),
        "UI": CodexServiceHandle(ui_spec, log_buffer=log_buffer),
    }


def _api_backend_args_from_env(env: Mapping[str, str]) -> List[str]:
    args: List[str] = []

    # Device defaults (required at backend bootstrap; explicit, no silent fallbacks).
    raw_core_device = str(env.get("CODEX_CORE_DEVICE", "") or "").strip().lower()
    if raw_core_device:
        args.append(f"--core-device={raw_core_device}")

    raw_te_device = str(env.get("CODEX_TE_DEVICE", "") or "").strip().lower()
    if raw_te_device:
        args.append(f"--te-device={raw_te_device}")

    raw_vae_device = str(env.get("CODEX_VAE_DEVICE", "") or "").strip().lower()
    if raw_vae_device:
        args.append(f"--vae-device={raw_vae_device}")

    raw_exec = str(env.get("CODEX_GGUF_EXEC", "") or "").strip().lower()
    if raw_exec:
        args.append(f"--gguf-exec={raw_exec}")

    raw_dequant_cache = str(env.get("CODEX_GGUF_DEQUANT_CACHE", "") or "").strip().lower()
    if raw_dequant_cache:
        args.append(f"--gguf-dequant-cache={raw_dequant_cache}")

    raw_dequant_cache_limit = str(env.get("CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB", "") or "").strip()
    if raw_dequant_cache_limit:
        try:
            limit_mb = int(raw_dequant_cache_limit)
        except Exception as exc:
            raise ValueError(f"CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB must be an integer (got {raw_dequant_cache_limit!r}).") from exc
        if limit_mb <= 0:
            raise ValueError(f"CODEX_GGUF_DEQUANT_CACHE_LIMIT_MB must be > 0 (got {limit_mb}).")
        args.append(f"--gguf-dequant-cache-limit-mb={limit_mb}")

    raw_lora_mode = str(env.get("CODEX_LORA_APPLY_MODE", "") or "").strip().lower()
    if raw_lora_mode:
        args.append(f"--lora-apply-mode={raw_lora_mode}")

    raw_lora_math = str(env.get("CODEX_LORA_ONLINE_MATH", "") or "").strip().lower()
    if raw_lora_math:
        args.append(f"--lora-online-math={raw_lora_math}")

    return args


def _extract_cli_port(command: List[str]) -> int | None:
    for idx, token in enumerate(command):
        if token == "--port" and idx + 1 < len(command):
            try:
                return int(command[idx + 1])
            except Exception:
                return None
        if token.startswith("--port="):
            try:
                return int(token.split("=", 1)[1])
            except Exception:
                return None
    return None


def _port_free_everywhere(port: int) -> tuple[bool, str]:
    def _can_bind(family: int, host: str) -> tuple[bool, str]:
        try:
            with closing(socket.socket(family, socket.SOCK_STREAM)) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if family == socket.AF_INET6:
                    s.bind((host, port, 0, 0))
                else:
                    s.bind((host, port))
                return True, ""
        except OSError as exc:
            if getattr(exc, "errno", None) in (errno.EAFNOSUPPORT, errno.EADDRNOTAVAIL):
                return True, ""
            code = getattr(exc, "errno", None)
            return False, f"host={host} errno={code}"

    for family, host in (
        (socket.AF_INET, "0.0.0.0"),
        (socket.AF_INET, "127.0.0.1"),
        (socket.AF_INET6, "::"),
        (socket.AF_INET6, "::1"),
    ):
        ok, detail = _can_bind(family, host)
        if not ok:
            return False, detail
    return True, ""


def _windows_no_activate():
    if os.name != "nt":
        return None
    try:
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = 4  # SW_SHOWNOACTIVATE
        return startupinfo
    except Exception:
        return None
