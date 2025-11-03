from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from queue import Empty, Queue
from typing import Dict, Iterator, List, Mapping, Optional


from .log_buffer import CodexLogBuffer

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
    _stdout_thread: Optional[threading.Thread] = None
    _queue: Queue[str] = field(default_factory=Queue, init=False)

    def start(self, overrides: Mapping[str, str], *, external_terminal: bool = False) -> None:
        if self.process and self.process.poll() is None:
            LOGGER.info("Service %s already running (pid=%s)", self.spec.name, self.process.pid)
            self.status = ServiceStatus.RUNNING
            self.pid = self.process.pid
            return

        overrides_map = dict(overrides)
        cli_args: list[str] = []
        if self.spec.name.upper() == "API":
            device_flags = {
                "CODEX_DIFFUSION_DEVICE": "--core-device",
                "CODEX_TE_DEVICE": "--te-device",
                "CODEX_VAE_DEVICE": "--vae-device",
            }
            dtype_flags = {
                "CODEX_DIFFUSION_DTYPE": "--core-dtype",
                "CODEX_TE_DTYPE": "--te-dtype",
                "CODEX_VAE_DTYPE": "--vae-dtype",
            }
            for mapping in (device_flags, dtype_flags):
                for env_key, flag in mapping.items():
                    raw = overrides_map.pop(env_key, None)
                    if raw is None:
                        continue
                    text = str(raw).strip().lower()
                    if not text or text == "auto":
                        continue
                    cli_args.extend([flag, text])
            smart_env = overrides_map.get("CODEX_SMART_OFFLOAD")
            if smart_env is not None:
                if str(smart_env).strip().lower() in {"1", "true", "yes", "on"}:
                    cli_args.append("--smart-offload")
            debug_cond_env = overrides_map.get("CODEX_DEBUG_COND")
            if debug_cond_env is not None:
                if str(debug_cond_env).strip().lower() in {"1", "true", "yes", "on"}:
                    cli_args.append("--debug-conditioning")
            pin_shared_env = overrides_map.get("CODEX_PIN_SHARED_MEMORY")
            if pin_shared_env is not None:
                if str(pin_shared_env).strip().lower() in {"1", "true", "yes", "on"}:
                    cli_args.append("--pin-shared-memory")

        env = os.environ.copy()
        env.update(self.spec.base_env)
        env.update(overrides_map)

        command = list(self.spec.command)
        if cli_args:
            command.extend(cli_args)
        flags = 0
        startupinfo = None
        use_external = external_terminal and self.spec.allow_external_terminal
        if use_external and os.name == "nt":
            flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            startupinfo = _windows_no_activate()
            command = ["cmd.exe", "/K", subprocess.list2cmdline(command)]

        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        stdin = subprocess.DEVNULL
        if use_external:
            stdout = None
            stderr = None
            stdin = None

        self.status = ServiceStatus.STARTING
        self.started_at = time.time()
        try:
            self.process = subprocess.Popen(
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
            )
        except Exception as exc:
            self.status = ServiceStatus.ERROR
            self.pid = None
            self.process = None
            LOGGER.error("Failed to start %s: %s", self.spec.name, exc)
            raise

        self.pid = self.process.pid
        self.status = ServiceStatus.RUNNING
        if stdout is subprocess.PIPE:
            self._stdout_thread = threading.Thread(target=self._capture_output, daemon=True)
            self._stdout_thread.start()
        threading.Thread(target=self._wait_for_exit, daemon=True).start()

    def stop(self, *, wait: float = 10.0) -> None:
        if not self.process or self.process.poll() is not None:
            self.status = ServiceStatus.STOPPED
            self.pid = None
            self.process = None
            return
        LOGGER.info("Stopping service %s", self.spec.name)
        try:
            if os.name == "nt":
                self.process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))  # type: ignore[attr-defined]
            else:
                self.process.terminate()
            self.process.wait(timeout=wait)
        except Exception:
            LOGGER.warning("Terminate failed, killing %s", self.spec.name)
            try:
                self.process.kill()
            except Exception:
                pass
        finally:
            self.status = ServiceStatus.STOPPED
            self.pid = None
            self.process = None
            self.started_at = None

    def kill(self) -> None:
        if not self.process or self.process.poll() is not None:
            return
        LOGGER.warning("Killing service %s", self.spec.name)
        try:
            self.process.kill()
        except Exception:
            pass

    def restart(self, overrides: Mapping[str, str], *, external_terminal: bool = False) -> None:
        self.stop()
        time.sleep(0.2)
        self.start(overrides, external_terminal=external_terminal)

    def iterate_live_output(self) -> Iterator[str]:
        while self.process and self.process.poll() is None:
            try:
                yield self._queue.get(timeout=0.1)
            except Empty:
                continue

    def _capture_output(self) -> None:
        assert self.process and self.process.stdout
        try:
            for line in self.process.stdout:
                cleaned = (line.rstrip("\n") or " ")
                if self.log_buffer:
                    self.log_buffer.append(f"[{_now()}] [{self.spec.name}] {cleaned}")
                self._queue.put(cleaned)
        except Exception:
            pass
        finally:
            try:
                self.process.stdout.close()  # type: ignore[union-attr]
            except Exception:
                pass

    def _wait_for_exit(self) -> None:
        if not self.process:
            return
        code = self.process.wait()
        if code == 0:
            status = ServiceStatus.STOPPED
            message = "exited cleanly"
        else:
            status = ServiceStatus.ERROR
            message = f"exited with code {code}"
        if self.log_buffer:
            self.log_buffer.append(f"[{_now()}] [{self.spec.name}] {message}")
        self.status = status
        self.pid = None
        self.process = None
        self.started_at = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_services(log_buffer: CodexLogBuffer | None = None) -> Dict[str, CodexServiceHandle]:
    root = _project_root()
    py_exe = Path(sys.executable)
    api_script = root / "apps" / "backend" / "interfaces" / "api" / "run_api.py"
    if not api_script.exists():
        raise RuntimeError(f"API entrypoint not found at {api_script}")
    api_spec = CodexServiceSpec(
        name="API",
        command=[str(py_exe), str(api_script)],
        cwd=root,
        base_env={"PYTHONUNBUFFERED": "1"},
        allow_external_terminal=True,
    )
    npm_cmd = "npm.cmd" if os.name == "nt" else "npm"
    ui_spec = CodexServiceSpec(
        name="UI",
        command=[npm_cmd, "run", "dev", "--", "--host"],
        cwd=root / "apps" / "interface",
        base_env={"FORCE_COLOR": "1", "API_HOST": "localhost", "SERVER_HOST": "localhost"},
        allow_external_terminal=os.name == "nt",
    )
    return {
        "API": CodexServiceHandle(api_spec, log_buffer=log_buffer),
        "UI": CodexServiceHandle(ui_spec, log_buffer=log_buffer),
    }


def _now() -> str:
    return time.strftime("%H:%M:%S")


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
