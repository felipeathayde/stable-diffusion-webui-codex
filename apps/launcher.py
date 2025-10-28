from __future__ import annotations

"""
Codex launcher infrastructure.

This module centralises process supervision, paths resolution, reusable
profiles and environment validation for the WebUI bootstrap.
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import signal
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from queue import Queue, Empty
from typing import Deque, Dict, Iterator, List, Mapping, Optional
from collections import deque

MIN_NODE_MAJOR = 18


LOGGER = logging.getLogger("codex.launcher")


# --------------------------------------------------------------------------------------
# Paths & normalisation helpers
# --------------------------------------------------------------------------------------


def _normalize_path(value: str | os.PathLike[str] | None, *, fallback: Path) -> Path:
    if value is None:
        return fallback
    try:
        p = Path(value).expanduser()
        return p.resolve()
    except Exception as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"Failed to normalise path '{value}': {exc}") from exc


@dataclass(frozen=True)
class CodexPaths:
    """Resolved canonical paths for the runtime."""

    project_root: Path
    data_dir: Path
    models_dir: Path
    extensions_dir: Path
    extensions_builtin_dir: Path
    outputs_dir: Path
    configs_dir: Path
    default_config: Path
    default_checkpoint: Path


def resolve_paths(
    *,
    project_root: Path,
    data_dir: str | os.PathLike[str] | None = None,
    models_dir: str | os.PathLike[str] | None = None,
) -> CodexPaths:
    data = _normalize_path(data_dir, fallback=project_root / "data")
    models = _normalize_path(models_dir, fallback=data / "models")
    extensions = data / "extensions"
    extensions_builtin = project_root / "extensions-builtin"
    outputs = data / "outputs"
    configs = project_root / "configs"
    default_cfg = configs / "v1-inference.yaml"
    default_ckpt = project_root / "model.ckpt"
    return CodexPaths(
        project_root=project_root,
        data_dir=data,
        models_dir=models,
        extensions_dir=extensions,
        extensions_builtin_dir=extensions_builtin,
        outputs_dir=outputs,
        configs_dir=configs,
        default_config=default_cfg,
        default_checkpoint=default_ckpt,
    )


# --------------------------------------------------------------------------------------
# Logging buffer (reusable across UIs and scripts)
# --------------------------------------------------------------------------------------


@dataclass
class CodexLogBuffer:
    """Thread-safe ring buffer that stores the last N log lines."""

    capacity: int = 4000
    _lines: Deque[str] = field(default_factory=deque, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line)
            while len(self._lines) > self.capacity:
                self._lines.popleft()

    def snapshot(self) -> List[str]:
        with self._lock:
            return list(self._lines)

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


# --------------------------------------------------------------------------------------
# Environment validation
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CodexLaunchCheck:
    name: str
    ok: bool
    detail: str


def _check_python_version() -> CodexLaunchCheck:
    major, minor = sys.version_info[:2]
    supported = (major == 3) and (minor in (10, 11))
    detail = f"Detected Python {major}.{minor}"
    if not supported:
        detail += " (expected 3.10 or 3.11)"
    return CodexLaunchCheck(name="python-version", ok=supported, detail=detail)

def _parse_semver(version: str, components: int = 3) -> tuple[int, ...]:
    parts: List[int] = []
    for part in version.split(".")[:components]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            parts.append(int(digits))
        else:
            parts.append(0)
    while len(parts) < components:
        parts.append(0)
    return tuple(parts)


def _check_node() -> CodexLaunchCheck:
    npm = shutil.which("npm")
    node = shutil.which("node")
    if not (node and npm):
        missing = ", ".join(x for x, ref in (("node", node), ("npm", npm)) if ref is None)
        detail = f"Missing tool(s): {missing or 'unknown'}"
        return CodexLaunchCheck(name="node/npm", ok=False, detail=detail)

    try:
        raw_node_version = subprocess.check_output([node, "--version"], text=True, stderr=subprocess.STDOUT).strip()
        node_version = raw_node_version.lstrip("v")
    except Exception as exc:
        return CodexLaunchCheck(
            name="node/npm",
            ok=False,
            detail=f"node detected at {node} but version check failed: {exc}",
        )

    node_major = _parse_semver(node_version, components=1)[0]
    try:
        npm_version = subprocess.check_output([npm, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    except Exception:
        npm_version = None

    ok = (node_major >= MIN_NODE_MAJOR) and (npm_version is not None)
    detail_bits = [
        f"node {node_version} (path={node}, requires >= {MIN_NODE_MAJOR})",
        f"npm {npm_version or 'unavailable'} (path={npm})",
    ]
    if node_major < MIN_NODE_MAJOR:
        detail_bits.append("upgrade Node.js to >=18.x")
    if npm_version is None:
        detail_bits.append("npm --version command failed")

    return CodexLaunchCheck(name="node/npm", ok=ok, detail="; ".join(detail_bits))


def _vite_requirement_satisfied(actual: str, requirement: str) -> bool:
    if not requirement:
        return True
    req = requirement.strip()
    if "||" in req:
        return any(_vite_requirement_satisfied(actual, part) for part in req.split("||"))
    if req.startswith("^"):
        target_major = _parse_semver(req[1:], components=1)[0]
        return _parse_semver(actual, components=1)[0] == target_major
    if req.startswith("~"):
        target = _parse_semver(req[1:], components=2)
        return _parse_semver(actual, components=2) == target
    if req.startswith(">="):
        target = _parse_semver(req[2:], components=3)
        return _parse_semver(actual, components=3) >= target
    return actual.startswith(req)


def _check_vite(project_root: Path) -> CodexLaunchCheck:
    interface_dir = project_root / "apps" / "interface"
    package_json = interface_dir / "package.json"
    if not package_json.exists():
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="apps/interface/package.json not found; frontend workspace missing.",
        )
    try:
        package_spec = json.loads(package_json.read_text(encoding="utf-8"))
    except Exception as exc:
        return CodexLaunchCheck(name="vite", ok=False, detail=f"Failed to read package.json: {exc}")

    dev_deps = package_spec.get("devDependencies") or {}
    requirement = str(dev_deps.get("vite", "")).strip()
    if not requirement:
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="vite is not listed under devDependencies; ensure frontend deps are declared.",
        )

    installed_pkg = interface_dir / "node_modules" / "vite" / "package.json"
    if not installed_pkg.exists():
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="vite not installed (apps/interface/node_modules/vite missing). Run 'npm install' in apps/interface.",
        )
    try:
        installed_spec = json.loads(installed_pkg.read_text(encoding="utf-8"))
        installed_version = str(installed_spec.get("version", "")).strip()
    except Exception as exc:
        return CodexLaunchCheck(name="vite", ok=False, detail=f"Failed to read installed Vite package: {exc}")

    if not installed_version:
        return CodexLaunchCheck(
            name="vite",
            ok=False,
            detail="Installed Vite package has no version field; reinstall dev dependencies.",
        )

    ok = _vite_requirement_satisfied(installed_version, requirement)
    detail = f"vite {installed_version} (expected {requirement})"
    if not ok:
        detail += " — reinstall dependencies (npm install in apps/interface)."
    return CodexLaunchCheck(name="vite", ok=ok, detail=detail)


def run_launch_checks() -> List[CodexLaunchCheck]:
    root = _project_root()
    return [_check_python_version(), _check_node(), _check_vite(root)]


# --------------------------------------------------------------------------------------
# Services and process supervision
# --------------------------------------------------------------------------------------


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

        env = os.environ.copy()
        env.update(self.spec.base_env)
        env.update(overrides)

        flags = 0
        startupinfo = None
        if external_terminal and self.spec.allow_external_terminal and os.name == "nt":
            flags |= getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
            startupinfo = _windows_no_activate()

        stdout = subprocess.PIPE
        stderr = subprocess.STDOUT
        if external_terminal and self.spec.allow_external_terminal:
            stdout = None
            stderr = None

        self.status = ServiceStatus.STARTING
        self.started_at = time.time()
        try:
            self.process = subprocess.Popen(
                self.spec.command,
                cwd=str(self.spec.cwd),
                env=env,
                stdin=subprocess.DEVNULL,
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

    def iterate_live_output(self) -> Iterator[str]:
        while self.process and self.process.poll() is None:
            try:
                yield self._queue.get(timeout=0.1)
            except Empty:
                continue


# --------------------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------------------


PROFILE_PATH = Path(".sangoi") / "tui-profile.json"


@dataclass
class CodexLaunchProfile:
    env: Dict[str, str] = field(default_factory=dict)
    external_terminal: bool = False
    sdpa_policy: str = "mem_efficient"
    tab_index: int = 0

    def to_json(self) -> Dict[str, object]:
        return {
            "env": self.env,
            "external_terminal": self.external_terminal,
            "sdpa_policy": self.sdpa_policy,
            "tab_index": self.tab_index,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, object]) -> "CodexLaunchProfile":
        env = dict(
            (str(k), str(v))
            for k, v in (data.get("env") or {}).items()  # type: ignore[union-attr]
            if isinstance(k, str)
        )
        return cls(
            env=env,
            external_terminal=bool(data.get("external_terminal", False)),
            sdpa_policy=str(data.get("sdpa_policy", "mem_efficient")),
            tab_index=int(data.get("tab_index", 0)),
        )


def load_profile(path: Path = PROFILE_PATH) -> CodexLaunchProfile:
    try:
        raw = json.loads(path.read_text())
        profile = CodexLaunchProfile.from_json(raw)
        LOGGER.debug("Loaded profile from %s", path)
        return profile
    except FileNotFoundError:
        LOGGER.info("Profile %s not found, using defaults", path)
        return CodexLaunchProfile()
    except Exception as exc:
        LOGGER.warning("Failed to load profile %s: %s", path, exc)
        return CodexLaunchProfile()


def save_profile(profile: CodexLaunchProfile, path: Path = PROFILE_PATH) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(profile.to_json(), indent=2))
        LOGGER.debug("Saved profile to %s", path)
    except Exception as exc:
        LOGGER.warning("Failed to save profile %s: %s", path, exc)


# --------------------------------------------------------------------------------------
# Default service definitions
# --------------------------------------------------------------------------------------


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_services(log_buffer: CodexLogBuffer | None = None) -> Dict[str, CodexServiceHandle]:
    root = _project_root()
    py_exe = Path(sys.executable)
    api_spec = CodexServiceSpec(
        name="API",
        command=[str(py_exe), str(root / "apps" / "server" / "run_api.py")],
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


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------


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


__all__ = [
    "CodexPaths",
    "resolve_paths",
    "CodexLogBuffer",
    "CodexServiceSpec",
    "CodexServiceHandle",
    "CodexLaunchProfile",
    "load_profile",
    "save_profile",
    "default_services",
    "run_launch_checks",
    "CodexLaunchCheck",
    "ServiceStatus",
]
