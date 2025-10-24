#!/usr/bin/env python
"""
Trace Python imports for a target module or script (Windows-friendly).

Highlights
- Runtime, not just static: installs a PEP 578 audit hook and wraps sys.meta_path
  to log real import resolution (works with custom loaders).
- Multiple modes:
  - audit (default): run target in-process with audit + finder proxy logging
  - verbose: spawn child with `-v` (CPython import verbose) and capture output
  - importtime: spawn child with `-X importtime` and capture timing waterfall
  - all: runs audit, verbose and importtime sequentially
- Outputs to file (text or JSON lines) and echoes to console.

Usage (examples)
- python scripts/trace_imports.py --module apps.server.run_api --mode audit
- python scripts/trace_imports.py --file apps\\server\\run_api.py --mode verbose
- python scripts/trace_imports.py --module apps.server.run_api --mode all --json --output import-trace.jsonl

Notes
- Run in the SAME interpreter/environment that you use to start your backend.
- For the ground truth, start with: --mode verbose and --mode importtime.
  Then use --mode audit for structured, filterable logs.
"""

from __future__ import annotations

import argparse
import builtins
import datetime as _dt
import io
import json
import os
import runpy
import subprocess
import sys
import time
from dataclasses import dataclass
from importlib.abc import MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import find_spec
from types import ModuleType
from typing import Any, Optional


# ------------------------------- Logging utils -------------------------------

def _now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="milliseconds")


@dataclass
class Logger:
    out: io.TextIOBase
    as_json: bool = False

    def _write(self, payload: dict[str, Any]):
        try:
            if self.as_json:
                self.out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            else:
                # simple human line
                ts = payload.get("ts", _now_iso())
                ev = payload.get("event", "log")
                msg = payload.get("msg", "")
                extra = {k: v for k, v in payload.items() if k not in {"ts", "event", "msg"}}
                line = f"[{ts}] {ev}: {msg}"
                if extra:
                    line += f" | {extra}"
                self.out.write(line + "\n")
            self.out.flush()
        except (ValueError, OSError):
            # Ignore writes after stream is closed (e.g., late audit events on atexit)
            pass

    def info(self, msg: str, **kw):
        self._write({"ts": _now_iso(), "event": "info", "msg": msg, **kw})

    def warn(self, msg: str, **kw):
        self._write({"ts": _now_iso(), "event": "warn", "msg": msg, **kw})

    def error(self, msg: str, **kw):
        self._write({"ts": _now_iso(), "event": "error", "msg": msg, **kw})

    def evt(self, event: str, **kw):
        self._write({"ts": _now_iso(), "event": event, **kw})


# ------------------------------- Finder proxy --------------------------------

class _FinderProxy(MetaPathFinder):
    """Proxy around an existing meta_path finder to log find_spec calls."""

    def __init__(self, inner: MetaPathFinder, log: Logger):
        self._inner = inner
        self._log = log

    def find_spec(self, fullname: str, path: Optional[list[str]] = None, target: Optional[ModuleType] = None) -> Optional[ModuleSpec]:
        t0 = time.perf_counter()
        try:
            spec = self._inner.find_spec(fullname, path, target)  # type: ignore[arg-type]
        except Exception as e:  # pragma: no cover
            dt_ms = round((time.perf_counter() - t0) * 1000, 3)
            self._log.evt(
                "find_spec_error",
                name=fullname,
                finder=type(self._inner).__name__,
                elapsed_ms=dt_ms,
                error=repr(e),
            )
            raise

        dt_ms = round((time.perf_counter() - t0) * 1000, 3)
        if spec is not None:
            self._log.evt(
                "find_spec_ok",
                name=fullname,
                finder=type(self._inner).__name__,
                origin=getattr(spec, "origin", None),
                loader=type(spec.loader).__name__ if getattr(spec, "loader", None) else None,
                is_package=bool(getattr(spec, "submodule_search_locations", None)),
                elapsed_ms=dt_ms,
            )
        else:
            self._log.evt(
                "find_spec_miss",
                name=fullname,
                finder=type(self._inner).__name__,
                elapsed_ms=dt_ms,
            )
        return spec

    def __getattr__(self, name: str):  # pass-through
        return getattr(self._inner, name)


# -------------------------------- Audit hook ---------------------------------

class _Audit:
    def __init__(self, log: Logger):
        self._log = log
        self._stack: list[str] = []

    def __call__(self, event: str, args: tuple[Any, ...]):  # pragma: no cover (runtime only)
        if event == "import":
            # args: (module_name, filename)
            try:
                name, filename = (args + (None, None))[:2]
            except Exception:
                name, filename = (None, None)
            self._log.evt("import", name=name, filename=filename, stack=list(self._stack))
            if isinstance(name, str):
                self._stack.append(name)
        elif event == "importlib.find_spec":
            # args: (name, path, target)
            try:
                name, path, target = (args + (None, None, None))[:3]
            except Exception:
                name, path, target = (None, None, None)
            self._log.evt("audit_find_spec", name=name, path=path)
        elif event == "importlib.ModuleSpec":
            try:
                spec = args[0]
                self._log.evt(
                    "modulespec",
                    name=getattr(spec, "name", None),
                    origin=getattr(spec, "origin", None),
                    loader=type(getattr(spec, "loader", None)).__name__ if getattr(spec, "loader", None) else None,
                    is_package=bool(getattr(spec, "submodule_search_locations", None)),
                )
            except Exception:
                pass


# ------------------------------- Main routines -------------------------------

def _wrap_meta_path(log: Logger):
    new_list = []
    for f in list(sys.meta_path):
        try:
            new_list.append(_FinderProxy(f, log))
        except Exception:
            new_list.append(f)
    sys.meta_path[:] = new_list


def _install_audit(log: Logger):
    try:
        sys.addaudithook(_Audit(log))
        log.info("Audit hook installed")
    except Exception as e:
        log.warn("Could not install audit hook", error=repr(e))


def _ensure_repo_root_on_path(log: Logger) -> None:
    """Ensure repository root (parent of this scripts/ dir) and CWD are on sys.path.

    When this tool is executed as a file (python scripts/trace_imports.py),
    Python sets sys.path[0] to the directory of this script (scripts/), not the
    repository root. Our codebase expects imports like `apps.server...` to
    resolve from repo root. This injects the parent directory early in sys.path.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    cwd = os.getcwd()

    inserted = []
    for p in (repo_root, cwd):
        if p not in sys.path:
            sys.path.insert(0, p)
            inserted.append(p)
    if inserted:
        log.info("sys.path.inject", inserted=inserted)


def _run_target_inprocess(args, log: Logger):
    # Show environment for reproducibility
    _ensure_repo_root_on_path(log)
    log.info("Interpreter", exe=sys.executable)
    log.info("CWD", cwd=os.getcwd())
    log.info("sys.path", path=sys.path)
    log.info("sys.meta_path", meta=[type(f).__name__ for f in sys.meta_path])
    log.info("sys.path_hooks", hooks=[getattr(h, "__name__", str(h)) for h in sys.path_hooks])

    _wrap_meta_path(log)
    _install_audit(log)

    # Run target
    t0 = time.perf_counter()
    try:
        if args.module:
            log.info("run_module", module=args.module)
            runpy.run_module(args.module, run_name="__main__", alter_sys=True)
        elif args.file:
            # For script path, ensure its parent-of-parent (package root) is on sys.path
            file_abspath = os.path.abspath(args.file)
            pkg_root = os.path.dirname(os.path.dirname(file_abspath))
            if pkg_root not in sys.path:
                sys.path.insert(0, pkg_root)
                log.info("sys.path.inject_file_root", added=pkg_root)
            log.info("run_path", file=file_abspath)
            runpy.run_path(file_abspath, run_name="__main__")
        else:
            raise SystemExit("Provide --module or --file")
    except SystemExit as e:
        # propagate exit code but log it
        code = e.code if isinstance(e.code, int) else 0
        log.warn("SystemExit", code=code)
        raise
    except Exception as e:
        log.error("Unhandled exception", error=repr(e))
        raise
    finally:
        dt_ms = round((time.perf_counter() - t0) * 1000, 3)
        log.info("target_completed", elapsed_ms=dt_ms)


def _spawn_child(args, flags: list[str], name: str, log: Logger):
    cmd = [sys.executable] + flags
    if args.module:
        cmd += ["-m", args.module]
    elif args.file:
        cmd += [args.file]
    else:
        raise SystemExit("Provide --module or --file")

    log.info("spawn", mode=name, cmd=" ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    for line in proc.stdout:
        # pass-through and capture
        log._write({"ts": _now_iso(), "event": name, "msg": line.rstrip()})
    rc = proc.wait()
    log.info("child_exit", mode=name, returncode=rc)
    if rc != 0:
        raise SystemExit(rc)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Trace imports for a target module or script")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--module", help="Module to run, e.g., apps.server.run_api")
    g.add_argument("--file", help="Script path to run, e.g., apps/server/run_api.py")
    ap.add_argument(
        "--mode",
        choices=["audit", "verbose", "importtime", "all"],
        default="audit",
        help="Tracing mode to use",
    )
    ap.add_argument("--output", help="Output log file path", default=None)
    ap.add_argument("--json", action="store_true", help="Emit JSON lines instead of text")

    args = ap.parse_args(argv)

    # Prepare output
    if args.output:
        out = open(args.output, "w", encoding="utf-8", newline="")
    else:
        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        suffix = "jsonl" if args.json else "log"
        out = open(f"import-trace-{ts}.{suffix}", "w", encoding="utf-8", newline="")

    log = Logger(out=out, as_json=args.json)
    log.info("trace_imports_start", mode=args.mode)

    try:
        if args.mode in ("audit", "all"):
            log.info("mode_audit_begin")
            try:
                _run_target_inprocess(args, log)
            except SystemExit as e:
                # child code used sys.exit
                if args.mode != "all":
                    return int(e.code) if isinstance(e.code, int) else 0
            log.info("mode_audit_end")

        if args.mode in ("verbose", "all"):
            _spawn_child(args, ["-v"], name="verbose", log=log)

        if args.mode in ("importtime", "all"):
            _spawn_child(args, ["-X", "importtime"], name="importtime", log=log)

        log.info("trace_imports_done")
        return 0
    finally:
        try:
            out.flush()
            out.close()
        except Exception:
            pass


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
