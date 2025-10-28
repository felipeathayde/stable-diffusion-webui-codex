from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

LOGGER = logging.getLogger("codex.launcher.checks")
MIN_NODE_MAJOR = 18


@dataclass(frozen=True)
class CodexLaunchCheck:
    name: str
    ok: bool
    detail: str


def _parse_semver(version: str, components: int = 3) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split(".")[:components]:
        digits = ""
        for ch in part:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits or 0))
    while len(parts) < components:
        parts.append(0)
    return tuple(parts)


def _check_python_version() -> CodexLaunchCheck:
    major, minor = sys.version_info[:2]
    supported = (major == 3) and (minor in (10, 11))
    detail = f"Detected Python {major}.{minor}"
    if not supported:
        detail += " (expected 3.10 or 3.11)"
    return CodexLaunchCheck(name="python-version", ok=supported, detail=detail)


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


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
    """Execute all launcher environment checks."""
    root = _project_root()
    checks = [_check_python_version(), _check_node(), _check_vite(root)]
    for check in checks:
        LOGGER.debug("Launch check %s ok=%s detail=%s", check.name, check.ok, check.detail)
    return checks
