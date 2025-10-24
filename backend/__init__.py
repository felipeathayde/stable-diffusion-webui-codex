"""Backend package for WebUI Codex.

Contains service abstractions and backend helpers decoupled from FastAPI.

Also initializes logging early so all submodules share consistent verbosity.
The default level is DEBUG, overridable via env vars
CODEX_LOG_LEVEL/SDWEBUI_LOG_LEVEL/WEBUI_LOG_LEVEL or webui.settings.bat.
"""

# Configure logging once per interpreter early in import chain
try:
    from .logging_utils import setup_logging as _codex_setup_logging  # type: ignore
    _codex_setup_logging()
except Exception:
    # Never block imports; downstream modules may configure their own logging
    pass

# Dynamic import redirector for legacy modules
# Maps `backend.*` -> `apps.server.backend.*` when a shim file is not present.
try:  # pragma: no cover
    import importlib
    import importlib.util
    import importlib.abc
    import sys as _sys
    import os as _os
    import logging as _logging

    _PREFIX_MAP = {
        # Top-level packages
        "core": "apps.server.backend.core",
        "engines": "apps.server.backend.engines",
        "huggingface": "apps.server.backend.huggingface",
        "services": "apps.server.backend.services",
        "runtime": "apps.server.backend.runtime",
        "video": "apps.server.backend.video",
        "wan_gguf": "apps.server.backend.wan_gguf",
        "wan_gguf_core": "apps.server.backend.wan_gguf_core",
        "gguf": "apps.server.backend.gguf",
        # Legacy group renames
        "diffusion_engine": "apps.server.backend.engines.diffusion",
        "text_processing": "apps.server.backend.runtime.text_processing",
        "sampling": "apps.server.backend.runtime.sampling",
        "misc": "apps.server.backend.runtime.misc",
        "modules": "apps.server.backend.runtime.modules",
        "nn": "apps.server.backend.runtime.nn",
        # Single-file legacy modules
        "operations": "apps.server.backend.runtime.ops.operations",
        "operations_bnb": "apps.server.backend.runtime.ops.operations_bnb",
        "operations_gguf": "apps.server.backend.runtime.ops.operations_gguf",
        "utils": "apps.server.backend.runtime.utils",
        "memory_management": "apps.server.backend.runtime.memory.memory_management",
        "stream": "apps.server.backend.runtime.memory.stream",
        "state_dict": "apps.server.backend.runtime.models.state_dict",
        "loader": "apps.server.backend.runtime.models.loader",
        "logging_utils": "apps.server.backend.runtime.logging",
        "torch_trace": "apps.server.backend.runtime.trace",
        # Subpackages with different anchor
        "video.interpolation": "apps.server.backend.video.interpolation",
    }

    def _map_alias(fullname: str) -> str | None:
        # fullname startswith 'backend.' here
        tail = fullname.split('.', 1)[1]
        # Exact or prefix match in the map
        for key, target in _PREFIX_MAP.items():
            if tail == key:
                return target
            if tail.startswith(key + "."):
                return target + "." + tail[len(key) + 1 :]
        # Default: apps.server.<fullname>
        return "apps.server." + fullname

    class _BackendShimFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
        _log = _logging.getLogger("backend.shim")

        def find_spec(self, fullname, path, target=None):  # noqa: D401
            # Only handle nested modules under 'backend.'; never the root 'backend'
            if not fullname.startswith("backend."):
                return None
            if fullname in _sys.modules:
                return None
            # If a concrete file/module exists under backend/, prefer it
            try:
                from importlib.machinery import PathFinder
                real_spec = PathFinder.find_spec(fullname, path)
                if real_spec is not None:
                    return None
            except Exception:
                pass
            # Defer alias resolution to exec_module to avoid importing parent packages here
            return importlib.util.spec_from_loader(fullname, self)

        def create_module(self, spec):
            return None  # default module creation

        def exec_module(self, module):
            fullname = module.__spec__.name  # type: ignore[assignment]
            alias = _map_alias(fullname)
            target = importlib.import_module(alias)
            _sys.modules[fullname] = target
            if (_os.environ.get("CODEX_SHIM_WARN", "1") == "1"):
                # Keep it at DEBUG to avoid noisy logs by default
                self._log.debug("shim redirect: %s -> %s", fullname, alias)

    # Install finder at the front so it wins when shim files are removed
    _sys.meta_path.insert(0, _BackendShimFinder())
except Exception:
    # Import system issues should never block runtime
    pass
