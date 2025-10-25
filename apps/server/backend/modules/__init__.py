from __future__ import annotations

"""Facade for legacy A1111 'modules' package under backend namespace.

Active backend code must import legacy submodules through this package:
    from apps.server.backend.modules import shared, processing, sd_models

Do not import 'modules.*' directly from backend code. This keeps our import
surface centralized and makes future migrations straightforward.
"""

import importlib
from types import ModuleType


def __getattr__(name: str) -> ModuleType:  # pragma: no cover - thin proxy
    return importlib.import_module(f"modules.{name}")

