from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

from apps.backend.runtime.model_registry.specs import ModelSignature

from .families import resolve_plan
from .plan import execute_plan
from .specs import CodexEstimatedConfig


def parse_state_dict(state_dict: MutableMapping[str, Any], signature: ModelSignature) -> CodexEstimatedConfig:
    bundle = resolve_plan(signature)
    context = execute_plan(bundle.plan, state_dict, signature=signature)
    return bundle.build_config(context)

__all__ = ["parse_state_dict", "CodexEstimatedConfig"]
