"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model parser exception types (component-aware, fail-fast).
Defines parser-specific errors raised by planners/converters/validators, optionally annotating the component name that failed to keep error
messages actionable at the API/UI layer.

Symbols (top-level; keep in sync; no ghosts):
- `ModelParserError` (class): Base parser error that can carry a `component` tag for diagnostics.
- `MissingComponentError` (class): Raised when a required component is missing from the parsed checkpoint.
- `ValidationError` (class): Raised when a component exists but fails validation or conversion checks.
- `UnsupportedFamilyError` (class): Raised when no parser plan exists for a given model family.
"""

from __future__ import annotations


class ModelParserError(RuntimeError):
    """Raised when the Codex model parser fails to interpret a checkpoint."""

    def __init__(self, message: str, *, component: str | None = None):
        ctx = f" [component={component}]" if component else ""
        super().__init__(f"{message}{ctx}")
        self.component = component


class MissingComponentError(ModelParserError):
    def __init__(self, component: str, detail: str | None = None):
        msg = f"Required component '{component}' is missing"
        if detail:
            msg = f"{msg}: {detail}"
        super().__init__(msg, component=component)


class ValidationError(ModelParserError):
    pass


class UnsupportedFamilyError(ModelParserError):
    def __init__(self, family: str):
        super().__init__(f"No parser plan for model family '{family}'", component="planner")
        self.family = family
