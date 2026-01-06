"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Custom exception hierarchy for backend engines and orchestration.
Provides explicit failure types for engine registration/lookup, unsupported tasks, load failures, and runtime execution errors.

Symbols (top-level; keep in sync; no ghosts):
- `EngineError` (class): Base error for engine-related failures.
- `EngineRegistrationError` (class): Raised when registering an engine with conflicting keys/aliases.
- `EngineNotFoundError` (class): Raised when resolving an unknown engine key or alias.
- `UnsupportedTaskError` (class): Raised when the engine does not implement a requested task.
- `EngineLoadError` (class): Raised when an engine fails to load required weights/resources.
- `EngineExecutionError` (class): Raised when an engine fails during inference execution.
"""

from __future__ import annotations


class EngineError(RuntimeError):
    """Base error for engine-related failures."""


class EngineRegistrationError(EngineError):
    """Raised when attempting to register an engine with conflicting keys."""


class EngineNotFoundError(EngineError):
    """Raised when resolving an unknown engine key or alias."""


class UnsupportedTaskError(EngineError):
    """Raised when the requested task is not implemented by the engine."""


class EngineLoadError(EngineError):
    """Raised when an engine fails to load required weights or resources."""


class EngineExecutionError(EngineError):
    """Raised when an engine encounters a runtime failure during inference."""
