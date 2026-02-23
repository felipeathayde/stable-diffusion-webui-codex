"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Declarative schema models for launcher Tk forms.
Defines typed descriptors for sections and fields so tabs can render settings declaratively instead of hand-writing each widget row.

Symbols (top-level; keep in sync; no ghosts):
- `FieldKind` (class): Supported form input kinds for launcher setting rows.
- `FormFieldDescriptor` (dataclass): Declarative field definition consumed by the shared form renderer.
- `FormSectionDescriptor` (dataclass): Declarative section definition containing ordered form fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import tkinter as tk
from typing import Callable, Sequence


class FieldKind(StrEnum):
    """Supported input kinds for launcher form descriptors."""

    CHOICE = "choice"
    CHECK = "check"
    ENTRY = "entry"
    ENTRY_COMMIT = "entry_commit"


@dataclass(frozen=True, slots=True)
class FormFieldDescriptor:
    """Declarative field definition rendered by `FormRenderer`."""

    field_id: str
    kind: FieldKind
    label: str
    variable: tk.Variable
    on_change: Callable[[], None]
    choices: Sequence[str] = ()
    width: int = 18
    advanced: bool = False
    help_text: str | None = None


@dataclass(frozen=True, slots=True)
class FormSectionDescriptor:
    """Declarative section definition with ordered fields."""

    title: str
    fields: Sequence[FormFieldDescriptor] = field(default_factory=tuple)
    help_texts: Sequence[str] = field(default_factory=tuple)
    advanced: bool = False
