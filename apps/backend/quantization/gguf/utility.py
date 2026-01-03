"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Utility helpers for GGUF naming and size labels.
Implements size rounding/labels and filename convention helpers used by conversion tooling.

Symbols (top-level; keep in sync; no ghosts):
- `fill_templated_filename` (function): Fills GGUF filename templates (e.g. `{ftype}` placeholders) using the output type.
- `model_weight_count_rounded_notation` (function): Formats a parameter count using K/M/B/T notation with stable rounding.
- `size_label` (function): Builds a size label string, including MoE-aware formatting when experts are present.
- `naming_convention` (function): Constructs a GGUF-style output filename following the upstream naming convention.
"""

from __future__ import annotations

from typing import Literal


def fill_templated_filename(filename: str, output_type: str | None) -> str:
    # Given a file name fill in any type templates e.g. 'some-model-name.{ftype}.gguf'
    ftype_lowercase: str = output_type.lower() if output_type is not None else ""
    ftype_uppercase: str = output_type.upper() if output_type is not None else ""
    return filename.format(ftype_lowercase,
                           outtype=ftype_lowercase, ftype=ftype_lowercase,
                           OUTTYPE=ftype_uppercase, FTYPE=ftype_uppercase)


def model_weight_count_rounded_notation(model_params_count: int, min_digits: int = 2) -> str:
    if model_params_count > 1e12 :
        # Trillions Of Parameters
        scaled_model_params = model_params_count * 1e-12
        scale_suffix = "T"
    elif model_params_count > 1e9 :
        # Billions Of Parameters
        scaled_model_params = model_params_count * 1e-9
        scale_suffix = "B"
    elif model_params_count > 1e6 :
        # Millions Of Parameters
        scaled_model_params = model_params_count * 1e-6
        scale_suffix = "M"
    else:
        # Thousands Of Parameters
        scaled_model_params = model_params_count * 1e-3
        scale_suffix = "K"

    fix = max(min_digits - len(str(round(scaled_model_params)).lstrip('0')), 0)

    return f"{scaled_model_params:.{fix}f}{scale_suffix}"


def size_label(total_params: int, shared_params: int, expert_params: int, expert_count: int) -> str:

    if expert_count > 0:
        pretty_size = model_weight_count_rounded_notation(abs(shared_params) + abs(expert_params), min_digits=2)
        size_class = f"{expert_count}x{pretty_size}"
    else:
        size_class = model_weight_count_rounded_notation(abs(total_params), min_digits=2)

    return size_class


def naming_convention(model_name: str | None, base_name: str | None, finetune_string: str | None, version_string: str | None, size_label: str | None, output_type: str | None, model_type: Literal['vocab', 'LoRA'] | None = None) -> str:
    # Reference: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md#gguf-naming-convention

    if base_name is not None:
        name = base_name.strip().replace(' ', '-').replace('/', '-')
    elif model_name is not None:
        name = model_name.strip().replace(' ', '-').replace('/', '-')
    else:
        name = "ggml-model"

    parameters = f"-{size_label}" if size_label is not None else ""

    finetune = f"-{finetune_string.strip().replace(' ', '-')}" if finetune_string is not None else ""

    version = f"-{version_string.strip().replace(' ', '-')}" if version_string is not None else ""

    encoding = f"-{output_type.strip().replace(' ', '-').upper()}" if output_type is not None else ""

    kind = f"-{model_type.strip().replace(' ', '-')}" if model_type is not None else ""

    return f"{name}{parameters}{finetune}{version}{encoding}{kind}"
