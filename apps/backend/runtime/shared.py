"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared runtime variable holder (legacy compatibility surface).
Provides a simple attribute-access container used as a global variable bag by legacy-adjacent modules.

Symbols (top-level; keep in sync; no ghosts):
- `VariableHolder` (class): Attribute-backed dictionary wrapper used for shared runtime variables.
- `global_variables` (constant): Singleton `VariableHolder` instance used as a global variable bag.
"""


class VariableHolder:
    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, name):
        return self.__dict__.get(name, None)


global_variables = VariableHolder()

__all__ = ["VariableHolder", "global_variables"]
