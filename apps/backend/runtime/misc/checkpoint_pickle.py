"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Custom pickle helpers for best-effort checkpoint loading (avoid importing unsupported classes).

Symbols (top-level; keep in sync; no ghosts):
- `load` (function): Alias for `pickle.load` (for `torch.load(..., pickle_module=checkpoint_pickle)`).
- `Empty` (class): Placeholder type returned for ignored modules/classes.
- `Unpickler` (class): Custom Unpickler that filters unsupported modules in `find_class`.
"""

import pickle

load = pickle.load


class Empty:
    pass


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # TODO: safe unpickle
        if module.startswith("pytorch_lightning"):
            return Empty
        return super().find_class(module, name)
