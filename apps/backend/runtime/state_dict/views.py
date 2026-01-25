"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Lightweight mapping views for state_dict handling.
Provides prefix/filter/remap/cast views plus a SafeTensors-backed lazy dict used to stream state_dict preprocessing.

Symbols (top-level; keep in sync; no ghosts):
- `KeyPrefixView` (class): Mapping view that exposes `base` keys under a fixed prefix without materializing values.
- `FilterPrefixView` (class): Mapping view that filters keys by prefix and optionally re-prefixes them lazily.
- `RemapKeysView` (class): Mapping view that remaps keys through a mapping dict (useful for on-the-fly state_dict key conversion).
- `CastOnGetView` (class): Mapping view that casts tensors/values on access (`__getitem__`) to a target dtype/device (no eager conversion).
- `LazySafetensorsDict` (class): Lazy SafeTensors-backed state_dict; keeps a single handle and loads tensors on demand (Windows: materializes on first access to avoid repeated opens).
"""

from __future__ import annotations

import sys
import threading
from collections.abc import MutableMapping
from typing import Dict

from safetensors.torch import safe_open


class KeyPrefixView(MutableMapping):
    """Lightweight mapping view that exposes `base` keys with a fixed prefix.

    - Does not materialize tensor values; delegates to `base[key_without_prefix]` on access.
    - Deletions and sets propagate to the underlying mapping.
    - Useful to avoid rebuilding huge state_dicts on CPU.
    """

    def __init__(self, base: MutableMapping, prefix: str):
        self._base = base
        self._prefix = prefix

    def _strip(self, k: str) -> str:
        if not k.startswith(self._prefix):
            raise KeyError(k)
        return k[len(self._prefix) :]

    def __getitem__(self, k: str):
        return self._base[self._strip(k)]

    def __setitem__(self, k: str, v):
        self._base[self._strip(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._strip(k)]

    def __iter__(self):
        for k in self._base.keys():
            yield f"{self._prefix}{k}"

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            # Fallback: iterate
            return sum(1 for _ in self.__iter__())


class FilterPrefixView(MutableMapping):
    """View over keys under a given prefix, optionally re-prefixed lazily.

    - base: mapping with original keys (e.g., LazySafetensorsDict or KeyPrefixView)
    - prefix: filter only keys that start with this
    - new_prefix: keys presented by this view will start with new_prefix instead
    """

    def __init__(self, base: MutableMapping, prefix: str, new_prefix: str = ""):
        self._base = base
        self._prefix = prefix
        self._new_prefix = new_prefix

    def _to_base_key(self, k: str) -> str:
        # Map presented key 'k' back to the underlying base mapping key.
        if self._new_prefix:
            if k.startswith(self._new_prefix):
                return self._prefix + k[len(self._new_prefix) :]
            # If caller already uses base prefix, pass-through
            if k.startswith(self._prefix):
                return k
            # Fallback: assume k is suffix; prepend prefix
            return self._prefix + k
        else:
            # Presented keys are suffix-only; base keys use prefix
            if k.startswith(self._prefix):
                return k
            return self._prefix + k

    def _present_key(self, base_key: str) -> str:
        if not self._prefix:
            if self._new_prefix:
                return f"{self._new_prefix}{base_key}"
            return base_key
        suffix = base_key[len(self._prefix) :]
        if self._new_prefix:
            return f"{self._new_prefix}{suffix}"
        return suffix

    def __getitem__(self, k: str):
        return self._base[self._to_base_key(k)]

    def __setitem__(self, k: str, v):
        self._base[self._to_base_key(k)] = v

    def __delitem__(self, k: str):
        del self._base[self._to_base_key(k)]

    def __iter__(self):
        for k in self._base.keys():
            if k.startswith(self._prefix):
                out = self._new_prefix + k[len(self._prefix) :]
                yield out

    def __len__(self) -> int:
        c = 0
        for _ in self.__iter__():
            c += 1
        return c

    def materialize(self, *, return_mapping: bool = False):
        """Realise all tensors matching the prefix into a concrete dict.

        Prefers calling the underlying mapping's `materialize` helper when
        available so SafeTensors files are streamed with a single handle open.
        """

        materializer = getattr(self._base, "materialize", None)
        if callable(materializer):
            try:
                return materializer(prefix=self._prefix, new_prefix=self._new_prefix, return_mapping=return_mapping)
            except TypeError:
                result = materializer(prefix=self._prefix, new_prefix=self._new_prefix)
                if return_mapping:
                    raise
                return result

        out: Dict[str, object] = {}
        mapping: Dict[str, str] = {}
        for key in self._base.keys():
            if not key.startswith(self._prefix):
                continue
            presented = self._present_key(key)
            out[presented] = self._base[key]
            mapping[presented] = key
        if return_mapping:
            return out, mapping
        return out


class RemapKeysView(MutableMapping):
    """Present a remapped keyspace over an underlying mapping lazily.

    - base: original mapping (e.g., LazySafetensorsDict)
    - mapping: dict[new_key] -> old_key in the base mapping
    - Does not materialize any tensor unless __getitem__ is called.
    """

    def __init__(self, base: MutableMapping, mapping: dict[str, str]):
        self._base = base
        self._map = dict(mapping)

    def __getitem__(self, k: str):
        return self._base[self._map[k]]

    def __setitem__(self, k: str, v):
        self._map[k] = k
        self._base[k] = v

    def __delitem__(self, k: str):
        old = self._map.pop(k, None)
        if old is not None and old in self._base:
            del self._base[old]

    def __iter__(self):
        return iter(self._map.keys())

    def __len__(self):
        return len(self._map)

    def __contains__(self, k: object) -> bool:
        return k in self._map

    def keys(self):
        return list(self._map.keys())

    def items(self):
        for k in self._map.keys():
            yield k, self._base[self._map[k]]


class CastOnGetView(MutableMapping):
    """Mapping view that casts tensor values on CPU to a target dtype on access.

    Useful to avoid fragile CPU bf16/fp16 ops during preprocessing. Only casts
    floating tensors matching `from_dtypes` and `device_type`.
    """

    def __init__(self, base: MutableMapping, *, device_type: str = "cpu", from_dtypes=None, to_dtype=None):
        import torch as _torch

        self._base = base
        self._device_type = device_type
        self._from = tuple(from_dtypes) if from_dtypes is not None else (_torch.bfloat16, _torch.float16)
        self._to = to_dtype or _torch.float32

    def __getitem__(self, k: str):
        import torch as _torch

        v = self._base[k]
        if isinstance(v, _torch.Tensor):
            try:
                if v.device.type == self._device_type and v.dtype in self._from:
                    return v.to(self._to)
            except Exception:
                return v
        return v

    def __setitem__(self, k: str, v):
        self._base[k] = v

    def __delitem__(self, k: str):
        del self._base[k]

    def __iter__(self):
        return iter(self._base.keys())

    def __len__(self) -> int:
        try:
            return len(self._base.keys())
        except Exception:
            return sum(1 for _ in self.__iter__())


class LazySafetensorsDict(MutableMapping):
    """Lazy, mutable mapping backed by a .safetensors file.

    - Keys come from the file; values are loaded on demand with safe_open.get_tensor.
    - Supports overlay writes and deletions without touching the underlying file.
    - Device: only CPU tensors are produced (parity with previous loader).

    Windows crash prevention: Once any tensor is accessed, the entire file is
    materialized into memory to avoid reopening the file repeatedly (which causes
    torch_cpu.dll crashes on Windows). On non-Windows platforms, keep a single
    SafeTensors handle open and load tensors on demand to stay truly lazy.
    """

    def __init__(self, filepath: str, device: str = "cpu"):
        self.filepath = filepath
        self.device = device or "cpu"
        self._platform_windows = sys.platform.startswith("win")
        self._overlay = {}  # in-memory writes/overrides
        self._deleted = set()  # keys logically removed
        self._keys_cache = None  # cached set of underlying keys
        self._materialized = None  # holds all tensors after first access
        self._materialized_triggered = False
        self._handle = None  # persistent SafeTensors handle (non-Windows only)
        self._handle_lock = threading.Lock()

    def _base_keys(self):
        if self._keys_cache is None:
            if self._platform_windows:
                with safe_open(self.filepath, framework="pt", device=self.device) as f:
                    self._keys_cache = set(f.keys())
            else:
                with self._handle_lock:
                    if self._handle is None:
                        self._handle = safe_open(self.filepath, framework="pt", device=self.device)
                    self._keys_cache = set(self._handle.keys())
        return self._keys_cache

    def _ensure_materialized(self):
        """Load all tensors from file once to avoid reopening repeatedly."""

        if not self._platform_windows:
            return
        if self._materialized is None and not self._materialized_triggered:
            self._materialized_triggered = True
            self._materialized = {}
            try:
                with safe_open(self.filepath, framework="pt", device=self.device) as f:
                    self._keys_cache = set(f.keys())
                    for key in f.keys():
                        self._materialized[key] = f.get_tensor(key)
            except Exception:
                # If materialization fails, clear and fall back to per-key loading
                self._materialized = None
                self._materialized_triggered = False

    # Mapping protocol
    def __getitem__(self, key):
        if key in self._overlay:
            return self._overlay[key]
        if key in self._deleted:
            raise KeyError(key)

        if self._platform_windows:
            # Materialize all tensors on first access to avoid repeated file opens.
            self._ensure_materialized()

            if self._materialized is not None:
                if key in self._materialized:
                    return self._materialized[key]
                raise KeyError(key)

            # Fallback for edge cases (should rarely happen)
            if key not in self._base_keys():
                raise KeyError(key)
            with safe_open(self.filepath, framework="pt", device=self.device) as f:
                return f.get_tensor(key)

        if key not in self._base_keys():
            raise KeyError(key)
        with self._handle_lock:
            if self._handle is None:
                self._handle = safe_open(self.filepath, framework="pt", device=self.device)
            return self._handle.get_tensor(key)

    def __setitem__(self, key, value):
        self._overlay[key] = value
        if self._keys_cache is None and key not in self._deleted:
            # do not expand base key set; overlay keys are separate
            pass
        if key in self._deleted:
            self._deleted.remove(key)

    def __delitem__(self, key):
        if key in self._overlay:
            del self._overlay[key]
        else:
            # mark as deleted logically
            self._deleted.add(key)

    def __iter__(self):
        base = (k for k in self._base_keys() if k not in self._deleted)
        # overlay can shadow base
        for k in base:
            if k not in self._overlay:
                yield k
        for k in self._overlay.keys():
            yield k

    def __len__(self):
        base_keys = self._base_keys()
        overlay_keys = set(self._overlay.keys())
        return len(base_keys - self._deleted - overlay_keys) + len(self._overlay)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, str):
            return False
        if key in self._overlay:
            return True
        if key in self._deleted:
            return False
        try:
            return key in self._base_keys()
        except Exception:
            return False

    # Convenience helpers
    def keys(self):
        return list(iter(self))

    def items(self):
        # Use materialization to avoid per-item file opens (Windows only)
        if self._platform_windows:
            self._ensure_materialized()
        for k in self:
            yield k, self[k]

    def materialize(
        self,
        *,
        prefix: str = "",
        new_prefix: str = "",
        return_mapping: bool = False,
    ):
        """Eagerly load tensors matching `prefix`, optionally re-prefixing keys."""

        def _translate(key: str) -> str:
            suffix = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            if new_prefix:
                return f"{new_prefix}{suffix}"
            if prefix:
                return suffix
            return key

        result: Dict[str, object] = {}
        mapping: Dict[str, str] = {}
        if self._platform_windows:
            with safe_open(self.filepath, framework="pt", device=self.device) as handle:
                for key in handle.keys():
                    if prefix and not key.startswith(prefix):
                        continue
                    if key in self._deleted:
                        continue
                    if key in self._overlay:
                        continue
                    presented = _translate(key)
                    result[presented] = handle.get_tensor(key)
                    mapping[presented] = key
        else:
            with self._handle_lock:
                if self._handle is None:
                    self._handle = safe_open(self.filepath, framework="pt", device=self.device)
                handle = self._handle
                for key in handle.keys():
                    if prefix and not key.startswith(prefix):
                        continue
                    if key in self._deleted:
                        continue
                    if key in self._overlay:
                        continue
                    presented = _translate(key)
                    result[presented] = handle.get_tensor(key)
                    mapping[presented] = key

        for key, value in self._overlay.items():
            if prefix and not key.startswith(prefix):
                continue
            if key in self._deleted:
                continue
            presented = _translate(key)
            result[presented] = value
            mapping[presented] = key

        if return_mapping:
            return result, mapping
        return result

    def __del__(self):
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.__exit__(None, None, None)
            except Exception:
                pass


__all__ = [
    "CastOnGetView",
    "FilterPrefixView",
    "KeyPrefixView",
    "LazySafetensorsDict",
    "RemapKeysView",
]
