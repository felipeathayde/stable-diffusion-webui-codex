"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: File metadata readers for UI/debug endpoints.
Provides lightweight, JSON-serializable metadata extraction for GGUF and SafeTensors files
so the Web UI can display provenance (e.g., repo commit/url) and basic file info.

Symbols (top-level; keep in sync; no ghosts):
- `read_file_metadata` (function): Read metadata from a supported weights file (GGUF/SafeTensors), returning both flat and nested views.
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Sequence


_GGUF_MAGIC = 0x46554747  # 'GGUF' little-endian

_GGUF_VALUE_UINT8 = 0
_GGUF_VALUE_INT8 = 1
_GGUF_VALUE_UINT16 = 2
_GGUF_VALUE_INT16 = 3
_GGUF_VALUE_UINT32 = 4
_GGUF_VALUE_INT32 = 5
_GGUF_VALUE_FLOAT32 = 6
_GGUF_VALUE_BOOL = 7
_GGUF_VALUE_STRING = 8
_GGUF_VALUE_ARRAY = 9
_GGUF_VALUE_UINT64 = 10
_GGUF_VALUE_INT64 = 11
_GGUF_VALUE_FLOAT64 = 12

_SAFE_MAX_GGUF_ARRAY_ITEMS = 256


def _read_exact(handle, n: int) -> bytes:
    data = handle.read(n)
    if len(data) != n:
        raise EOFError(f"Unexpected EOF (wanted {n} bytes, got {len(data)}).")
    return data


def _read_u32(handle) -> int:
    return struct.unpack("<I", _read_exact(handle, 4))[0]


def _read_i32(handle) -> int:
    return struct.unpack("<i", _read_exact(handle, 4))[0]


def _read_u64(handle) -> int:
    return struct.unpack("<Q", _read_exact(handle, 8))[0]


def _read_i64(handle) -> int:
    return struct.unpack("<q", _read_exact(handle, 8))[0]


def _read_f32(handle) -> float:
    return struct.unpack("<f", _read_exact(handle, 4))[0]


def _read_f64(handle) -> float:
    return struct.unpack("<d", _read_exact(handle, 8))[0]


def _read_string(handle) -> str:
    n = _read_u64(handle)
    raw = _read_exact(handle, int(n))
    return raw.decode("utf-8", errors="replace")


def _read_gguf_value(handle, vtype: int) -> object:
    if vtype == _GGUF_VALUE_UINT8:
        return _read_exact(handle, 1)[0]
    if vtype == _GGUF_VALUE_INT8:
        return struct.unpack("<b", _read_exact(handle, 1))[0]
    if vtype == _GGUF_VALUE_UINT16:
        return struct.unpack("<H", _read_exact(handle, 2))[0]
    if vtype == _GGUF_VALUE_INT16:
        return struct.unpack("<h", _read_exact(handle, 2))[0]
    if vtype == _GGUF_VALUE_UINT32:
        return _read_u32(handle)
    if vtype == _GGUF_VALUE_INT32:
        return _read_i32(handle)
    if vtype == _GGUF_VALUE_FLOAT32:
        return _read_f32(handle)
    if vtype == _GGUF_VALUE_BOOL:
        return bool(_read_exact(handle, 1)[0])
    if vtype == _GGUF_VALUE_STRING:
        return _read_string(handle)
    if vtype == _GGUF_VALUE_UINT64:
        return _read_u64(handle)
    if vtype == _GGUF_VALUE_INT64:
        return _read_i64(handle)
    if vtype == _GGUF_VALUE_FLOAT64:
        return _read_f64(handle)
    if vtype == _GGUF_VALUE_ARRAY:
        elem_type = _read_u32(handle)
        count = _read_u64(handle)
        preview: list[object] = []
        for i in range(int(count)):
            if i >= _SAFE_MAX_GGUF_ARRAY_ITEMS:
                # Consume remaining items without storing to avoid huge payloads.
                _ = _read_gguf_value(handle, int(elem_type))
                continue
            preview.append(_read_gguf_value(handle, int(elem_type)))
        if count > _SAFE_MAX_GGUF_ARRAY_ITEMS:
            return {"__truncated__": True, "count": int(count), "preview": preview, "elem_type": int(elem_type)}
        return preview
    raise ValueError(f"Unsupported GGUF value type: {vtype}")


def _read_gguf_kv(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        magic = _read_u32(handle)
        if magic != _GGUF_MAGIC:
            raise ValueError(f"Not a GGUF file (magic={hex(magic)}).")
        version = _read_u32(handle)
        n_tensors = _read_u64(handle)
        n_kv = _read_u64(handle)

        kv: dict[str, object] = {}
        for _ in range(int(n_kv)):
            key = _read_string(handle)
            vtype = _read_u32(handle)
            kv[key] = _read_gguf_value(handle, int(vtype))

    kv["gguf.version"] = int(version)
    kv["gguf.tensor_count"] = int(n_tensors)
    kv["gguf.kv_count"] = int(n_kv)
    return kv


def _read_safetensors_header(path: Path) -> dict[str, object]:
    with path.open("rb") as handle:
        header_len = _read_u64(handle)
        if header_len <= 0 or header_len > 64 * 1024 * 1024:
            raise ValueError(f"Invalid safetensors header length: {header_len}")
        raw = _read_exact(handle, int(header_len))
    data = json.loads(raw.decode("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Invalid safetensors header (expected a JSON object).")
    return data


def _nest_dotted_keys(flat: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for raw_key, value in flat.items():
        key = str(raw_key)
        if "." not in key:
            if key not in out:
                out[key] = value
            else:
                existing = out.get(key)
                if isinstance(existing, dict):
                    existing["__value__"] = value
                else:
                    out[key] = {"__value__": existing, "__new__": value}
            continue

        parts = [p for p in key.split(".") if p]
        if not parts:
            continue
        cur: MutableMapping[str, object] = out
        for part in parts[:-1]:
            node = cur.get(part)
            if not isinstance(node, dict):
                node = {} if node is None else {"__value__": node}
                cur[part] = node
            cur = node  # type: ignore[assignment]
        leaf = parts[-1]
        existing_leaf = cur.get(leaf)
        if isinstance(existing_leaf, dict):
            existing_leaf["__value__"] = value
        elif leaf in cur:
            cur[leaf] = {"__value__": existing_leaf, "__new__": value}
        else:
            cur[leaf] = value
    return out


@dataclass(frozen=True, slots=True)
class FileMetadataResult:
    path: str
    kind: str
    flat: Dict[str, object]
    nested: Dict[str, object]
    summary: Dict[str, object]

    def as_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "kind": self.kind,
            "flat": dict(self.flat),
            "nested": dict(self.nested),
            "summary": dict(self.summary),
        }


def read_file_metadata(raw_path: str) -> FileMetadataResult:
    value = str(raw_path or "").strip()
    if not value:
        raise ValueError("path is required")

    p = Path(os.path.expanduser(value))
    if not p.is_absolute():
        from apps.backend.infra.config.repo_root import get_repo_root

        p = (get_repo_root() / p)

    resolved = p.resolve(strict=False)
    if not resolved.exists() or not resolved.is_file():
        raise FileNotFoundError(str(resolved))

    suffix = resolved.suffix.lower()
    if suffix == ".gguf":
        flat = _read_gguf_kv(resolved)
        summary = {
            "file": {"size_bytes": int(resolved.stat().st_size), "name": resolved.name},
            "contains_codex_keys": any(str(k).startswith("codex.") for k in flat.keys()),
        }
        return FileMetadataResult(
            path=str(resolved),
            kind="gguf",
            flat=flat,
            nested=_nest_dotted_keys(flat),
            summary=summary,
        )

    if suffix in {".safetensors", ".safetensor"}:
        header = _read_safetensors_header(resolved)
        meta = header.get("__metadata__", {})
        if not isinstance(meta, dict):
            meta = {}

        dtype_counts: Dict[str, int] = {}
        tensor_count = 0
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(entry, dict):
                continue
            tensor_count += 1
            dtype = entry.get("dtype")
            if isinstance(dtype, str) and dtype:
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1

        flat = {
            "__metadata__": meta,
            "tensor_count": tensor_count,
            "dtype_counts": dtype_counts,
            "file_name": resolved.name,
            "file_size_bytes": int(resolved.stat().st_size),
        }
        return FileMetadataResult(
            path=str(resolved),
            kind="safetensors",
            flat=dict(flat),
            nested=_nest_dotted_keys(flat),
            summary={"file": {"size_bytes": int(resolved.stat().st_size), "name": resolved.name}},
        )

    raise ValueError(f"Unsupported file type for metadata: {suffix}")


__all__ = ["read_file_metadata"]
