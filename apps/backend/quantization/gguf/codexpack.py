"""
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: CodexPack GGUF schema helpers (contract-level; no kernels).
Defines the strict metadata keys and manifest parsing/validation for `*.codexpack.gguf` files, which are Codex-only GGUF
containers carrying kernel-friendly packed execution payloads.

Symbols (top-level; keep in sync; no ghosts):
- `CODEXPACK_SCHEMA` (constant): Required `codex.pack.schema` value for CodexPack GGUF files.
- `CODEXPACK_SCHEMA_VERSION` (constant): Supported CodexPack schema version.
- `KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1` (constant): CodexPack v1 kernel id (Q4_K tile-packed linear; CUDA).
- `CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1` (constant): Minimum CUDA SM required by `tilepack_v1` (SM86 baseline).
- `CodexPackError` (exception): Raised when a CodexPack GGUF fails schema/manifest validation.
- `CodexPackManifestV1` (dataclass): Parsed v1 CodexPack manifest wrapper.
- `is_codexpack_gguf` (function): Returns True when a GGUF file declares CodexPack schema keys.
- `read_codexpack_schema_version` (function): Reads and validates `codex.pack.schema_version`.
- `read_codexpack_manifest_json` (function): Reads and parses the `codex.pack.manifest_json` blob (strict JSON).
- `validate_codexpack_manifest_v1` (function): Validates the v1 manifest envelope and required entry fields.
- `load_codexpack_manifest_v1` (function): Convenience loader that enforces schema + v1 manifest validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .constants import GGUFValueType, GGMLQuantizationType, GGML_QUANT_SIZES
from .reader import GGUFReader, ReaderField

CODEXPACK_SCHEMA = "codexpack.gguf"
CODEXPACK_SCHEMA_VERSION = 1

KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 = "cuda.ggml_q4_k.linear.tilepack_v1"
CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 = 86
_SUPPORTED_KERNEL_IDS_V1 = {KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1}

TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 = 128
TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 = 256
Q4K_BLOCK_BYTES = GGML_QUANT_SIZES[GGMLQuantizationType.Q4_K][1]

_KEY_SCHEMA = "codex.pack.schema"
_KEY_SCHEMA_VERSION = "codex.pack.schema_version"
_KEY_KEYMAP_ID = "codex.pack.keymap_id"
_KEY_KERNEL_ID = "codex.pack.kernel_id"
_KEY_MANIFEST_JSON = "codex.pack.manifest_json"
_KEY_CUDA_SM_MIN = "codex.pack.cuda_sm_min"


class CodexPackError(ValueError):
    """Invalid CodexPack GGUF schema or manifest."""


@dataclass(frozen=True, slots=True)
class CodexPackManifestV1:
    schema_version: int
    keymap_id: str
    kernel_id: str
    manifest: Mapping[str, Any]


def _require_field(reader: GGUFReader, key: str) -> ReaderField:
    field = reader.get_field(key)
    if field is None:
        raise CodexPackError(f"Missing required GGUF metadata key: {key!r}")
    return field


def _decode_scalar(field: ReaderField, *, key: str) -> object:
    if not field.types:
        raise CodexPackError(f"GGUF metadata key {key!r} has no type information.")
    vtype = field.types[0]
    if vtype is GGUFValueType.STRING:
        if not field.data:
            raise CodexPackError(f"GGUF metadata key {key!r} has no data index.")
        raw = field.parts[field.data[0]]
        try:
            blob = raw.tobytes()  # numpy array of uint8
        except Exception as exc:  # noqa: BLE001
            raise CodexPackError(f"GGUF metadata key {key!r} could not be decoded as bytes: {exc}") from exc
        try:
            return blob.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise CodexPackError(f"GGUF metadata key {key!r} is not valid UTF-8: {exc}") from exc

    if vtype is GGUFValueType.ARRAY:
        raise CodexPackError(f"GGUF metadata key {key!r} is an array; expected scalar.")

    if not field.data:
        raise CodexPackError(f"GGUF metadata key {key!r} has no data index.")
    raw = field.parts[field.data[0]]
    try:
        if getattr(raw, "size", None) != 1:
            raise CodexPackError(f"GGUF metadata key {key!r} has non-scalar payload.")
        return raw[0].item() if hasattr(raw[0], "item") else raw[0]
    except CodexPackError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise CodexPackError(f"GGUF metadata key {key!r} could not be decoded as scalar: {exc}") from exc


def _read_string(reader: GGUFReader, key: str) -> str:
    field = _require_field(reader, key)
    value = _decode_scalar(field, key=key)
    if not isinstance(value, str) or not value.strip():
        raise CodexPackError(f"GGUF metadata key {key!r} must be a non-empty string; got: {value!r}")
    return value


def _read_uint32(reader: GGUFReader, key: str) -> int:
    field = _require_field(reader, key)
    if not field.types or field.types[0] is not GGUFValueType.UINT32:
        raw_type = field.types[0] if field.types else None
        raise CodexPackError(f"GGUF metadata key {key!r} must be UINT32; got: {raw_type}")
    value = _decode_scalar(field, key=key)
    try:
        return int(value)
    except Exception as exc:  # noqa: BLE001
        raise CodexPackError(f"GGUF metadata key {key!r} must be an int; got: {value!r}") from exc


def is_codexpack_gguf(reader: GGUFReader) -> bool:
    try:
        schema = reader.get_field(_KEY_SCHEMA)
        if schema is None:
            return False
        value = _decode_scalar(schema, key=_KEY_SCHEMA)
        return isinstance(value, str) and value.strip() == CODEXPACK_SCHEMA
    except Exception:
        return False


def read_codexpack_schema_version(reader: GGUFReader) -> int:
    schema = _read_string(reader, _KEY_SCHEMA)
    if schema != CODEXPACK_SCHEMA:
        raise CodexPackError(f"GGUF file is not a CodexPack schema: {_KEY_SCHEMA}={schema!r}")
    version = _read_uint32(reader, _KEY_SCHEMA_VERSION)
    if version != CODEXPACK_SCHEMA_VERSION:
        raise CodexPackError(
            f"Unsupported CodexPack schema version: {version} (expected {CODEXPACK_SCHEMA_VERSION}).",
        )
    return version


def read_codexpack_manifest_json(reader: GGUFReader) -> Mapping[str, Any]:
    raw = _read_string(reader, _KEY_MANIFEST_JSON)
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CodexPackError(f"Invalid JSON in {_KEY_MANIFEST_JSON}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise CodexPackError(f"{_KEY_MANIFEST_JSON} must decode to an object; got: {type(parsed).__name__}")
    return parsed


def validate_codexpack_manifest_v1(manifest: Mapping[str, Any]) -> None:
    schema_version = manifest.get("schema_version")
    if schema_version != CODEXPACK_SCHEMA_VERSION:
        raise CodexPackError(f"manifest.schema_version must be {CODEXPACK_SCHEMA_VERSION}; got: {schema_version!r}")

    keymap_id = manifest.get("keymap_id")
    if not isinstance(keymap_id, str) or not keymap_id.strip():
        raise CodexPackError(f"manifest.keymap_id must be a non-empty string; got: {keymap_id!r}")

    kernel_id = manifest.get("kernel_id")
    if not isinstance(kernel_id, str) or not kernel_id.strip():
        raise CodexPackError(f"manifest.kernel_id must be a non-empty string; got: {kernel_id!r}")
    if kernel_id not in _SUPPORTED_KERNEL_IDS_V1:
        allowed = ", ".join(sorted(_SUPPORTED_KERNEL_IDS_V1))
        raise CodexPackError(f"manifest.kernel_id must be one of: {allowed}; got: {kernel_id!r}")

    targets = manifest.get("targets")
    if not isinstance(targets, dict):
        raise CodexPackError(f"manifest.targets must be an object; got: {type(targets).__name__}")
    backend = targets.get("backend")
    if backend != "cuda":
        raise CodexPackError(f"manifest.targets.backend must be 'cuda' for v1; got: {backend!r}")
    cuda_sm_min = targets.get("cuda_sm_min")
    if cuda_sm_min != CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1:
        raise CodexPackError(
            f"manifest.targets.cuda_sm_min must be {CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1} for v1; got: {cuda_sm_min!r}"
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise CodexPackError(f"manifest.entries must be an array; got: {type(entries).__name__}")
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise CodexPackError(f"manifest.entries[{idx}] must be an object; got: {type(entry).__name__}")
        kind = entry.get("kind")
        if kind != "linear_weight":
            raise CodexPackError(f"manifest.entries[{idx}].kind must be 'linear_weight'; got: {kind!r}")

        param_key = entry.get("param_key")
        if not isinstance(param_key, str) or not param_key.strip():
            raise CodexPackError(f"manifest.entries[{idx}].param_key must be a non-empty string; got: {param_key!r}")

        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 2 or not all(isinstance(x, int) and x > 0 for x in shape):
            raise CodexPackError(f"manifest.entries[{idx}].shape must be a 2-int array; got: {shape!r}")
        out_features, in_features = int(shape[0]), int(shape[1])
        if out_features % TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 != 0:
            raise CodexPackError(
                f"manifest.entries[{idx}].shape[0] (out_features) must be a multiple of "
                f"{TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1} for tilepack_v1; got: {out_features!r}"
            )
        if in_features % TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1 != 0:
            raise CodexPackError(
                f"manifest.entries[{idx}].shape[1] (in_features) must be a multiple of "
                f"{TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1} for tilepack_v1; got: {in_features!r}"
            )

        qtype = entry.get("qtype")
        if qtype != "Q4_K":
            raise CodexPackError(f"manifest.entries[{idx}].qtype must be 'Q4_K' for v1; got: {qtype!r}")

        tensors = entry.get("tensors")
        if not isinstance(tensors, dict):
            raise CodexPackError(f"manifest.entries[{idx}].tensors must be an object; got: {type(tensors).__name__}")
        packed = tensors.get("packed")
        if not isinstance(packed, str) or not packed.strip():
            raise CodexPackError(
                f"manifest.entries[{idx}].tensors.packed must be a non-empty string; got: {packed!r}",
            )
        if not packed.startswith("__codexpack__."):
            raise CodexPackError(
                f"manifest.entries[{idx}].tensors.packed must start with '__codexpack__.'; got: {packed!r}",
            )
        dora_norm_out = tensors.get("dora_norm_out")
        if not isinstance(dora_norm_out, str) or not dora_norm_out.strip():
            raise CodexPackError(
                f"manifest.entries[{idx}].tensors.dora_norm_out must be a non-empty string; got: {dora_norm_out!r}",
            )
        if not dora_norm_out.startswith("__codexpack__."):
            raise CodexPackError(
                f"manifest.entries[{idx}].tensors.dora_norm_out must start with '__codexpack__.'; got: {dora_norm_out!r}",
            )

    float_keys = manifest.get("float_keys", [])
    if float_keys is not None:
        if not isinstance(float_keys, list) or not all(isinstance(k, str) and k.strip() for k in float_keys):
            raise CodexPackError(f"manifest.float_keys must be an array of strings; got: {float_keys!r}")

    fallback_fp16_keys = manifest.get("fallback_fp16_keys", [])
    if fallback_fp16_keys is not None:
        if not isinstance(fallback_fp16_keys, list) or not all(isinstance(k, str) and k.strip() for k in fallback_fp16_keys):
            raise CodexPackError(f"manifest.fallback_fp16_keys must be an array of strings; got: {fallback_fp16_keys!r}")


def load_codexpack_manifest_v1(reader: GGUFReader) -> CodexPackManifestV1:
    read_codexpack_schema_version(reader)
    kv_keymap = _read_string(reader, _KEY_KEYMAP_ID)
    kv_kernel = _read_string(reader, _KEY_KERNEL_ID)
    kv_sm_min = _read_uint32(reader, _KEY_CUDA_SM_MIN)
    manifest = read_codexpack_manifest_json(reader)
    validate_codexpack_manifest_v1(manifest)

    if manifest.get("keymap_id") != kv_keymap:
        raise CodexPackError(
            f"codex.pack.keymap_id mismatch: KV={kv_keymap!r} manifest={manifest.get('keymap_id')!r}",
        )
    if manifest.get("kernel_id") != kv_kernel:
        raise CodexPackError(
            f"codex.pack.kernel_id mismatch: KV={kv_kernel!r} manifest={manifest.get('kernel_id')!r}",
        )
    manifest_sm_min = manifest.get("targets", {}).get("cuda_sm_min")
    if int(manifest_sm_min) != int(kv_sm_min):
        raise CodexPackError(
            f"codex.pack.cuda_sm_min mismatch: KV={kv_sm_min!r} manifest={manifest_sm_min!r}",
        )

    tensors_by_name = {t.name: t for t in reader.tensors}

    for idx, entry in enumerate(manifest.get("entries", [])):
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "linear_weight":
            continue
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) != 2:
            continue
        out_features, in_features = int(shape[0]), int(shape[1])
        k_tiles = in_features // TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1
        expected_packed_bytes = out_features * k_tiles * Q4K_BLOCK_BYTES

        tensors = entry.get("tensors", {})
        packed_name = tensors.get("packed")
        if isinstance(packed_name, str):
            packed_tensor = tensors_by_name.get(packed_name)
            if packed_tensor is None:
                raise CodexPackError(f"Missing packed tensor referenced by manifest.entries[{idx}]: {packed_name!r}")
            if packed_tensor.tensor_type is not GGMLQuantizationType.I8:
                raise CodexPackError(
                    f"manifest.entries[{idx}].tensors.packed must be an I8 GGUF tensor; got: {packed_tensor.tensor_type.name}"
                )
            if int(packed_tensor.n_bytes) != int(expected_packed_bytes):
                raise CodexPackError(
                    f"manifest.entries[{idx}].tensors.packed size mismatch for {packed_name!r}: "
                    f"expected {expected_packed_bytes} bytes, got {packed_tensor.n_bytes}"
                )

        dora_norm_name = tensors.get("dora_norm_out")
        if isinstance(dora_norm_name, str):
            dora_tensor = tensors_by_name.get(dora_norm_name)
            if dora_tensor is None:
                raise CodexPackError(
                    f"Missing dora_norm_out tensor referenced by manifest.entries[{idx}]: {dora_norm_name!r}"
                )
            if dora_tensor.tensor_type is not GGMLQuantizationType.F32:
                raise CodexPackError(
                    f"manifest.entries[{idx}].tensors.dora_norm_out must be an F32 GGUF tensor; got: {dora_tensor.tensor_type.name}"
                )
            if int(dora_tensor.n_elements) != int(out_features):
                raise CodexPackError(
                    f"manifest.entries[{idx}].tensors.dora_norm_out length mismatch for {dora_norm_name!r}: "
                    f"expected {out_features}, got {dora_tensor.n_elements}"
                )

    return CodexPackManifestV1(
        schema_version=int(manifest["schema_version"]),
        keymap_id=str(kv_keymap),
        kernel_id=str(kv_kernel),
        manifest=manifest,
    )


__all__ = [
    "CODEXPACK_SCHEMA",
    "CODEXPACK_SCHEMA_VERSION",
    "KERNEL_ID_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "CUDA_SM_MIN_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "TILE_M_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "TILE_K_CUDA_GGML_Q4K_LINEAR_TILEPACK_V1",
    "CodexPackError",
    "CodexPackManifestV1",
    "is_codexpack_gguf",
    "load_codexpack_manifest_v1",
    "read_codexpack_manifest_json",
    "read_codexpack_schema_version",
    "validate_codexpack_manifest_v1",
]
