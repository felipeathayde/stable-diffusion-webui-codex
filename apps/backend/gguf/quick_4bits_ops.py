# Codex wrapper around the Forge 4-bit helper routines.

from __future__ import annotations

from apps.backend.gguf.quants.kernels import forge_quick_4bits_ops as _forge_ops

disable_all_optimizations = _forge_ops.disable_all_optimizations
native_4bits_lookup_table = getattr(_forge_ops, "native_4bits_lookup_table", None)
native_4bits_lookup_table_u = getattr(_forge_ops, "native_4bits_lookup_table_u", None)


def native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits(x):
    return _forge_ops.native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits(x)


def native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits_u(x):
    return _forge_ops.native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits_u(x)


def quick_unpack_4bits(x):
    return _forge_ops.quick_unpack_4bits(x)


def quick_unpack_4bits_u(x):
    return _forge_ops.quick_unpack_4bits_u(x)


def change_4bits_order(x):
    return _forge_ops.change_4bits_order(x)


__all__ = [
    "disable_all_optimizations",
    "native_4bits_lookup_table",
    "native_4bits_lookup_table_u",
    "native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits",
    "native_unpack_4x4bits_in_1x16bits_to_4x8bits_in_1x32bits_u",
    "quick_unpack_4bits",
    "quick_unpack_4bits_u",
    "change_4bits_order",
]
