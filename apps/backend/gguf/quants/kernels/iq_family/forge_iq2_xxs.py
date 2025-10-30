from __future__ import annotations

import numpy as np

from ....constants import QK_K

__all__ = [
    "KSIGNS",
    "GRID_SHAPE",
    "GRID_MAP",
    "GRID_HEX",
    "dequantize_numpy",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — IQ2_XXS
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


KSIGNS: bytes = (
    b"\x00\x81\x82\x03\x84\x05\x06\x87\x88\x09\x0a\x8b\x0c\x8d\x8e\x0f"
    b"\x90\x11\x12\x93\x14\x95\x96\x17\x18\x99\x9a\x1b\x9c\x1d\x1e\x9f"
    b"\xa0\x21\x22\xa3\x24\xa5\xa6\x27\x28\xa9\xaa\x2b\xac\x2d\x2e\xaf"
    b"\x30\xb1\xb2\x33\xb4\x35\x36\xb7\xb8\x39\x3a\xbb\x3c\xbd\xbe\x3f"
    b"\xc0\x41\x42\xc3\x44\xc5\xc6\x47\x48\xc9\xca\x4b\xcc\x4d\x4e\xcf"
    b"\x50\xd1\xd2\x53\xd4\x55\x56\xd7\xd8\x59\x5a\xdb\x5c\xdd\xde\x5f"
    b"\x60\xe1\xe2\x63\xe4\x65\x66\xe7\xe8\x69\x6a\xeb\x6c\xed\xee\x6f"
    b"\xf0\x71\x72\xf3\x74\xf5\xf6\x77\x78\xf9\xfa\x7b\xfc\x7d\x7e\xff"
)

# iq2xxs_grid, but with each byte of the original packed in 2 bits,
# by mapping 0x08 to 0, 0x19 to 1, and 0x2b to 2.
GRID_SHAPE: tuple[int, int] = (256, 8)
GRID_MAP: tuple[int, int, int] = (0x08, 0x19, 0x2b)
GRID_HEX: bytes = (
    b"00000200050008000a00110014002000220028002a0041004400500058006100"
    b"6400800082008a00a20001010401100115014001840198010002020222028202"
    b"010404041004210424044004420448046004810484049004a404000502050805"
    b"200546056905800591050906100640068406a406000805080808140828084108"
    b"440850085208880804094009020a140a01100410101021104010601084109010"
    b"951000110811201150115a118011241245120014081420142514491480141815"
    b"6215001616160118041810184018811800190519a019511a002002200a204420"
    b"6120802082202921482100220222012404241024402456240025412564259026"
    b"082820289428442a014004401040184021402440404048405640604081408440"
    b"9040004120416141804185410142104248425642684200440844204480449944"
    b"124524450046014804481048404845480049584961498249454a904a00500850"
    b"1150195020508050885004514251a4519152905492540a550156545600581158"
    b"195864584059085a046010604060686000615561186260620064056410651265"
    b"84654268008002800a8041808280048118814081118201840484108415844084"
    b"608400854685948509864086608602880489118a0490109024904090a1901691"
    b"8091459200942294449451958198209902a050a085a009a100a218a450a804a9"
)


def dequantize_numpy(blocks: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    d, qs = np.hsplit(blocks, [2])

    d = d.view(np.float16).astype(np.float32)

    qs = qs.view(np.uint32).reshape(n_blocks, -1, 2)

    db = d * (np.float32(0.5) + (qs[..., 1] >> 28).astype(np.float32)) * np.float32(0.25)
    db = db.reshape((n_blocks, -1, 1, 1))

    # get the sign indices and unpack the bits
    signs = qs[..., 1].reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
    ksigns = np.frombuffer(KSIGNS, dtype=np.uint8).reshape((1, 1, 1, 128))
    signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
    signs = np.take_along_axis(ksigns, signs, axis=-1)
    signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
    signs = signs & np.uint8(0x01)
    signs = np.where(signs == 0, np.float32(1), np.float32(-1))
    signs = signs.reshape((n_blocks, -1, 4, 8))

    grid = np.take_along_axis(grid, qs[..., 0].copy().view(np.uint8).reshape((n_blocks, -1, 1, 1)), axis=-2)
    grid = grid.reshape((n_blocks, -1, 4, 8))

    return (db * grid * signs).reshape((n_blocks, QK_K))
