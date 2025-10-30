from __future__ import annotations

import numpy as np

from ....constants import QK_K
from .forge_iq2_xxs import KSIGNS

__all__ = [
    "GRID_SHAPE",
    "GRID_MAP",
    "GRID_HEX",
    "dequantize_numpy",
]

# -----------------------------------------------------------------------------
# Codex Forge Port — IQ3_XXS
# Extracted from AUTOMATIC1111 Forge (packages_3rdparty/gguf/quants.py)
# Original license: MIT License (c) 2023 Georgi Gerganov
# -----------------------------------------------------------------------------


GRID_SHAPE: tuple[int, int] = (256, 4)
GRID_MAP: tuple[int, int, int, int, int, int, int, int] = (0x04, 0x0C, 0x14, 0x1C, 0x24, 0x2C, 0x34, 0x3E)
GRID_HEX: bytes = (
    b"0000020004001100130017002000220031004200730075000101030110011201"
    b"2101250130013201410154017001000202020402110220022202310233023702"
    b"5102570275020103070310031203250370031304370444045704730475040105"
    b"0705320552053506640610071407160743076107011003101010121021102310"
    b"3010321034104710501000110211111120112211011203121012121221123012"
    b"7212001302132013311346136613011405145014201524154615711505162217"
    b"4017002002201120132020202220262031204220012103210521102112212121"
    b"3021632167217021002202221122172220222222372240225522012310231423"
    b"7023742335245324032527254125742501270327162745270130103012302130"
    b"2330503065307230003102312031313144314631013203321032253252327232"
    b"1133333330344734723400350635223555351436363663363337603704401740"
    b"3540374053405740744120423742404260426642074345430444514464442545"
    b"4345704505471047124730471250415070500051065126515551145232527252"
    b"0253535310542354275472540255315550562457425724604460466064602161"
    b"6161176264623063366344640565526533660367216703700570077010703270"
    b"5270267140711272457252720073157333736073217441740075027524753076"
)


def dequantize_numpy(blocks: np.ndarray, grid: np.ndarray) -> np.ndarray:
    n_blocks = blocks.shape[0]

    d, rest = np.hsplit(blocks, [2])
    qs, scales = np.hsplit(rest, [QK_K // 4])

    d = d.view(np.float16).astype(np.float32)
    scales = scales.view(np.uint32)

    db = d * (np.float32(0.5) + (scales >> 28).astype(np.float32)) * np.float32(0.5)
    db = db.reshape((n_blocks, -1, 1, 1))

    # get the sign indices and unpack the bits
    signs = scales.reshape((n_blocks, -1, 1)) >> np.array([0, 7, 14, 21], dtype=np.uint32).reshape((1, 1, 4))
    ksigns = np.frombuffer(KSIGNS, dtype=np.uint8).reshape((1, 1, 1, 128))
    signs = (signs & np.uint32(0x7F)).reshape((n_blocks, -1, 4, 1))
    signs = np.take_along_axis(ksigns, signs, axis=-1)
    signs = signs.reshape((n_blocks, -1, 4, 1)) >> np.array([i for i in range(8)], dtype=np.uint8).reshape((1, 1, 1, 8))
    signs = signs & np.uint8(0x01)
    signs = np.where(signs == 0, np.float32(1), np.float32(-1))
    signs = signs.reshape((n_blocks, -1, 4, 8))

    grid = np.take_along_axis(grid, qs.reshape((n_blocks, -1, 1, 1)), axis=-2)
    grid = grid.reshape((n_blocks, -1, 4, 8))

    return (db * grid * signs).reshape((n_blocks, QK_K))
