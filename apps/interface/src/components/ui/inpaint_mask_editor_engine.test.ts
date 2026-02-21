/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Deterministic regression tests for the inpaint mask editor engine.
Validates draw operations, history policy behavior, undo/redo semantics, and RGBA conversion helpers.

Symbols (top-level; keep in sync; no ghosts):
- `describe('inpaint_mask_editor_engine')` (test suite): Verifies mask editing operations and invariants.
*/

import { describe, expect, it } from 'vitest'
import {
  InpaintMaskEditorEngine,
  MASK_VALUE_EMPTY,
  MASK_VALUE_FILLED,
  maskPlaneToRgba,
  rgbaToMaskPlane,
} from './inpaint_mask_editor_engine'

function pixel(mask: Uint8Array, width: number, x: number, y: number): number {
  return mask[(y * width) + x]
}

describe('inpaint_mask_editor_engine', () => {
  it('applies stroke and supports multi-step undo/redo', () => {
    const engine = new InpaintMaskEditorEngine({ width: 8, height: 8 })

    engine.applyStroke([{ x: 1.5, y: 1.5 }, { x: 6.5, y: 1.5 }], 1.4, MASK_VALUE_FILLED)
    const afterFirst = engine.currentMask
    expect(pixel(afterFirst, 8, 2, 1)).toBe(MASK_VALUE_FILLED)

    engine.applyStroke([{ x: 1.5, y: 5.5 }, { x: 6.5, y: 5.5 }], 1.4, MASK_VALUE_FILLED)
    const afterSecond = engine.currentMask
    expect(pixel(afterSecond, 8, 2, 5)).toBe(MASK_VALUE_FILLED)

    expect(engine.undo()).toBe(true)
    expect(pixel(engine.currentMask, 8, 2, 5)).toBe(MASK_VALUE_EMPTY)
    expect(pixel(engine.currentMask, 8, 2, 1)).toBe(MASK_VALUE_FILLED)

    expect(engine.undo()).toBe(true)
    expect(pixel(engine.currentMask, 8, 2, 1)).toBe(MASK_VALUE_EMPTY)

    expect(engine.redo()).toBe(true)
    expect(pixel(engine.currentMask, 8, 2, 1)).toBe(MASK_VALUE_FILLED)
  })

  it('invalidates redo chain on new commit', () => {
    const engine = new InpaintMaskEditorEngine({ width: 8, height: 8 })
    engine.applyCircle({ x: 2, y: 2 }, 1.5, MASK_VALUE_FILLED)
    engine.applyCircle({ x: 5, y: 5 }, 1.5, MASK_VALUE_FILLED)

    expect(engine.undo()).toBe(true)
    expect(engine.canRedo).toBe(true)

    engine.applyCircle({ x: 1, y: 6 }, 1.2, MASK_VALUE_FILLED)
    expect(engine.canRedo).toBe(false)
  })

  it('enforces history maxSnapshots policy', () => {
    const engine = new InpaintMaskEditorEngine({
      width: 4,
      height: 4,
      historyPolicy: { maxSnapshots: 3, maxBytes: 1024 * 1024 },
    })

    engine.applyCircle({ x: 0.5, y: 0.5 }, 1, MASK_VALUE_FILLED)
    engine.applyCircle({ x: 1.5, y: 1.5 }, 1, MASK_VALUE_FILLED)
    engine.applyCircle({ x: 2.5, y: 2.5 }, 1, MASK_VALUE_FILLED)
    engine.applyCircle({ x: 3.0, y: 3.0 }, 1, MASK_VALUE_FILLED)

    expect(engine.snapshotCount).toBe(3)
    expect(engine.canUndo).toBe(true)

    expect(engine.undo()).toBe(true)
    expect(engine.undo()).toBe(true)
    expect(engine.canUndo).toBe(false)
  })

  it('enforces history maxBytes policy', () => {
    const engine = new InpaintMaskEditorEngine({
      width: 16,
      height: 16,
      historyPolicy: { maxSnapshots: 100, maxBytes: 256 },
    })

    engine.applyCircle({ x: 1, y: 1 }, 1, MASK_VALUE_FILLED)
    engine.applyCircle({ x: 3, y: 3 }, 1, MASK_VALUE_FILLED)
    engine.applyCircle({ x: 5, y: 5 }, 1, MASK_VALUE_FILLED)

    expect(engine.snapshotCount).toBe(1)
    expect(engine.canUndo).toBe(false)
  })

  it('requires at least three points for polygon', () => {
    const engine = new InpaintMaskEditorEngine({ width: 8, height: 8 })
    expect(() => engine.applyPolygon([{ x: 1, y: 1 }, { x: 6, y: 1 }], MASK_VALUE_FILLED)).toThrow(
      'Polygon requires at least 3 points.',
    )
  })

  it('fills polygon interior', () => {
    const engine = new InpaintMaskEditorEngine({ width: 12, height: 12 })
    engine.applyPolygon(
      [
        { x: 2, y: 2 },
        { x: 9, y: 2 },
        { x: 9, y: 9 },
        { x: 2, y: 9 },
      ],
      MASK_VALUE_FILLED,
    )

    const mask = engine.currentMask
    expect(pixel(mask, 12, 5, 5)).toBe(MASK_VALUE_FILLED)
    expect(pixel(mask, 12, 0, 0)).toBe(MASK_VALUE_EMPTY)
  })

  it('clear is undoable', () => {
    const engine = new InpaintMaskEditorEngine({ width: 8, height: 8 })
    engine.applyCircle({ x: 3, y: 3 }, 2, MASK_VALUE_FILLED)
    engine.clear(MASK_VALUE_EMPTY)

    const afterClear = engine.currentMask
    expect(pixel(afterClear, 8, 3, 3)).toBe(MASK_VALUE_EMPTY)

    expect(engine.undo()).toBe(true)
    const afterUndo = engine.currentMask
    expect(pixel(afterUndo, 8, 3, 3)).toBe(MASK_VALUE_FILLED)
  })

  it('replaceMask is undoable and redoable', () => {
    const engine = new InpaintMaskEditorEngine({ width: 4, height: 4 })
    const uploadedMask = new Uint8Array(16)
    uploadedMask[(1 * 4) + 1] = MASK_VALUE_FILLED
    uploadedMask[(2 * 4) + 2] = MASK_VALUE_FILLED

    engine.replaceMask(uploadedMask)
    expect(pixel(engine.currentMask, 4, 1, 1)).toBe(MASK_VALUE_FILLED)
    expect(pixel(engine.currentMask, 4, 2, 2)).toBe(MASK_VALUE_FILLED)

    expect(engine.undo()).toBe(true)
    expect(pixel(engine.currentMask, 4, 1, 1)).toBe(MASK_VALUE_EMPTY)
    expect(pixel(engine.currentMask, 4, 2, 2)).toBe(MASK_VALUE_EMPTY)

    expect(engine.redo()).toBe(true)
    expect(pixel(engine.currentMask, 4, 1, 1)).toBe(MASK_VALUE_FILLED)
    expect(pixel(engine.currentMask, 4, 2, 2)).toBe(MASK_VALUE_FILLED)
  })

  it('uses alpha channel precedence when rgba has transparency', () => {
    const rgba = new Uint8ClampedArray([
      0, 0, 0, 0,
      255, 255, 255, 255,
      255, 255, 255, 120,
      0, 0, 0, 200,
    ])

    const mask = rgbaToMaskPlane(rgba, 2, 2, 128)
    expect(Array.from(mask)).toEqual([
      MASK_VALUE_EMPTY,
      MASK_VALUE_FILLED,
      MASK_VALUE_EMPTY,
      MASK_VALUE_FILLED,
    ])
  })

  it('falls back to rgb luminance threshold when alpha is fully opaque', () => {
    const rgba = new Uint8ClampedArray([
      255, 255, 255, 255,
      32, 32, 32, 255,
    ])

    const mask = rgbaToMaskPlane(rgba, 2, 1, 100)
    expect(Array.from(mask)).toEqual([MASK_VALUE_FILLED, MASK_VALUE_EMPTY])
  })

  it('exports mask plane as opaque grayscale rgba', () => {
    const mask = new Uint8Array([
      MASK_VALUE_EMPTY,
      MASK_VALUE_FILLED,
      MASK_VALUE_FILLED,
      MASK_VALUE_EMPTY,
    ])

    const rgba = maskPlaneToRgba(mask, 2, 2)
    expect(Array.from(rgba)).toEqual([
      0, 0, 0, 255,
      255, 255, 255, 255,
      255, 255, 255, 255,
      0, 0, 0, 255,
    ])
  })
})
