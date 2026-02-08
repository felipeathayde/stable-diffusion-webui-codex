/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Regression tests for image parameter normalization helpers used by img2img/inpaint controls.

Symbols (top-level; keep in sync; no ghosts):
- `describe('image_params normalizers')` (test suite): Validates mask enforcement, fill-mode clamp, and non-negative int normalization semantics.
*/

import { describe, expect, it } from 'vitest'
import { normalizeInpaintingFill, normalizeMaskEnforcement, normalizeNonNegativeInt } from './image_params'

describe('image_params normalizers', () => {
  it('normalizes mask enforcement into strict enum', () => {
    expect(normalizeMaskEnforcement('per_step_clamp')).toBe('per_step_clamp')
    expect(normalizeMaskEnforcement('post_blend')).toBe('post_blend')
    expect(normalizeMaskEnforcement('anything-else')).toBe('post_blend')
  })

  it('clamps and truncates inpainting fill mode to [0, 3]', () => {
    expect(normalizeInpaintingFill(-9)).toBe(0)
    expect(normalizeInpaintingFill(1.8)).toBe(1)
    expect(normalizeInpaintingFill(99)).toBe(3)
  })

  it('keeps NaN behavior explicit for inpainting fill', () => {
    expect(Number.isNaN(normalizeInpaintingFill(Number.NaN))).toBe(true)
  })

  it('normalizes non-negative integer fields', () => {
    expect(normalizeNonNegativeInt(-2.7)).toBe(0)
    expect(normalizeNonNegativeInt(7.9)).toBe(7)
  })

  it('keeps NaN behavior explicit for non-negative int', () => {
    expect(Number.isNaN(normalizeNonNegativeInt(Number.NaN))).toBe(true)
  })
})
