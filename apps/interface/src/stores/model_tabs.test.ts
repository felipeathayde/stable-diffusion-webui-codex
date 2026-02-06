/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for model tab type normalization and required tab derivation.
Locks fail-loud behavior for unknown tab types and validates capability-driven Anima tab auto-inclusion.

Symbols (top-level; keep in sync; no ghosts):
- `model_tabs.test` (module): Model tabs normalization/required-types tests.
*/

import { describe, expect, it } from 'vitest'

import { normalizeTabType, requiredTypesFromCapabilities } from './model_tabs'

describe('normalizeTabType', () => {
  it('normalizes known aliases', () => {
    expect(normalizeTabType('wan22_5b')).toBe('wan')
    expect(normalizeTabType('flux1_chroma')).toBe('chroma')
    expect(normalizeTabType('anima')).toBe('anima')
  })

  it('fails loud on unknown type', () => {
    expect(() => normalizeTabType('mystery_engine')).toThrow(/Unsupported model tab type/)
  })
})

describe('requiredTypesFromCapabilities', () => {
  it('includes anima when capabilities expose anima', () => {
    const types = requiredTypesFromCapabilities({ anima: {}, sd15: {} })
    expect(types).toContain('anima')
  })

  it('does not include anima when capabilities omit anima', () => {
    const types = requiredTypesFromCapabilities({ sd15: {}, sdxl: {} })
    expect(types).not.toContain('anima')
  })
})
