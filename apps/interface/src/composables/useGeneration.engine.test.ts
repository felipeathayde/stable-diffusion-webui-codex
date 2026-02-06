/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unit tests for canonical engine mapping used by generation preflight and request dispatch.
Locks alias parity so UI disable-state and request engine selection remain consistent.

Symbols (top-level; keep in sync; no ghosts):
- `useGeneration.engine.test` (module): resolveEngineForRequest mapping tests.
*/

import { describe, expect, it } from 'vitest'

import { resolveEngineForRequest } from './useGeneration'

describe('resolveEngineForRequest', () => {
  it('maps wan and chroma aliases', () => {
    expect(resolveEngineForRequest('wan', false)).toBe('wan22')
    expect(resolveEngineForRequest('chroma', false)).toBe('flux1_chroma')
  })

  it('routes flux1 img2img through kontext', () => {
    expect(resolveEngineForRequest('flux1', false)).toBe('flux1')
    expect(resolveEngineForRequest('flux1', true)).toBe('flux1_kontext')
  })

  it('keeps anima unchanged', () => {
    expect(resolveEngineForRequest('anima', false)).toBe('anima')
    expect(resolveEngineForRequest('anima', true)).toBe('anima')
  })
})
