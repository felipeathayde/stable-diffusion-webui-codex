/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for canonical frontend engine taxonomy fallback defaults.
Locks Anima fallback sampler/scheduler defaults to backend-supported values.

Symbols (top-level; keep in sync; no ghosts):
- `engine_taxonomy.test` (module): Frontend taxonomy fallback default regressions.
*/

import { describe, expect, it } from 'vitest'

import { fallbackSamplingDefaultsForTabFamily } from './engine_taxonomy'

describe('engine taxonomy fallback sampling defaults', () => {
  it('uses native-supported defaults for anima', () => {
    const defaults = fallbackSamplingDefaultsForTabFamily('anima')
    expect(defaults).toEqual({ sampler: 'euler', scheduler: 'simple' })
    expect(defaults.sampler).not.toBe('er sde')
  })
})
