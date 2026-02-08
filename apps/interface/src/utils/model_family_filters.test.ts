/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for shared model-family checkpoint filtering helpers.
Locks path-root precedence and fallback heuristics used by QuickSettings and Image swap-model selectors.

Symbols (top-level; keep in sync; no ghosts):
- `model_family_filters.test` (module): Regression coverage for model-family filtering helpers.
*/

import { describe, expect, it } from 'vitest'

import type { ModelInfo } from '../api/types'
import { enginePrefixForFamily, filterModelTitlesForFamily, modelMatchesFamily } from './model_family_filters'

function model(title: string, filename: string, metadata: Record<string, unknown> = {}): ModelInfo {
  return {
    title,
    name: title,
    model_name: title,
    hash: null,
    filename,
    metadata,
    core_only: false,
    core_only_reason: null,
    family_hint: null,
  }
}

describe('enginePrefixForFamily', () => {
  it('maps family aliases used by path keys', () => {
    expect(enginePrefixForFamily('wan')).toBe('wan22')
    expect(enginePrefixForFamily('chroma')).toBe('flux1')
    expect(enginePrefixForFamily('sd15')).toBe('sd15')
  })
})

describe('model family filtering', () => {
  it('uses configured ckpt roots first when available', () => {
    const paths = { flux1_ckpt: ['/models/flux'] }
    const hit = model('Flux Model', '/models/flux/dev.safetensors')
    const miss = model('Flux Other', '/models/other/dev.safetensors')
    expect(modelMatchesFamily(hit, 'flux1', paths)).toBe(true)
    expect(modelMatchesFamily(miss, 'flux1', paths)).toBe(false)
  })

  it('falls back to metadata/title heuristics when roots are absent', () => {
    const paths: Record<string, string[]> = {}
    const zByMetadata = model('Any', '/weights/custom.safetensors', { model_family: 'zimage' })
    const zByTitle = model('Z-Image Turbo', '/weights/custom.safetensors')
    expect(modelMatchesFamily(zByMetadata, 'zimage', paths)).toBe(true)
    expect(modelMatchesFamily(zByTitle, 'zimage', paths)).toBe(true)
  })

  it('returns filtered titles for one family', () => {
    const models: ModelInfo[] = [
      model('SDXL Base', '/models/sdxl/base.safetensors'),
      model('Flux Dev', '/models/flux/dev.safetensors'),
      model('Z-Image', '/models/zimage/base.safetensors'),
    ]
    const paths = { sdxl_ckpt: ['/models/sdxl'] }
    expect(filterModelTitlesForFamily(models, 'sdxl', paths)).toEqual(['SDXL Base'])
  })
})
