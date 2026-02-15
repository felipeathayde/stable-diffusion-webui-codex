/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unit tests for generation request helpers (engine mapping, payload sanitization, revision-conflict parsing).
Locks alias parity so UI disable-state and request engine selection remain consistent, and verifies stale-revision conflict helper behavior.

Symbols (top-level; keep in sync; no ghosts):
- `useGeneration.engine.test` (module): resolveEngineForRequest mapping + img2img payload sanitization + revision-conflict helper tests.
*/

import { describe, expect, it } from 'vitest'

import { resolveEngineForRequest, sanitizeImg2ImgPayload } from './useGeneration'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'

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

describe('sanitizeImg2ImgPayload', () => {
  it('removes all img2img hires keys', () => {
    const payload = sanitizeImg2ImgPayload({
      img2img_prompt: 'base',
      img2img_hires_enable: true,
      img2img_hires_scale: 2,
      img2img_hires_resize_x: 0,
      img2img_hires_resize_y: 0,
      img2img_hires_steps: 0,
      img2img_hires_denoise: 0.4,
      img2img_hires_upscaler: 'latent:bicubic-aa',
      img2img_hires_sampling: 'euler',
      img2img_hires_scheduler: 'simple',
      img2img_hires_prompt: 'override',
      img2img_hires_neg_prompt: '',
      img2img_hires_cfg: 7,
      img2img_hires_distilled_cfg: undefined,
      img2img_hires_tile: { tile: 256, overlap: 16 },
    })

    expect(payload.img2img_prompt).toBe('base')
    expect(payload.img2img_hires_enable).toBeUndefined()
    expect(payload.img2img_hires_scale).toBeUndefined()
    expect(payload.img2img_hires_resize_x).toBeUndefined()
    expect(payload.img2img_hires_resize_y).toBeUndefined()
    expect(payload.img2img_hires_steps).toBeUndefined()
    expect(payload.img2img_hires_denoise).toBeUndefined()
    expect(payload.img2img_hires_upscaler).toBeUndefined()
    expect(payload.img2img_hires_sampling).toBeUndefined()
    expect(payload.img2img_hires_scheduler).toBeUndefined()
    expect(payload.img2img_hires_prompt).toBeUndefined()
    expect(payload.img2img_hires_neg_prompt).toBeUndefined()
    expect(payload.img2img_hires_cfg).toBeUndefined()
    expect(payload.img2img_hires_distilled_cfg).toBeUndefined()
    expect(payload.img2img_hires_tile).toBeUndefined()
  })
})

describe('settings revision conflict helpers', () => {
  it('extracts current_revision from 409 conflict payloads', () => {
    const error = Object.assign(new Error('conflict'), {
      status: 409,
      body: {
        detail: {
          current_revision: 19,
        },
      },
    })

    expect(resolveSettingsRevisionConflict(error)).toBe(19)
  })

  it('ignores non-conflict errors', () => {
    const error = Object.assign(new Error('bad request'), {
      status: 400,
      body: { detail: { current_revision: 3 } },
    })
    expect(resolveSettingsRevisionConflict(error)).toBeNull()
  })

  it('formats an actionable retry message', () => {
    expect(formatSettingsRevisionConflictMessage(23)).toContain('retry')
    expect(formatSettingsRevisionConflictMessage(23)).toContain('23')
  })
})
