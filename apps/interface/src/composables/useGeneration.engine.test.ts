/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unit tests for generation request helpers (engine mapping, img2img payload contract, revision-conflict parsing).
Locks alias parity so UI disable-state and request engine selection remain consistent, and verifies stale-revision conflict helper behavior
plus img2img payload invariants (no hires-prefixed keys) and advanced-guidance payload wiring.

Symbols (top-level; keep in sync; no ghosts):
- `useGeneration.engine.test` (module): resolveEngineForRequest mapping + img2img payload contract + advanced-guidance payload + revision-conflict helper tests.
*/

import { describe, expect, it } from 'vitest'

import type { ImageBaseParams } from '../stores/model_tabs'
import { buildGuidancePayload, buildImg2ImgPayload, resolveEngineForRequest } from './useGeneration'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'

function makeImageParams(overrides: Partial<ImageBaseParams> = {}): ImageBaseParams {
  return {
    prompt: 'portrait of a woman',
    negativePrompt: 'low quality',
    width: 1024,
    height: 896,
    sampler: 'dpm++ 2m',
    scheduler: 'karras',
    steps: 32,
    cfgScale: 7,
    seed: 123456,
    clipSkip: 0,
    batchSize: 1,
    batchCount: 1,
    img2imgResizeMode: 'just_resize',
    img2imgUpscaler: 'latent:bicubic-aa',
    guidanceAdvanced: {
      enabled: false,
      apgEnabled: false,
      apgStartStep: 0,
      apgEta: 0,
      apgMomentum: 0,
      apgNormThreshold: 15,
      apgRescale: 0,
      guidanceRescale: 0,
      cfgTruncEnabled: false,
      cfgTruncRatio: 0.8,
      renormCfg: 0,
    },
    hires: {
      enabled: true,
      denoise: 0.35,
      scale: 2,
      resizeX: 0,
      resizeY: 0,
      steps: 0,
      upscaler: 'latent:bicubic-aa',
      tile: { tile: 1024, overlap: 64 },
      sampler: 'euler',
      scheduler: 'simple',
      prompt: 'override prompt',
      negativePrompt: 'override negative',
      cfg: 6,
      distilledCfg: undefined,
    },
    refiner: {
      enabled: false,
      swapAtStep: 28,
      cfg: 7,
      seed: -1,
    },
    checkpoint: 'model.safetensors',
    textEncoders: [],
    useInitImage: true,
    initImageData: 'data:image/png;base64,AAAA',
    initImageName: 'init.png',
    denoiseStrength: 0.5,
    useMask: false,
    maskImageData: '',
    maskImageName: '',
    maskEnforcement: 'post_blend',
    inpaintFullRes: true,
    inpaintFullResPadding: 32,
    inpaintingFill: 1,
    maskInvert: false,
    maskBlur: 4,
    maskRound: false,
    zimageTurbo: true,
    ...overrides,
  }
}

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

describe('buildImg2ImgPayload', () => {
  it('builds payload without any img2img_hires_* keys', () => {
    const params = makeImageParams({
      useMask: true,
      maskImageData: 'data:image/png;base64,BBBB',
      maskImageName: 'mask.png',
    })

    const payload = buildImg2ImgPayload({
      params,
      supportsNegativePrompt: true,
      isDistilledCfgModel: false,
      batchCount: 2,
      batchSize: 1,
      device: 'cuda',
      settingsRevision: 17,
      engineId: 'sdxl',
      modelOverride: 'sha256:abc',
      extras: { lora_sha: 'sha256:lora' },
    })

    expect(Object.keys(payload).some((key) => key.startsWith('img2img_hires_'))).toBe(false)
    expect(payload.img2img_hires_enable).toBeUndefined()
    expect(payload.img2img_init_image).toBe(params.initImageData)
    expect(payload.img2img_extras).toEqual({ lora_sha: 'sha256:lora' })
    expect(payload.img2img_mask).toBe(params.maskImageData)
  })

  it('routes distilled cfg and negative prompt according to engine support', () => {
    const params = makeImageParams({ cfgScale: 5.5, negativePrompt: 'should be dropped' })

    const payload = buildImg2ImgPayload({
      params,
      supportsNegativePrompt: false,
      isDistilledCfgModel: true,
      batchCount: 1,
      batchSize: 1,
      device: 'cuda',
      settingsRevision: 3,
      engineId: 'flux1_kontext',
      modelOverride: 'sha256:def',
      extras: {},
    })

    expect(payload.img2img_cfg_scale).toBe(1)
    expect(payload.img2img_distilled_cfg_scale).toBe(5.5)
    expect(payload.img2img_neg_prompt).toBe('')
  })
})

describe('buildGuidancePayload', () => {
  it('returns null when advanced guidance is disabled', () => {
    const payload = buildGuidancePayload(
      {
        enabled: false,
        apgEnabled: true,
        apgStartStep: 2,
        apgEta: 0.15,
        apgMomentum: 0.2,
        apgNormThreshold: 12,
        apgRescale: 0.3,
        guidanceRescale: 0.4,
        cfgTruncEnabled: true,
        cfgTruncRatio: 0.7,
        renormCfg: 1.1,
      },
      {
        apg_enabled: true,
        apg_start_step: true,
        apg_eta: true,
        apg_momentum: true,
        apg_norm_threshold: true,
        apg_rescale: true,
        guidance_rescale: true,
        cfg_trunc_ratio: true,
        renorm_cfg: true,
      },
    )
    expect(payload).toBeNull()
  })

  it('keeps only supported controls and clamps values', () => {
    const payload = buildGuidancePayload(
      {
        enabled: true,
        apgEnabled: true,
        apgStartStep: -4,
        apgEta: 0.2,
        apgMomentum: 2.5,
        apgNormThreshold: -3,
        apgRescale: 4,
        guidanceRescale: -1,
        cfgTruncEnabled: false,
        cfgTruncRatio: 1.4,
        renormCfg: -5,
      },
      {
        apg_enabled: true,
        apg_start_step: true,
        apg_eta: false,
        apg_momentum: true,
        apg_norm_threshold: true,
        apg_rescale: false,
        guidance_rescale: true,
        cfg_trunc_ratio: true,
        renorm_cfg: true,
      },
    )

    expect(payload).toEqual({
      apg_enabled: true,
      apg_start_step: 0,
      apg_momentum: 0.999999,
      apg_norm_threshold: 0,
      guidance_rescale: 0,
      renorm_cfg: 0,
    })
    expect(payload?.cfg_trunc_ratio).toBeUndefined()
    expect(payload?.apg_eta).toBeUndefined()
    expect(payload?.apg_rescale).toBeUndefined()
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
