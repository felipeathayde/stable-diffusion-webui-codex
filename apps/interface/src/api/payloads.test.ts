/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for image payload builders (txt2img).
Ensures engine-specific guidance fields (`cfg` vs `distilled_cfg`) and extras mapping (including Z-Image variant selection)
are emitted as expected.

Symbols (top-level; keep in sync; no ghosts):
- `payloads.test` (module): Image payload builder tests.
*/

import { describe, expect, it } from 'vitest'

import { buildTxt2ImgPayload } from './payloads'

describe('buildTxt2ImgPayload', () => {
  it('includes base and hires refiner when enabled', () => {
    const payload = buildTxt2ImgPayload({
      prompt: 'test prompt',
      negativePrompt: 'neg',
      width: 512,
      height: 512,
      steps: 20,
      guidanceScale: 7,
      sampler: 'euler',
      scheduler: 'karras',
      seed: 42,
      clipSkip: 0,
      batchSize: 1,
      batchCount: 1,
      device: 'cuda',
      hires: {
        enabled: true,
        denoise: 0.4,
        scale: 1.5,
        resizeX: 0,
        resizeY: 0,
        steps: 10,
        upscaler: 'latent:nearest',
        tile: { tile: 256, overlap: 16 },
        refiner: {
          enabled: true,
          steps: 4,
          cfg: 5,
          seed: -1,
          model: 'sdxl_refiner_1.0',
          vae: 'vae.safetensors',
        },
      },
      refiner: {
        enabled: true,
        steps: 6,
        cfg: 6.5,
        seed: 123,
        model: 'base-refiner',
        vae: undefined,
      },
    })

    expect(payload.extras?.refiner).toBeDefined()
    expect(payload.extras?.refiner).toMatchObject({ steps: 6, cfg: 6.5, seed: 123 })
    expect(payload.extras?.hires?.refiner).toBeDefined()
    expect(payload.extras?.hires?.refiner).toMatchObject({ steps: 4, cfg: 5 })
  })

  it('propagates hiresFallbackOnOom and hiresMinTile into hires tile config', () => {
    const payload = buildTxt2ImgPayload({
      prompt: 'test prompt',
      negativePrompt: 'neg',
      width: 512,
      height: 512,
      steps: 20,
      guidanceScale: 7,
      sampler: 'euler',
      scheduler: 'karras',
      seed: 42,
      clipSkip: 0,
      batchSize: 1,
      batchCount: 1,
      device: 'cuda',
      hires: {
        enabled: true,
        denoise: 0.4,
        scale: 1.5,
        resizeX: 0,
        resizeY: 0,
        steps: 0,
        upscaler: 'latent:nearest',
        tile: { tile: 256, overlap: 16 },
      },
    }, { hiresFallbackOnOom: false, hiresMinTile: 64 })

    expect((payload.extras as any)?.hires?.tile?.fallback_on_oom).toBe(false)
    expect((payload.extras as any)?.hires?.tile?.min_tile).toBe(64)
  })

  it('clamps hiresMinTile to hires tile size', () => {
    const payload = buildTxt2ImgPayload({
      prompt: 'test prompt',
      negativePrompt: 'neg',
      width: 512,
      height: 512,
      steps: 20,
      guidanceScale: 7,
      sampler: 'euler',
      scheduler: 'karras',
      seed: 42,
      clipSkip: 0,
      batchSize: 1,
      batchCount: 1,
      device: 'cuda',
      hires: {
        enabled: true,
        denoise: 0.4,
        scale: 1.5,
        resizeX: 0,
        resizeY: 0,
        steps: 0,
        upscaler: 'latent:nearest',
        tile: { tile: 256, overlap: 16 },
      },
    }, { hiresMinTile: 9999 })

    expect((payload.extras as any)?.hires?.tile?.min_tile).toBe(256)
  })

  it('uses cfg (not distilled_cfg) for zimage', () => {
    const payload = buildTxt2ImgPayload({
      prompt: 'test prompt',
      negativePrompt: 'neg',
      width: 1024,
      height: 1024,
      steps: 9,
      guidanceScale: 2.5,
      sampler: 'euler',
      scheduler: 'simple',
      seed: 42,
      clipSkip: 0,
      batchSize: 1,
      batchCount: 1,
      device: 'cuda',
      engine: 'zimage',
      model: 'dummy.safetensors',
      extras: { zimage_variant: 'turbo' },
    })

    expect(payload.cfg).toBe(2.5)
    expect((payload as any).distilled_cfg).toBeUndefined()
    expect(payload.negative_prompt).toBe('neg')
    expect(payload.extras?.zimage_variant).toBe('turbo')
  })
})
