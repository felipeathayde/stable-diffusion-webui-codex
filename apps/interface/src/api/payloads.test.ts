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
      batchSize: 1,
      batchCount: 1,
      device: 'cuda',
      highres: {
        enabled: true,
        denoise: 0.4,
        scale: 1.5,
        resizeX: 0,
        resizeY: 0,
        steps: 10,
        upscaler: 'Latent (nearest)',
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
    expect(payload.extras?.highres?.refiner).toBeDefined()
    expect(payload.extras?.highres?.refiner).toMatchObject({ steps: 4, cfg: 5 })
  })
})

