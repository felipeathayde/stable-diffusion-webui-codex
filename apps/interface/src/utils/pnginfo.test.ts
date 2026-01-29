import { describe, expect, it } from 'vitest'

import { mapCheckpointTitle, mapSamplerScheduler, parseComfyPromptJson, parseInfotext } from './pnginfo'

describe('pnginfo utils', () => {
  it('parses common A1111 infotext', () => {
    const infotext = [
      'a cat sitting on a chair',
      'Negative prompt: blurry',
      'Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 123, Size: 512x768, Clip skip: 2, Model: sdxl.safetensors, Model hash: abcdef1234, VAE: vae-ft-mse',
    ].join('\n')

    const { parsed, warnings } = parseInfotext(infotext)
    expect(warnings).toEqual([])
    expect(parsed.prompt).toBe('a cat sitting on a chair')
    expect(parsed.negativePrompt).toBe('blurry')
    expect(parsed.hasNegativePrompt).toBe(true)
    expect(parsed.steps).toBe(20)
    expect(parsed.sampler).toBe('Euler a')
    expect(parsed.scheduler).toBe('Karras')
    expect(parsed.cfgScale).toBe(7)
    expect(parsed.seed).toBe(123)
    expect(parsed.width).toBe(512)
    expect(parsed.height).toBe(768)
    expect(parsed.clipSkip).toBe(2)
    expect(parsed.model).toBe('sdxl.safetensors')
    expect(parsed.modelHash).toBe('abcdef1234')
    expect(parsed.vae).toBe('vae-ft-mse')
  })

  it("treats infotext without 'Steps:' as prompt-only", () => {
    const { parsed, warnings } = parseInfotext('just a prompt')
    expect(parsed.prompt).toBe('just a prompt')
    expect(parsed.rawKv).toEqual({})
    expect(warnings.join('\n')).toMatch(/Steps:/i)
  })

  it('maps sampler/scheduler names case-insensitively', () => {
    const res = mapSamplerScheduler(
      'Euler a',
      'Karras',
      [{ name: 'euler a', allowed_schedulers: ['karras', 'simple'] }],
      [{ name: 'karras' }, { name: 'simple' }],
    )
    expect(res.warnings).toEqual([])
    expect(res.sampler).toBe('euler a')
    expect(res.scheduler).toBe('karras')
  })

  it('infers scheduler suffix from sampler label', () => {
    const res = mapSamplerScheduler(
      'DPM++ 2M Karras',
      undefined,
      [{ name: 'dpm++ 2m', allowed_schedulers: ['karras'] }],
      [{ name: 'karras' }],
    )
    expect(res.warnings).toEqual([])
    expect(res.sampler).toBe('dpm++ 2m')
    expect(res.scheduler).toBe('karras')
  })

  it('maps ComfyUI sampler aliases', () => {
    const res = mapSamplerScheduler(
      'dpmpp_2m',
      'normal',
      [{ name: 'dpm++ 2m', allowed_schedulers: ['simple'] }],
      [{ name: 'simple' }, { name: 'karras' }],
    )
    expect(res.warnings).toEqual([])
    expect(res.sampler).toBe('dpm++ 2m')
    expect(res.scheduler).toBe('simple')
  })

  it('maps euler_ancestral to euler a', () => {
    const res = mapSamplerScheduler(
      'euler_ancestral',
      'karras',
      [{ name: 'euler a', allowed_schedulers: ['karras'] }],
      [{ name: 'karras' }],
    )
    expect(res.warnings).toEqual([])
    expect(res.sampler).toBe('euler a')
    expect(res.scheduler).toBe('karras')
  })

  it('rejects unsupported sampler/scheduler pair', () => {
    const res = mapSamplerScheduler(
      'Euler a',
      'Karras',
      [{ name: 'euler a', allowed_schedulers: ['simple'] }],
      [{ name: 'karras' }, { name: 'simple' }],
    )
    expect(res.sampler).toBeUndefined()
    expect(res.scheduler).toBeUndefined()
    expect(res.warnings.join('\n')).toMatch(/not supported/i)
  })

  it('maps checkpoint title by model hash when unambiguous', () => {
    const res = mapCheckpointTitle(
      { model: undefined, modelHash: 'abcdef1234' },
      [
        { title: 'SDXL 1.0', hash: 'abcdef1234', filename: 'sdxl.safetensors', model_name: 'sdxl', name: 'sdxl' },
        { title: 'Other', hash: '1234567890', filename: 'other.safetensors', model_name: 'other', name: 'other' },
      ],
    )
    expect(res.warnings).toEqual([])
    expect(res.checkpoint).toBe('SDXL 1.0')
  })

  it('extracts key fields from ComfyUI prompt JSON with a single KSampler', () => {
    const prompt = {
      '3': {
        class_type: 'KSampler',
        inputs: {
          seed: 123,
          steps: 20,
          cfg: 7,
          sampler_name: 'euler_ancestral',
          scheduler: 'karras',
          denoise: 0.75,
          latent_image: ['5', 0],
          positive: ['4', 0],
          negative: ['6', 0],
        },
      },
      '4': { class_type: 'CLIPTextEncode', inputs: { text: 'a cat' } },
      '6': { class_type: 'CLIPTextEncode', inputs: { text: 'blurry' } },
      '5': { class_type: 'EmptyLatentImage', inputs: { width: 512, height: 768 } },
    }

    const res = parseComfyPromptJson(JSON.stringify(prompt))
    expect(res.graph).not.toBeNull()
    expect(res.warnings).toEqual([])
    expect(res.extracted.prompt).toBe('a cat')
    expect(res.extracted.negativePrompt).toBe('blurry')
    expect(res.extracted.hasNegativePrompt).toBe(true)
    expect(res.extracted.steps).toBe(20)
    expect(res.extracted.seed).toBe(123)
    expect(res.extracted.cfgScale).toBe(7)
    expect(res.extracted.sampler).toBe('euler_ancestral')
    expect(res.extracted.scheduler).toBe('karras')
    expect(res.extracted.denoiseStrength).toBe(0.75)
    expect(res.extracted.width).toBe(512)
    expect(res.extracted.height).toBe(768)
  })
})
