import { describe, expect, it } from 'vitest'

import { mapSamplerScheduler, parseInfotext } from './pnginfo'

describe('pnginfo utils', () => {
  it('parses common A1111 infotext', () => {
    const infotext = [
      'a cat sitting on a chair',
      'Negative prompt: blurry',
      'Steps: 20, Sampler: Euler a, Schedule type: Karras, CFG scale: 7, Seed: 123, Size: 512x768, Clip skip: 2',
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
})

