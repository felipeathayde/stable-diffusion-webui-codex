import { describe, expect, it } from 'vitest'

import { parseAxisValues, buildCombos, labelOf } from './xyz'

describe('xyz utils', () => {
  it('parses text values from comma/newline input', () => {
    const raw = 'sampler-a, sampler-b\n  sampler-c '
    const values = parseAxisValues(raw, 'text')
    expect(values).toEqual(['sampler-a', 'sampler-b', 'sampler-c'])
  })

  it('parses numeric values and drops invalids', () => {
    const raw = '7, 12.5, nope, 3'
    const values = parseAxisValues(raw, 'number')
    expect(values).toEqual([7, 12.5, 3])
  })

  it('builds combos across xyz axes', () => {
    const combos = buildCombos(['a', 'b'], [1], [])
    expect(combos).toHaveLength(2)
    expect(combos[0]).toMatchObject({ x: 'a', y: 1, z: null })
    expect(combos[1]).toMatchObject({ x: 'b', y: 1, z: null })
  })

  it('formats labels for numbers and empties', () => {
    expect(labelOf(4)).toBe('4')
    expect(labelOf(1.2345)).toBe('1.23')
    expect(labelOf(null)).toBe('—')
  })
})
