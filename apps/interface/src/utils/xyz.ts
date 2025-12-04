// tags: xyz, parsing, sweeps
import type { GeneratedImage } from '../api/types'

export type AxisParam =
  | 'prompt'
  | 'negative'
  | 'cfg'
  | 'steps'
  | 'sampler'
  | 'scheduler'
  | 'seed'
  | 'width'
  | 'height'
  | 'highres_scale'
  | 'highres_steps'
  | 'refiner_model'
  | 'refiner_steps'
  | 'refiner_cfg'

export type AxisValue = string | number

export interface AxisConfig {
  param: AxisParam
  values: AxisValue[]
}

export interface Combo {
  x: AxisValue
  y: AxisValue | null
  z: AxisValue | null
  index: number
}

export interface XyzCell {
  x: AxisValue
  y: AxisValue | null
  z: AxisValue | null
  status: 'queued' | 'running' | 'done' | 'error' | 'stopped'
  image?: GeneratedImage
  info?: unknown
  error?: string
}

export const AXIS_OPTIONS: { id: AxisParam; label: string; kind: 'text' | 'number' }[] = [
  { id: 'prompt', label: 'Prompt', kind: 'text' },
  { id: 'negative', label: 'Negative prompt', kind: 'text' },
  { id: 'cfg', label: 'CFG scale', kind: 'number' },
  { id: 'steps', label: 'Steps', kind: 'number' },
  { id: 'sampler', label: 'Sampler', kind: 'text' },
  { id: 'scheduler', label: 'Scheduler', kind: 'text' },
  { id: 'seed', label: 'Seed', kind: 'number' },
  { id: 'width', label: 'Width', kind: 'number' },
  { id: 'height', label: 'Height', kind: 'number' },
  { id: 'highres_scale', label: 'Highres scale', kind: 'number' },
  { id: 'highres_steps', label: 'Highres steps', kind: 'number' },
  { id: 'refiner_model', label: 'Refiner model', kind: 'text' },
  { id: 'refiner_steps', label: 'Refiner steps', kind: 'number' },
  { id: 'refiner_cfg', label: 'Refiner CFG', kind: 'number' },
]

export function parseAxisValues(raw: string, kind: 'text' | 'number'): AxisValue[] {
  if (!raw.trim()) return []
  const tokens = raw
    .split(/\r?\n|,/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0)

  if (kind === 'text') return tokens

  const numbers: number[] = []
  for (const token of tokens) {
    const num = Number(token)
    if (Number.isFinite(num)) numbers.push(num)
  }
  return numbers
}

export function buildCombos(x: AxisValue[], y: AxisValue[], z: AxisValue[]): Combo[] {
  const xs = x.length ? x : ['(base)']
  const ys = y.length ? y : [null]
  const zs = z.length ? z : [null]
  const combos: Combo[] = []
  let idx = 0
  for (const zVal of zs) {
    for (const yVal of ys) {
      for (const xVal of xs) {
        combos.push({ x: xVal, y: yVal, z: zVal, index: idx++ })
      }
    }
  }
  return combos
}

export function labelOf(value: AxisValue | null): string {
  if (value === null) return '—'
  if (typeof value === 'number') return Number.isInteger(value) ? `${value}` : value.toFixed(2)
  return value || '—'
}
