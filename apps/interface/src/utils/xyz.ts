/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: XYZ sweep parsing and combination helpers.
Defines axis types and helpers to parse axis values, build X/Y/Z combinations, and format labels used by the XYZ plot store and view.

Symbols (top-level; keep in sync; no ghosts):
- `AxisParam` (type): Allowed axis parameter identifiers.
- `AxisValue` (type): Axis value union (`string | number`).
- `AxisConfig` (interface): Axis configuration (param + values array).
- `Combo` (interface): Single X/Y/Z combination entry with stable index.
- `XyzCell` (interface): UI cell state for a sweep combo (status + optional image/info/error).
- `AXIS_OPTIONS` (const): Supported axis option list with display labels and kinds.
- `parseAxisValues` (function): Parses multiline/comma-separated axis values as text or numbers.
- `buildCombos` (function): Builds the cartesian product of X/Y/Z values into `Combo` entries.
- `labelOf` (function): Formats an axis value for display.
*/

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
  | 'hires_scale'
  | 'hires_steps'
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
  { id: 'hires_scale', label: 'Hires scale', kind: 'number' },
  { id: 'hires_steps', label: 'Hires steps', kind: 'number' },
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
