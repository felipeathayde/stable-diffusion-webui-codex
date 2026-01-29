/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: PNG Info infotext parsing + send-to mapping helpers.
Parses common A1111/Forge-style `parameters` infotext into structured fields and maps sampler/scheduler strings into Codex canonical names.

Symbols (top-level; keep in sync; no ghosts):
- `ParsedInfotext` (interface): Structured subset of parsed infotext fields.
- `SamplerLike` (interface): Minimal sampler entry used for name + allowlist mapping.
- `SchedulerLike` (interface): Minimal scheduler entry used for name matching.
- `parseInfotext` (function): Parses infotext into structured fields + raw kv map.
- `mapSamplerScheduler` (function): Maps raw sampler/scheduler strings to canonical names with allowlist validation.
*/

export interface ParsedInfotext {
  prompt: string
  negativePrompt: string
  hasNegativePrompt: boolean
  steps?: number
  sampler?: string
  scheduler?: string
  cfgScale?: number
  seed?: number
  width?: number
  height?: number
  clipSkip?: number
  denoiseStrength?: number
  rawKv: Record<string, string>
}

export interface SamplerLike {
  name: string
  allowed_schedulers?: string[]
}

export interface SchedulerLike {
  name: string
}

function normalizeComparable(value: string): string {
  return String(value || '')
    .trim()
    .toLowerCase()
    .replace(/[_]+/g, ' ')
    .replace(/\s+/g, ' ')
}

function parseIntStrict(raw: string): number | null {
  const text = String(raw || '').trim()
  if (!text) return null
  if (!/^-?\d+$/.test(text)) return null
  const n = Number(text)
  return Number.isFinite(n) ? n : null
}

function parseFloatStrict(raw: string): number | null {
  const text = String(raw || '').trim()
  if (!text) return null
  if (!/^-?\d+(\.\d+)?$/.test(text)) return null
  const n = Number(text)
  return Number.isFinite(n) ? n : null
}

function parseSize(raw: string): { width: number; height: number } | null {
  const text = String(raw || '').trim()
  if (!text) return null
  const m = text.match(/^(\d+)\s*[x×]\s*(\d+)$/i)
  if (!m) return null
  const w = parseIntStrict(m[1])
  const h = parseIntStrict(m[2])
  if (w === null || h === null) return null
  if (w <= 0 || h <= 0) return null
  return { width: w, height: h }
}

export function parseInfotext(infotext: string): { parsed: ParsedInfotext; warnings: string[] } {
  const warnings: string[] = []
  const raw = String(infotext || '').trim()
  if (!raw) {
    return {
      parsed: { prompt: '', negativePrompt: '', hasNegativePrompt: false, rawKv: {} },
      warnings: [],
    }
  }

  const idxSteps = raw.toLowerCase().indexOf('steps:')
  if (idxSteps === -1) {
    warnings.push("Infotext: couldn't find 'Steps:' block; treating content as prompt only.")
    return {
      parsed: { prompt: raw, negativePrompt: '', hasNegativePrompt: false, rawKv: {} },
      warnings,
    }
  }

  const head = raw.slice(0, idxSteps).trimEnd()
  const kvBlock = raw.slice(idxSteps).trim()

  const negMarker = 'negative prompt:'
  const negIdx = head.toLowerCase().indexOf(negMarker)
  const hasNegativePrompt = negIdx !== -1
  const prompt = (hasNegativePrompt ? head.slice(0, negIdx) : head).trimEnd()
  const negativePrompt = hasNegativePrompt ? head.slice(negIdx + negMarker.length).trim() : ''

  const rawKv: Record<string, string> = {}
  const parts = kvBlock.split(',').map(p => p.trim()).filter(Boolean)
  for (const part of parts) {
    const idx = part.indexOf(':')
    if (idx === -1) continue
    const key = part.slice(0, idx).trim()
    const value = part.slice(idx + 1).trim()
    if (!key) continue
    rawKv[key] = value
  }

  const parsed: ParsedInfotext = {
    prompt,
    negativePrompt,
    hasNegativePrompt,
    rawKv,
  }

  const get = (key: string): string | undefined => {
    const exact = rawKv[key]
    if (exact !== undefined) return exact
    const low = key.toLowerCase()
    for (const [k, v] of Object.entries(rawKv)) {
      if (k.toLowerCase() === low) return v
    }
    return undefined
  }

  const steps = get('Steps')
  if (steps !== undefined) {
    const n = parseIntStrict(steps)
    if (n === null || n < 0) warnings.push(`Invalid Steps value: ${steps}`)
    else parsed.steps = n
  }

  const sampler = get('Sampler')
  if (sampler !== undefined && sampler.trim()) parsed.sampler = sampler.trim()

  const scheduler = get('Schedule type') ?? get('Scheduler')
  if (scheduler !== undefined && scheduler.trim()) parsed.scheduler = scheduler.trim()

  const cfg = get('CFG scale') ?? get('CFG')
  if (cfg !== undefined) {
    const n = parseFloatStrict(cfg)
    if (n === null) warnings.push(`Invalid CFG scale value: ${cfg}`)
    else parsed.cfgScale = n
  }

  const seed = get('Seed')
  if (seed !== undefined) {
    const n = parseIntStrict(seed)
    if (n === null) warnings.push(`Invalid Seed value: ${seed}`)
    else parsed.seed = n
  }

  const size = get('Size')
  if (size !== undefined) {
    const dims = parseSize(size)
    if (!dims) warnings.push(`Invalid Size value: ${size}`)
    else {
      parsed.width = dims.width
      parsed.height = dims.height
    }
  }

  const clipSkip = get('Clip skip')
  if (clipSkip !== undefined) {
    const n = parseIntStrict(clipSkip)
    if (n === null || n < 0) warnings.push(`Invalid Clip skip value: ${clipSkip}`)
    else parsed.clipSkip = n
  }

  const denoise = get('Denoising strength') ?? get('Denoising Strength')
  if (denoise !== undefined) {
    const n = parseFloatStrict(denoise)
    if (n === null || n < 0 || n > 1) warnings.push(`Invalid Denoising strength value: ${denoise}`)
    else parsed.denoiseStrength = n
  }

  return { parsed, warnings }
}

export function mapSamplerScheduler(
  rawSampler: string | undefined,
  rawScheduler: string | undefined,
  samplers: SamplerLike[],
  schedulers: SchedulerLike[],
): { sampler?: string; scheduler?: string; warnings: string[] } {
  const warnings: string[] = []

  const samplerMap = new Map<string, string>()
  for (const s of samplers) {
    if (!s?.name) continue
    samplerMap.set(normalizeComparable(s.name), s.name)
  }

  const schedulerMap = new Map<string, string>()
  const schedulerLabels: string[] = []
  for (const sch of schedulers) {
    if (!sch?.name) continue
    const canonical = sch.name
    const key = normalizeComparable(canonical)
    schedulerMap.set(key, canonical)
    const label = normalizeComparable(canonical.replace(/_/g, ' '))
    schedulerMap.set(label, canonical)
    schedulerLabels.push(label)
  }
  schedulerLabels.sort((a, b) => b.length - a.length)

  const resolveSampler = (value: string): string | null => {
    const key = normalizeComparable(value)
    const found = samplerMap.get(key)
    return found ?? null
  }

  const resolveScheduler = (value: string): string | null => {
    const key = normalizeComparable(value)
    const found = schedulerMap.get(key)
    return found ?? null
  }

  let sampler = rawSampler ? resolveSampler(rawSampler) : null
  let scheduler = rawScheduler ? resolveScheduler(rawScheduler) : null

  // Heuristic: when schedule type isn't present, it may be appended to sampler name
  // e.g. "DPM++ 2M Karras" → sampler="dpm++ 2m", scheduler="karras".
  if (scheduler === null && rawScheduler && normalizeComparable(rawScheduler) === 'automatic') {
    warnings.push("Scheduler is 'Automatic' (not reproducible); leaving sampler/scheduler unchanged.")
  }

  if (scheduler === null && rawScheduler && normalizeComparable(rawScheduler) !== 'automatic') {
    warnings.push(`Scheduler '${rawScheduler}' not recognized; leaving sampler/scheduler unchanged.`)
  }

  if ((sampler === null || scheduler === null) && rawSampler) {
    const rawKey = normalizeComparable(rawSampler)
    for (const label of schedulerLabels) {
      const suffix = ` ${label}`
      if (!rawKey.endsWith(suffix)) continue
      const samplerPart = rawKey.slice(0, -suffix.length).trim()
      const inferredSampler = samplerMap.get(samplerPart) ?? null
      const inferredScheduler = schedulerMap.get(label) ?? null
      if (inferredSampler && inferredScheduler) {
        sampler = inferredSampler
        scheduler = inferredScheduler
      }
      break
    }
  }

  if (sampler === null && rawSampler) {
    warnings.push(`Sampler '${rawSampler}' not recognized; leaving sampler/scheduler unchanged.`)
  }

  if (!sampler || !scheduler) {
    return { warnings }
  }

  const samplerEntry = samplers.find(s => s.name === sampler) ?? null
  const allowed = samplerEntry?.allowed_schedulers ?? []
  if (Array.isArray(allowed) && allowed.length > 0 && !allowed.includes(scheduler)) {
    warnings.push(`Sampler/scheduler pair '${sampler}' / '${scheduler}' is not supported; leaving unchanged.`)
    return { warnings }
  }

  return { sampler, scheduler, warnings }
}

