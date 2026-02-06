/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model Tabs store (tab definitions + per-tab params + ordering) for the WebUI.
Owns the list of engine tabs, persists tab CRUD/reorder via `/api/ui/tabs`, normalizes/validates tab payloads from the backend, and provides
default parameter shapes per tab type (image vs WAN video) using engine defaults and form-state schemas. Hires upscaler values are stable ids
(`latent:*` / `spandrel:*`) for hires-fix wiring.

Symbols (top-level; keep in sync; no ghosts):
- `BaseTabType` (type): API tab type discriminator (from backend `ApiTab['type']`).
- `ImageTabType` (type): Image-only tab type discriminator (`BaseTabType` without `wan`).
- `BaseTabMeta` (interface): Tab metadata timestamps (created/updated) tracked client-side.
- `ModelTabsErrorCode` (type): Error code taxonomy for model-tabs store failures.
- `ModelTabsStoreError` (class): Typed store error thrown for tab lookup/API/contract/reorder failures.
- `WanStageParams` (interface): UI WAN stage params (high/low) used by video tabs and payload builders.
- `WanVideoParams` (interface): UI WAN video params (prompt/dims/fps/frames + optional init media + overrides).
- `WanAssetsParams` (interface): WAN asset selectors (metadata/text encoder/VAE) used by WAN requests.
- `BaseTab` (interface): Generic tab record persisted in the store (id/type/label + params + meta).
- `ImageBaseParams` (interface): Common image-tab params (prompt, seed, steps, CFG, dims, etc.) shared across SD/Flux.1/Chroma/ZImage
  (includes optional family-specific fields like `zimageTurbo`).
- `TabParamsByType` (type): Canonical params map by tab type.
- `TabByType` (type): Typed tab shape (`type` + matching params payload).
- `MODEL_TABS_STORAGE_KEY` (const): LocalStorage key used for persisted tabs state (bump when schema changes).
- `nowIso` (function): Returns current time in ISO string form for metadata timestamps.
- `defaultParams` (function): Returns default params for a given tab type (image vs WAN video), merging engine defaults where applicable.
- `defaultImageParamsForType` (function): Returns canonical image-tab defaults for a specific image tab type.
- `normalizeTabType` (function): Validates/coerces raw type values into `BaseTabType`.
- `BASE_REQUIRED_TYPES` (const): Baseline tab types always auto-created by the UI store.
- `requiredTypesFromCapabilities` (function): Derives required tab types from backend capability map (adds `anima` only when exposed).
- `asRecordObject` (function): Narrowing helper that normalizes unknown values into plain records for merge-safe processing.
- `asParamsRecord` (function): Explicit boundary cast helper from typed tab params to persisted `Record<string, unknown>`.
- `normalizeWanParams` (function): Applies WAN-specific nested merge normalization for `high/low/video/assets` params.
- `normalizeImageParams` (function): Applies image-tab nested merge normalization (`hires/refiner`) with sampler/scheduler fallback.
- `normalizeParamsForType` (function): Normalizes raw params payload based on tab type (shape checking; discards invalid fields).
- `normalizeTab` (function): Normalizes a raw tab record (id/type/params/meta) into the store shape.
- `useModelTabsStore` (store): Pinia store for tabs; loads/syncs with backend, provides CRUD/reorder actions, and exposes computed helpers.
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { fetchTabs, createTabApi, updateTabApi, reorderTabsApi, deleteTabApi } from '../api/client'
import type { ApiTab } from '../api/types'
import type { HiresFormState, RefinerFormState } from '../api/payloads'
import { type EngineType, getEngineConfig, getEngineDefaults } from './engine_config'
import { useEngineCapabilitiesStore } from './engine_capabilities'
import { fallbackSamplingDefaultsForTabFamily, normalizeTabFamily, type TabFamily } from '../utils/engine_taxonomy'

export type BaseTabType = ApiTab['type']
export type ImageTabType = Exclude<BaseTabType, 'wan'>

export interface BaseTabMeta {
  createdAt: string
  updatedAt: string
}

export type ModelTabsErrorCode = 'tab_not_found' | 'api_failure' | 'invalid_response' | 'invalid_reorder'

export class ModelTabsStoreError extends Error {
  readonly code: ModelTabsErrorCode
  readonly cause: unknown
  readonly details: Record<string, unknown> | null

  constructor(
    code: ModelTabsErrorCode,
    message: string,
    options?: { cause?: unknown; details?: Record<string, unknown> | null },
  ) {
    super(message)
    this.name = 'ModelTabsStoreError'
    this.code = code
    this.cause = options?.cause
    this.details = options?.details ?? null
  }
}

export interface WanStageParams {
  modelDir: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  loraSha: string
  loraWeight: number
}

export interface WanVideoParams {
  // Core generation fields (txt2vid/img2vid shared)
  prompt: string
  negativePrompt: string
  width: number
  height: number
  fps: number
  frames: number
  // Optional initial image (img2vid)
  useInitImage: boolean
  initImageData: string
  initImageName: string
  // Optional initial video (vid2vid)
  useInitVideo: boolean
  initVideoPath: string
  initVideoName: string
  // vid2vid controls
  vid2vidStrength: number
  vid2vidMethod: 'native' | 'flow_chunks'
  vid2vidUseSourceFps: boolean
  vid2vidUseSourceFrames: boolean
  vid2vidChunkFrames: number
  vid2vidOverlapFrames: number
  vid2vidPreviewFrames: number
  vid2vidFlowEnabled: boolean
  vid2vidFlowUseLarge: boolean
  vid2vidFlowDownscale: number
  // Export options
  filenamePrefix: string
  format: string
  pixFmt: string
  crf: number
  loopCount: number
  pingpong: boolean
  trimToAudio: boolean
  saveMetadata: boolean
  saveOutput: boolean
  returnFrames: boolean
  // Interpolation (RIFE)
  rifeEnabled: boolean
  rifeModel: string
  rifeTimes: number
}

export interface WanAssetsParams {
  metadata: string
  textEncoder: string
  vae: string
}

export interface BaseTab {
  id: string
  type: BaseTabType
  title: string
  order: number
  enabled: boolean
  params: Record<string, unknown>
  meta: BaseTabMeta
}

export interface ImageBaseParams {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  clipSkip: number
  batchSize: number
  batchCount: number
  hires: HiresFormState
  refiner: RefinerFormState
  checkpoint: string
  textEncoders: string[]
  useInitImage: boolean
  initImageData: string
  initImageName: string
  denoiseStrength: number
  useMask: boolean
  maskImageData: string
  maskImageName: string
  maskEnforcement: 'post_blend' | 'per_step_clamp'
  inpaintFullRes: boolean
  inpaintFullResPadding: number
  inpaintingFill: number
  maskInvert: boolean
  maskBlur: number
  maskRound: boolean
  zimageTurbo?: boolean
}

export type TabParamsByType = {
  sd15: ImageBaseParams
  sdxl: ImageBaseParams
  flux1: ImageBaseParams
  zimage: ImageBaseParams
  chroma: ImageBaseParams
  anima: ImageBaseParams
  wan: {
    high: WanStageParams
    low: WanStageParams
    video: WanVideoParams
    assets: WanAssetsParams
    lightx2v: boolean
    lowFollowsHigh: boolean
  }
}

export type TabByType<T extends BaseTabType = BaseTabType> = Omit<BaseTab, 'type' | 'params'> & {
  type: T
  params: TabParamsByType[T]
}

export const MODEL_TABS_STORAGE_KEY = 'codex:model-tabs:v2'
const STORAGE_KEY = MODEL_TABS_STORAGE_KEY

function nowIso(): string {
  return new Date().toISOString()
}

function requirePersistedTabId(value: unknown, context: string): string {
  const id = typeof value === 'string' ? value.trim() : ''
  if (!id) {
    throw new ModelTabsStoreError(
      'invalid_response',
      `Invalid '/api/ui/tabs' contract: ${context} returned an empty 'id'.`,
    )
  }
  return id
}

function defaultParams<T extends BaseTabType>(
  type: T,
  opts?: { sampler?: string; scheduler?: string },
): TabParamsByType[T] {
  // Video tabs (WAN)
  if (type === 'wan') {
    const stage = (): WanStageParams => ({
      modelDir: '',
      sampler: '',
      scheduler: '',
      steps: 30,
      cfgScale: 7,
      seed: -1,
      loraSha: '',
      loraWeight: 1.0,
    })
    const video: WanVideoParams = {
      prompt: '',
      negativePrompt: '',
      width: 768,
      height: 432,
      fps: 24,
      frames: 16,
      useInitImage: false,
      initImageData: '',
      initImageName: '',
      useInitVideo: false,
      initVideoPath: '',
      initVideoName: '',
      vid2vidStrength: 0.8,
      vid2vidMethod: 'flow_chunks',
      vid2vidUseSourceFps: true,
      vid2vidUseSourceFrames: true,
      vid2vidChunkFrames: 16,
      vid2vidOverlapFrames: 4,
      vid2vidPreviewFrames: 48,
      vid2vidFlowEnabled: true,
      vid2vidFlowUseLarge: false,
      vid2vidFlowDownscale: 2,
      filenamePrefix: 'wan22',
      format: 'video/h264-mp4',
      pixFmt: 'yuv420p',
      crf: 15,
      loopCount: 0,
      pingpong: false,
      trimToAudio: false,
      saveMetadata: true,
      saveOutput: true,
      returnFrames: false,
      rifeEnabled: true,
      rifeModel: 'rife47.pth',
      rifeTimes: 2,
    }
    const assets: WanAssetsParams = { metadata: '', textEncoder: '', vae: '' }
    const wanDefaults: TabParamsByType['wan'] = {
      high: stage(),
      low: stage(),
      video,
      assets,
      lightx2v: false,
      lowFollowsHigh: false,
    }
    return wanDefaults as TabParamsByType[T]
  }

  // Image tabs (SD15, SDXL, FLUX.1)
  const config = getEngineConfig(type as EngineType)
  const defaults = getEngineDefaults(type as EngineType)
  const guidance = config.capabilities.usesDistilledCfg && defaults.distilledCfg !== undefined ? defaults.distilledCfg : defaults.cfg
  const samplingDefaults = fallbackSamplingDefaultsForTabFamily(type as TabFamily)
  const resolvedSampler = String(opts?.sampler || '').trim() || samplingDefaults.sampler
  const resolvedScheduler = String(opts?.scheduler || '').trim() || samplingDefaults.scheduler
  const refinerDefaults: RefinerFormState = {
    enabled: false,
    steps: 0,
    cfg: 3.5,
    seed: -1,
    model: undefined,
    vae: undefined,
  }
  const hiresDefaults: HiresFormState = {
    enabled: false,
    denoise: 0.4,
    scale: 1.5,
    resizeX: 0,
    resizeY: 0,
    steps: 0,
    upscaler: 'latent:bicubic-aa',
    tile: { tile: 256, overlap: 16 },
    checkpoint: undefined,
    modules: [],
    sampler: undefined,
    scheduler: undefined,
    prompt: undefined,
    negativePrompt: undefined,
    cfg: undefined,
    distilledCfg: undefined,
    refiner: { ...refinerDefaults },
  }
  const imageDefaults: ImageBaseParams = {
    prompt: '',
    negativePrompt: config.capabilities.usesNegativePrompt ? '' : '',
    width: defaults.width,
    height: defaults.height,
    sampler: resolvedSampler,
    scheduler: resolvedScheduler,
    steps: defaults.steps,
    cfgScale: guidance,
    seed: -1,
    clipSkip: 0,
    batchSize: 1,
    batchCount: 1,
    hires: { ...hiresDefaults },
    refiner: { ...refinerDefaults },
    checkpoint: '',
    textEncoders: [],
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    denoiseStrength: 0.75,
    useMask: false,
    maskImageData: '',
    maskImageName: '',
    maskEnforcement: 'post_blend',
    inpaintFullRes: true,
    inpaintFullResPadding: 0,
    inpaintingFill: 1,
    maskInvert: false,
    maskBlur: 4,
    maskRound: true,
  }
  if (type === 'zimage') {
    imageDefaults.zimageTurbo = true
  }
  return imageDefaults as TabParamsByType[T]
}

export function defaultImageParamsForType(
  type: ImageTabType,
  opts?: { sampler?: string; scheduler?: string },
): ImageBaseParams
export function defaultImageParamsForType(
  type: BaseTabType,
  opts?: { sampler?: string; scheduler?: string },
): ImageBaseParams {
  if (type === 'wan') {
    const msg = "defaultImageParamsForType received 'wan'; expected an image tab type."
    console.error(`[model_tabs] ${msg}`, { type })
    throw new Error(msg)
  }
  return defaultParams(type, opts)
}

export function normalizeTabType(type: unknown): BaseTabType {
  const raw = String(type || '').trim()
  if (!raw) {
    const msg = 'Model tab type is required, got empty value.'
    console.error(`[model_tabs] ${msg}`, { type })
    throw new Error(msg)
  }
  const normalized = normalizeTabFamily(raw)
  if (normalized) return normalized
  const msg = `Unsupported model tab type '${raw}'.`
  console.error(`[model_tabs] ${msg}`, { type })
  throw new Error(msg)
}

function asRecordObject(value: unknown): Record<string, unknown> {
  if (value && typeof value === 'object' && !Array.isArray(value)) return value as Record<string, unknown>
  return {}
}

function asParamsRecord(params: TabParamsByType[BaseTabType]): Record<string, unknown> {
  return params as unknown as Record<string, unknown>
}

function normalizeWanParams(raw: unknown, defaults: TabParamsByType['wan']): TabParamsByType['wan'] {
  const patch = asRecordObject(raw) as Partial<TabParamsByType['wan']>
  const highPatch = asRecordObject(patch.high)
  const lowPatch = asRecordObject(patch.low)
  const videoPatch = asRecordObject(patch.video)
  const assetsPatch = asRecordObject(patch.assets)
  return {
    ...defaults,
    ...patch,
    high: { ...defaults.high, ...(highPatch as Partial<WanStageParams>) },
    low: { ...defaults.low, ...(lowPatch as Partial<WanStageParams>) },
    video: { ...defaults.video, ...(videoPatch as Partial<WanVideoParams>) },
    assets: { ...defaults.assets, ...(assetsPatch as Partial<WanAssetsParams>) },
  }
}

function normalizeImageParams(raw: unknown, defaults: ImageBaseParams): ImageBaseParams {
  const patch = asRecordObject(raw) as Partial<ImageBaseParams>
  const hiresPatch = asRecordObject(patch.hires)
  const hiresRefinerPatch = asRecordObject(hiresPatch.refiner)
  const hiresTilePatch = asRecordObject(hiresPatch.tile)
  const refinerPatch = asRecordObject(patch.refiner)

  const mergedHires: HiresFormState = {
    ...defaults.hires,
    ...(hiresPatch as Partial<HiresFormState>),
    refiner: {
      ...(asRecordObject(defaults.hires.refiner) as Partial<RefinerFormState>),
      ...(hiresRefinerPatch as Partial<RefinerFormState>),
    } as RefinerFormState,
    tile: {
      ...defaults.hires.tile,
      ...(hiresTilePatch as Partial<HiresFormState['tile']>),
    },
  }

  const merged: ImageBaseParams = {
    ...defaults,
    ...patch,
    hires: mergedHires,
    refiner: {
      ...defaults.refiner,
      ...(refinerPatch as Partial<RefinerFormState>),
    },
  }

  if (typeof merged.sampler !== 'string' || !merged.sampler.trim()) {
    merged.sampler = defaults.sampler
  }
  if (typeof merged.scheduler !== 'string' || !merged.scheduler.trim()) {
    merged.scheduler = defaults.scheduler
  }
  return merged
}

function normalizeParamsForType<T extends BaseTabType>(
  type: T,
  raw: unknown,
  defaultsOverride?: TabParamsByType[T],
): TabParamsByType[T] {
  const defaults = defaultsOverride ?? defaultParams(type)
  if (type === 'wan') {
    return normalizeWanParams(raw, defaults as TabParamsByType['wan']) as TabParamsByType[T]
  }
  return normalizeImageParams(raw, defaults as ImageBaseParams) as TabParamsByType[T]
}

type RawTab = Omit<BaseTab, 'type' | 'params'> & {
  type: unknown
  params?: unknown
}

function normalizeTab(
  tab: RawTab,
  resolveDefaults?: <T extends BaseTabType>(type: T) => TabParamsByType[T],
): BaseTab {
  const type = normalizeTabType(tab.type)
  const defaults = resolveDefaults ? resolveDefaults(type) : undefined
  return {
    ...tab,
    type,
    params: asParamsRecord(normalizeParamsForType(type, tab.params, defaults)),
  }
}

const BASE_REQUIRED_TYPES: BaseTabType[] = ['sd15', 'sdxl', 'flux1', 'chroma', 'zimage', 'wan']

export function requiredTypesFromCapabilities(engines: Record<string, unknown>): BaseTabType[] {
  const types: BaseTabType[] = [...BASE_REQUIRED_TYPES]
  if (Object.prototype.hasOwnProperty.call(engines, 'anima')) {
    types.push('anima')
  }
  return types
}

export const useModelTabsStore = defineStore('modelTabs', () => {
  const tabs = ref<BaseTab[]>([])
  const activeId = ref<string>('')

  async function resolveRequiredTypesFromCapabilities(): Promise<BaseTabType[]> {
    const capsStore = useEngineCapabilitiesStore()
    await capsStore.init()
    const engines = capsStore.engines
    return requiredTypesFromCapabilities(engines as Record<string, unknown>)
  }

  function save(): void {
    const payload = { tabs: tabs.value, activeId: activeId.value }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  }

  function ensureTabOrThrow(id: string): BaseTab {
    const tab = tabs.value.find((entry) => entry.id === id)
    if (!tab) {
      throw new ModelTabsStoreError('tab_not_found', `Tab not found: '${id}'.`, { details: { id } })
    }
    return tab
  }

  function mapApiError(operation: string, error: unknown, details?: Record<string, unknown>): ModelTabsStoreError {
    const message = error instanceof Error ? error.message : String(error)
    return new ModelTabsStoreError(
      'api_failure',
      `${operation}: ${message}`,
      { cause: error, details: details ?? null },
    )
  }

  function preferredSamplingDefaultsForType(type: BaseTabType): { sampler: string; scheduler: string } | null {
    if (type === 'wan') return null
    const capsStore = useEngineCapabilitiesStore()
    const fallback = fallbackSamplingDefaultsForTabFamily(type as TabFamily)
    return capsStore.resolveSamplingDefaults(type, {
      fallbackSampler: fallback.sampler,
      fallbackScheduler: fallback.scheduler,
    })
  }

  function defaultParamsForType<T extends BaseTabType>(type: T): TabParamsByType[T] {
    const preferredSampling = preferredSamplingDefaultsForType(type)
    return defaultParams(type, preferredSampling ?? undefined)
  }

  async function ensureRequiredTabs(requiredTypes: BaseTabType[]): Promise<void> {
    const existing = new Set<BaseTabType>(tabs.value.map(t => t.type))
    let nextOrder = tabs.value.length ? (Math.max(...tabs.value.map(t => t.order)) + 1) : 0
    for (const type of requiredTypes) {
      if (existing.has(type)) continue
      const title = type === 'wan' ? 'WAN 2.2' : getEngineConfig(type as EngineType).label
      const params = asParamsRecord(defaultParamsForType(type))
      let createdId = ''
      try {
        const created = await createTabApi({ type, title, params })
        createdId = requirePersistedTabId(created?.id, `create required tab '${type}'`)
      } catch (error) {
        throw mapApiError(`Failed to ensure required tab '${type}'`, error, { type, title })
      }
      const createdAt = nowIso()
      tabs.value.push({
        id: createdId,
        type,
        title,
        order: nextOrder++,
        enabled: true,
        params,
        meta: { createdAt, updatedAt: createdAt },
      })
      existing.add(type)
    }
  }

  async function load(): Promise<void> {
    const requiredTypes = await resolveRequiredTypesFromCapabilities()
    const preferredActiveId = activeId.value || (() => {
      try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return ''
        const parsed = JSON.parse(raw) as { activeId?: unknown }
        return typeof parsed.activeId === 'string' ? parsed.activeId : ''
      } catch {
        return ''
      }
    })()

    const res = await fetchTabs()
    if (!res || !Array.isArray(res.tabs)) {
      const msg = "Invalid '/api/ui/tabs' response: missing 'tabs' array."
      console.error(`[model_tabs] ${msg}`, res)
      throw new ModelTabsStoreError('invalid_response', msg, { details: { response: res as unknown as Record<string, unknown> } })
    }

    tabs.value = (res.tabs as unknown as BaseTab[]).map((tab) => normalizeTab(tab, defaultParamsForType))
    tabs.value.sort((a, b) => a.order - b.order)
    activeId.value = (preferredActiveId && tabs.value.some(t => t.id === preferredActiveId)) ? preferredActiveId : (tabs.value[0]?.id ?? '')
    await ensureRequiredTabs(requiredTypes)
    tabs.value.sort((a, b) => a.order - b.order)
    if (activeId.value && !tabs.value.some(t => t.id === activeId.value)) activeId.value = tabs.value[0]?.id ?? ''
    save()
  }

  async function create(type: BaseTabType, title?: string): Promise<string> {
    const resolvedTitle = title?.trim() || (type === 'wan' ? 'WAN 2.2' : getEngineConfig(type as EngineType).label)
    const params = asParamsRecord(defaultParamsForType(type))
    let createdId = ''
    try {
      const created = await createTabApi({ type, title: resolvedTitle, params })
      createdId = requirePersistedTabId(created?.id, `create tab '${resolvedTitle}'`)
    } catch (error) {
      throw mapApiError(`Failed to create tab '${resolvedTitle}'`, error, { type, title: resolvedTitle })
    }
    const createdAt = nowIso()
    const nextOrder = tabs.value.length ? Math.max(...tabs.value.map(t => t.order)) + 1 : 0
    tabs.value.push({
      id: createdId,
      type,
      title: resolvedTitle,
      order: nextOrder,
      enabled: true,
      params,
      meta: { createdAt, updatedAt: createdAt },
    })
    save()
    return createdId
  }

  async function duplicate(id: string): Promise<string> {
    const src = ensureTabOrThrow(id)
    const copy: BaseTab = JSON.parse(JSON.stringify(src))
    copy.title = src.title + ' (copy)'
    let createdId = ''
    try {
      const created = await createTabApi({ type: copy.type as BaseTabType, title: copy.title, params: copy.params })
      createdId = requirePersistedTabId(created?.id, `duplicate tab '${id}'`)
    } catch (error) {
      throw mapApiError(`Failed to duplicate tab '${id}'`, error, { id, sourceType: src.type })
    }
    copy.id = createdId
    copy.order = (Math.max(...tabs.value.map(t => t.order)) || 0) + 1
    copy.meta.createdAt = nowIso()
    copy.meta.updatedAt = copy.meta.createdAt
    tabs.value.push(copy)
    save()
    return copy.id
  }

  async function remove(id: string): Promise<void> {
    ensureTabOrThrow(id)
    try {
      await deleteTabApi(id)
    } catch (error) {
      throw mapApiError(`Failed to remove tab '${id}'`, error, { id })
    }
    tabs.value = tabs.value.filter(t => t.id !== id)
    if (activeId.value === id) activeId.value = tabs.value[0]?.id ?? ''
    normalizeOrder()
    save()
  }

  async function rename(id: string, title: string): Promise<void> {
    const t = ensureTabOrThrow(id)
    try {
      await updateTabApi(id, { title })
    } catch (error) {
      throw mapApiError(`Failed to rename tab '${id}'`, error, { id, title })
    }
    t.title = title
    t.meta.updatedAt = nowIso()
    save()
  }

  async function setEnabled(id: string, value: boolean): Promise<void> {
    const t = ensureTabOrThrow(id)
    try {
      await updateTabApi(id, { enabled: value })
    } catch (error) {
      throw mapApiError(`Failed to update enabled flag for tab '${id}'`, error, { id, enabled: value })
    }
    t.enabled = value
    t.meta.updatedAt = nowIso()
    save()
  }

  async function reorder(ids: string[]): Promise<void> {
    const expectedIds = tabs.value.map(t => t.id)
    if (ids.length !== expectedIds.length) {
      throw new ModelTabsStoreError(
        'invalid_reorder',
        'Invalid reorder payload: id list length does not match tabs length.',
        { details: { expected: expectedIds.length, received: ids.length } },
      )
    }
    if (new Set(ids).size !== ids.length) {
      throw new ModelTabsStoreError(
        'invalid_reorder',
        'Invalid reorder payload: duplicate tab ids are not allowed.',
        { details: { ids } },
      )
    }
    const expectedSet = new Set(expectedIds)
    for (const id of ids) {
      if (!expectedSet.has(id)) {
        throw new ModelTabsStoreError(
          'invalid_reorder',
          `Invalid reorder payload: unknown tab id '${id}'.`,
          { details: { id, expectedIds } },
        )
      }
    }
    try {
      await reorderTabsApi(ids)
    } catch (error) {
      throw mapApiError('Failed to reorder tabs', error, { ids })
    }
    const map = new Map<string, number>()
    ids.forEach((id, idx) => map.set(id, idx))
    tabs.value.forEach(t => { t.order = map.get(t.id) ?? t.order })
    tabs.value.sort((a, b) => a.order - b.order)
    save()
  }

  function setActive(id: string): void { activeId.value = id; save() }

  async function updateParams<T extends Record<string, unknown>>(id: string, patch: Partial<T>): Promise<void> {
    const t = ensureTabOrThrow(id)
    const current = (t.params && typeof t.params === 'object' && !Array.isArray(t.params)) ? (t.params as T) : ({} as T)
    if (current !== t.params) t.params = current
    const previousEntries = new Map<string, { existed: boolean; value: unknown }>()
    for (const [key, value] of Object.entries(patch as Record<string, unknown>)) {
      previousEntries.set(key, {
        existed: Object.prototype.hasOwnProperty.call(current, key),
        value: (current as Record<string, unknown>)[key],
      })
    }
    const previousUpdatedAt = t.meta.updatedAt
    Object.assign(current, patch)
    t.meta.updatedAt = nowIso()
    try {
      await updateTabApi(id, { params: t.params })
    } catch (error) {
      for (const [key, snapshot] of previousEntries.entries()) {
        if (snapshot.existed) {
          ;(current as Record<string, unknown>)[key] = snapshot.value
        } else {
          delete (current as Record<string, unknown>)[key]
        }
      }
      t.meta.updatedAt = previousUpdatedAt
      save()
      throw mapApiError(`Failed to update params for tab '${id}'`, error, { id })
    }
    save()
  }

  function normalizeOrder(): void {
    tabs.value.sort((a, b) => a.order - b.order)
    tabs.value.forEach((t, idx) => { t.order = idx })
  }

  const orderedTabs = computed(() => [...tabs.value].sort((a, b) => a.order - b.order))
  const activeTab = computed(() => tabs.value.find(t => t.id === activeId.value) || null)

  return {
    tabs,
    orderedTabs,
    activeId,
    activeTab,
    load,
    save,
    create,
    duplicate,
    remove,
    rename,
    reorder,
    setEnabled,
    setActive,
    updateParams,
  }
})
