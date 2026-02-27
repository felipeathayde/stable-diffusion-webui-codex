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
(`latent:*` / `spandrel:*`) for hires-fix wiring, and img2img UI keeps an explicit resize/upscaler layout state (`img2imgResizeMode`,
`img2imgUpscaler`) decoupled from backend hires dispatch.

Symbols (top-level; keep in sync; no ghosts):
- `BaseTabType` (type): API tab type discriminator (from backend `ApiTab['type']`).
- `ImageTabType` (type): Image-only tab type discriminator (`BaseTabType` without `wan`).
- `BaseTabMeta` (interface): Tab metadata timestamps (created/updated) tracked client-side.
- `ModelTabsErrorCode` (type): Error code taxonomy for model-tabs store failures.
- `ModelTabsStoreError` (class): Typed store error thrown for tab lookup/API/contract/reorder/serialization failures.
- `WanStageParams` (interface): UI WAN stage params (high/low), including stage prompt/negative prompt and optional explicit `flowShift`, used by video tabs and payload builders.
- `WanImg2VidMode` (type): WAN img2vid temporal mode discriminator (`solo|chunk|sliding|svi2|svi2_pro`).
- `WanChunkSeedMode` (type): WAN chunk/sliding per-window seed strategy (`fixed|increment|random`).
- `WanVideoParams` (interface): UI WAN video params (dims/fps/frames + optional init image + chunking/output controls + interpolation target FPS).
- `WanAssetsParams` (interface): WAN asset selectors (metadata/text encoder/VAE) used by WAN requests.
- `BaseTab` (interface): Generic tab record persisted in the store (id/type/label + params + meta).
- `ImageBaseParams` (interface): Common image-tab params (prompt, seed, steps, CFG, dims, etc.) shared across SD/Flux.1/Chroma/ZImage
  (includes optional family-specific fields like `zimageTurbo`, img2img layout state `img2imgResizeMode`/`img2imgUpscaler`,
  and advanced guidance policy controls).
- `GuidanceAdvancedParams` (interface): Per-tab advanced guidance policy state (APG/rescale/trunc/renorm).
- `DEFAULT_GUIDANCE_ADVANCED_PARAMS` (constant): Canonical defaults for `ImageBaseParams.guidanceAdvanced`.
- `TabParamsByType` (type): Canonical params map by tab type.
- `TabByType` (type): Typed tab shape (`type` + matching params payload).
- `ModelTabsStorageState` (type): LocalStorage payload contract for light model-tabs state (`activeId` + tab refs).
- `MODEL_TABS_STORAGE_KEY` (const): LocalStorage key used for persisted tabs state (bump when schema changes).
- `buildStoragePayload` (function): Builds a small localStorage payload without heavy per-tab params blobs.
- `isQuotaExceededStorageError` (function): Detects storage quota-exceeded failures across browser variants.
- `nowIso` (function): Returns current time in ISO string form for metadata timestamps.
- `defaultParams` (function): Returns default params for a given tab type (image vs WAN video), merging engine defaults where applicable.
- `defaultImageParamsForType` (function): Returns canonical image-tab defaults for a specific image tab type.
- `normalizeTabType` (function): Validates/coerces raw type values into `BaseTabType`.
- `BASE_REQUIRED_TYPES` (const): Baseline tab types always auto-created by the UI store.
- `requiredTypesFromCapabilities` (function): Derives required tab types from backend capability map (adds `anima` only when exposed).
- `asRecordObject` (function): Narrowing helper that normalizes unknown values into plain records for merge-safe processing.
- `isPlainRecord` (function): Validates object values as plain record payloads (no arrays/class instances) for patch serialization safety.
- `PersistSerializationPhase` (type): Serialization boundary phases used by params persistence snapshots and rollback.
- `serializationFailure` (function): Factory for typed fail-loud params serialization errors with contextual details.
- `normalizeSerializableForPersist` (function): Recursively unwraps reactive/proxy branches into plain clone-safe structures for persistence.
- `asParamsRecord` (function): Explicit boundary cast helper from typed tab params to persisted `Record<string, unknown>`.
- `normalizeWanFrameCount` (function): Clamps/snap-normalizes WAN frame counts to the `4n+1` domain.
- `normalizeWanVideoParams` (function): Sanitizes WAN video nested params (frames/chunk/sliding/attention controls) with `img2vidMode` as source of truth.
- `normalizeWanParams` (function): Applies WAN-specific nested merge normalization for `high/low/video/assets` params.
- `normalizeImageParams` (function): Applies image-tab nested merge normalization (`hires/refiner`) with sampler/scheduler and mask-enforcement fallback.
- `normalizeParamsForType` (function): Normalizes raw params payload based on tab type (shape checking; discards invalid fields).
- `normalizeTab` (function): Normalizes a raw tab record (id/type/params/meta) into the store shape.
- `cloneParamsForPersist` (function): Proxy-safe `structuredClone` boundary for params snapshots/payloads; throws typed serialization failures.
- `restorePendingParamsSnapshot` (function): Restores tab params/meta from pending snapshot after failed persistence attempts.
- `scheduleParamsPersist` (function): Schedules debounced `/api/ui/tabs/:id` params PATCH calls.
- `flushParamsPersist` (function): Flushes pending params PATCH for a tab, resolves queued promises, and rolls back on API failure.
- `useModelTabsStore` (store): Pinia store for tabs; loads/syncs with backend, provides CRUD/reorder actions, and exposes computed helpers.
*/

import { defineStore } from 'pinia'
import { ref, computed, toRaw } from 'vue'
import { fetchTabs, createTabApi, updateTabApi, reorderTabsApi, deleteTabApi } from '../api/client'
import type { ApiTab } from '../api/types'
import type { HiresFormState, RefinerFormState } from '../api/payloads'
import { type EngineType, getEngineConfig, getEngineDefaults } from './engine_config'
import { useEngineCapabilitiesStore } from './engine_capabilities'
import { fallbackSamplingDefaultsForTabFamily, normalizeTabFamily, type TabFamily } from '../utils/engine_taxonomy'
import { DEFAULT_IMG2IMG_RESIZE_MODE, normalizeImg2ImgResizeMode, type Img2ImgResizeMode } from '../utils/img2img_resize'
import { normalizeMaskEnforcement } from '../utils/image_params'
import {
  normalizeWanChunkOverlap,
  normalizeWanWindowCommit,
  normalizeWanWindowStride,
  type WanImg2VidMode as WanImg2VidModeInternal,
} from '../utils/wan_img2vid_temporal'

export type BaseTabType = ApiTab['type']
export type ImageTabType = Exclude<BaseTabType, 'wan'>

export interface BaseTabMeta {
  createdAt: string
  updatedAt: string
}

export type ModelTabsErrorCode =
  | 'tab_not_found'
  | 'api_failure'
  | 'invalid_response'
  | 'invalid_reorder'
  | 'serialization_failure'

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
  prompt: string
  negativePrompt: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  loraSha: string
  loraWeight: number
  flowShift?: number
}

export type WanImg2VidMode = WanImg2VidModeInternal
export type WanChunkSeedMode = 'fixed' | 'increment' | 'random'

export interface WanVideoParams {
  // Core generation fields (txt2vid/img2vid shared)
  width: number
  height: number
  fps: number
  frames: number
  attentionMode: 'global' | 'sliding'
  // Optional initial image (img2vid)
  useInitImage: boolean
  initImageData: string
  initImageName: string
  img2vidMode: WanImg2VidMode
  img2vidChunkFrames: number
  img2vidOverlapFrames: number
  img2vidAnchorAlpha: number
  img2vidResetAnchorToBase: boolean
  img2vidChunkSeedMode: WanChunkSeedMode
  img2vidWindowFrames: number
  img2vidWindowStride: number
  img2vidWindowCommitFrames: number
  // Export options
  format: string
  pixFmt: string
  crf: number
  loopCount: number
  pingpong: boolean
  returnFrames: boolean
  // Interpolation (RIFE target FPS; 0 disables interpolation)
  interpolationFps: number
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
  img2imgResizeMode: Img2ImgResizeMode
  img2imgUpscaler: string
  guidanceAdvanced: GuidanceAdvancedParams
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

export interface GuidanceAdvancedParams {
  enabled: boolean
  apgEnabled: boolean
  apgStartStep: number
  apgEta: number
  apgMomentum: number
  apgNormThreshold: number
  apgRescale: number
  guidanceRescale: number
  cfgTruncEnabled: boolean
  cfgTruncRatio: number
  renormCfg: number
}

export const DEFAULT_GUIDANCE_ADVANCED_PARAMS: GuidanceAdvancedParams = {
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

type ModelTabsStorageTabRef = Pick<BaseTab, 'id' | 'type'>

type ModelTabsStorageState = {
  tabs: ModelTabsStorageTabRef[]
  activeId: string
}

export const MODEL_TABS_STORAGE_KEY = 'codex:model-tabs:v2'
const STORAGE_KEY = MODEL_TABS_STORAGE_KEY

function buildStoragePayload(tabList: BaseTab[], currentActiveId: string): ModelTabsStorageState {
  const tabRefs: ModelTabsStorageTabRef[] = tabList.map((tab) => ({
    id: tab.id,
    type: tab.type,
  }))
  return {
    tabs: tabRefs,
    activeId: currentActiveId,
  }
}

function isQuotaExceededStorageError(error: unknown): boolean {
  if (!(error instanceof DOMException)) return false
  if (error.name === 'QuotaExceededError' || error.name === 'NS_ERROR_DOM_QUOTA_REACHED') return true
  const domError = error as DOMException & { code?: number }
  return domError.code === 22 || domError.code === 1014
}

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
      prompt: '',
      negativePrompt: '',
      sampler: '',
      scheduler: '',
      steps: 30,
      cfgScale: 7,
      seed: -1,
      loraSha: '',
      loraWeight: 1.0,
    })
    const video: WanVideoParams = {
      width: 768,
      height: 432,
      fps: 15,
      frames: 17,
      attentionMode: 'global',
      useInitImage: false,
      initImageData: '',
      initImageName: '',
      img2vidMode: 'solo',
      img2vidChunkFrames: 13,
      img2vidOverlapFrames: 4,
      img2vidAnchorAlpha: 0.2,
      img2vidResetAnchorToBase: false,
      img2vidChunkSeedMode: 'increment',
      img2vidWindowFrames: 13,
      img2vidWindowStride: 8,
      img2vidWindowCommitFrames: 12,
      format: 'video/h264-mp4',
      pixFmt: 'yuv420p',
      crf: 15,
      loopCount: 0,
      pingpong: false,
      returnFrames: false,
      interpolationFps: 0,
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
    swapAtStep: 1,
    cfg: 3.5,
    seed: -1,
    model: undefined,
    vae: undefined,
  }
  const hiresDefaults: HiresFormState = {
    enabled: false,
    denoise: 0.4,
    scale: 2,
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
    img2imgResizeMode: DEFAULT_IMG2IMG_RESIZE_MODE,
    img2imgUpscaler: 'latent:bicubic-aa',
    guidanceAdvanced: { ...DEFAULT_GUIDANCE_ADVANCED_PARAMS },
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
    maskEnforcement: 'per_step_clamp',
    inpaintFullRes: true,
    inpaintFullResPadding: 32,
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

function isPlainRecord(value: unknown): value is Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false
  const prototype = Object.getPrototypeOf(toRaw(value))
  return prototype === Object.prototype || prototype === null
}

function asParamsRecord(params: TabParamsByType[BaseTabType]): Record<string, unknown> {
  return params as unknown as Record<string, unknown>
}

function normalizeWanFrameCount(rawValue: number, min = 9, max = 401): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : min
  const clamped = Math.min(max, Math.max(min, numeric))
  if ((clamped - 1) % 4 === 0) return clamped

  const down = clamped - (((clamped - 1) % 4 + 4) % 4)
  const up = down + 4
  const downInRange = down >= min
  const upInRange = up <= max
  if (downInRange && upInRange) {
    const downDistance = Math.abs(clamped - down)
    const upDistance = Math.abs(up - clamped)
    return downDistance <= upDistance ? down : up
  }
  if (downInRange) return down
  if (upInRange) return up
  return min
}

function normalizeInterpolationTargetFps(rawValue: unknown, fallback: number): number {
  const maxFps = 240
  const fallbackNumeric = Number.isFinite(Number(fallback)) ? Math.trunc(Number(fallback)) : 0
  const fallbackNormalized = Math.max(0, Math.min(maxFps, fallbackNumeric))
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) return fallbackNormalized
  return Math.max(0, Math.min(maxFps, Math.trunc(numeric)))
}

function normalizeWanVideoParams(raw: Partial<WanVideoParams>, defaults: WanVideoParams): WanVideoParams {
  const merged: WanVideoParams = { ...defaults, ...raw }
  merged.frames = normalizeWanFrameCount(Number(merged.frames))

  const attnMode = String(merged.attentionMode || '').trim().toLowerCase()
  merged.attentionMode = attnMode === 'sliding' ? 'sliding' : 'global'

  const rawRecord = raw as Record<string, unknown>
  const rawMode = String(merged.img2vidMode || '').trim().toLowerCase()
  if (rawMode === 'solo' || rawMode === 'chunk' || rawMode === 'sliding' || rawMode === 'svi2' || rawMode === 'svi2_pro') {
    merged.img2vidMode = rawMode
  } else {
    const legacyEnabled = rawRecord.img2vidChunkingEnabled
    const hasLegacyChunkToggle = typeof legacyEnabled === 'boolean'
    const legacyChunkEnabled = hasLegacyChunkToggle ? Boolean(legacyEnabled) : false
    const hasSlidingWindow = Number.isFinite(Number(rawRecord.img2vidWindowFrames)) && Number(rawRecord.img2vidWindowFrames) > 0
    const hasChunkFrames = Number.isFinite(Number(rawRecord.img2vidChunkFrames)) && Number(rawRecord.img2vidChunkFrames) > 0
    if (hasLegacyChunkToggle) {
      merged.img2vidMode = legacyChunkEnabled ? 'chunk' : 'solo'
    } else if (hasSlidingWindow) {
      merged.img2vidMode = 'sliding'
    } else if (hasChunkFrames) {
      merged.img2vidMode = 'chunk'
    } else {
      merged.img2vidMode = 'solo'
    }
  }

  const chunkRaw = Number(merged.img2vidChunkFrames)
  if (!Number.isFinite(chunkRaw) || chunkRaw <= 0) {
    merged.img2vidChunkFrames = defaults.img2vidChunkFrames
  } else {
    merged.img2vidChunkFrames = normalizeWanFrameCount(chunkRaw, 9, 401)
  }

  const anchorRaw = Number(merged.img2vidAnchorAlpha)
  merged.img2vidAnchorAlpha = Number.isFinite(anchorRaw) ? Math.min(1, Math.max(0, anchorRaw)) : defaults.img2vidAnchorAlpha

  const modeDefaultResetAnchor = merged.img2vidMode === 'chunk'
  const hasExplicitResetAnchor = Object.prototype.hasOwnProperty.call(rawRecord, 'img2vidResetAnchorToBase')
  if (merged.img2vidMode === 'svi2' || merged.img2vidMode === 'svi2_pro') {
    merged.img2vidResetAnchorToBase = false
  } else if (hasExplicitResetAnchor) {
    merged.img2vidResetAnchorToBase = Boolean(rawRecord.img2vidResetAnchorToBase)
  } else {
    merged.img2vidResetAnchorToBase = modeDefaultResetAnchor
  }

  const seedMode = String(merged.img2vidChunkSeedMode || '').trim().toLowerCase()
  const modeDefaultSeed = merged.img2vidMode === 'sliding' ? 'fixed' : 'increment'
  if (seedMode !== 'fixed' && seedMode !== 'increment' && seedMode !== 'random') {
    merged.img2vidChunkSeedMode = modeDefaultSeed
  } else {
    merged.img2vidChunkSeedMode = seedMode
  }

  const windowRaw = Number(merged.img2vidWindowFrames)
  if (!Number.isFinite(windowRaw) || windowRaw <= 0) {
    merged.img2vidWindowFrames = defaults.img2vidWindowFrames
  } else {
    merged.img2vidWindowFrames = normalizeWanFrameCount(windowRaw, 9, 401)
  }

  const temporalUpperBound = normalizeWanFrameCount(Math.max(9, merged.frames - 4), 9, 401)
  if (temporalUpperBound < merged.frames) {
    if (merged.img2vidChunkFrames >= merged.frames) {
      merged.img2vidChunkFrames = temporalUpperBound
    }
    if (merged.img2vidWindowFrames >= merged.frames) {
      merged.img2vidWindowFrames = temporalUpperBound
    }
  }

  const overlapRaw = Number(merged.img2vidOverlapFrames)
  merged.img2vidOverlapFrames = normalizeWanChunkOverlap(
    overlapRaw,
    merged.img2vidChunkFrames,
    defaults.img2vidOverlapFrames,
  )

  const strideRaw = Number(merged.img2vidWindowStride)
  merged.img2vidWindowStride = normalizeWanWindowStride(
    strideRaw,
    merged.img2vidWindowFrames,
    defaults.img2vidWindowStride,
  )

  const commitRaw = Number(merged.img2vidWindowCommitFrames)
  merged.img2vidWindowCommitFrames = normalizeWanWindowCommit(
    commitRaw,
    merged.img2vidWindowFrames,
    merged.img2vidWindowStride,
    defaults.img2vidWindowCommitFrames,
  )

  merged.interpolationFps = normalizeInterpolationTargetFps(
    merged.interpolationFps,
    defaults.interpolationFps,
  )

  return {
    width: merged.width,
    height: merged.height,
    fps: merged.fps,
    frames: merged.frames,
    attentionMode: merged.attentionMode,
    useInitImage: merged.useInitImage,
    initImageData: merged.initImageData,
    initImageName: merged.initImageName,
    img2vidMode: merged.img2vidMode,
    img2vidChunkFrames: merged.img2vidChunkFrames,
    img2vidOverlapFrames: merged.img2vidOverlapFrames,
    img2vidAnchorAlpha: merged.img2vidAnchorAlpha,
    img2vidResetAnchorToBase: merged.img2vidResetAnchorToBase,
    img2vidChunkSeedMode: merged.img2vidChunkSeedMode,
    img2vidWindowFrames: merged.img2vidWindowFrames,
    img2vidWindowStride: merged.img2vidWindowStride,
    img2vidWindowCommitFrames: merged.img2vidWindowCommitFrames,
    format: merged.format,
    pixFmt: merged.pixFmt,
    crf: merged.crf,
    loopCount: merged.loopCount,
    pingpong: merged.pingpong,
    returnFrames: merged.returnFrames,
    interpolationFps: merged.interpolationFps,
  }
}

function normalizeWanParams(raw: unknown, defaults: TabParamsByType['wan']): TabParamsByType['wan'] {
  const patch = asRecordObject(raw) as Partial<TabParamsByType['wan']>
  const highPatch = asRecordObject(patch.high)
  const lowPatch = asRecordObject(patch.low)
  const videoPatch = asRecordObject(patch.video)
  const assetsPatch = asRecordObject(patch.assets)
  const normalizedVideo = normalizeWanVideoParams(videoPatch as Partial<WanVideoParams>, defaults.video)
  return {
    ...defaults,
    ...patch,
    high: { ...defaults.high, ...(highPatch as Partial<WanStageParams>) },
    low: { ...defaults.low, ...(lowPatch as Partial<WanStageParams>) },
    video: normalizedVideo,
    assets: { ...defaults.assets, ...(assetsPatch as Partial<WanAssetsParams>) },
  }
}

function normalizeGuidanceAdvancedParams(raw: unknown, defaults: GuidanceAdvancedParams): GuidanceAdvancedParams {
  const patch = asRecordObject(raw)
  const toFiniteNumber = (value: unknown, fallback: number): number => {
    const numeric = Number(value)
    return Number.isFinite(numeric) ? numeric : fallback
  }
  const clampNumber = (value: unknown, fallback: number, min?: number, max?: number): number => {
    const numeric = toFiniteNumber(value, fallback)
    if (min !== undefined && numeric < min) return min
    if (max !== undefined && numeric > max) return max
    return numeric
  }
  const clampInteger = (value: unknown, fallback: number, min?: number, max?: number): number => {
    const numeric = Math.trunc(clampNumber(value, fallback, min, max))
    if (min !== undefined && numeric < min) return min
    if (max !== undefined && numeric > max) return max
    return numeric
  }
  return {
    enabled: typeof patch.enabled === 'boolean' ? patch.enabled : defaults.enabled,
    apgEnabled: typeof patch.apgEnabled === 'boolean' ? patch.apgEnabled : defaults.apgEnabled,
    apgStartStep: clampInteger(patch.apgStartStep, defaults.apgStartStep, 0),
    apgEta: clampNumber(patch.apgEta, defaults.apgEta),
    apgMomentum: clampNumber(patch.apgMomentum, defaults.apgMomentum, 0, 0.999999),
    apgNormThreshold: clampNumber(patch.apgNormThreshold, defaults.apgNormThreshold, 0),
    apgRescale: clampNumber(patch.apgRescale, defaults.apgRescale, 0, 1),
    guidanceRescale: clampNumber(patch.guidanceRescale, defaults.guidanceRescale, 0, 1),
    cfgTruncEnabled: typeof patch.cfgTruncEnabled === 'boolean' ? patch.cfgTruncEnabled : defaults.cfgTruncEnabled,
    cfgTruncRatio: clampNumber(patch.cfgTruncRatio, defaults.cfgTruncRatio, 0, 1),
    renormCfg: clampNumber(patch.renormCfg, defaults.renormCfg, 0),
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

  const globalSwapAtStep = Number(merged.refiner.swapAtStep)
  merged.refiner.swapAtStep = Number.isFinite(globalSwapAtStep) && globalSwapAtStep >= 1
    ? Math.trunc(globalSwapAtStep)
    : 1
  if (merged.hires.refiner) {
    const hiresSwapAtStep = Number(merged.hires.refiner.swapAtStep)
    merged.hires.refiner.swapAtStep = Number.isFinite(hiresSwapAtStep) && hiresSwapAtStep >= 1
      ? Math.trunc(hiresSwapAtStep)
      : 1
  }

  if (typeof merged.sampler !== 'string' || !merged.sampler.trim()) {
    merged.sampler = defaults.sampler
  }
  if (typeof merged.scheduler !== 'string' || !merged.scheduler.trim()) {
    merged.scheduler = defaults.scheduler
  }
  merged.maskEnforcement = normalizeMaskEnforcement(
    typeof merged.maskEnforcement === 'string' ? merged.maskEnforcement : defaults.maskEnforcement,
  )
  merged.img2imgResizeMode = normalizeImg2ImgResizeMode(merged.img2imgResizeMode)
  merged.img2imgUpscaler = String(merged.img2imgUpscaler || '').trim() || defaults.img2imgUpscaler
  merged.guidanceAdvanced = normalizeGuidanceAdvancedParams(
    patch.guidanceAdvanced,
    defaults.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED_PARAMS,
  )
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
  const pendingParamsPersists = new Map<string, PendingParamsPersist>()

  const PARAMS_PERSIST_DEBOUNCE_MS = 220

  type PersistDeferred = {
    version: number
    resolve: () => void
    reject: (reason?: unknown) => void
  }

  type PendingParamsPersist = {
    timer: ReturnType<typeof setTimeout> | null
    inFlight: boolean
    version: number
    persistedVersion: number
    deferreds: PersistDeferred[]
    snapshotParams: Record<string, unknown> | null
    snapshotUpdatedAt: string
  }

  type PersistSerializationPhase = 'snapshot' | 'patch' | 'persist' | 'rollback'

  async function resolveRequiredTypesFromCapabilities(): Promise<BaseTabType[]> {
    const capsStore = useEngineCapabilitiesStore()
    await capsStore.init()
    const engines = capsStore.engines
    return requiredTypesFromCapabilities(engines as Record<string, unknown>)
  }

  function save(): void {
    const payload = buildStoragePayload(tabs.value, activeId.value)
    const serializedPayload = JSON.stringify(payload)
    try {
      localStorage.setItem(STORAGE_KEY, serializedPayload)
      return
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      if (!isQuotaExceededStorageError(error)) {
        console.warn(`[model_tabs] Failed to persist local state '${STORAGE_KEY}': ${message}`)
        return
      }
      console.warn(`[model_tabs] LocalStorage quota exceeded for '${STORAGE_KEY}'. Writing minimal fallback state.`)
    }

    try {
      localStorage.removeItem(STORAGE_KEY)
      localStorage.setItem(STORAGE_KEY, serializedPayload)
      return
    } catch (fallbackError) {
      const message = fallbackError instanceof Error ? fallbackError.message : String(fallbackError)
      console.warn(`[model_tabs] Failed to persist lightweight state retry for '${STORAGE_KEY}': ${message}`)
    }

    try {
      const fallbackPayload: ModelTabsStorageState = { tabs: [], activeId: activeId.value }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(fallbackPayload))
    } catch (fallbackError) {
      const message = fallbackError instanceof Error ? fallbackError.message : String(fallbackError)
      console.warn(`[model_tabs] Failed to persist minimal fallback state for '${STORAGE_KEY}': ${message}`)
    }
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

  function serializationFailure(
    tabId: string,
    phase: PersistSerializationPhase,
    message: string,
    options?: { cause?: unknown; details?: Record<string, unknown> },
  ): ModelTabsStoreError {
    return new ModelTabsStoreError(
      'serialization_failure',
      `Failed to serialize params for tab '${tabId}' (${phase}): ${message}`,
      {
        cause: options?.cause,
        details: {
          tabId,
          phase,
          ...(options?.details ?? {}),
        },
      },
    )
  }

  function normalizeSerializableForPersist(
    tabId: string,
    phase: PersistSerializationPhase,
    value: unknown,
    path: string,
    seen: WeakSet<object>,
  ): unknown {
    if (value === null) return null
    const kind = typeof value
    if (kind === 'string' || kind === 'number' || kind === 'boolean' || kind === 'bigint') return value
    if (kind === 'undefined') return undefined
    if (kind === 'function' || kind === 'symbol') {
      throw serializationFailure(tabId, phase, `Unsupported value type '${kind}' at '${path}'.`, { details: { path, kind } })
    }
    if (kind !== 'object') return value

    const raw = toRaw(value as object)
    if (Array.isArray(raw)) {
      if (seen.has(raw)) {
        throw serializationFailure(tabId, phase, `Circular reference found at '${path}'.`, { details: { path, kind: 'array' } })
      }
      seen.add(raw)
      const normalizedArray = raw.map((entry, index) =>
        normalizeSerializableForPersist(tabId, phase, entry, `${path}[${index}]`, seen),
      )
      seen.delete(raw)
      return normalizedArray
    }

    const prototype = Object.getPrototypeOf(raw)
    if (prototype !== Object.prototype && prototype !== null) {
      const ctorName = (raw as { constructor?: { name?: string } }).constructor?.name ?? 'unknown'
      throw serializationFailure(tabId, phase, `Unsupported object type '${ctorName}' at '${path}'.`, {
        details: { path, ctorName },
      })
    }

    if (seen.has(raw as object)) {
      throw serializationFailure(tabId, phase, `Circular reference found at '${path}'.`, { details: { path, kind: 'object' } })
    }
    seen.add(raw as object)
    const normalizedRecord: Record<string, unknown> = {}
    for (const [key, entry] of Object.entries(raw as Record<string, unknown>)) {
      normalizedRecord[key] = normalizeSerializableForPersist(tabId, phase, entry, `${path}.${key}`, seen)
    }
    seen.delete(raw as object)
    return normalizedRecord
  }

  function cloneParamsForPersist(
    tabId: string,
    phase: PersistSerializationPhase,
    value: Record<string, unknown>,
  ): Record<string, unknown> {
    if (typeof structuredClone !== 'function') {
      throw serializationFailure(tabId, phase, 'structuredClone is unavailable.')
    }

    let normalizedValue: unknown
    try {
      normalizedValue = normalizeSerializableForPersist(tabId, phase, value, '$', new WeakSet<object>())
    } catch (error) {
      if (error instanceof ModelTabsStoreError) throw error
      const message = error instanceof Error ? error.message : String(error)
      throw serializationFailure(tabId, phase, message, { cause: error })
    }
    if (!isPlainRecord(normalizedValue)) {
      throw serializationFailure(tabId, phase, "Root params payload must be a plain object.", { details: { path: '$' } })
    }

    try {
      return structuredClone(normalizedValue)
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)
      throw serializationFailure(tabId, phase, message, { cause: error })
    }
  }

  function restorePendingParamsSnapshot(tabId: string, tab: BaseTab, pending: PendingParamsPersist): void {
    if (pending.snapshotParams) {
      tab.params = cloneParamsForPersist(tabId, 'rollback', pending.snapshotParams)
    }
    tab.meta.updatedAt = pending.snapshotUpdatedAt
    pending.version = pending.persistedVersion
    pending.snapshotParams = null
  }

  function getPendingParamsPersist(tabId: string, tab: BaseTab): PendingParamsPersist {
    const existing = pendingParamsPersists.get(tabId)
    if (existing) return existing
    const created: PendingParamsPersist = {
      timer: null,
      inFlight: false,
      version: 0,
      persistedVersion: 0,
      deferreds: [],
      snapshotParams: null,
      snapshotUpdatedAt: tab.meta.updatedAt,
    }
    pendingParamsPersists.set(tabId, created)
    return created
  }

  function clearPendingParamsPersist(tabId: string, reason: unknown): void {
    const pending = pendingParamsPersists.get(tabId)
    if (!pending) return
    if (pending.timer !== null) {
      clearTimeout(pending.timer)
      pending.timer = null
    }
    const rejectList = pending.deferreds
    pending.deferreds = []
    rejectList.forEach((entry) => entry.reject(reason))
    pendingParamsPersists.delete(tabId)
  }

  function scheduleParamsPersist(tabId: string): void {
    const pending = pendingParamsPersists.get(tabId)
    if (!pending) return
    if (pending.timer !== null) {
      clearTimeout(pending.timer)
    }
    pending.timer = setTimeout(() => {
      pending.timer = null
      void flushParamsPersist(tabId)
    }, PARAMS_PERSIST_DEBOUNCE_MS)
  }

  async function flushParamsPersist(tabId: string): Promise<void> {
    const pending = pendingParamsPersists.get(tabId)
    if (!pending) return
    if (pending.inFlight) {
      scheduleParamsPersist(tabId)
      return
    }
    if (pending.version <= pending.persistedVersion) {
      if (pending.deferreds.length === 0 && pending.timer === null) {
        pendingParamsPersists.delete(tabId)
      }
      return
    }

    let tab: BaseTab
    try {
      tab = ensureTabOrThrow(tabId)
    } catch (error) {
      clearPendingParamsPersist(tabId, error)
      return
    }
    let paramsToPersist: Record<string, unknown>
    const updatedAtSnapshot = tab.meta.updatedAt
    try {
      paramsToPersist = cloneParamsForPersist(tabId, 'persist', tab.params as Record<string, unknown>)
    } catch (error) {
      const mapped = error instanceof ModelTabsStoreError
        ? error
        : mapApiError(`Failed to serialize params for tab '${tabId}' before persistence`, error, { id: tabId })
      try {
        restorePendingParamsSnapshot(tabId, tab, pending)
        save()
      } catch (rollbackError) {
        clearPendingParamsPersist(tabId, rollbackError)
        return
      }
      clearPendingParamsPersist(tabId, mapped)
      return
    }
    const versionToPersist = pending.version

    pending.inFlight = true
    try {
      await updateTabApi(tabId, { params: paramsToPersist })
      pending.persistedVersion = versionToPersist

      const resolveList = pending.deferreds.filter((entry) => entry.version <= versionToPersist)
      pending.deferreds = pending.deferreds.filter((entry) => entry.version > versionToPersist)
      resolveList.forEach((entry) => entry.resolve())

      if (pending.version > pending.persistedVersion) {
        pending.snapshotParams = paramsToPersist
        pending.snapshotUpdatedAt = updatedAtSnapshot
      } else {
        pending.snapshotParams = null
        pending.snapshotUpdatedAt = tab.meta.updatedAt
      }
      save()
    } catch (error) {
      try {
        restorePendingParamsSnapshot(tabId, tab, pending)
      } catch (rollbackError) {
        clearPendingParamsPersist(tabId, rollbackError)
        return
      }

      const mapped = error instanceof ModelTabsStoreError
        ? error
        : mapApiError(`Failed to update params for tab '${tabId}'`, error, { id: tabId })
      const rejectList = pending.deferreds
      pending.deferreds = []
      rejectList.forEach((entry) => entry.reject(mapped))
      save()
    } finally {
      pending.inFlight = false
      if (pending.version > pending.persistedVersion) {
        scheduleParamsPersist(tabId)
      } else if (pending.deferreds.length === 0 && pending.timer === null) {
        pendingParamsPersists.delete(tabId)
      }
    }
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
    if (pendingParamsPersists.size > 0) {
      for (const tabId of pendingParamsPersists.keys()) {
        clearPendingParamsPersist(
          tabId,
          new ModelTabsStoreError('invalid_response', 'Tabs were reloaded while param updates were pending.'),
        )
      }
    }
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
    clearPendingParamsPersist(
      id,
      new ModelTabsStoreError('tab_not_found', `Tab not found: '${id}'.`, { details: { id } }),
    )
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

    if (!patch || !isPlainRecord(patch)) {
      throw new ModelTabsStoreError(
        'serialization_failure',
        `Failed to serialize params for tab '${id}' (patch): patch must be a plain object record.`,
        { details: { tabId: id, phase: 'patch' } },
      )
    }
    const patchSnapshot = cloneParamsForPersist(id, 'patch', patch as Record<string, unknown>)

    const pending = getPendingParamsPersist(id, t)
    if (pending.snapshotParams === null) {
      pending.snapshotParams = cloneParamsForPersist(id, 'snapshot', current as unknown as Record<string, unknown>)
      pending.snapshotUpdatedAt = t.meta.updatedAt
    }

    Object.assign(current, patchSnapshot)
    t.meta.updatedAt = nowIso()

    pending.version += 1
    const targetVersion = pending.version
    const persistPromise = new Promise<void>((resolve, reject) => {
      pending.deferreds.push({ version: targetVersion, resolve, reject })
    })
    scheduleParamsPersist(id)
    save()
    return persistPromise
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
