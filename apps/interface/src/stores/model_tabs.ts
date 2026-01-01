import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { fetchTabs, createTabApi, updateTabApi, reorderTabsApi, deleteTabApi } from '../api/client'
import type { ApiTab } from '../api/types'
import type { HighresFormState, RefinerFormState } from '../api/payloads'
import { type EngineType, getEngineConfig, getEngineDefaults } from './engine_config'

export type BaseTabType = ApiTab['type']

export interface BaseTabMeta {
  createdAt: string
  updatedAt: string
}

export interface WanStageParams {
  modelDir: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  loraPath: string
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
  // Interpolation (RIFE)
  rifeEnabled: boolean
  rifeModel: string
  rifeTimes: number
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
  highres: HighresFormState
  refiner: RefinerFormState
  checkpoint: string
  textEncoders: string[]
  useInitImage: boolean
  initImageData: string
  initImageName: string
  denoiseStrength: number
}

export const MODEL_TABS_STORAGE_KEY = 'codex:model-tabs:v1'
const STORAGE_KEY = MODEL_TABS_STORAGE_KEY

function nowIso(): string {
  return new Date().toISOString()
}

function uuid(): string {
  // Suficiente para ids locais; backend ganhará ids estáveis na Fase 4
  return 'tab-' + Math.random().toString(36).slice(2, 10)
}

function defaultParams(type: BaseTabType): Record<string, unknown> {
  // Video tabs (WAN)
  if (type === 'wan') {
    const stage = (): WanStageParams => ({
      modelDir: '',
      sampler: '',
      scheduler: '',
      steps: 30,
      cfgScale: 7,
      seed: -1,
      loraPath: '',
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
      rifeEnabled: true,
      rifeModel: 'rife47.pth',
      rifeTimes: 2,
    }
    const assets = { metadata: '', textEncoder: '', vae: '' }
    return { high: stage(), low: stage(), video, assets, lightx2v: false, lowFollowsHigh: false }
  }

  // Image tabs (SD15, SDXL, Flux)
  const config = getEngineConfig(type as EngineType)
  const defaults = getEngineDefaults(type as EngineType)
  const guidance = config.capabilities.usesDistilledCfg && defaults.distilledCfg !== undefined ? defaults.distilledCfg : defaults.cfg
  const refinerDefaults: RefinerFormState = {
    enabled: false,
    steps: 0,
    cfg: 3.5,
    seed: -1,
    model: undefined,
    vae: undefined,
  }
  const highresDefaults: HighresFormState = {
    enabled: false,
    denoise: 0.4,
    scale: 1.5,
    resizeX: 0,
    resizeY: 0,
    steps: 0,
    upscaler: 'Use same upscaler',
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
    sampler: '',
    scheduler: '',
    steps: defaults.steps,
    cfgScale: guidance,
    seed: -1,
    clipSkip: type === 'sdxl' ? 2 : 1,
    batchSize: 1,
    batchCount: 1,
    highres: { ...highresDefaults },
    refiner: { ...refinerDefaults },
    checkpoint: '',
    textEncoders: [],
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    denoiseStrength: 0.75,
  }
  return imageDefaults
}

function normalizeTabType(type: unknown): BaseTabType {
  const raw = String(type || '').trim()
  if (!raw) return 'sd15'
  const value = raw.toLowerCase()
  if (value === 'wan22' || value === 'wan22_14b' || value === 'wan22_5b') return 'wan'
  if (value === 'sd15' || value === 'sdxl' || value === 'flux' || value === 'zimage' || value === 'wan') return value as BaseTabType
  // Fail closed: unknown tab types fall back to a safe image tab.
  return 'sd15'
}

function normalizeParamsForType(type: BaseTabType, raw: unknown): Record<string, unknown> {
  const params = (raw && typeof raw === 'object' && !Array.isArray(raw)) ? (raw as Record<string, unknown>) : {}
  const defaults = defaultParams(type)
  if (type === 'wan') {
    const merged: Record<string, unknown> = { ...defaults, ...params }
    const d = defaults as any
    const p = params as any
    merged.high = { ...(d.high || {}), ...(p.high || {}) }
    merged.low = { ...(d.low || {}), ...(p.low || {}) }
    merged.video = { ...(d.video || {}), ...(p.video || {}) }
    merged.assets = { ...(d.assets || {}), ...(p.assets || {}) }
    return merged
  }

  const merged: Record<string, unknown> = { ...defaults, ...params }
  const d = defaults as any
  const p = params as any
  if (d.highres && typeof d.highres === 'object') {
    merged.highres = { ...(d.highres || {}), ...(p.highres || {}) }
    if ((d.highres as any).refiner && typeof (d.highres as any).refiner === 'object') {
      ;(merged.highres as any).refiner = { ...((d.highres as any).refiner || {}), ...((p.highres as any)?.refiner || {}) }
    }
  }
  if (d.refiner && typeof d.refiner === 'object') {
    merged.refiner = { ...(d.refiner || {}), ...(p.refiner || {}) }
  }
  return merged
}

function normalizeTab(tab: BaseTab): BaseTab {
  const type = normalizeTabType((tab as any).type)
  return { ...tab, type, params: normalizeParamsForType(type, (tab as any).params) }
}

export const useModelTabsStore = defineStore('modelTabs', () => {
  const tabs = ref<BaseTab[]>([])
  const activeId = ref<string>('')

  const requiredTypes: BaseTabType[] = ['sd15', 'sdxl', 'flux', 'zimage', 'wan']

  function save(): void {
    const payload = { tabs: tabs.value, activeId: activeId.value }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  }

  async function ensureRequiredTabs(): Promise<void> {
    const existing = new Set<BaseTabType>(tabs.value.map(t => t.type))
    let nextOrder = tabs.value.length ? (Math.max(...tabs.value.map(t => t.order)) + 1) : 0
    const createdAt = nowIso()
    for (const type of requiredTypes) {
      if (existing.has(type)) continue
      const id = uuid()
      const title = type === 'wan' ? 'WAN 2.2' : getEngineConfig(type as EngineType).label
      const params = defaultParams(type)
      tabs.value.push({
        id,
        type,
        title,
        order: nextOrder++,
        enabled: true,
        params,
        meta: { createdAt, updatedAt: createdAt },
      })
      existing.add(type)
      try { await createTabApi({ id, type, title, params }) } catch { /* ignore */ }
    }
  }

  async function load(): Promise<void> {
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

    // Try backend first
    try {
      const res = await fetchTabs()
      if (res && Array.isArray(res.tabs)) {
        tabs.value = (res.tabs as unknown as BaseTab[]).map(normalizeTab)
        tabs.value.sort((a, b) => a.order - b.order)
        activeId.value = (preferredActiveId && tabs.value.some(t => t.id === preferredActiveId)) ? preferredActiveId : (tabs.value[0]?.id ?? '')
        await ensureRequiredTabs()
        tabs.value.sort((a, b) => a.order - b.order)
        if (activeId.value && !tabs.value.some(t => t.id === activeId.value)) activeId.value = tabs.value[0]?.id ?? ''
        save()
        return
      }
    } catch {
      // fallback to local
    }
    const raw = localStorage.getItem(STORAGE_KEY)
    if (raw) {
      try {
        const parsed = JSON.parse(raw) as { tabs: BaseTab[]; activeId: string }
        tabs.value = (parsed.tabs || []).map(normalizeTab)
        const stored = typeof parsed.activeId === 'string' ? parsed.activeId : ''
        const nextActive = activeId.value || stored
        activeId.value = (nextActive && tabs.value.some(t => t.id === nextActive)) ? nextActive : (tabs.value[0]?.id ?? '')
        tabs.value.sort((a, b) => a.order - b.order)
        await ensureRequiredTabs()
        tabs.value.sort((a, b) => a.order - b.order)
        if (activeId.value && !tabs.value.some(t => t.id === activeId.value)) activeId.value = tabs.value[0]?.id ?? ''
        save()
        return
      } catch { /* ignore */ }
    }
    bootstrap()
  }

  function bootstrap(): void {
    // Create minimal default tabs when backend persistence is unavailable.
    const createdAt = nowIso()
    const types: BaseTabType[] = ['sd15', 'sdxl', 'flux', 'zimage', 'wan']
    tabs.value = types.map((type, idx) => ({
      id: uuid(),
      type,
      title: type === 'wan' ? 'WAN 2.2' : getEngineConfig(type as EngineType).label,
      order: idx,
      enabled: true,
      params: defaultParams(type),
      meta: { createdAt, updatedAt: createdAt },
    }))
    activeId.value = tabs.value[0]?.id ?? ''
    save()
  }

  async function create(type: BaseTabType, title?: string): Promise<string> {
    const id = uuid()
    const createdAt = nowIso()
    const defaultTitle = type === 'wan' ? 'WAN 2.2' : getEngineConfig(type as EngineType).label
    const nextOrder = tabs.value.length ? Math.max(...tabs.value.map(t => t.order)) + 1 : 0
    tabs.value.push({
      id,
      type,
      title: title || defaultTitle,
      order: nextOrder,
      enabled: true,
      params: defaultParams(type),
      meta: { createdAt, updatedAt: createdAt },
    })
    try { await createTabApi({ id, type, title: title || defaultTitle, params: defaultParams(type) }) } catch { /* ignore */ }
    save()
    return id
  }

  async function duplicate(id: string): Promise<string> {
    const src = tabs.value.find(t => t.id === id)
    if (!src) return ''
    const copy: BaseTab = JSON.parse(JSON.stringify(src))
    copy.id = uuid()
    copy.title = src.title + ' (copy)'
    copy.order = (Math.max(...tabs.value.map(t => t.order)) || 0) + 1
    copy.meta.createdAt = nowIso()
    copy.meta.updatedAt = copy.meta.createdAt
    tabs.value.push(copy)
    try { await createTabApi({ id: copy.id, type: copy.type as BaseTabType, title: copy.title, params: copy.params }) } catch { /* ignore */ }
    save()
    return copy.id
  }

  async function remove(id: string): Promise<void> {
    tabs.value = tabs.value.filter(t => t.id !== id)
    if (activeId.value === id) activeId.value = tabs.value[0]?.id ?? ''
    normalizeOrder()
    try { await deleteTabApi(id) } catch { /* ignore */ }
    save()
  }

  async function rename(id: string, title: string): Promise<void> {
    const t = tabs.value.find(x => x.id === id)
    if (!t) return
    t.title = title
    t.meta.updatedAt = nowIso()
    try { await updateTabApi(id, { title }) } catch { /* ignore */ }
    save()
  }

  async function setEnabled(id: string, value: boolean): Promise<void> {
    const t = tabs.value.find(x => x.id === id)
    if (!t) return
    t.enabled = value
    t.meta.updatedAt = nowIso()
    try { await updateTabApi(id, { enabled: value }) } catch { /* ignore */ }
    save()
  }

  async function reorder(ids: string[]): Promise<void> {
    const map = new Map<string, number>()
    ids.forEach((id, idx) => map.set(id, idx))
    tabs.value.forEach(t => { t.order = map.get(t.id) ?? t.order })
    tabs.value.sort((a, b) => a.order - b.order)
    try { await reorderTabsApi(ids) } catch { /* ignore */ }
    save()
  }

  function setActive(id: string): void { activeId.value = id; save() }

  async function updateParams<T extends Record<string, unknown>>(id: string, patch: Partial<T>): Promise<void> {
    const t = tabs.value.find(x => x.id === id)
    if (!t) return
    const current = (t.params && typeof t.params === 'object' && !Array.isArray(t.params)) ? (t.params as T) : ({} as T)
    if (current !== t.params) t.params = current
    Object.assign(current, patch)
    t.meta.updatedAt = nowIso()
    try { await updateTabApi(id, { params: t.params }) } catch { /* ignore */ }
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
