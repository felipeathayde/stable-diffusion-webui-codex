import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { fetchTabs, createTabApi, updateTabApi, reorderTabsApi, deleteTabApi } from '../api/client'
import type { ApiTab } from '../api/types'
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
  lightning: boolean
  loraEnabled: boolean
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
  useInitImage: boolean
  initImageData: string
  initImageName: string
}

const STORAGE_KEY = 'codex:model-tabs:v1'

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
      lightning: false,
      loraEnabled: false,
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
    return { high: stage(), low: stage(), video, assets, modelFormat: 'auto' }
  }

  // Image tabs (SD15, SDXL, Flux)
  const config = getEngineConfig(type as EngineType)
  const defaults = getEngineDefaults(type as EngineType)
  const imageDefaults: ImageBaseParams = {
    prompt: '',
    negativePrompt: config.capabilities.usesNegativePrompt ? '' : '',
    width: defaults.width,
    height: defaults.height,
    sampler: '',
    scheduler: '',
    steps: defaults.steps,
    cfgScale: defaults.cfg,
    seed: -1,
    useInitImage: false,
    initImageData: '',
    initImageName: '',
  }
  
  // Add distilledCfg for flow engines
  if (config.capabilities.usesDistilledCfg && defaults.distilledCfg !== undefined) {
    return { ...imageDefaults, distilledCfgScale: defaults.distilledCfg }
  }
  
  return imageDefaults
}

function normalizeTabType(type: unknown): BaseTabType {
  const raw = String(type || '').trim()
  if (!raw) return 'sd15'
  const value = raw.toLowerCase()
  if (value === 'wan22' || value === 'wan22_14b' || value === 'wan22_5b') return 'wan'
  if (value === 'sd15' || value === 'sdxl' || value === 'flux' || value === 'wan') return value as BaseTabType
  // Fail closed: unknown tab types fall back to a safe image tab.
  return 'sd15'
}

function normalizeTab(tab: BaseTab): BaseTab {
  return { ...tab, type: normalizeTabType((tab as any).type) }
}

export const useModelTabsStore = defineStore('modelTabs', () => {
  const tabs = ref<BaseTab[]>([])
  const activeId = ref<string>('')

  function save(): void {
    const payload = { tabs: tabs.value, activeId: activeId.value }
    localStorage.setItem(STORAGE_KEY, JSON.stringify(payload))
  }

  async function load(): Promise<void> {
    // Try backend first
    try {
      const res = await fetchTabs()
      if (res && Array.isArray(res.tabs)) {
        tabs.value = (res.tabs as unknown as BaseTab[]).map(normalizeTab)
        tabs.value.sort((a, b) => a.order - b.order)
        activeId.value = tabs.value[0]?.id ?? ''
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
        activeId.value = parsed.activeId || (tabs.value[0]?.id ?? '')
        tabs.value.sort((a, b) => a.order - b.order)
        return
      } catch { /* ignore */ }
    }
    bootstrap()
  }

  function bootstrap(): void {
    // Create minimal default tabs when backend persistence is unavailable.
    const createdAt = nowIso()
    const types: BaseTabType[] = ['sd15', 'sdxl', 'flux', 'wan']
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
    try { await createTabApi({ type, title: title || defaultTitle, params: defaultParams(type) }) } catch { /* ignore */ }
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
    try { await createTabApi({ type: copy.type as BaseTabType, title: copy.title, params: copy.params }) } catch { /* ignore */ }
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
    t.params = { ...(t.params as T), ...patch }
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
