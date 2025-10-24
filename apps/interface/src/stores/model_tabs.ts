import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import { fetchTabs, createTabApi, updateTabApi, reorderTabsApi, deleteTabApi } from '../api/client'

export type BaseTabType = 'sd15' | 'sdxl' | 'flux' | 'wan'

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
  if (type === 'wan') {
    const stage = (): WanStageParams => ({
      modelDir: '', sampler: '', scheduler: '', steps: 30, cfgScale: 7, seed: -1,
      lightning: false, loraEnabled: false, loraPath: '', loraWeight: 1.0,
    })
    const video: WanVideoParams = {
      prompt: '', negativePrompt: '', width: 768, height: 432, fps: 24, frames: 16,
      useInitImage: false, initImageData: '', initImageName: '',
      filenamePrefix: 'wan22', format: 'video/h264-mp4', pixFmt: 'yuv420p', crf: 15,
      loopCount: 0, pingpong: false, trimToAudio: false, saveMetadata: true, saveOutput: true,
      rifeEnabled: true, rifeModel: 'rife47.pth', rifeTimes: 2,
    }
    return { high: stage(), low: stage(), video }
  }
  // SD15/SDXL/FLUX defaults for image generation
  const imageDefaults: ImageBaseParams = {
    prompt: '', negativePrompt: '', width: 1024, height: 1024,
    sampler: '', scheduler: '', steps: 30, cfgScale: 7, seed: -1,
    useInitImage: false, initImageData: '', initImageName: '',
  }
  return imageDefaults
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
        tabs.value = res.tabs as unknown as BaseTab[]
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
        tabs.value = parsed.tabs || []
        activeId.value = parsed.activeId || (tabs.value[0]?.id ?? '')
        tabs.value.sort((a, b) => a.order - b.order)
        return
      } catch { /* ignore */ }
    }
    bootstrap()
  }

  function bootstrap(): void {
    // Cria 4 bases padrão: sd15, sdxl, flux, wan
    const createdAt = nowIso()
    const list: Array<{ type: BaseTabType; title: string }> = [
      { type: 'sd15', title: 'SD 1.5' },
      { type: 'sdxl', title: 'SDXL' },
      { type: 'flux', title: 'FLUX' },
      { type: 'wan', title: 'WAN 2.2' },
    ]
    tabs.value = list.map((it, idx) => ({
      id: uuid(), type: it.type, title: it.title, order: idx, enabled: true,
      params: defaultParams(it.type), meta: { createdAt, updatedAt: createdAt },
    }))
    activeId.value = tabs.value[0]?.id ?? ''
    save()
  }

  async function create(type: BaseTabType, title?: string): Promise<string> {
    const id = uuid()
    const createdAt = nowIso()
    const nextOrder = tabs.value.length ? Math.max(...tabs.value.map(t => t.order)) + 1 : 0
    tabs.value.push({
      id,
      type,
      title: title || type.toUpperCase(),
      order: nextOrder,
      enabled: true,
      params: defaultParams(type),
      meta: { createdAt, updatedAt: createdAt },
    })
    try { await createTabApi({ type, title: title || type.toUpperCase(), params: defaultParams(type) }) } catch { /* ignore */ }
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
