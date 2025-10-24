import type {
  ModelsResponse,
  SamplersResponse,
  SchedulersResponse,
  OptionsResponse,
  OptionsUpdateResponse,
  Txt2ImgStartResponse,
  TaskResult,
  TaskEvent,
  VaesResponse,
  TextEncodersResponse,
  MemoryResponse,
  VersionResponse,
  EmbeddingsResponse,
  LoraListResponse,
  PathsResponse,
  PathsUpdateResponse,
  SettingsSchemaResponse,
  UiBlocksResponse,
  UiPresetsResponse,
  UiPresetApplyResponse,
} from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `${res.status} ${res.statusText}`)
  }
  return (await res.json()) as T
}

export function fetchModels(): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>('/models')
}

export function fetchSamplers(): Promise<SamplersResponse> {
  return requestJson<SamplersResponse>('/samplers')
}

export function fetchSchedulers(): Promise<SchedulersResponse> {
  return requestJson<SchedulersResponse>('/schedulers')
}

export function fetchVaes(): Promise<VaesResponse> {
  return requestJson<VaesResponse>('/vaes')
}

export function fetchTextEncoders(): Promise<TextEncodersResponse> {
  return requestJson<TextEncodersResponse>('/text-encoders')
}

export function fetchOptions(): Promise<OptionsResponse> {
  return requestJson<OptionsResponse>('/options')
}

export function updateOptions(payload: Record<string, unknown>): Promise<OptionsUpdateResponse> {
  return requestJson<OptionsUpdateResponse>('/options', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startTxt2Img(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/txt2img', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startImg2Img(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/img2img', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startTxt2Vid(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/txt2vid', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startImg2Vid(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/img2vid', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchTaskResult(taskId: string): Promise<TaskResult> {
  return requestJson<TaskResult>(`/tasks/${taskId}`)
}

export function subscribeTask(taskId: string, onEvent: (event: TaskEvent) => void): () => void {
  const es = new EventSource(`${API_BASE}/tasks/${taskId}/events`)
  es.onmessage = (msg: MessageEvent<string>) => {
    try {
      const payload = JSON.parse(msg.data) as TaskEvent
      onEvent(payload)
    } catch (error) {
      console.error('[task-events] failed to parse event', error)
    }
  }
  es.onerror = (err) => {
    console.error('[task-events] stream error', err)
  }
  return () => es.close()
}

export function fetchMemory(): Promise<MemoryResponse> {
  return requestJson<MemoryResponse>('/memory')
}

export function fetchVersion(): Promise<VersionResponse> {
  return requestJson<VersionResponse>('/version')
}

export function fetchEmbeddings(): Promise<EmbeddingsResponse> {
  return requestJson<EmbeddingsResponse>('/embeddings')
}

export function fetchLoras(): Promise<LoraListResponse> {
  return requestJson<LoraListResponse>('/loras')
}

export function fetchPaths(): Promise<PathsResponse> {
  return requestJson<PathsResponse>('/paths')
}

export function updatePaths(paths: Record<string, string[]>): Promise<PathsUpdateResponse> {
  return requestJson<PathsUpdateResponse>('/paths', { method: 'POST', body: JSON.stringify({ paths }) })
}

export function fetchSettingsSchema(): Promise<SettingsSchemaResponse> {
  return requestJson<SettingsSchemaResponse>('/settings/schema').catch(async (err) => {
    console.warn('[api] /settings/schema not available, falling back to static /settings_schema.json', err)
    const res = await fetch('/settings_schema.json')
    if (!res.ok) throw err
    return (await res.json()) as SettingsSchemaResponse
  })
}

export function fetchUiBlocks(tab?: string): Promise<UiBlocksResponse> {
  const q = tab ? `?tab=${encodeURIComponent(tab)}` : ''
  return requestJson<UiBlocksResponse>(`/ui/blocks${q}`)
}

export function fetchUiPresets(tab?: string): Promise<UiPresetsResponse> {
  const q = tab ? `?tab=${encodeURIComponent(tab)}` : ''
  return requestJson<UiPresetsResponse>(`/ui/presets${q}`)
}

export function applyUiPreset(id: string, tab: string): Promise<UiPresetApplyResponse> {
  return requestJson<UiPresetApplyResponse>('/ui/presets/apply', {
    method: 'POST',
    body: JSON.stringify({ id, tab }),
  })
}

// Tabs/workflows persistence
import type { TabsResponse, ApiTab, WorkflowsResponse } from './types'

export function fetchTabs(): Promise<TabsResponse> {
  return requestJson<TabsResponse>('/ui/tabs')
}

export function createTabApi(payload: Partial<ApiTab> & { type: ApiTab['type']; title?: string; params?: Record<string, unknown> }): Promise<{ id: string }> {
  return requestJson<{ id: string }>('/ui/tabs', { method: 'POST', body: JSON.stringify(payload) })
}

export function updateTabApi(tabId: string, payload: Partial<Pick<ApiTab, 'title' | 'enabled' | 'params'>>): Promise<{ updated: string }> {
  return requestJson<{ updated: string }>(`/ui/tabs/${encodeURIComponent(tabId)}`, { method: 'PATCH', body: JSON.stringify(payload) })
}

export function reorderTabsApi(ids: string[]): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>('/ui/tabs/reorder', { method: 'POST', body: JSON.stringify({ ids }) })
}

export function deleteTabApi(tabId: string): Promise<{ deleted: string }> {
  return requestJson<{ deleted: string }>(`/ui/tabs/${encodeURIComponent(tabId)}`, { method: 'DELETE' })
}

export function fetchWorkflows(): Promise<WorkflowsResponse> {
  return requestJson<WorkflowsResponse>('/ui/workflows')
}

export function createWorkflow(payload: { name: string; source_tab_id: string; type: string; engine_semantics?: string; params_snapshot: Record<string, unknown> }): Promise<{ id: string }> {
  return requestJson<{ id: string }>('/ui/workflows', { method: 'POST', body: JSON.stringify(payload) })
}

export function deleteWorkflow(id: string): Promise<{ deleted: string }> {
  return requestJson<{ deleted: string }>(`/ui/workflows/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

export function loadModelsForTab(tabId: string): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>('/models/load', { method: 'POST', body: JSON.stringify({ tab_id: tabId }) })
}

export function unloadModelsForTab(tabId: string): Promise<{ ok: boolean }> {
  return requestJson<{ ok: boolean }>('/models/unload', { method: 'POST', body: JSON.stringify({ tab_id: tabId }) })
}
