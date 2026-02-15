/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend API client (typed fetch helpers + endpoint wrappers).
Provides JSON/Form fetch helpers and exports functions for models/options/inventory/tasks, UI tabs/workflows persistence, and UI schema/preset
endpoints under `VITE_API_BASE` (default `/api`). Also caches `/api/options` revision and preserves structured HTTP error metadata (`status/detail/body`)
for conflict-aware generation UX.
Task SSE subscriptions support resume via `after=<event_id>` and expose the latest `lastEventId` for reconnect/replay persistence.

Symbols (top-level; keep in sync; no ghosts):
- `API_BASE` (const): Base URL prefix for backend endpoints (from Vite env, default `/api`).
- `requestJson` (function): JSON request helper with consistent error handling.
- `requestForm` (function): Form POST helper for multipart endpoints.
- `readErrorDetail` (function): Extracts structured error detail (`message/detail/body`) from failed backend responses.
- `getApiErrorStatus` (function): Reads an HTTP status code from request errors emitted by this client.
- `getCurrentRevisionFromError` (function): Extracts `current_revision` from backend conflict errors (`409`) when present.
- `getCachedOptionsRevision` (function): Returns the cached `/api/options` revision used by generation payload builders.
- `fetchModels` (function): Fetches the model list (`/models`).
- `refreshModels` (function): Forces a checkpoint rescan (`/models?refresh=1`).
- `fetchModelInventory` (function): Fetches the inventory cache (`/models/inventory`).
- `fetchFileMetadata` (function): Reads GGUF/SafeTensors file metadata (`/models/file-metadata`).
- `fetchCheckpointMetadata` (function): Fetches the metadata modal payload for a checkpoint selection (`/models/checkpoint-metadata`).
- `refreshModelInventory` (function): Forces an inventory rescan (`/models/inventory/refresh`).
- `fetchSamplers` (function): Fetches supported samplers (`/samplers`) and filters out unsupported entries.
- `fetchSchedulers` (function): Fetches supported schedulers (`/schedulers`) and filters out unsupported entries.
- `analyzePngInfo` (function): Extracts PNG text metadata for the PNG Info view (`POST /tools/pnginfo/analyze` multipart).
- `fetchOptions` (function): Fetches runtime options (`/options`).
- `updateOptions` (function): Updates runtime options (`POST /options`).
- `startTxt2Img` (function): Starts a txt2img task (`POST /txt2img`).
- `startImg2Img` (function): Starts an img2img task (`POST /img2img`).
- `startTxt2Vid` (function): Starts a txt2vid task (`POST /txt2vid`).
- `startImg2Vid` (function): Starts an img2vid task (`POST /img2vid`).
- `startVid2Vid` (function): Starts a vid2vid task (`POST /vid2vid` multipart).
- `fetchUpscalers` (function): Fetches the local upscalers list (`/upscalers`).
- `refreshUpscalers` (function): Forces an upscalers re-fetch (clears cache, then calls `/upscalers`).
- `fetchRemoteUpscalers` (function): Fetches curated HF upscalers list (`/upscalers/remote`).
- `downloadUpscalers` (function): Starts an upscalers download task (`POST /upscalers/download`).
- `startUpscale` (function): Starts a standalone upscale task (`POST /upscale` multipart).
- `fetchTaskResult` (function): Fetches a task result (`/tasks/:id`).
- `cancelTask` (function): Requests task cancellation (`/tasks/:id/cancel`).
- `subscribeTask` (function): Subscribes to task SSE events and returns an unsubscribe closure.
- `fetchMemory` (function): Fetches memory stats (`/memory`).
- `fetchVersion` (function): Fetches backend version (`/version`).
- `fetchEmbeddings` (function): Fetches embeddings list (`/embeddings`).
- `fetchEngineCapabilities` (function): Fetches engine capabilities (`/engines/capabilities`).
- `fetchPromptTokenCount` (function): Counts prompt tokens via backend tokenizer (`POST /models/prompt-token-count`).
- `fetchPaths` (function): Fetches configured paths (`/paths`).
- `updatePaths` (function): Updates configured paths (`POST /paths`).
- `fetchSettingsSchema` (function): Fetches settings schema (`/settings/schema`).
- `fetchUiBlocks` (function): Fetches UI blocks schema (`/ui/blocks`).
- `fetchUiPresets` (function): Fetches UI presets (`/ui/presets`).
- `applyUiPreset` (function): Applies a UI preset (`POST /ui/presets/apply`).
- `fetchTabs` (function): Fetches persisted tabs (`/ui/tabs`).
- `createTabApi` (function): Creates a tab (`POST /ui/tabs`).
- `updateTabApi` (function): Updates a tab (`PATCH /ui/tabs/:id`).
- `reorderTabsApi` (function): Reorders tabs (`POST /ui/tabs/reorder`).
- `deleteTabApi` (function): Deletes a tab (`DELETE /ui/tabs/:id`).
- `fetchWorkflows` (function): Fetches workflows (`/ui/workflows`).
- `createWorkflow` (function): Creates a workflow (`POST /ui/workflows`).
- `deleteWorkflow` (function): Deletes a workflow (`DELETE /ui/workflows/:id`).
- `loadModelsForTab` (function): Loads models for a tab (`POST /models/load`).
- `unloadModelsForTab` (function): Unloads models for a tab (`POST /models/unload`).
*/

import type {
  ModelsResponse,
  SamplersResponse,
  SchedulersResponse,
  OptionsResponse,
  OptionsUpdateResponse,
  Txt2ImgStartResponse,
  TaskStartResponse,
  TaskResult,
  TaskEvent,
  MemoryResponse,
  VersionResponse,
  EmbeddingsResponse,
  PathsResponse,
  PathsUpdateResponse,
  SettingsSchemaResponse,
  UiBlocksResponse,
  UiPresetsResponse,
  UiPresetApplyResponse,
  InventoryResponse,
  EngineCapabilitiesResponse,
  PromptTokenCountRequest,
  PromptTokenCountResponse,
  FileMetadataResponse,
  CheckpointMetadataResponse,
  PngInfoAnalyzeResponse,
  UpscalersResponse,
  RemoteUpscalersResponse,
} from './types'
import type { Txt2ImgRequest } from './payloads'

const API_BASE = import.meta.env.VITE_API_BASE ?? '/api'

const _jsonCache = new Map<string, unknown>()
const _jsonInflight = new Map<string, Promise<unknown>>()
let _cachedOptionsRevision = 0

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function normalizeRevision(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value))
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    if (/^-?\d+$/.test(trimmed)) {
      return Math.max(0, Math.trunc(Number(trimmed)))
    }
  }
  return null
}

function cacheOptionsRevisionFromPayload(payload: unknown): void {
  if (!isRecordObject(payload)) return
  const direct = normalizeRevision(payload.revision)
  if (direct !== null) {
    _cachedOptionsRevision = direct
    return
  }
  const values = payload.values
  if (!isRecordObject(values)) return
  const fromValues = normalizeRevision(values.codex_options_revision)
  if (fromValues !== null) _cachedOptionsRevision = fromValues
}

export function getCachedOptionsRevision(): number {
  return _cachedOptionsRevision
}

function readCurrentRevision(value: unknown, depth = 0): number | null {
  if (depth > 5) return null
  if (value === null || value === undefined) return null

  if (typeof value === 'string') {
    const match = value.match(/current[_\s-]?revision[^0-9-]*(-?\d+)/i)
    if (!match) return null
    return normalizeRevision(match[1])
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      const found = readCurrentRevision(item, depth + 1)
      if (found !== null) return found
    }
    return null
  }

  if (!isRecordObject(value)) return null

  for (const key of ['current_revision', 'currentRevision'] as const) {
    const found = normalizeRevision(value[key])
    if (found !== null) return found
  }

  for (const nested of Object.values(value)) {
    const found = readCurrentRevision(nested, depth + 1)
    if (found !== null) return found
  }
  return null
}

export function getApiErrorStatus(error: unknown): number | null {
  if (!isRecordObject(error)) return null
  return normalizeRevision(error.status)
}

export function getCurrentRevisionFromError(error: unknown): number | null {
  if (error instanceof Error) {
    const fromMessage = readCurrentRevision(error.message)
    if (fromMessage !== null) return fromMessage
  }
  return readCurrentRevision(error)
}

function detailToMessage(detail: unknown): string {
  if (typeof detail === 'string' && detail.trim()) return detail.trim()
  if (Array.isArray(detail)) {
    const msgs = detail
      .map((item) => {
        const msg = (item && typeof item === 'object') ? (item as any).msg : null
        return typeof msg === 'string' ? msg : String(item)
      })
      .filter((s) => String(s || '').trim())
    if (msgs.length) return msgs.join('\n')
  }
  if (detail !== undefined) return JSON.stringify(detail)
  return ''
}

async function readErrorDetail(res: Response): Promise<{ message: string; detail: unknown; body: unknown }> {
  const text = await res.text()
  if (!text) return { message: '', detail: null, body: null }
  try {
    const data = JSON.parse(text) as unknown
    if (isRecordObject(data)) {
      const detail = data.detail
      return { message: detailToMessage(detail), detail, body: data }
    }
    return { message: text, detail: null, body: data }
  } catch {
    // not JSON; fall through
  }
  return { message: text, detail: null, body: null }
}

function invalidateJsonCache(prefixPath: string): void {
  for (const key of Array.from(_jsonCache.keys())) {
    if (key === prefixPath || key.startsWith(`${prefixPath}?`)) _jsonCache.delete(key)
  }
  for (const key of Array.from(_jsonInflight.keys())) {
    if (key === prefixPath || key.startsWith(`${prefixPath}?`)) _jsonInflight.delete(key)
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: {
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    ...init,
  })
  if (!res.ok) {
    const detail = await readErrorDetail(res)
    const err = new Error(detail.message || `HTTP ${res.status} ${res.statusText}`) as Error & {
      status?: number
      detail?: unknown
      body?: unknown
    }
    err.status = res.status
    err.detail = detail.detail
    err.body = detail.body
    throw err
  }
  return (await res.json()) as T
}

function requestJsonCached<T>(path: string): Promise<T> {
  const cached = _jsonCache.get(path)
  if (cached !== undefined) return Promise.resolve(cached as T)

  const inflight = _jsonInflight.get(path)
  if (inflight) return inflight as Promise<T>

  const p = requestJson<T>(path)
    .then((value) => {
      _jsonCache.set(path, value)
      return value
    })
    .finally(() => {
      _jsonInflight.delete(path)
    })
  _jsonInflight.set(path, p as Promise<unknown>)
  return p
}

async function requestForm<T>(path: string, form: FormData): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) {
    const detail = await readErrorDetail(res)
    const err = new Error(detail.message || `HTTP ${res.status} ${res.statusText}`) as Error & {
      status?: number
      detail?: unknown
      body?: unknown
    }
    err.status = res.status
    err.detail = detail.detail
    err.body = detail.body
    throw err
  }
  return (await res.json()) as T
}

function withSettingsRevision(payload: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = { ...payload }
  const existing = normalizeRevision(out.settings_revision)
  out.settings_revision = existing ?? getCachedOptionsRevision()
  return out
}

export function fetchModels(): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>('/models')
}

export function refreshModels(): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>('/models?refresh=1')
}

export function fetchModelInventory(): Promise<InventoryResponse> {
  return requestJsonCached<InventoryResponse>('/models/inventory')
}

export function fetchFileMetadata(path: string): Promise<FileMetadataResponse> {
  return requestJson<FileMetadataResponse>(`/models/file-metadata?path=${encodeURIComponent(path)}`)
}

export function fetchCheckpointMetadata(value: string): Promise<CheckpointMetadataResponse> {
  return requestJson<CheckpointMetadataResponse>(`/models/checkpoint-metadata?value=${encodeURIComponent(value)}`)
}

export async function refreshModelInventory(): Promise<InventoryResponse> {
  invalidateJsonCache('/models/inventory')
  const inv = await requestJson<InventoryResponse>('/models/inventory/refresh', { method: 'POST' })
  _jsonCache.set('/models/inventory', inv)
  return inv
}

export async function fetchSamplers(): Promise<SamplersResponse> {
  const res = await requestJsonCached<SamplersResponse>('/samplers')
  const supported = res.samplers.filter((sampler) => sampler.supported !== false)
  return { samplers: supported }
}

export async function fetchSchedulers(): Promise<SchedulersResponse> {
  const res = await requestJsonCached<SchedulersResponse>('/schedulers')
  const supported = res.schedulers.filter((scheduler) => scheduler.supported !== false)
  return { schedulers: supported }
}

export function analyzePngInfo(file: File): Promise<PngInfoAnalyzeResponse> {
  const form = new FormData()
  form.append('file', file)
  return requestForm<PngInfoAnalyzeResponse>('/tools/pnginfo/analyze', form)
}

export async function fetchOptions(): Promise<OptionsResponse> {
  const res = await requestJson<OptionsResponse>('/options')
  cacheOptionsRevisionFromPayload(res)
  return res
}

export async function updateOptions(payload: Record<string, unknown>): Promise<OptionsUpdateResponse> {
  const res = await requestJson<OptionsUpdateResponse>('/options', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
  cacheOptionsRevisionFromPayload(res)
  return res
}

export function startTxt2Img(payload: Txt2ImgRequest): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/txt2img', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function startImg2Img(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/img2img', {
    method: 'POST',
    body: JSON.stringify(withSettingsRevision(payload)),
  })
}

export function startTxt2Vid(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/txt2vid', {
    method: 'POST',
    body: JSON.stringify(withSettingsRevision(payload)),
  })
}

export function startImg2Vid(payload: Record<string, unknown>): Promise<Txt2ImgStartResponse> {
  return requestJson<Txt2ImgStartResponse>('/img2vid', {
    method: 'POST',
    body: JSON.stringify(withSettingsRevision(payload)),
  })
}

export function startVid2Vid(form: FormData): Promise<Txt2ImgStartResponse> {
  return requestForm<Txt2ImgStartResponse>('/vid2vid', form)
}

export function fetchUpscalers(): Promise<UpscalersResponse> {
  return requestJsonCached<UpscalersResponse>('/upscalers')
}

export async function refreshUpscalers(): Promise<UpscalersResponse> {
  invalidateJsonCache('/upscalers')
  const res = await requestJson<UpscalersResponse>('/upscalers')
  _jsonCache.set('/upscalers', res)
  return res
}

export function fetchRemoteUpscalers(opts: { repo_id?: string; revision?: string } = {}): Promise<RemoteUpscalersResponse> {
  const params = new URLSearchParams()
  if (typeof opts.repo_id === 'string' && opts.repo_id.trim()) params.set('repo_id', opts.repo_id.trim())
  if (typeof opts.revision === 'string' && opts.revision.trim()) params.set('revision', opts.revision.trim())
  const q = params.toString()
  return requestJson<RemoteUpscalersResponse>(`/upscalers/remote${q ? `?${q}` : ''}`)
}

export function downloadUpscalers(payload: { repo_id?: string; revision?: string | null; files: string[] }): Promise<TaskStartResponse> {
  return requestJson<TaskStartResponse>('/upscalers/download', { method: 'POST', body: JSON.stringify(payload) })
}

export function startUpscale(image: File, payload: Record<string, unknown>): Promise<TaskStartResponse> {
  const form = new FormData()
  form.append('image', image)
  form.append('payload', JSON.stringify(payload))
  return requestForm<TaskStartResponse>('/upscale', form)
}

export function fetchTaskResult(taskId: string): Promise<TaskResult> {
  return requestJson<TaskResult>(`/tasks/${taskId}`)
}

export function cancelTask(taskId: string, mode: 'immediate' | 'after_current' = 'immediate'): Promise<{ status: string; mode: string }> {
  return requestJson<{ status: string; mode: string }>(`/tasks/${encodeURIComponent(taskId)}/cancel`, {
    method: 'POST',
    body: JSON.stringify({ mode }),
  })
}

export function subscribeTask(
  taskId: string,
  onEvent: (event: TaskEvent) => void,
  onError?: (err: unknown) => void,
  opts: { after?: number; onMeta?: (meta: { eventId?: number }) => void } = {},
): () => void {
  const params = new URLSearchParams()
  if (typeof opts.after === 'number' && Number.isFinite(opts.after) && opts.after > 0) {
    params.set('after', String(Math.trunc(opts.after)))
  }
  const q = params.toString()
  const es = new EventSource(`${API_BASE}/tasks/${taskId}/events${q ? `?${q}` : ''}`)
  let ended = false
  es.onmessage = (msg: MessageEvent<string>) => {
    try {
      const payload = JSON.parse(msg.data) as TaskEvent
      const idRaw = (msg as any).lastEventId
      const id = typeof idRaw === 'string' && idRaw.trim() ? Number(idRaw) : null
      if (id !== null && Number.isFinite(id)) {
        try { opts.onMeta?.({ eventId: Math.trunc(id) }) } catch (_) { /* ignore */ }
      }
      // Mark graceful end so we don’t log a browser “error” on normal close
      if ((payload as any)?.type === 'end') {
        ended = true
        // Let consumers receive the end event before closing
        onEvent(payload)
        es.close()
        return
      }
      onEvent(payload)
    } catch (error) {
      console.error('[task-events] failed to parse event', error)
    }
  }
  es.onerror = (err) => {
    // EventSource fires onerror on normal close; suppress noisy logs when ended or closed
    if (ended || (es as any).readyState === 2 /* CLOSED */) return
    console.error('[task-events] stream error', err)
    try { onError?.(err) } catch (_) { /* ignore */ }
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

export function fetchEngineCapabilities(): Promise<EngineCapabilitiesResponse> {
  return requestJson<EngineCapabilitiesResponse>('/engines/capabilities')
}

export function fetchPromptTokenCount(payload: PromptTokenCountRequest): Promise<PromptTokenCountResponse> {
  return requestJson<PromptTokenCountResponse>('/models/prompt-token-count', {
    method: 'POST',
    body: JSON.stringify(payload),
  })
}

export function fetchPaths(): Promise<PathsResponse> {
  return requestJsonCached<PathsResponse>('/paths')
}

export function updatePaths(paths: Record<string, string[]>): Promise<PathsUpdateResponse> {
  invalidateJsonCache('/paths')
  // Inventory resolution depends on roots; clear the cached snapshot so callers can re-fetch.
  invalidateJsonCache('/models/inventory')
  return requestJson<PathsUpdateResponse>('/paths', { method: 'POST', body: JSON.stringify({ paths }) })
}

export function fetchSettingsSchema(): Promise<SettingsSchemaResponse> {
  return requestJson<SettingsSchemaResponse>('/settings/schema')
}

export function fetchUiBlocks(tab?: string): Promise<UiBlocksResponse> {
  const q = tab ? `?tab=${encodeURIComponent(tab)}` : ''
  return requestJsonCached<UiBlocksResponse>(`/ui/blocks${q}`)
}

export function fetchUiPresets(tab?: string): Promise<UiPresetsResponse> {
  const q = tab ? `?tab=${encodeURIComponent(tab)}` : ''
  return requestJsonCached<UiPresetsResponse>(`/ui/presets${q}`)
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
  return requestJsonCached<TabsResponse>('/ui/tabs')
}

export function createTabApi(payload: Partial<ApiTab> & { type: ApiTab['type']; title?: string; params?: Record<string, unknown> }): Promise<{ id: string }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ id: string }>('/ui/tabs', { method: 'POST', body: JSON.stringify(payload) })
}

export function updateTabApi(tabId: string, payload: Partial<Pick<ApiTab, 'title' | 'enabled' | 'params'>>): Promise<{ updated: string }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ updated: string }>(`/ui/tabs/${encodeURIComponent(tabId)}`, { method: 'PATCH', body: JSON.stringify(payload) })
}

export function reorderTabsApi(ids: string[]): Promise<{ ok: boolean }> {
  invalidateJsonCache('/ui/tabs')
  return requestJson<{ ok: boolean }>('/ui/tabs/reorder', { method: 'POST', body: JSON.stringify({ ids }) })
}

export function deleteTabApi(tabId: string): Promise<{ deleted: string }> {
  invalidateJsonCache('/ui/tabs')
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
