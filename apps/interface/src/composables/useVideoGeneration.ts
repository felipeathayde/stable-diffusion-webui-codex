/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

	Purpose: Unified video generation composable for WAN (txt2vid/img2vid/vid2vid).
	Owns per-tab video generation state (progress/frames/video result/history/queue), builds typed WAN payloads, starts tasks, and consumes task SSE events
	to update UI state and fetch final results. Every start payload includes `settings_revision`, and stale-revision conflicts (`409` + `current_revision`)
	trigger revision refresh + manual-retry UX. Persists a minimal resume marker to `localStorage` and auto-reattaches to in-flight tasks after reload
	via SSE replay (`after` / `lastEventId`) and snapshot refresh on `gap`. Includes `output.returnFrames` and stage `flowShift` pass-through in common WAN payload input.

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): Video generation status state (`idle|running|error|done`).
- `VideoMode` (type): Supported video modes (`txt2vid|img2vid|vid2vid`).
- `VideoRunStatus` (type): Terminal status for history entries (`completed|error|cancelled`).
- `VideoRunHistoryItem` (interface): Persisted run history entry (task id, status, summary, params snapshot, error message).
- `PreparedWanRun` (type): Mode-discriminated prepared run payload (`txt2vid|img2vid|vid2vid`) with typed WAN request bodies.
- `VideoQueuedRun` (type): Queued run entry (`PreparedWanRun` + `id`) to support sequential submissions.
- `VideoProgressState` (interface): Progress payload shape (stage/percent/eta/step/totalSteps).
- `VideoGenerationState` (interface): Per-tab runtime state (status/progress/frames/video result/history/queue/cancel flags).
- `ResumeState` (type): Persisted task resume marker (task id + last event id + params snapshot).
- `freshState` (function): Creates a new `VideoGenerationState` with empty progress/history/queue.
- `getTabState` (function): Returns (and initializes) the per-tab `VideoGenerationState` for a given tab id.
- `defaultStage` (function): Creates default `WanStageParams` for UI state initialization.
- `defaultVideo` (function): Creates default `WanVideoParams` for UI state initialization.
- `defaultAssets` (function): Creates default `WanAssetsParams`.
- `resumeKey` (function): Returns the localStorage key used to persist a WAN task resume marker for a tab.
- `isRecordObject` (function): Type guard for plain-object payloads used in resume-state parsing.
- `assertRunPayloadObject` (function): Runtime invariant guard that fails loud when a prepared run carries a non-object payload.
- `assertNeverMode` (function): Exhaustiveness guard for prepared-run dispatch by `mode`.
- `useVideoGeneration` (function): Main composable API; wires payload building, task start/cancel, SSE handling, queued runs, and history updates
  (contains nested handlers for events, queue progression, and per-mode payload assembly).
*/

import { computed, ref } from 'vue'

import { cancelTask, fetchTaskResult, startImg2Vid, startTxt2Vid, startVid2Vid, subscribeTask } from '../api/client'
import { formatZodError } from '../api/payloads'
import {
  buildWanImg2VidPayload,
  buildWanTxt2VidPayload,
  buildWanVid2VidPayload,
  type WanImg2VidPayload,
  type WanTxt2VidPayload,
  type WanVideoCommonInput,
  type WanVid2VidPayload,
  type WanVid2VidInput,
} from '../api/payloads_video'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { useModelTabsStore, type TabByType, type WanAssetsParams, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'

type Status = 'idle' | 'running' | 'error' | 'done'
type VideoMode = 'txt2vid' | 'img2vid' | 'vid2vid'
type VideoRunStatus = 'completed' | 'error' | 'cancelled'

export interface VideoRunHistoryItem {
  taskId: string
  mode: VideoMode
  createdAtMs: number
  status: VideoRunStatus
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
  errorMessage?: string
}

export type PreparedWanRun =
  | {
      mode: 'txt2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      payload: WanTxt2VidPayload
    }
  | {
      mode: 'img2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      payload: WanImg2VidPayload
    }
  | {
      mode: 'vid2vid'
      createdAtMs: number
      summary: string
      promptPreview: string
      paramsSnapshot: Record<string, unknown>
      payload: WanVid2VidPayload
      file?: File | null
    }

export type VideoQueuedRun = PreparedWanRun & {
  id: string
}

function assertRunPayloadObject(payload: unknown, mode: VideoMode): asserts payload is Record<string, unknown> {
  if (!isRecordObject(payload)) {
    throw new Error(`useVideoGeneration: invalid payload for mode '${mode}'.`)
  }
}

function assertNeverMode(mode: never): never {
  throw new Error(`useVideoGeneration: unsupported prepared run mode '${String(mode)}'.`)
}

export interface VideoProgressState {
  stage: string
  percent: number | null
  etaSeconds: number | null
  step: number | null
  totalSteps: number | null
}

export interface VideoGenerationState {
  status: Status
  progress: VideoProgressState
  frames: GeneratedImage[]
  info: unknown | null
  video: { rel_path?: string | null; mime?: string | null } | null
  errorMessage: string
  taskId: string
  cancelRequested: boolean
  currentRun: VideoRunHistoryItem | null
  history: VideoRunHistoryItem[]
  selectedTaskId: string
  historyLoadingTaskId: string
  queue: VideoQueuedRun[]
}

const DEFAULT_PROGRESS: VideoProgressState = { stage: 'idle', percent: null, etaSeconds: null, step: null, totalSteps: null }
const MAX_HISTORY = 8
const MAX_QUEUE = 3

// Per-tab generation state (keyed by tab ID)
const tabStates = new Map<string, VideoGenerationState>()
const unsubscribers = new Map<string, () => void>()
const initVideos = new Map<string, File | null>()
const resumeAttempts = new Set<string>()
const resumeToastShown = new Set<string>()

type ResumeState = {
  taskId: string
  lastEventId: number
  createdAtMs: number
  mode: VideoMode
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
}

function resumeKey(tabId: string): string {
  return `codex.resume.wan.${tabId}`
}

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function loadResumeState(key: string): ResumeState | null {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (!isRecordObject(parsed)) return null
    if (typeof parsed.taskId !== 'string' || !parsed.taskId.trim()) return null
    const taskId = String(parsed.taskId).trim()

    const lastEventId = typeof parsed.lastEventId === 'number' && Number.isFinite(parsed.lastEventId) ? Math.trunc(parsed.lastEventId) : 0
    const createdAtMs = typeof parsed.createdAtMs === 'number' && Number.isFinite(parsed.createdAtMs) ? Math.trunc(parsed.createdAtMs) : 0
    const summary = typeof parsed.summary === 'string' ? parsed.summary : ''
    const promptPreview = typeof parsed.promptPreview === 'string' ? parsed.promptPreview : ''
    const paramsSnapshot = isRecordObject(parsed.paramsSnapshot) ? parsed.paramsSnapshot : {}

    const modeRaw = String(parsed.mode || '').trim()
    const mode: VideoMode = modeRaw === 'vid2vid' || modeRaw === 'img2vid' ? modeRaw : 'txt2vid'

    return {
      taskId,
      lastEventId: Math.max(0, lastEventId),
      createdAtMs,
      mode,
      summary,
      promptPreview,
      paramsSnapshot,
    }
  } catch {
    return null
  }
}

function saveResumeState(key: string, state: ResumeState): void {
  try {
    localStorage.setItem(key, JSON.stringify(state))
  } catch {
    // ignore localStorage failures (private mode/quota)
  }
}

function clearResumeState(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore
  }
}

function updateResumeEventId(key: string, eventId: number): void {
  const v = Math.trunc(Number(eventId))
  if (!Number.isFinite(v) || v <= 0) return
  const cur = loadResumeState(key)
  if (!cur) return
  if (v <= cur.lastEventId) return
  saveResumeState(key, { ...cur, lastEventId: v })
}

function freshState(): VideoGenerationState {
  return {
    status: 'idle',
    progress: { ...DEFAULT_PROGRESS },
    frames: [],
    info: null,
    video: null,
    errorMessage: '',
    taskId: '',
    cancelRequested: false,
    currentRun: null,
    history: [],
    selectedTaskId: '',
    historyLoadingTaskId: '',
    queue: [],
  }
}

function getTabState(tabId: string): VideoGenerationState {
  if (!tabStates.has(tabId)) tabStates.set(tabId, freshState())
  return tabStates.get(tabId)!
}

function defaultStage(): WanStageParams {
  return {
    modelDir: '',
    sampler: '',
    scheduler: '',
    steps: 30,
    cfgScale: 7,
    seed: -1,
    loraSha: '',
    loraWeight: 1.0,
  }
}

function defaultVideo(): WanVideoParams {
  return {
    prompt: '',
    negativePrompt: '',
    width: 768,
    height: 432,
    fps: 24,
    frames: 17,
    attentionMode: 'global',
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    img2vidChunkFrames: 0,
    img2vidOverlapFrames: 4,
    img2vidAnchorAlpha: 0.2,
    img2vidChunkSeedMode: 'increment',
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
}

function defaultAssets(): WanAssetsParams {
  return { metadata: '', textEncoder: '', vae: '' }
}

export function useVideoGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()

  const state = ref(getTabState(tabId))
  const resumeNotice = ref('')

  const tab = computed<TabByType<'wan'> | null>(() => {
    const candidate = modelTabs.tabs.find((entry) => entry.id === tabId) || null
    if (!candidate || candidate.type !== 'wan') return null
    return candidate as TabByType<'wan'>
  })
  const params = computed<TabByType<'wan'>['params'] | null>(() => tab.value?.params || null)

  const video = computed<WanVideoParams>(() => params.value?.video || defaultVideo())
  const high = computed<WanStageParams>(() => params.value?.high || defaultStage())
  const low = computed<WanStageParams>(() => params.value?.low || defaultStage())
  const lightx2v = computed<boolean>(() => Boolean(params.value?.lightx2v))
  const assets = computed<WanAssetsParams>(() => params.value?.assets || defaultAssets())
  const mode = computed<VideoMode>(() => {
    if (video.value.useInitVideo) return 'vid2vid'
    if (video.value.useInitImage) return 'img2vid'
    return 'txt2vid'
  })

  function normalizeWanMetadataRepo(raw: string): string | null {
    const v = String(raw || '').trim()
    if (!v) return null
    // Back-compat: ignore legacy path values persisted in older tabs.
    if (v.startsWith('/') || v.includes('\\') || v.includes(':')) return null
    if (!v.includes('/')) return null
    return v
  }

  function inferWanMetadataRepo(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): string {
    const hint = `${hi.modelDir || ''} ${lo.modelDir || ''}`.toLowerCase()
    if (hint.includes('ti2v') || hint.includes('5b')) return 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
    if (v.useInitImage || v.useInitVideo) return 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
    return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
  }

  function effectiveWanMetadataRepo(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): string {
    const repo = normalizeWanMetadataRepo(assets.value.metadata)
    const repoLower = (repo || '').toLowerCase()
    const is5b = repoLower.includes('wan2.2-ti2v-5b')
    if (is5b) return 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'

    const isKnown14b = repoLower.includes('wan2.2-i2v-a14b') || repoLower.includes('wan2.2-t2v-a14b')
    if (isKnown14b) {
      if (v.useInitImage || v.useInitVideo) return 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
      return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
    }

    // If a different (valid) repo id is pinned in the tab, respect it.
    if (repo) return repo

    return inferWanMetadataRepo(v, hi, lo)
  }

  function blockedReasonFor(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams, initVideo: File | null): string {
    const prompt = String(v.prompt || '').trim()
    if (!prompt) return 'Prompt must not be empty.'
    if (v.useInitImage && !v.initImageData) {
      return 'Image mode requires an initial image; select a file or switch to Text mode.'
    }
    if (v.useInitVideo) {
      const path = String(v.initVideoPath || '').trim()
      if (!initVideo && !path) return 'Video mode requires an input video; upload a file or provide a path.'
    }
    if (!hi.modelDir || !lo.modelDir) {
      return 'WAN requires both High and Low stage models. Set them in QuickSettings.'
    }
    if (!quicksettings.resolveWanGgufSha(hi.modelDir)) {
      return 'WAN High model must resolve to a sha256. Click Refresh and re-select the High model.'
    }
    if (!quicksettings.resolveWanGgufSha(lo.modelDir)) {
      return 'WAN Low model must resolve to a sha256. Click Refresh and re-select the Low model.'
    }

    const teLabel = String(assets.value.textEncoder || '').trim()
    if (!teLabel) {
      return 'WAN requires a text encoder (.safetensors). Set WAN Text Encoder in QuickSettings.'
    }
    if (!quicksettings.resolveTextEncoderSha(teLabel)) {
      return 'WAN Text Encoder must resolve to a sha256. Click Refresh and re-select the text encoder.'
    }

    const vaeLabel = String(assets.value.vae || '').trim()
    if (!vaeLabel) {
      return 'WAN requires a VAE selection. Set WAN VAE in QuickSettings.'
    }
    if (!quicksettings.resolveVaeSha(vaeLabel)) {
      return 'WAN VAE must resolve to a sha256. Click Refresh and re-select the VAE.'
    }
    return ''
  }

  function currentInitVideo(): File | null {
    return initVideos.get(tabId) ?? null
  }

  const blockedReason = computed(() => blockedReasonFor(video.value, high.value, low.value, currentInitVideo()))
  const canGenerate = computed(() => blockedReason.value.length === 0)

  function uuid(): string {
    return `q-${Math.random().toString(36).slice(2, 10)}`
  }

  function stopStream(): void {
    const unsub = unsubscribers.get(tabId)
    if (!unsub) return
    unsub()
    unsubscribers.delete(tabId)
  }

  function resetProgress(): void {
    state.value.progress = { ...DEFAULT_PROGRESS }
  }

  function setError(message: string): void {
    state.value.status = 'error'
    state.value.errorMessage = message
  }

  function setErrorMessage(message: string): void {
    state.value.errorMessage = message
  }

  function buildRunSummary(v: WanVideoParams, hi: WanStageParams): string {
    const w = Number(v.width) || 0
    const h = Number(v.height) || 0
    const frames = Number(v.frames) || 0
    const fps = Number(v.fps) || 0
    const seconds = fps > 0 ? (frames / fps) : 0
    const strength = v.useInitVideo ? ` · strength ${Number(v.vid2vidStrength).toFixed(2)}` : ''
    const loraTag = lightx2v.value && String(hi.loraSha || '').trim() ? ' · lightx2v' : ''
    return `${w}×${h} · ${frames}f @ ${fps}fps (~${seconds.toFixed(2)}s) · steps ${hi.steps} · cfg ${hi.cfgScale}${strength}${loraTag}`
  }

  function buildParamsSnapshot(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): Record<string, unknown> {
    return {
      mode: v.useInitVideo ? 'vid2vid' : (v.useInitImage ? 'img2vid' : 'txt2vid'),
      initImageName: v.initImageName || '',
      initVideoName: v.initVideoName || '',
      initVideoPath: String(v.initVideoPath || ''),
      prompt: String(v.prompt || ''),
      negativePrompt: String(v.negativePrompt || ''),
      width: v.width,
      height: v.height,
      frames: v.frames,
      fps: v.fps,
      attentionMode: v.attentionMode,
      img2vid: {
        chunkFrames: v.img2vidChunkFrames,
        overlapFrames: v.img2vidOverlapFrames,
        anchorAlpha: v.img2vidAnchorAlpha,
        chunkSeedMode: v.img2vidChunkSeedMode,
      },
      vid2vid: {
        strength: v.vid2vidStrength,
        method: v.vid2vidMethod,
        useSourceFps: v.vid2vidUseSourceFps,
        useSourceFrames: v.vid2vidUseSourceFrames,
        chunkFrames: v.vid2vidChunkFrames,
        overlapFrames: v.vid2vidOverlapFrames,
        previewFrames: v.vid2vidPreviewFrames,
        flowEnabled: v.vid2vidFlowEnabled,
        flowUseLarge: v.vid2vidFlowUseLarge,
        flowDownscale: v.vid2vidFlowDownscale,
      },
      lightx2v: lightx2v.value,
      assets: {
        metadata: String(assets.value.metadata || ''),
        textEncoder: String(assets.value.textEncoder || ''),
        vae: String(assets.value.vae || ''),
      },
      high: {
        modelDir: hi.modelDir,
        sampler: hi.sampler,
        scheduler: hi.scheduler,
        steps: hi.steps,
        cfgScale: hi.cfgScale,
        seed: hi.seed,
        loraSha: lightx2v.value ? hi.loraSha : '',
        loraWeight: hi.loraWeight,
        flowShift: hi.flowShift,
      },
      low: {
        modelDir: lo.modelDir,
        sampler: lo.sampler,
        scheduler: lo.scheduler,
        steps: lo.steps,
        cfgScale: lo.cfgScale,
        seed: lo.seed,
        loraSha: lightx2v.value ? lo.loraSha : '',
        loraWeight: lo.loraWeight,
        flowShift: lo.flowShift,
      },
      output: {
        filenamePrefix: v.filenamePrefix,
        format: v.format,
        pixFmt: v.pixFmt,
        crf: v.crf,
        loopCount: v.loopCount,
        pingpong: v.pingpong,
        trimToAudio: v.trimToAudio,
        saveMetadata: v.saveMetadata,
        saveOutput: v.saveOutput,
        returnFrames: v.returnFrames,
      },
      interpolation: {
        enabled: v.rifeEnabled,
        model: v.rifeModel,
        times: v.rifeTimes,
      },
    }
  }

  function pushHistory(item: VideoRunHistoryItem): void {
    state.value.history.unshift(item)
    if (state.value.history.length > MAX_HISTORY) state.value.history.length = MAX_HISTORY
  }

  function buildCommonInput(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): WanVideoCommonInput {
    const metaRepo = effectiveWanMetadataRepo(v, hi, lo)
    const teLabel = String(assets.value.textEncoder || '').trim()
    const vaeLabel = String(assets.value.vae || '').trim()

    const hiSha = quicksettings.resolveWanGgufSha(hi.modelDir) || ''
    const loSha = quicksettings.resolveWanGgufSha(lo.modelDir) || ''
    const tencSha = quicksettings.resolveTextEncoderSha(teLabel) || ''
    const vaeSha = quicksettings.resolveVaeSha(vaeLabel) || ''

    return {
      device: quicksettings.currentDevice || 'cpu',
      settingsRevision: quicksettings.getSettingsRevision(),
      prompt: v.prompt,
      negativePrompt: v.negativePrompt,
      width: v.width,
      height: v.height,
      fps: v.fps,
      frames: v.frames,
      attentionMode: v.attentionMode,
      high: {
        modelSha: hiSha,
        sampler: hi.sampler,
        scheduler: hi.scheduler,
        steps: hi.steps,
        cfgScale: hi.cfgScale,
        seed: hi.seed,
        loraSha: lightx2v.value ? hi.loraSha : '',
        loraWeight: hi.loraWeight,
        flowShift: hi.flowShift,
      },
      low: {
        modelSha: loSha,
        sampler: lo.sampler,
        scheduler: lo.scheduler,
        steps: lo.steps,
        cfgScale: lo.cfgScale,
        seed: lo.seed,
        loraSha: lightx2v.value ? lo.loraSha : '',
        loraWeight: lo.loraWeight,
        flowShift: lo.flowShift,
      },
      format: 'auto' as const,
      assets: {
        metadataRepo: metaRepo,
        textEncoderSha: tencSha,
        vaeSha: vaeSha,
      },
      output: {
        filenamePrefix: v.filenamePrefix,
        format: v.format,
        pixFmt: v.pixFmt,
        crf: v.crf,
        loopCount: v.loopCount,
        pingpong: v.pingpong,
        trimToAudio: v.trimToAudio,
        saveMetadata: v.saveMetadata,
        saveOutput: v.saveOutput,
        returnFrames: v.returnFrames,
      },
      interpolation: {
        enabled: v.rifeEnabled,
        model: v.rifeModel,
        times: v.rifeTimes,
      },
    }
  }

  function prepareRunFromValues(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): PreparedWanRun {
    const promptPreview = String(v.prompt || '').trim().slice(0, 120)
    const createdAtMs = Date.now()
    const summary = buildRunSummary(v, hi)
    const paramsSnapshot = buildParamsSnapshot(v, hi, lo)

    const common = buildCommonInput(v, hi, lo)

    if (v.useInitVideo) {
      const payload = buildWanVid2VidPayload({
        ...common,
        strength: v.vid2vidStrength,
        method: v.vid2vidMethod,
        useSourceFps: v.vid2vidUseSourceFps,
        useSourceFrames: v.vid2vidUseSourceFrames,
        chunkFrames: v.vid2vidChunkFrames,
        overlapFrames: v.vid2vidOverlapFrames,
        previewFrames: v.vid2vidPreviewFrames,
        flowEnabled: v.vid2vidFlowEnabled,
        flowUseLarge: v.vid2vidFlowUseLarge,
        flowDownscale: v.vid2vidFlowDownscale,
        videoPath: v.initVideoPath,
      } satisfies WanVid2VidInput)
      return { mode: 'vid2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload, file: currentInitVideo() }
    }

    if (v.useInitImage) {
      const img2vidChunkInput = v.img2vidChunkFrames > 0
        ? {
            chunkFrames: v.img2vidChunkFrames,
            overlapFrames: v.img2vidOverlapFrames,
            anchorAlpha: v.img2vidAnchorAlpha,
            chunkSeedMode: v.img2vidChunkSeedMode,
          }
        : {}
      const payload = buildWanImg2VidPayload({
        ...common,
        initImageData: v.initImageData,
        ...img2vidChunkInput,
      })
      return { mode: 'img2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload }
    }

    const payload = buildWanTxt2VidPayload(common)
    return { mode: 'txt2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload }
  }

  async function startPreparedRun(run: PreparedWanRun): Promise<void> {
    stopStream()
    state.value.errorMessage = ''
    state.value.frames = []
    state.value.info = null
    state.value.video = null
    resetProgress()
    state.value.progress.stage = 'starting'
    state.value.cancelRequested = false
    state.value.currentRun = null

    state.value.status = 'running'

    try {
      const res = await (async () => {
        const mode = run.mode
        switch (mode) {
          case 'vid2vid': {
            assertRunPayloadObject(run.payload, mode)
            const form = new FormData()
            form.append('payload', JSON.stringify(run.payload))
            if (run.file) form.append('video', run.file)
            return startVid2Vid(form)
          }
          case 'img2vid':
            assertRunPayloadObject(run.payload, mode)
            return startImg2Vid(run.payload)
          case 'txt2vid':
            assertRunPayloadObject(run.payload, mode)
            return startTxt2Vid(run.payload)
          default:
            return assertNeverMode(mode)
        }
      })()
      const task_id = res.task_id
      state.value.taskId = task_id
      state.value.currentRun = {
        taskId: task_id,
        mode: run.mode,
        createdAtMs: run.createdAtMs,
        status: 'completed',
        summary: run.summary,
        promptPreview: run.promptPreview,
        paramsSnapshot: run.paramsSnapshot,
      }
      const key = resumeKey(tabId)
      saveResumeState(key, {
        taskId: task_id,
        lastEventId: 0,
        createdAtMs: run.createdAtMs,
        mode: run.mode,
        summary: run.summary,
        promptPreview: run.promptPreview,
        paramsSnapshot: run.paramsSnapshot,
      })
      const unsub = subscribeTask(task_id, onTaskEvent, undefined, {
        onMeta: ({ eventId }) => {
          if (typeof eventId === 'number') updateResumeEventId(key, eventId)
        },
      })
      unsubscribers.set(tabId, unsub)
    } catch (err) {
      clearResumeState(resumeKey(tabId))
      const conflictRevision = resolveSettingsRevisionConflict(err)
      if (conflictRevision !== null) {
        try {
          await quicksettings.refreshSettingsRevision(conflictRevision)
        } catch {
          // Ignore refresh failures; fallback revision is already applied.
        }
        setError(formatSettingsRevisionConflictMessage(quicksettings.getSettingsRevision()))
        return
      }
      setError(formatZodError(err))
    }
  }

  async function startNextQueued(): Promise<void> {
    if (state.value.status === 'running') return
    const next = state.value.queue.shift()
    if (!next) return
    await startPreparedRun(next)
  }

  async function generate(): Promise<void> {
    if (!tab.value) {
      setError(`useVideoGeneration: tab '${tabId}' not found or not available.`)
      return
    }
    if (tab.value.type !== 'wan') {
      setError(`useVideoGeneration: unsupported tab type '${String(tab.value.type)}'`)
      return
    }

    const v = video.value
    const hi = high.value
    const lo = low.value

    const blocked = blockedReasonFor(v, hi, lo, currentInitVideo())
    if (blocked) {
      setError(blocked)
      return
    }

    try {
      const run = prepareRunFromValues(v, hi, lo)
      await startPreparedRun(run)
    } catch (err) {
      setError(formatZodError(err))
    }
  }

  function onTaskEvent(event: TaskEvent): void {
    const key = resumeKey(tabId)
    switch (event.type) {
      case 'status':
        state.value.progress.stage = event.stage
        break
      case 'progress':
        state.value.progress = {
          stage: event.stage,
          percent: event.percent ?? null,
          etaSeconds: event.eta_seconds ?? null,
          step: event.step ?? null,
          totalSteps: event.total_steps ?? null,
        }
        break
      case 'gap':
        if (state.value.taskId) void refreshTaskSnapshot(state.value.taskId)
        break
      case 'result':
        state.value.frames = event.images
        state.value.info = event.info ?? null
        state.value.video = event.video ?? null
        state.value.status = 'done'
        if (state.value.currentRun && state.value.currentRun.taskId) state.value.currentRun.status = 'completed'
        break
      case 'error':
        setError(event.message)
        clearResumeState(key)
        if (state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = state.value.cancelRequested ? 'cancelled' : 'error'
          state.value.currentRun.errorMessage = event.message
        }
        break
      case 'end':
        clearResumeState(key)
        if (state.value.cancelRequested && state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = 'cancelled'
        }
        if (state.value.status !== 'error') state.value.status = 'done'
        if (state.value.currentRun && state.value.currentRun.taskId) {
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        void startNextQueued()
        break
    }
  }

  async function refreshTaskSnapshot(taskId: string): Promise<void> {
    try {
      const res = await fetchTaskResult(taskId)
      if (res.status !== 'running') return
      if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
      const p = res.progress
      if (p && typeof p === 'object') {
        state.value.progress = {
          stage: String(p.stage ?? state.value.progress.stage),
          percent: p.percent ?? null,
          etaSeconds: p.eta_seconds ?? null,
          step: p.step ?? null,
          totalSteps: p.total_steps ?? null,
        }
      }
    } catch {
      // ignore snapshot refresh failures
    }
  }

  async function tryAutoResume(): Promise<void> {
    if (resumeAttempts.has(tabId)) return
    resumeAttempts.add(tabId)

    const key = resumeKey(tabId)
    const saved = loadResumeState(key)
    if (!saved) return

    let res
    try {
      res = await fetchTaskResult(saved.taskId)
    } catch {
      clearResumeState(key)
      return
    }

    if (res.status === 'running') {
      stopStream()
      state.value.status = 'running'
      state.value.taskId = saved.taskId
      state.value.errorMessage = ''
      state.value.frames = []
      state.value.info = null
      state.value.video = null
      resetProgress()
      state.value.cancelRequested = false
      state.value.currentRun = {
        taskId: saved.taskId,
        mode: saved.mode,
        createdAtMs: saved.createdAtMs,
        status: 'completed',
        summary: saved.summary,
        promptPreview: saved.promptPreview,
        paramsSnapshot: saved.paramsSnapshot,
      }

      if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
      const p = res.progress
      if (p && typeof p === 'object') {
        state.value.progress = {
          stage: String(p.stage ?? state.value.progress.stage),
          percent: p.percent ?? null,
          etaSeconds: p.eta_seconds ?? null,
          step: p.step ?? null,
          totalSteps: p.total_steps ?? null,
        }
      }

      const unsub = subscribeTask(saved.taskId, onTaskEvent, undefined, {
        after: saved.lastEventId,
        onMeta: ({ eventId }) => {
          if (typeof eventId === 'number') updateResumeEventId(key, eventId)
        },
      })
      unsubscribers.set(tabId, unsub)

      if (!resumeToastShown.has(saved.taskId)) {
        resumeNotice.value = 'Reconnected (resumed task).'
        resumeToastShown.add(saved.taskId)
      }
      return
    }

    clearResumeState(key)

    if (res.status === 'completed' && res.result) {
      state.value.frames = res.result.images
      state.value.info = res.result.info ?? null
      state.value.video = res.result.video ?? null
      state.value.status = 'done'
      pushHistory({
        taskId: saved.taskId,
        mode: saved.mode,
        createdAtMs: saved.createdAtMs,
        status: 'completed',
        summary: saved.summary,
        promptPreview: saved.promptPreview,
        paramsSnapshot: saved.paramsSnapshot,
      })
      state.value.selectedTaskId = saved.taskId
      return
    }
    if (res.status === 'error') {
      state.value.status = 'error'
      state.value.errorMessage = String(res.error || 'Task failed.')
      pushHistory({
        taskId: saved.taskId,
        mode: saved.mode,
        createdAtMs: saved.createdAtMs,
        status: 'error',
        summary: saved.summary,
        promptPreview: saved.promptPreview,
        paramsSnapshot: saved.paramsSnapshot,
        errorMessage: String(res.error || 'Task failed.'),
      })
      state.value.selectedTaskId = saved.taskId
    }
  }

  void tryAutoResume()

  async function cancel(mode: 'immediate' | 'after_current' = 'immediate'): Promise<void> {
    const taskId = state.value.taskId
    if (!taskId || state.value.status !== 'running') return
    state.value.cancelRequested = true
    try {
      await cancelTask(taskId, mode)
    } catch (err) {
      // Keep running; backend may still send end/error shortly.
      setErrorMessage(err instanceof Error ? err.message : String(err))
    }
  }

  async function enqueue(): Promise<void> {
    if (state.value.queue.length >= MAX_QUEUE) throw new Error(`Queue is full (max ${MAX_QUEUE}).`)

    const v = video.value
    const hi = high.value
    const lo = low.value
    const blocked = blockedReasonFor(v, hi, lo, currentInitVideo())
    if (blocked) throw new Error(blocked)

    const run = prepareRunFromValues(v, hi, lo)
    state.value.queue.push({ id: uuid(), ...run })
  }

  function clearQueue(): void {
    state.value.queue = []
  }

  async function loadHistory(taskId: string): Promise<void> {
    if (!taskId || state.value.status === 'running') return
    state.value.historyLoadingTaskId = taskId
    try {
      const result = await fetchTaskResult(taskId)
      if (result.status === 'error') {
        setError(result.error || 'Task failed.')
        return
      }
      if (result.status === 'completed' && result.result) {
        state.value.frames = result.result.images
        state.value.info = result.result.info ?? null
        state.value.video = result.result.video ?? null
        state.value.status = 'done'
        state.value.selectedTaskId = taskId
        return
      }
      setError('Task is still running.')
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      state.value.historyLoadingTaskId = ''
    }
  }

  function clearHistory(): void {
    state.value.history = []
    state.value.selectedTaskId = ''
  }

  function setInitVideoFile(file: File): void {
    initVideos.set(tabId, file)
  }

  function clearInitVideoFile(): void {
    initVideos.delete(tabId)
  }

  function outputUrl(relPath: string): string {
    const clean = String(relPath || '').replace(/\\+/g, '/').replace(/^\/+/, '')
    const encoded = clean.split('/').map((p) => encodeURIComponent(p)).join('/')
    return `/api/output/${encoded}`
  }

  const videoExport = computed(() => state.value.video)
  const videoUrl = computed(() => {
    const rel = state.value.video?.rel_path
    if (!rel) return ''
    return outputUrl(rel)
  })

  return {
    // State
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    frames: computed(() => state.value.frames),
    info: computed(() => state.value.info),
    videoExport,
    videoUrl,
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    isRunning: computed(() => state.value.status === 'running'),
    cancelRequested: computed(() => state.value.cancelRequested),
    history: computed(() => state.value.history),
    selectedTaskId: computed(() => state.value.selectedTaskId),
    historyLoadingTaskId: computed(() => state.value.historyLoadingTaskId),
    queue: computed(() => state.value.queue),
    queueMax: computed(() => MAX_QUEUE),

    // Tab info
    tab,
    params,
    video,
    high,
    low,
    lightx2v,
    assets,
    mode,
    blockedReason,
    canGenerate,

    // Actions
    generate,
    stopStream,
    cancel,
    loadHistory,
    clearHistory,
    enqueue,
    clearQueue,
    setInitVideoFile,
    clearInitVideoFile,
    resumeNotice,
  }
}
