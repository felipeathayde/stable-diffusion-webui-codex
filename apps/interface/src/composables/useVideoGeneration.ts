import { computed, ref } from 'vue'

import { cancelTask, fetchTaskResult, startImg2Vid, startTxt2Vid, startVid2Vid, subscribeTask } from '../api/client'
import { formatZodError } from '../api/payloads'
import { buildWanImg2VidPayload, buildWanTxt2VidPayload, buildWanVid2VidPayload, type WanVid2VidInput } from '../api/payloads_video'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { useModelTabsStore, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'

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

export interface VideoQueuedRun {
  id: string
  mode: VideoMode
  createdAtMs: number
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
  payload: Record<string, unknown>
  file?: File | null
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
    loraPath: '',
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
    rifeEnabled: false,
    rifeModel: '',
    rifeTimes: 2,
  }
}

interface WanAssetsParams {
  metadata: string
  textEncoder: string
  vae: string
}

function defaultAssets(): WanAssetsParams {
  return { metadata: '', textEncoder: '', vae: '' }
}

export function useVideoGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()

  const state = ref(getTabState(tabId))

  const tab = computed(() => modelTabs.tabs.find(t => t.id === tabId) || null)
  const params = computed(() => (tab.value?.params as any) as Record<string, unknown> | null)

  const video = computed<WanVideoParams>(() => (params.value?.video as WanVideoParams) || defaultVideo())
  const high = computed<WanStageParams>(() => (params.value?.high as WanStageParams) || defaultStage())
  const low = computed<WanStageParams>(() => (params.value?.low as WanStageParams) || defaultStage())
  const lightx2v = computed<boolean>(() => Boolean((params.value as any)?.lightx2v))
  const assets = computed<WanAssetsParams>(() => (params.value?.assets as WanAssetsParams) || defaultAssets())
  const mode = computed<VideoMode>(() => {
    if (video.value.useInitVideo) return 'vid2vid'
    if (video.value.useInitImage) return 'img2vid'
    return 'txt2vid'
  })

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
    if (!hi.modelDir && !lo.modelDir) {
      return 'WAN model directory is empty. Set WAN High/Low model dirs in QuickSettings.'
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
    const loraTag = lightx2v.value && String(hi.loraPath || '').trim() ? ' · lightx2v' : ''
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
        loraPath: lightx2v.value ? hi.loraPath : '',
        loraWeight: hi.loraWeight,
      },
      low: {
        modelDir: lo.modelDir,
        sampler: lo.sampler,
        scheduler: lo.scheduler,
        steps: lo.steps,
        cfgScale: lo.cfgScale,
        seed: lo.seed,
        loraPath: lightx2v.value ? lo.loraPath : '',
        loraWeight: lo.loraWeight,
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

  function buildCommonInput(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams) {
    const metaDir = String(assets.value.metadata || '').trim()
    const tePath = String(assets.value.textEncoder || '').trim()
    const vaePath = String(assets.value.vae || '').trim()

    return {
      device: quicksettings.currentDevice || 'cpu',
      prompt: v.prompt,
      negativePrompt: v.negativePrompt,
      width: v.width,
      height: v.height,
      fps: v.fps,
      frames: v.frames,
      high: {
        modelDir: hi.modelDir,
        sampler: hi.sampler,
        scheduler: hi.scheduler,
        steps: hi.steps,
        cfgScale: hi.cfgScale,
        seed: hi.seed,
        loraPath: lightx2v.value ? hi.loraPath : '',
        loraWeight: hi.loraWeight,
      },
      low: {
        modelDir: lo.modelDir,
        sampler: lo.sampler,
        scheduler: lo.scheduler,
        steps: lo.steps,
        cfgScale: lo.cfgScale,
        seed: lo.seed,
        loraPath: lightx2v.value ? lo.loraPath : '',
        loraWeight: lo.loraWeight,
      },
      format: 'auto' as const,
      assets: {
        metadataDir: metaDir,
        textEncoder: tePath,
        vaePath: vaePath,
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
      },
      interpolation: {
        enabled: v.rifeEnabled,
        model: v.rifeModel,
        times: v.rifeTimes,
      },
    } as const
  }

  function prepareRunFromValues(v: WanVideoParams, hi: WanStageParams, lo: WanStageParams): Omit<VideoQueuedRun, 'id'> & { mode: VideoMode } {
    const promptPreview = String(v.prompt || '').trim().slice(0, 120)
    const createdAtMs = Date.now()
    const summary = buildRunSummary(v, hi)
    const paramsSnapshot = buildParamsSnapshot(v, hi, lo)

    const common = buildCommonInput(v, hi, lo)

    if (v.useInitVideo) {
      const payload = buildWanVid2VidPayload({
        ...(common as any),
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
      } satisfies WanVid2VidInput) as unknown as Record<string, unknown>
      return { mode: 'vid2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload, file: currentInitVideo() }
    }

    if (v.useInitImage) {
      const payload = buildWanImg2VidPayload({ ...(common as any), initImageData: v.initImageData }) as unknown as Record<string, unknown>
      return { mode: 'img2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload }
    }

    const payload = buildWanTxt2VidPayload(common as any) as unknown as Record<string, unknown>
    return { mode: 'txt2vid', createdAtMs, summary, promptPreview, paramsSnapshot, payload }
  }

  async function startPreparedRun(run: { mode: VideoMode; createdAtMs: number; summary: string; promptPreview: string; paramsSnapshot: Record<string, unknown>; payload: Record<string, unknown>; file?: File | null }): Promise<void> {
    stopStream()
    state.value.errorMessage = ''
    state.value.frames = []
    state.value.info = null
    state.value.video = null
    resetProgress()
    state.value.cancelRequested = false
    state.value.currentRun = null

    state.value.status = 'running'

    try {
      const res =
        run.mode === 'vid2vid'
          ? await (async () => {
              const form = new FormData()
              form.append('payload', JSON.stringify(run.payload || {}))
              if (run.file) form.append('video', run.file)
              return startVid2Vid(form)
            })()
          : (run.mode === 'img2vid' ? await startImg2Vid(run.payload as any) : await startTxt2Vid(run.payload as any))
      const task_id = (res as any).task_id as string
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
      const unsub = subscribeTask(task_id, onTaskEvent)
      unsubscribers.set(tabId, unsub)
    } catch (err) {
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
    if (!tab.value) return
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
      case 'result':
        state.value.frames = event.images
        state.value.info = event.info ?? null
        state.value.video = event.video ?? null
        state.value.status = 'done'
        if (state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = 'completed'
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        void startNextQueued()
        break
      case 'error':
        setError(event.message)
        if (state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = 'error'
          state.value.currentRun.errorMessage = event.message
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        void startNextQueued()
        break
      case 'end':
        if (state.value.status !== 'error') state.value.status = 'done'
        if (state.value.cancelRequested && state.value.currentRun && state.value.currentRun.taskId) {
          state.value.currentRun.status = 'cancelled'
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        void startNextQueued()
        break
    }
  }

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
  }
}
