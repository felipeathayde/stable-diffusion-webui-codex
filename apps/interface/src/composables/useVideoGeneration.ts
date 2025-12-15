import { computed, ref } from 'vue'

import { startImg2Vid, startTxt2Vid, subscribeTask } from '../api/client'
import { formatZodError } from '../api/payloads'
import { buildWanImg2VidPayload, buildWanTxt2VidPayload } from '../api/payloads_video'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { useModelTabsStore, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'

type Status = 'idle' | 'running' | 'error' | 'done'

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
  errorMessage: string
  taskId: string
}

const DEFAULT_PROGRESS: VideoProgressState = { stage: 'idle', percent: null, etaSeconds: null, step: null, totalSteps: null }
const DEFAULT_STATE: VideoGenerationState = {
  status: 'idle',
  progress: { ...DEFAULT_PROGRESS },
  frames: [],
  info: null,
  errorMessage: '',
  taskId: '',
}

// Per-tab generation state (keyed by tab ID)
const tabStates = new Map<string, VideoGenerationState>()
const unsubscribers = new Map<string, () => void>()

function freshState(): VideoGenerationState {
  return { ...DEFAULT_STATE, progress: { ...DEFAULT_PROGRESS }, frames: [] }
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
    lightning: false,
    loraEnabled: false,
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
  const wanFormat = computed<string>(() => String((params.value as any)?.modelFormat || 'auto'))
  const assets = computed<WanAssetsParams>(() => (params.value?.assets as WanAssetsParams) || defaultAssets())

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

  async function generate(): Promise<void> {
    if (!tab.value) return
    if (tab.value.type !== 'wan') {
      setError(`useVideoGeneration: unsupported tab type '${String(tab.value.type)}'`)
      return
    }

    stopStream()
    state.value.errorMessage = ''
    state.value.frames = []
    state.value.info = null
    resetProgress()

    const v = video.value
    const hi = high.value
    const lo = low.value

    if (v.useInitImage && !v.initImageData) {
      setError('img2vid requires an initial image; select a file or disable the toggle.')
      return
    }
    if (!hi.modelDir && !lo.modelDir) {
      setError('WAN model directory is empty. Set WAN High/Low model dirs in QuickSettings.')
      return
    }
    if (hi.loraEnabled && !hi.loraPath) {
      setError('High stage: LoRA enabled but path is empty.')
      return
    }
    if (lo.loraEnabled && !lo.loraPath) {
      setError('Low stage: LoRA enabled but path is empty.')
      return
    }

    state.value.status = 'running'

    const metaDir = String(assets.value.metadata || '').trim()
    const tePath = String(assets.value.textEncoder || '').trim()
    const vaePath = String(assets.value.vae || '').trim()

    try {
      const fmt = (wanFormat.value === 'diffusers' || wanFormat.value === 'gguf') ? wanFormat.value : 'auto'
      const common = {
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
          lightning: hi.lightning,
          loraEnabled: hi.loraEnabled,
          loraPath: hi.loraPath,
          loraWeight: hi.loraWeight,
        },
        low: {
          modelDir: lo.modelDir,
          sampler: lo.sampler,
          scheduler: lo.scheduler,
          steps: lo.steps,
          cfgScale: lo.cfgScale,
          seed: lo.seed,
          lightning: lo.lightning,
          loraEnabled: lo.loraEnabled,
          loraPath: lo.loraPath,
          loraWeight: lo.loraWeight,
        },
        format: fmt,
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

      if (v.useInitImage) {
        const payload = buildWanImg2VidPayload({ ...common, initImageData: v.initImageData })
        const { task_id } = await startImg2Vid(payload as any)
        state.value.taskId = task_id
        const unsub = subscribeTask(task_id, onTaskEvent)
        unsubscribers.set(tabId, unsub)
        return
      }

      const payload = buildWanTxt2VidPayload(common)
      const { task_id } = await startTxt2Vid(payload as any)
      state.value.taskId = task_id
      const unsub = subscribeTask(task_id, onTaskEvent)
      unsubscribers.set(tabId, unsub)
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
        state.value.status = 'done'
        stopStream()
        break
      case 'error':
        setError(event.message)
        stopStream()
        break
      case 'end':
        if (state.value.status !== 'error') state.value.status = 'done'
        stopStream()
        break
    }
  }

  return {
    // State
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    frames: computed(() => state.value.frames),
    info: computed(() => state.value.info),
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    isRunning: computed(() => state.value.status === 'running'),

    // Tab info
    tab,
    params,
    video,
    high,
    low,
    wanFormat,
    assets,

    // Actions
    generate,
    stopStream,
  }
}
