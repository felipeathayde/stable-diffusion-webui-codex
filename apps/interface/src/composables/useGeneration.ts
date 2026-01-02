/**
 * Unified generation composable.
 * Provides generate(), progress, gallery, status for any engine type.
 * State is stored per-tab in model_tabs.
 */

import { computed, ref } from 'vue'
import { useModelTabsStore, type BaseTab, type ImageBaseParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { getEngineConfig, type EngineType } from '../stores/engine_config'
import { buildTxt2ImgPayload, deriveFluxTextEncoderOverrideFromLabels, type Txt2ImgRequest } from '../api/payloads'
import { fetchTaskResult, startImg2Img, startTxt2Img, subscribeTask } from '../api/client'
import type { GeneratedImage, TaskEvent } from '../api/types'

export interface ImageRunHistoryItem {
  taskId: string
  mode: 'txt2img' | 'img2img'
  createdAtMs: number
  status: 'completed' | 'error' | 'cancelled'
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
  errorMessage?: string
}

export interface GenerationState {
  status: 'idle' | 'running' | 'done' | 'error'
  progress: {
    stage: string
    percent: number | null
    etaSeconds: number | null
    step: number | null
    totalSteps: number | null
  }
  previewImage: GeneratedImage | null
  previewStep: number | null
  gallery: GeneratedImage[]
  info: unknown | null
  errorMessage: string
  taskId: string
  lastSeed: number | null
  startedAtMs: number | null
  finishedAtMs: number | null
  history: ImageRunHistoryItem[]
  selectedTaskId: string
  historyLoadingTaskId: string
  currentRun: ImageRunHistoryItem | null
}

const MAX_HISTORY = 8

function defaultState(): GenerationState {
  return {
    status: 'idle',
    progress: { stage: 'none', percent: null, etaSeconds: null, step: null, totalSteps: null },
    previewImage: null,
    previewStep: null,
    gallery: [],
    info: null,
    errorMessage: '',
    taskId: '',
    lastSeed: null,
    startedAtMs: null,
    finishedAtMs: null,
    history: [],
    selectedTaskId: '',
    historyLoadingTaskId: '',
    currentRun: null,
  }
}

// Per-tab generation state (keyed by tab ID)
const tabStates = new Map<string, GenerationState>()
const unsubscribers = new Map<string, () => void>()

function getTabState(tabId: string): GenerationState {
  if (!tabStates.has(tabId)) {
    tabStates.set(tabId, defaultState())
  }
  return tabStates.get(tabId)!
}

export function useGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()
  
  // Reactive state for this tab
  const state = ref(getTabState(tabId))
  
  // Tab info
  const tab = computed(() => modelTabs.tabs.find(t => t.id === tabId) as BaseTab | undefined)
  const params = computed(() => tab.value?.params as ImageBaseParams | undefined)
  const engineType = computed(() => tab.value?.type as EngineType | undefined)
  const engineConfig = computed(() => engineType.value ? getEngineConfig(engineType.value) : null)

  function buildRunSummary(p: ImageBaseParams): string {
    const sampler = p.sampler || 'automatic'
    const scheduler = p.scheduler || 'automatic'
    const seedLabel = p.seed === -1 ? 'seed random' : `seed ${p.seed}`
    const clipSkipLabel = Number.isFinite(p.clipSkip) && p.clipSkip > 0 && p.clipSkip !== 1 ? ` · clip-skip ${p.clipSkip}` : ''
    return `${p.width}×${p.height} px · ${p.steps} steps · cfg ${p.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel}${clipSkipLabel} · batch ${p.batchCount}×${p.batchSize}`
  }

  function buildParamsSnapshot(p: ImageBaseParams): Record<string, unknown> {
    return {
      checkpoint: p.checkpoint,
      textEncoders: p.textEncoders,

      prompt: p.prompt,
      negativePrompt: p.negativePrompt,

      width: p.width,
      height: p.height,
      sampler: p.sampler,
      scheduler: p.scheduler,
      steps: p.steps,
      cfgScale: p.cfgScale,
      seed: p.seed,
      clipSkip: p.clipSkip,

      batchSize: p.batchSize,
      batchCount: p.batchCount,

      highres: p.highres,
      refiner: p.refiner,

      useInitImage: p.useInitImage,
      initImageName: p.initImageName,
      denoiseStrength: p.denoiseStrength,
    }
  }

  function pushHistory(item: ImageRunHistoryItem): void {
    state.value.history.unshift(item)
    if (state.value.history.length > MAX_HISTORY) state.value.history.length = MAX_HISTORY
  }
  
  function stopStream(): void {
    const unsub = unsubscribers.get(tabId)
    if (unsub) {
      unsub()
      unsubscribers.delete(tabId)
    }
  }
  
  function resetProgress(): void {
    state.value.progress = { stage: 'none', percent: null, etaSeconds: null, step: null, totalSteps: null }
  }
  
  async function generate(): Promise<void> {
    if (!tab.value || !params.value || !engineType.value) {
      state.value.status = 'error'
      state.value.errorMessage = 'Tab not found'
      return
    }
    
    stopStream()
    state.value.status = 'running'
    state.value.errorMessage = ''
    state.value.gallery = []
    state.value.info = null
    state.value.previewImage = null
    state.value.previewStep = null
    resetProgress()
    state.value.progress.stage = 'starting'
    state.value.startedAtMs = performance.now()
    state.value.finishedAtMs = null

    const p = params.value
    const config = engineConfig.value!
    const checkpoint = String((p as any).checkpoint || '').trim()
    const modelIsGguf = quicksettings.isModelGguf(checkpoint)
    const resolvedModelSha = quicksettings.resolveModelSha(checkpoint)
    const modelOverride = resolvedModelSha || checkpoint
    if (!modelOverride) {
      state.value.status = 'error'
      state.value.errorMessage = 'Select a checkpoint to generate.'
      return
    }

    const createdAtMs = Date.now()
    const promptPreview = String(p.prompt || '').trim().slice(0, 120)
    const summary = buildRunSummary(p)
    const paramsSnapshot = buildParamsSnapshot(p)

    const batchSize = Math.max(1, Math.trunc(Number(p.batchSize)))
    const batchCount = Math.max(1, Math.trunc(Number(p.batchCount)))

    if (p.useInitImage) {
      if (!config.capabilities.tasks.includes('img2img')) {
        state.value.status = 'error'
        state.value.errorMessage = 'This engine does not support img2img.'
        return
      }
      if (!p.initImageData) {
        state.value.status = 'error'
        state.value.errorMessage = 'Select an initial image for img2img.'
        return
      }
    }

    const textEncoders = Array.isArray((p as any).textEncoders)
      ? (p as any).textEncoders.map((it: unknown) => String(it || '').trim()).filter((it: string) => it.length > 0)
      : []

    const teOverride = engineType.value === 'flux'
      ? deriveFluxTextEncoderOverrideFromLabels(textEncoders)
      : undefined
    
    // Build extras based on engine capabilities (e.g. tenc_sha)
    const extras: Record<string, unknown> = {}

    const needsTencSha = config.capabilities.requiresTenc || modelIsGguf
    if (needsTencSha) {
      const shas: string[] = []
      for (const label of textEncoders) {
        const sha = quicksettings.resolveTextEncoderSha(label)
        if (!sha) {
          state.value.status = 'error'
          state.value.errorMessage = `Text encoder SHA not found for '${label}'.`
          return
        }
        shas.push(sha)
      }
      if (shas.length === 0) {
        state.value.status = 'error'
        state.value.errorMessage = 'Select a text encoder so the request can include tenc_sha.'
        return
      }
      extras.tenc_sha = shas.length === 1 ? shas[0] : shas
    }
    
    const device = (quicksettings.currentDevice || 'cpu') as any
    let engineOverrideForRequest = String(engineType.value)
    // Flux img2img should run via the Kontext workflow engine.
    if (p.useInitImage && engineOverrideForRequest === 'flux') {
      engineOverrideForRequest = 'kontext'
    }

    try {
      let taskId = ''
      if (p.useInitImage) {
        const isFlowModel = Boolean(config.capabilities.usesDistilledCfg) && !config.capabilities.usesCfg
        const img2imgExtras: Record<string, unknown> = { ...extras }
        if (teOverride) img2imgExtras.text_encoder_override = teOverride

        const payload: any = {
          img2img_init_image: p.initImageData,
          img2img_mask: '',
          img2img_prompt: p.prompt,
          img2img_neg_prompt: config.capabilities.usesNegativePrompt ? p.negativePrompt : '',
          img2img_styles: [],
          img2img_batch_count: batchCount,
          img2img_batch_size: batchSize,
          img2img_steps: p.steps,
          img2img_cfg_scale: isFlowModel ? 1.0 : p.cfgScale,
          img2img_distilled_cfg_scale: isFlowModel ? p.cfgScale : undefined,
          img2img_denoising_strength: p.denoiseStrength,
          img2img_width: p.width,
          img2img_height: p.height,
          img2img_sampling: p.sampler || 'automatic',
          img2img_scheduler: p.scheduler || 'automatic',
          img2img_seed: p.seed,
          img2img_clip_skip: p.clipSkip,
          device,
          engine: engineOverrideForRequest,
          model: modelOverride,
          smart_offload: quicksettings.smartOffload,
          smart_fallback: quicksettings.smartFallback,
          smart_cache: quicksettings.smartCache,
          img2img_extras: img2imgExtras,
        }
        const { task_id } = await startImg2Img(payload)
        taskId = task_id
      } else {
        let payload: Txt2ImgRequest
        try {
          payload = buildTxt2ImgPayload({
            prompt: p.prompt,
            negativePrompt: config.capabilities.usesNegativePrompt ? p.negativePrompt : '',
            width: p.width,
            height: p.height,
            steps: p.steps,
            guidanceScale: p.cfgScale,
            sampler: p.sampler || 'automatic',
            scheduler: p.scheduler || 'automatic',
            seed: p.seed,
            clipSkip: p.clipSkip,
            batchSize,
            batchCount,
            styles: [],
            device,
            engine: engineOverrideForRequest,
            model: modelOverride,
            smartOffload: quicksettings.smartOffload,
            smartFallback: quicksettings.smartFallback,
            smartCache: quicksettings.smartCache,
            highres: p.highres,
            refiner: p.refiner,
            textEncoderOverride: teOverride,
            extras,
          })
        } catch (error) {
          state.value.status = 'error'
          state.value.errorMessage = error instanceof Error ? error.message : String(error)
          return
        }
        const { task_id } = await startTxt2Img(payload)
        taskId = task_id
      }

      state.value.taskId = taskId
      state.value.currentRun = {
        taskId,
        mode: p.useInitImage ? 'img2img' : 'txt2img',
        createdAtMs,
        status: 'completed',
        summary,
        promptPreview,
        paramsSnapshot,
      }

      state.value.progress.stage = 'submitted'

      const unsub = subscribeTask(state.value.taskId, handleTaskEvent)
      unsubscribers.set(tabId, unsub)
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
    }
  }
  
  function handleTaskEvent(event: TaskEvent): void {
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
        if (event.preview_image) {
          state.value.previewImage = event.preview_image
          state.value.previewStep = event.preview_step ?? null
        }
        break
      case 'result':
        state.value.gallery = event.images || []
        state.value.info = event.info ?? null
        state.value.previewImage = null
        state.value.previewStep = null
        if (state.value.currentRun?.taskId) {
          state.value.currentRun.status = 'completed'
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        try {
          const infoObj = (typeof event.info === 'string' ? JSON.parse(event.info) : event.info) as any
          const rawSeed = infoObj?.seed ?? infoObj?.all_seeds?.[0]
          const resolvedSeed = typeof rawSeed === 'number' ? rawSeed : Number(rawSeed)
          if (Number.isFinite(resolvedSeed)) {
            state.value.lastSeed = resolvedSeed
          }
        } catch {
          // ignore seed parsing; keep lastSeed as-is
        }
        state.value.finishedAtMs = performance.now()
        state.value.status = 'done'
        break
      case 'error':
        state.value.status = 'error'
        state.value.errorMessage = event.message
        state.value.finishedAtMs = performance.now()
        state.value.previewImage = null
        state.value.previewStep = null
        if (state.value.currentRun?.taskId) {
          state.value.currentRun.status = 'error'
          state.value.currentRun.errorMessage = event.message
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        stopStream()
        break
      case 'end':
        if (state.value.status !== 'error') {
          state.value.status = 'done'
        }
        if (state.value.finishedAtMs === null) {
          state.value.finishedAtMs = performance.now()
        }
        state.value.previewImage = null
        state.value.previewStep = null
        stopStream()
        break
    }
  }

  async function loadHistory(taskId: string): Promise<void> {
    if (!taskId || state.value.status === 'running') return
    stopStream()
    state.value.historyLoadingTaskId = taskId
    try {
      const result = await fetchTaskResult(taskId)
      if (result.status === 'error') {
        state.value.status = 'error'
        state.value.errorMessage = result.error || 'Task failed.'
        return
      }
      if (result.status === 'completed' && result.result) {
        state.value.gallery = result.result.images || []
        state.value.info = result.result.info ?? null
        state.value.status = 'done'
        state.value.selectedTaskId = taskId
        return
      }
      state.value.status = 'error'
      state.value.errorMessage = 'Task is still running.'
    } catch (err) {
      state.value.status = 'error'
      state.value.errorMessage = err instanceof Error ? err.message : String(err)
    } finally {
      state.value.historyLoadingTaskId = ''
    }
  }

  function clearHistory(): void {
    state.value.history = []
    state.value.selectedTaskId = ''
  }
  
  // Expose reactive state and methods
  return {
    // State
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    previewImage: computed(() => state.value.previewImage),
    previewStep: computed(() => state.value.previewStep),
    gallery: computed(() => state.value.gallery),
    info: computed(() => state.value.info),
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    lastSeed: computed(() => state.value.lastSeed),
    history: computed(() => state.value.history),
    selectedTaskId: computed(() => state.value.selectedTaskId),
    historyLoadingTaskId: computed(() => state.value.historyLoadingTaskId),
    gentimeMs: computed(() => {
      if (state.value.startedAtMs === null || state.value.finishedAtMs === null) return null
      return Math.max(0, state.value.finishedAtMs - state.value.startedAtMs)
    }),
    isRunning: computed(() => state.value.status === 'running'),
    
    // Tab info
    tab,
    params,
    engineType,
    engineConfig,
    
    // Actions
    generate,
    stopStream,
    loadHistory,
    clearHistory,
  }
}
