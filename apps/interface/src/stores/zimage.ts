/**
 * Z Image Turbo store for txt2img generation.
 * Based on Flux store pattern with Z Image-specific defaults.
 */
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  ModelInfo,
  SamplerInfo,
  SchedulerInfo,
  GeneratedImage,
  TaskEvent,
} from '../api/types'
import {
  fetchModels,
  fetchSamplers,
  fetchSchedulers,
  updateOptions,
  startTxt2Img,
  subscribeTask,
} from '../api/client'
import { buildTxt2ImgPayload, formatZodError } from '../api/payloads'
import type { Txt2ImgRequest } from '../api/payloads'
import { useQuicksettingsStore } from './quicksettings'
import { getEngineDefaults } from './engine_config'

const ENGINE_ID = 'zimage'
const ENGINE_DEFAULTS = getEngineDefaults('zimage')
const STORAGE_KEY = 'codex.zimage.profile'

type Status = 'idle' | 'running' | 'done' | 'error'

interface ProgressState {
  stage: 'none' | 'prepare' | 'sampling' | 'decode' | 'submitted'
  step: number
  totalSteps: number
}

const DEFAULT_PROGRESS: ProgressState = {
  stage: 'none',
  step: 0,
  totalSteps: 0,
}

export const useZImageStore = defineStore('zimage', () => {
  const quicksettings = useQuicksettingsStore()

  // Z Image defaults from engine_config
  const prompt = ref('')
  const negativePrompt = ref('')
  const steps = ref(ENGINE_DEFAULTS.steps)
  const cfgScale = ref(ENGINE_DEFAULTS.distilledCfg ?? ENGINE_DEFAULTS.cfg)
  const width = ref(ENGINE_DEFAULTS.width)
  const height = ref(ENGINE_DEFAULTS.height)
  const seed = ref(-1)
  const batchSize = ref(1)
  const batchCount = ref(1)

  const models = ref<ModelInfo[]>([])
  const samplers = ref<SamplerInfo[]>([])
  const schedulers = ref<SchedulerInfo[]>([])

  const selectedModel = ref<string>('')
  const selectedSampler = ref<string>('')
  const selectedScheduler = ref<string>('')

  const status = ref<Status>('idle')
  const running = ref(false)
  const startedAt = ref<number | null>(null)
  const finishedAt = ref<number | null>(null)
  const errorMessage = ref('')
  const taskId = ref<string>('')
  const progress = ref<ProgressState>({ ...DEFAULT_PROGRESS })
  const gallery = ref<GeneratedImage[]>([])
  const info = ref<Record<string, unknown> | null>(null)
  const profileMessage = ref('')
  const lastSeed = ref<number | null>(null)

  let unsubscribe: (() => void) | null = null

  const isRunning = computed(() => status.value === 'running')

  const aspectLabel = computed(() => {
    const g = (a: number, b: number): number => (b ? g(b, a % b) : a)
    const d = g(width.value, height.value) || 1
    return `${width.value / d}:${height.value / d}`
  })

  function resetProgress(): void {
    progress.value = { ...DEFAULT_PROGRESS }
  }

  function stopStream(): void {
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
  }

  async function ensureEngine(): Promise<void> {
    if (quicksettings.currentEngine !== ENGINE_ID) {
      quicksettings.currentEngine = ENGINE_ID
    }
  }

  async function init(): Promise<void> {
    await ensureEngine()
    await Promise.all([loadModels(), loadSamplers(), loadSchedulers()])
    loadProfile()
  }

  async function loadModels(): Promise<void> {
    const res = await fetchModels()
    models.value = res.models
    if (!selectedModel.value && res.current) {
      selectedModel.value = res.current
    } else if (!selectedModel.value && res.models.length > 0) {
      selectedModel.value = res.models[0].title
      await updateModel(res.models[0].title)
    }
  }

  async function updateModel(title: string): Promise<void> {
    selectedModel.value = title
    await updateOptions({ sd_model_checkpoint: title })
  }

  async function loadSamplers(): Promise<void> {
    const res = await fetchSamplers()
    samplers.value = res.samplers
    if (!selectedSampler.value && res.samplers.length > 0) {
      selectedSampler.value = res.samplers[0].name
    }
  }

  async function loadSchedulers(): Promise<void> {
    const res = await fetchSchedulers()
    schedulers.value = res.schedulers
    if (!selectedScheduler.value && res.schedulers.length > 0) {
      selectedScheduler.value = res.schedulers[0].name
    }
  }

  function loadProfile(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const snapshot = JSON.parse(raw) as {
        prompt: string
        negativePrompt: string
        steps: number
        cfgScale: number
        width: number
        height: number
        seed: number
        batchSize: number
        batchCount: number
        selectedModel?: string
        selectedSampler?: string
        selectedScheduler?: string
      }
      prompt.value = snapshot.prompt ?? ''
      negativePrompt.value = snapshot.negativePrompt ?? ''
      steps.value = snapshot.steps ?? 8
      cfgScale.value = snapshot.cfgScale ?? 4.0
      width.value = snapshot.width ?? 1024
      height.value = snapshot.height ?? 1024
      seed.value = snapshot.seed ?? -1
      batchSize.value = snapshot.batchSize ?? 1
      batchCount.value = snapshot.batchCount ?? 1
      if (snapshot.selectedModel) selectedModel.value = snapshot.selectedModel
      if (snapshot.selectedSampler) selectedSampler.value = snapshot.selectedSampler
      if (snapshot.selectedScheduler) selectedScheduler.value = snapshot.selectedScheduler
    } catch (error) {
      console.error('Failed to load Z Image profile', error)
    }
  }

  function saveProfile(): void {
    try {
      const snapshot = {
        prompt: prompt.value,
        negativePrompt: negativePrompt.value,
        steps: steps.value,
        cfgScale: cfgScale.value,
        width: width.value,
        height: height.value,
        seed: seed.value,
        batchSize: batchSize.value,
        batchCount: batchCount.value,
        selectedModel: selectedModel.value,
        selectedSampler: selectedSampler.value,
        selectedScheduler: selectedScheduler.value,
      }
      localStorage.setItem(STORAGE_KEY, JSON.stringify(snapshot))
      profileMessage.value = 'Profile saved.'
    } catch (error) {
      profileMessage.value = error instanceof Error ? error.message : String(error)
    }
  }

  async function generate(): Promise<void> {
    if (!models.value.length) await loadModels()
    if (!samplers.value.length) await loadSamplers()
    if (!schedulers.value.length) await loadSchedulers()

    if (!selectedSampler.value) {
      status.value = 'error'
      running.value = false
      errorMessage.value = 'Sampler is required.'
      return
    }
    if (!selectedScheduler.value) {
      status.value = 'error'
      running.value = false
      errorMessage.value = 'Scheduler is required.'
      return
    }

    stopStream()
    status.value = 'running'
    running.value = true
    startedAt.value = performance.now()
    finishedAt.value = null
    errorMessage.value = ''
    gallery.value = []
    info.value = null
    resetProgress()
    profileMessage.value = ''
    lastSeed.value = seed.value

    const promptText = prompt.value.trim()
    if (!promptText) {
      status.value = 'error'
      running.value = false
      errorMessage.value = 'Prompt is required.'
      return
    }

    const extras: Record<string, unknown> = {}
    // Z Image models (GGUF, FP8, BF16) are all core-only and require external text encoder
    const tencLabel = quicksettings.currentTextEncoders[0]
    const tencSha = quicksettings.resolveTextEncoderSha(tencLabel)
    if (tencSha) {
      extras.tenc_sha = tencSha
    } else {
      status.value = 'error'
      running.value = false
      errorMessage.value = 'Select a Z Image text encoder.'
      return
    }

    let payload: Txt2ImgRequest
    try {
      payload = buildTxt2ImgPayload({
        prompt: promptText,
        negativePrompt: negativePrompt.value || '',
        width: width.value,
        height: height.value,
        steps: steps.value,
        guidanceScale: cfgScale.value,
        sampler: selectedSampler.value || 'automatic',
        scheduler: selectedScheduler.value || 'automatic',
        seed: seed.value,
        batchSize: batchSize.value,
        batchCount: batchCount.value,
        styles: [],
        device: quicksettings.currentDevice || 'cpu',
        smartOffload: quicksettings.smartOffload,
        smartFallback: quicksettings.smartFallback,
        smartCache: quicksettings.smartCache,
        engine: ENGINE_ID,
        model: quicksettings.currentModel || selectedModel.value,
        extras,
      })
    } catch (error) {
      status.value = 'error'
      running.value = false
      errorMessage.value = formatZodError(error)
      return
    }

    try {
      const { task_id } = await startTxt2Img(payload)
      taskId.value = task_id
      resetProgress()
      progress.value.stage = 'submitted'
      unsubscribe = subscribeTask(task_id, handleTaskEvent)
    } catch (error) {
      status.value = 'error'
      errorMessage.value = error instanceof Error ? error.message : String(error)
      stopStream()
    }
  }

  function handleTaskEvent(event: TaskEvent): void {
    switch (event.type) {
      case 'progress':
        progress.value.stage = event.stage
        progress.value.step = event.step ?? 0
        progress.value.totalSteps = event.total_steps ?? 0
        break
      case 'result':
        // TaskEvent 'result' has images and info directly on the event
        if (event.images) {
          gallery.value = event.images
        }
        if (event.info) {
          try {
            // info may already be parsed object or string
            const infoObj = typeof event.info === 'string' 
              ? JSON.parse(event.info) as Record<string, unknown>
              : event.info as Record<string, unknown>
            info.value = infoObj
            const raw = (infoObj as any).seed ?? (infoObj as any).all_seeds?.[0]
            const resolvedSeed = typeof raw === 'number' ? raw : Number(raw)
            if (Number.isFinite(resolvedSeed)) {
              lastSeed.value = resolvedSeed
            }
          } catch {
            info.value = { raw: event.info }
          }
        }
        if (status.value === 'running') {
          status.value = 'done'
          finishedAt.value = performance.now()
        }
        break
      case 'error':
        status.value = 'error'
        errorMessage.value = event.message
        stopStream()
        break
      case 'end':
        if (status.value !== 'error') {
          status.value = 'done'
        }
        stopStream()
        break
    }
  }

  function randomizeSeed(): void {
    seed.value = Math.floor(Math.random() * 2 ** 31)
  }

  function reuseSeed(): void {
    if (lastSeed.value !== null) {
      seed.value = lastSeed.value
    }
  }

  const gentimeMs = computed(() => {
    if (startedAt.value === null || finishedAt.value === null) return null
    return Math.max(0, finishedAt.value - startedAt.value)
  })

  return {
    ENGINE_ID,
    prompt,
    negativePrompt,
    steps,
    cfgScale,
    width,
    height,
    seed,
    batchSize,
    batchCount,
    models,
    samplers,
    schedulers,
    selectedModel,
    selectedSampler,
    selectedScheduler,
    status,
    running,
    errorMessage,
    taskId,
    progress,
    gallery,
    info,
    profileMessage,
    aspectLabel,
    gentimeMs,
    isRunning,
    init,
    loadProfile,
    loadModels,
    loadSamplers,
    loadSchedulers,
    updateModel,
    stopStream,
    generate,
    saveProfile,
    randomizeSeed,
    reuseSeed,
  }
})

export type ZImageStore = ReturnType<typeof useZImageStore>
