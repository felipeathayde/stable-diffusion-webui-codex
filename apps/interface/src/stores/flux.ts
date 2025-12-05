import { defineStore } from 'pinia'
import { ref, computed, reactive } from 'vue'
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
import type { Txt2ImgRequest, HighresFormState, RefinerFormState } from '../api/payloads'
import { useQuicksettingsStore } from './quicksettings'

const ENGINE_ID = 'flux'
const STORAGE_KEY = 'codex.flux.profile'

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

export const useFluxStore = defineStore('flux', () => {
  const quicksettings = useQuicksettingsStore()

  const prompt = ref('')
  const negativePrompt = ref('')
  const steps = ref(20)
  const cfgScale = ref(3.5)
  const width = ref(1024)
  const height = ref(1024)
  const seed = ref(-1)
  const batchSize = ref(1)
  const batchCount = ref(1)
  const styles = ref<string[]>([])
  const highres = reactive<HighresFormState>({
    enabled: false,
    denoise: 0.4,
    scale: 1.5,
    resizeX: 0,
    resizeY: 0,
    steps: 0,
    upscaler: 'Latent (bicubic antialiased)',
    checkpoint: undefined,
    modules: [],
    sampler: undefined,
    scheduler: undefined,
    prompt: undefined,
    negativePrompt: undefined,
    cfg: undefined,
    distilledCfg: undefined,
    refiner: {
      enabled: false,
      steps: 0,
      cfg: 3.5,
      seed: -1,
      model: undefined,
      vae: undefined,
    },
  })
  const refiner = reactive<RefinerFormState>({
    enabled: false,
    steps: 0,
    cfg: 3.5,
    seed: -1,
    model: undefined,
    vae: undefined,
  })

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
        styles: string[]
        selectedModel?: string
        selectedSampler?: string
        selectedScheduler?: string
      }
      prompt.value = snapshot.prompt ?? ''
      negativePrompt.value = snapshot.negativePrompt ?? ''
      steps.value = snapshot.steps ?? 20
      cfgScale.value = snapshot.cfgScale ?? 3.5
      width.value = snapshot.width ?? 1024
      height.value = snapshot.height ?? 1024
      seed.value = snapshot.seed ?? -1
      batchSize.value = snapshot.batchSize ?? 1
      batchCount.value = snapshot.batchCount ?? 1
      styles.value = snapshot.styles ?? []
      if (snapshot.selectedModel) selectedModel.value = snapshot.selectedModel
      if (snapshot.selectedSampler) selectedSampler.value = snapshot.selectedSampler
      if (snapshot.selectedScheduler) selectedScheduler.value = snapshot.selectedScheduler
    } catch (error) {
      console.error('Failed to load FLUX profile', error)
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
        styles: styles.value,
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
    // Make sure essentials are loaded before building the payload.
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
    const negativeText = negativePrompt.value.trim()

    let payload: Txt2ImgRequest
    try {
      payload = buildTxt2ImgPayload({
        prompt: promptText,
        negativePrompt: negativeText,
        steps: steps.value,
        cfgScale: cfgScale.value,
        width: width.value,
        height: height.value,
        seed: seed.value,
        batchSize: batchSize.value,
        batchCount: batchCount.value,
        styles: styles.value,
        device: quicksettings.currentDevice || 'cpu',
        smartOffload: quicksettings.smartOffload,
        smartFallback: quicksettings.smartFallback,
        smartCache: quicksettings.smartCache,
        engine: ENGINE_ID,
        model: selectedModel.value,
        highres,
        refiner,
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
      return
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
        if (event.payload && event.payload.images) {
          gallery.value = event.payload.images
        }
        if (event.payload && event.payload.info) {
          try {
            const parsed = JSON.parse(event.payload.info) as Record<string, unknown>
            info.value = parsed
            const raw = (parsed as any).seed ?? (parsed as any).all_seeds?.[0]
            const resolvedSeed = typeof raw === 'number' ? raw : Number(raw)
            if (Number.isFinite(resolvedSeed)) {
              lastSeed.value = resolvedSeed
            }
          } catch (e) {
            info.value = { raw: event.payload.info }
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
      default:
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
    styles,
    highres,
    refiner,
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

export type FluxStore = ReturnType<typeof useFluxStore>
