import { defineStore } from 'pinia'
import { ref, computed, watch, reactive } from 'vue'
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

const ENGINE_ID = 'sdxl'
const DEFAULT_WIDTH = 1024
const DEFAULT_HEIGHT = 1024
const STORAGE_KEY = 'codex.sdxl.profile.v1'

interface ProgressState {
  stage: string
  percent: number | null
  etaSeconds: number | null
  step: number | null
  totalSteps: number | null
}

type Status = 'idle' | 'running' | 'error' | 'done'

const DEFAULT_PROGRESS: ProgressState = {
  stage: 'idle',
  percent: null,
  etaSeconds: null,
  step: null,
  totalSteps: null,
}

export const useSdxlStore = defineStore('sdxl', () => {
  const prompt = ref('')
  const negativePrompt = ref('')
  const steps = ref(30)
  const cfgScale = ref(7)
  const width = ref(DEFAULT_WIDTH)
  const height = ref(DEFAULT_HEIGHT)
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
    upscaler: 'Use same upscaler',
  })
  const refiner = reactive<RefinerFormState>({
    enabled: false,
    steps: 20,
    cfg: cfgScale.value,
    seed: -1,
  })
  const lastSeed = ref<number | null>(null)

  const models = ref<ModelInfo[]>([])
  const samplers = ref<SamplerInfo[]>([])
  const schedulers = ref<SchedulerInfo[]>([])

  const selectedModel = ref<string>('')
  const selectedSampler = ref<string>('')
  const selectedScheduler = ref<string>('')

  const status = ref<Status>('idle')
  const running = ref(false)
  const errorMessage = ref('')
  const taskId = ref<string>('')
  const progress = ref<ProgressState>({ ...DEFAULT_PROGRESS })
  const gallery = ref<GeneratedImage[]>([])
  const info = ref<unknown>(null)
  const profileMessage = ref('')
  let unsubscribe: (() => void) | null = null
  const quicksettings = useQuicksettingsStore()

  function resetProgress(): void {
    progress.value = { ...DEFAULT_PROGRESS }
  }

  async function ensureEngine(): Promise<void> {
    await updateOptions({ codex_engine: ENGINE_ID })
  }

  function loadProfile(): void {
    try {
      const raw = localStorage.getItem(STORAGE_KEY)
      if (!raw) return
      const snap = JSON.parse(raw) as Record<string, unknown>

      const numberOr = (value: unknown, fallback: number): number => {
        const n = Number(value)
        return Number.isFinite(n) ? n : fallback
      }

      if (typeof snap.prompt === 'string') prompt.value = snap.prompt
      if (typeof snap.negativePrompt === 'string') negativePrompt.value = snap.negativePrompt
      steps.value = numberOr(snap.steps, steps.value)
      cfgScale.value = numberOr(snap.cfgScale, cfgScale.value)
      width.value = numberOr(snap.width, width.value)
      height.value = numberOr(snap.height, height.value)
      seed.value = numberOr(snap.seed, seed.value)
      batchSize.value = numberOr(snap.batchSize, batchSize.value)
      batchCount.value = numberOr(snap.batchCount, batchCount.value)

      if (Array.isArray(snap.styles)) {
        styles.value = snap.styles.map((entry) => String(entry))
      }

      if (typeof snap.selectedModel === 'string') selectedModel.value = snap.selectedModel
      if (typeof snap.selectedSampler === 'string') selectedSampler.value = snap.selectedSampler
      if (typeof snap.selectedScheduler === 'string') selectedScheduler.value = snap.selectedScheduler

      profileMessage.value = 'Loaded saved profile.'
    } catch (error) {
      console.warn('[sdxl] failed to load profile', error)
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
    await updateOptions({ sd_model_checkpoint: title, codex_engine: ENGINE_ID })
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

  function setSampler(name: string): void {
    selectedSampler.value = name
  }

  function setScheduler(name: string): void {
    selectedScheduler.value = name
  }

  function stopStream(): void {
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
    running.value = false
  }

  // Clear validation errors when the user edits the prompt/negative
  watch([prompt, negativePrompt], () => {
    if (status.value === 'error') {
      status.value = 'idle'
      errorMessage.value = ''
    }
  })

  async function generate(): Promise<void> {
    stopStream()
    status.value = 'running'
    running.value = true
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
      errorMessage.value = 'Prompt must not be empty'
      return
    }

    await ensureEngine()

    if (selectedModel.value) {
      await updateOptions({ sd_model_checkpoint: selectedModel.value })
    }

    let payload: Txt2ImgRequest
    try {
      payload = buildTxt2ImgPayload({
        prompt: promptText,
        negativePrompt: negativePrompt.value,
        width: width.value,
        height: height.value,
        steps: steps.value,
        guidanceScale: cfgScale.value,
        sampler: selectedSampler.value || 'automatic',
        scheduler: selectedScheduler.value || 'automatic',
        seed: seed.value,
        batchSize: batchSize.value,
        batchCount: batchCount.value,
        styles: styles.value,
        device: quicksettings.currentDevice,
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
      case 'status':
        progress.value.stage = event.stage
        break
      case 'progress':
        progress.value = {
          stage: event.stage,
          percent: event.percent ?? null,
          etaSeconds: event.eta_seconds ?? null,
          step: event.step ?? null,
          totalSteps: event.total_steps ?? null,
        }
        break
      case 'result':
        gallery.value = event.images

        const baseInfo = event.info && typeof event.info === 'object' ? { ...(event.info as Record<string, unknown>) } : {}
        const promptText = prompt.value
        const negativeText = negativePrompt.value

        const seedCandidates: Array<unknown> = []
        const asRecord = baseInfo as Record<string, unknown>
        if ('seed' in asRecord) seedCandidates.push(asRecord.seed)
        if (Array.isArray(asRecord.all_seeds) && asRecord.all_seeds.length > 0) seedCandidates.push(asRecord.all_seeds[0])
        if (Array.isArray(asRecord.seeds) && asRecord.seeds.length > 0) seedCandidates.push(asRecord.seeds[0])
        seedCandidates.push(lastSeed.value)
        if (seed.value !== -1) seedCandidates.push(seed.value)

        const resolvedSeed = seedCandidates.map((v) => Number(v)).find((n) => Number.isFinite(n) && n !== -1)

        if (typeof resolvedSeed === 'number') {
          baseInfo.seed = resolvedSeed
          lastSeed.value = resolvedSeed
        }

        info.value = {
          ...baseInfo,
          prompt: promptText,
          negative_prompt: negativeText,
          output_dir: 'outputs/txt2img-images',
        }
        break
      case 'error':
        status.value = 'error'
        errorMessage.value = event.message ?? 'Unknown error'
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

  const isRunning = computed(() => running.value)

  function randomizeSeed(): void {
    if (seed.value !== -1) {
      lastSeed.value = seed.value
    }
    seed.value = -1
  }

  function reuseSeed(): void {
    if (lastSeed.value !== null) seed.value = lastSeed.value
  }

  const aspectLabel = computed(() => {
    const gcd = (a: number, b: number): number => (b === 0 ? a : gcd(b, a % b))
    const d = gcd(width.value, height.value)
    return `${width.value / d}:${height.value / d}`
  })

  return {
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
    lastSeed,
    models,
    samplers,
    schedulers,
    selectedModel,
    selectedSampler,
    selectedScheduler,
    status,
    errorMessage,
    taskId,
    progress,
    gallery,
    info,
    profileMessage,
    running,
    init,
    loadProfile,
    loadModels,
    loadSamplers,
    loadSchedulers,
    updateModel,
    setSampler,
    setScheduler,
    stopStream,
    generate,
    saveProfile,
    randomizeSeed,
    reuseSeed,
    isRunning,
    aspectLabel,
  }
})

export type SdxlStore = ReturnType<typeof useSdxlStore>
