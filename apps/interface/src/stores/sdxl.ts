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

const ENGINE_ID = 'sdxl'
const DEFAULT_WIDTH = 1024
const DEFAULT_HEIGHT = 1024

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
  let unsubscribe: (() => void) | null = null

  function resetProgress(): void {
    progress.value = { ...DEFAULT_PROGRESS }
  }

  async function ensureEngine(): Promise<void> {
    await updateOptions({ codex_engine: ENGINE_ID })
  }

  async function init(): Promise<void> {
    await ensureEngine()
    await Promise.all([loadModels(), loadSamplers(), loadSchedulers()])
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

  function buildPayload(): Record<string, unknown> {
    return {
      __strict_version: 1,
      codex_engine: ENGINE_ID,
      engine: ENGINE_ID,
      model: selectedModel.value,
      txt2img_prompt: prompt.value,
      txt2img_neg_prompt: negativePrompt.value,
      txt2img_styles: styles.value,
      txt2img_batch_count: batchCount.value,
      txt2img_batch_size: batchSize.value,
      txt2img_cfg_scale: cfgScale.value,
      txt2img_distilled_cfg_scale: 3.5,
      txt2img_height: height.value,
      txt2img_width: width.value,
      txt2img_hr_enable: false,
      txt2img_steps: steps.value,
      txt2img_sampling: selectedSampler.value,
      txt2img_scheduler: selectedScheduler.value,
      txt2img_seed: seed.value,
      txt2img_denoising_strength: 0.0,
      txt2img_hr_scale: 1.0,
      txt2img_hr_upscaler: 'Latent',
      txt2img_hires_steps: 0,
      txt2img_hr_resize_x: width.value,
      txt2img_hr_resize_y: height.value,
      hr_checkpoint: 'Use same checkpoint',
      hr_vae_te: ['Use same choices'],
      hr_sampler: 'Use same sampler',
      hr_scheduler: 'Use same scheduler',
      txt2img_hr_prompt: '',
      txt2img_hr_neg_prompt: '',
      txt2img_hr_cfg: cfgScale.value,
      txt2img_hr_distilled_cfg: 3.5,
    }
  }

  async function generate(): Promise<void> {
    stopStream()
    status.value = 'running'
    running.value = true
    errorMessage.value = ''
    gallery.value = []
    info.value = null
    resetProgress()
    lastSeed.value = seed.value

    await ensureEngine()

    if (selectedModel.value) {
      await updateOptions({ sd_model_checkpoint: selectedModel.value })
    }

    const payload = buildPayload()
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
      throw error
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
        info.value = event.info
        if (event.info && typeof event.info === 'object') {
          const infoSeed = (event.info as Record<string, unknown>)['seed']
          if (typeof infoSeed === 'number') lastSeed.value = infoSeed
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
    running,
    init,
    loadModels,
    loadSamplers,
    loadSchedulers,
    updateModel,
    setSampler,
    setScheduler,
    stopStream,
    generate,
    randomizeSeed,
    reuseSeed,
    isRunning,
    aspectLabel,
  }
})

export type SdxlStore = ReturnType<typeof useSdxlStore>
