import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { GeneratedImage, ModelInfo, SamplerInfo, SchedulerInfo, TaskEvent } from '../api/types'
import {
  fetchModels,
  fetchSamplers,
  fetchSchedulers,
  updateOptions,
  startTxt2Vid,
  startImg2Vid,
  subscribeTask,
} from '../api/client'

type Status = 'idle' | 'running' | 'error' | 'done'

interface ProgressState {
  stage: string
  percent: number | null
  etaSeconds: number | null
  step: number | null
  totalSteps: number | null
}

const DEFAULT_PROGRESS: ProgressState = {
  stage: 'idle',
  percent: null,
  etaSeconds: null,
  step: null,
  totalSteps: null,
}

const DEFAULT_WIDTH = 768
const DEFAULT_HEIGHT = 432
const DEFAULT_VIDEO_FILENAME_PREFIX = 'wan22'
const DEFAULT_VIDEO_FORMAT = 'video/h264-mp4'
const DEFAULT_VIDEO_PIX_FMT = 'yuv420p'
const DEFAULT_VIDEO_CRF = 15
const DEFAULT_VIDEO_LOOP_COUNT = 0
const DEFAULT_RIFE_MODEL = 'rife47.pth'
const DEFAULT_RIFE_TIMES = 2

function cloneProgress(): ProgressState {
  return { ...DEFAULT_PROGRESS }
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

async function readFileAsDataURL(file: File): Promise<string> {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

export const useTxt2VidStore = defineStore('txt2vid', () => {
  const prompt = ref('')
  const negativePrompt = ref('')
  const width = ref(DEFAULT_WIDTH)
  const height = ref(DEFAULT_HEIGHT)
  const frames = ref(16)
  const fps = ref(24)
  const steps = ref(30)
  const cfgScale = ref(7)
  const seed = ref(-1)
  const sampler = ref('')
  const scheduler = ref('')
  // WAN 2.2 dual-stage params
  const hiSampler = ref('')
  const hiScheduler = ref('')
  const hiSteps = ref(30)
  const hiCfgScale = ref(7)
  const hiSeed = ref(-1)
  const loSampler = ref('')
  const loScheduler = ref('')
  const loSteps = ref(15)
  const loCfgScale = ref(5)
  const loSeed = ref(-1)
  // WAN per-stage models/LoRAs
  const hiModelDir = ref('')
  const loModelDir = ref('')
  const hiUseLora = ref(false)
  const hiLoraPath = ref('')
  const hiLoraWeight = ref(1.0)
  const loUseLora = ref(false)
  const loLoraPath = ref('')
  const loLoraWeight = ref(1.0)
  const batchSize = ref(1)
  const styles = ref<string[]>([])
  const filenamePrefix = ref(DEFAULT_VIDEO_FILENAME_PREFIX)
  const videoFormat = ref(DEFAULT_VIDEO_FORMAT)
  const videoPixFormat = ref(DEFAULT_VIDEO_PIX_FMT)
  const videoCrf = ref(DEFAULT_VIDEO_CRF)
  const videoLoopCount = ref(DEFAULT_VIDEO_LOOP_COUNT)
  const videoPingpong = ref(false)
  const videoSaveMetadata = ref(true)
  const videoSaveOutput = ref(true)
  const videoTrimToAudio = ref(false)
  const rifeEnabled = ref(true)
  const rifeModel = ref(DEFAULT_RIFE_MODEL)
  const rifeTimes = ref(DEFAULT_RIFE_TIMES)
  const lastSeed = ref<number | null>(null)
  const lastSeedHi = ref<number | null>(null)
  const lastSeedLo = ref<number | null>(null)

  const models = ref<ModelInfo[]>([])
  const samplers = ref<SamplerInfo[]>([])
  const schedulers = ref<SchedulerInfo[]>([])
  const selectedModel = ref('')

  const status = ref<Status>('idle')
  const errorMessage = ref('')
  const taskId = ref('')
  const progress = ref<ProgressState>(cloneProgress())
  const framesResult = ref<GeneratedImage[]>([])
  const info = ref<unknown>(null)
  let unsubscribe: (() => void) | null = null

  async function init(): Promise<void> {
    await Promise.all([loadModels(), loadSamplers(), loadSchedulers()])
  }

  async function loadModels(): Promise<void> {
    const res = await fetchModels()
    models.value = res.models
    if (!selectedModel.value && res.current) {
      selectedModel.value = res.current
    } else if (!selectedModel.value && res.models.length > 0) {
      selectedModel.value = res.models[0].title
      await updateOptions({ sd_model_checkpoint: selectedModel.value })
    }
  }

  async function setModel(title: string): Promise<void> {
    selectedModel.value = title
    await updateOptions({ sd_model_checkpoint: title })
  }

  async function loadSamplers(): Promise<void> {
    const res = await fetchSamplers()
    samplers.value = res.samplers
    if (!sampler.value && res.samplers.length > 0) sampler.value = res.samplers[0].name
    if (!hiSampler.value && res.samplers.length > 0) hiSampler.value = res.samplers[0].name
    if (!loSampler.value && res.samplers.length > 0) loSampler.value = res.samplers[0].name
  }

  async function loadSchedulers(): Promise<void> {
    const res = await fetchSchedulers()
    schedulers.value = res.schedulers
    if (!scheduler.value && res.schedulers.length > 0) scheduler.value = res.schedulers[0].name
  }

  function stopStream(): void {
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
  }

  async function generate(): Promise<void> {
    stopStream()
    status.value = 'running'
    errorMessage.value = ''
    framesResult.value = []
    info.value = null
    progress.value = cloneProgress()
    lastSeed.value = seed.value

    if (selectedModel.value) {
      await updateOptions({ sd_model_checkpoint: selectedModel.value })
    }

    const payload: Record<string, unknown> = {
      __strict_version: 1,
      txt2vid_prompt: prompt.value,
      txt2vid_neg_prompt: negativePrompt.value,
      txt2vid_width: width.value,
      txt2vid_height: height.value,
      txt2vid_num_frames: frames.value,
      txt2vid_fps: fps.value,
      txt2vid_steps: steps.value,
      txt2vid_cfg_scale: cfgScale.value,
      txt2vid_seed: seed.value,
      txt2vid_sampler: sampler.value,
      txt2vid_scheduler: scheduler.value,
      txt2vid_batch_size: batchSize.value,
      txt2vid_styles: styles.value,
      video_filename_prefix: filenamePrefix.value,
      video_format: videoFormat.value,
      video_pix_fmt: videoPixFormat.value,
      video_crf: videoCrf.value,
      video_loop_count: videoLoopCount.value,
      video_pingpong: videoPingpong.value,
      video_save_metadata: videoSaveMetadata.value,
      video_save_output: videoSaveOutput.value,
      video_trim_to_audio: videoTrimToAudio.value,
      video_interpolation: {
        enabled: rifeEnabled.value,
        model: rifeModel.value,
        times: rifeTimes.value,
      },
      wan_high: {
        sampler: hiSampler.value,
        scheduler: hiScheduler.value,
        steps: hiSteps.value,
        cfg_scale: hiCfgScale.value,
        seed: hiSeed.value,
        model_dir: hiModelDir.value || undefined,
        lora_path: hiUseLora.value ? (hiLoraPath.value || undefined) : undefined,
        lora_weight: hiLoraWeight.value,
      },
      wan_low: {
        sampler: loSampler.value,
        scheduler: loScheduler.value,
        steps: loSteps.value,
        cfg_scale: loCfgScale.value,
        seed: loSeed.value,
        model_dir: loModelDir.value || undefined,
        lora_path: loUseLora.value ? (loLoraPath.value || undefined) : undefined,
        lora_weight: loLoraWeight.value,
      },
    }

    try {
      const { task_id } = await startTxt2Vid(payload)
      taskId.value = task_id
      progress.value.stage = 'submitted'
      unsubscribe = subscribeTask(task_id, handleTaskEvent)
    } catch (error) {
      status.value = 'error'
      errorMessage.value = error instanceof Error ? error.message : String(error)
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
        framesResult.value = event.images
        info.value = event.info
        status.value = 'done'
        if (event.info && typeof event.info === 'object') {
          const val = Number((event.info as Record<string, unknown>).seed)
          if (!Number.isNaN(val)) lastSeed.value = val
        }
        break
      case 'error':
        status.value = 'error'
        errorMessage.value = event.message
        stopStream()
        break
      case 'end':
        if (status.value !== 'error') status.value = 'done'
        stopStream()
        break
      default:
        break
    }
  }

  const isRunning = computed(() => status.value === 'running')

  function randomizeSeed(): void {
    if (seed.value !== -1) lastSeed.value = seed.value
    seed.value = -1
  }

  function reuseSeed(): void {
    if (lastSeed.value !== null) seed.value = lastSeed.value
  }

  function randomizeSeedHi(): void { if (hiSeed.value !== -1) lastSeedHi.value = hiSeed.value; hiSeed.value = -1 }
  function randomizeSeedLo(): void { if (loSeed.value !== -1) lastSeedLo.value = loSeed.value; loSeed.value = -1 }
  function reuseSeedHi(): void { if (lastSeedHi.value !== null) hiSeed.value = lastSeedHi.value }
  function reuseSeedLo(): void { if (lastSeedLo.value !== null) loSeed.value = lastSeedLo.value }

  return {
    prompt,
    negativePrompt,
    width,
    height,
    frames,
    fps,
    steps,
    cfgScale,
    seed,
    sampler,
    scheduler,
    hiSampler,
    hiScheduler,
    hiSteps,
    hiCfgScale,
    hiSeed,
    loSampler,
    loScheduler,
    loSteps,
    loCfgScale,
    loSeed,
    hiModelDir,
    loModelDir,
    hiUseLora,
    hiLoraPath,
    hiLoraWeight,
    loUseLora,
    loLoraPath,
    loLoraWeight,
    batchSize,
    styles,
    filenamePrefix,
    videoFormat,
    videoPixFormat,
    videoCrf,
    videoLoopCount,
    videoPingpong,
    videoSaveMetadata,
    videoSaveOutput,
    videoTrimToAudio,
    rifeEnabled,
    rifeModel,
    rifeTimes,
    models,
    samplers,
    schedulers,
    selectedModel,
    status,
    errorMessage,
    taskId,
    progress,
    framesResult,
    info,
    init,
    loadModels,
    loadSamplers,
    loadSchedulers,
    generate,
    stopStream,
    isRunning,
    toDataUrl,
    setModel,
    randomizeSeed,
    reuseSeed,
    randomizeSeedHi,
    randomizeSeedLo,
    reuseSeedHi,
    reuseSeedLo,
  }
})

export const useImg2VidStore = defineStore('img2vid', () => {
  const prompt = ref('')
  const negativePrompt = ref('')
  const width = ref(DEFAULT_WIDTH)
  const height = ref(DEFAULT_HEIGHT)
  const frames = ref(16)
  const fps = ref(24)
  const steps = ref(30)
  const cfgScale = ref(7)
  const seed = ref(-1)
  const sampler = ref('')
  const scheduler = ref('')
  // WAN 2.2 dual-stage params
  const hiSampler = ref('')
  const hiScheduler = ref('')
  const hiSteps = ref(30)
  const hiCfgScale = ref(7)
  const hiSeed = ref(-1)
  const loSampler = ref('')
  const loScheduler = ref('')
  const loSteps = ref(15)
  const loCfgScale = ref(5)
  const loSeed = ref(-1)
  // WAN per-stage models/LoRAs
  const hiModelDir = ref('')
  const loModelDir = ref('')
  const hiLoraPath = ref('')
  const hiLoraWeight = ref(1.0)
  const loLoraPath = ref('')
  const loLoraWeight = ref(1.0)
  const styles = ref<string[]>([])
  const filenamePrefix = ref(DEFAULT_VIDEO_FILENAME_PREFIX)
  const videoFormat = ref(DEFAULT_VIDEO_FORMAT)
  const videoPixFormat = ref(DEFAULT_VIDEO_PIX_FMT)
  const videoCrf = ref(DEFAULT_VIDEO_CRF)
  const videoLoopCount = ref(DEFAULT_VIDEO_LOOP_COUNT)
  const videoPingpong = ref(false)
  const videoSaveMetadata = ref(true)
  const videoSaveOutput = ref(true)
  const videoTrimToAudio = ref(false)
  const rifeEnabled = ref(true)
  const rifeModel = ref(DEFAULT_RIFE_MODEL)
  const rifeTimes = ref(DEFAULT_RIFE_TIMES)
  const lastSeed = ref<number | null>(null)
  const lastSeedHi = ref<number | null>(null)
  const lastSeedLo = ref<number | null>(null)

  const models = ref<ModelInfo[]>([])
  const samplers = ref<SamplerInfo[]>([])
  const schedulers = ref<SchedulerInfo[]>([])
  const selectedModel = ref('')

  const initImageData = ref('')
  const initImageName = ref('')

  const status = ref<Status>('idle')
  const errorMessage = ref('')
  const taskId = ref('')
  const progress = ref<ProgressState>(cloneProgress())
  const framesResult = ref<GeneratedImage[]>([])
  const info = ref<unknown>(null)
  let unsubscribe: (() => void) | null = null

  async function init(): Promise<void> {
    await Promise.all([loadModels(), loadSamplers(), loadSchedulers()])
  }

  async function loadModels(): Promise<void> {
    const res = await fetchModels()
    models.value = res.models
    if (!selectedModel.value && res.current) {
      selectedModel.value = res.current
    } else if (!selectedModel.value && res.models.length > 0) {
      selectedModel.value = res.models[0].title
      await updateOptions({ sd_model_checkpoint: selectedModel.value })
    }
  }

  async function setModel(title: string): Promise<void> {
    selectedModel.value = title
    await updateOptions({ sd_model_checkpoint: title })
  }

  async function loadSamplers(): Promise<void> {
    const res = await fetchSamplers()
    samplers.value = res.samplers
    if (!sampler.value && res.samplers.length > 0) sampler.value = res.samplers[0].name
    if (!hiSampler.value && res.samplers.length > 0) hiSampler.value = res.samplers[0].name
    if (!loSampler.value && res.samplers.length > 0) loSampler.value = res.samplers[0].name
  }

  async function loadSchedulers(): Promise<void> {
    const res = await fetchSchedulers()
    schedulers.value = res.schedulers
    if (!scheduler.value && res.schedulers.length > 0) scheduler.value = res.schedulers[0].name
  }

  async function setInitImage(file: File): Promise<void> {
    const dataUrl = await readFileAsDataURL(file)
    initImageData.value = dataUrl
    initImageName.value = file.name
  }

  function clearInitImage(): void {
    initImageData.value = ''
    initImageName.value = ''
  }

  function stopStream(): void {
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
  }

  async function generate(): Promise<void> {
    if (!initImageData.value) {
      status.value = 'error'
      errorMessage.value = 'Initial image required.'
      throw new Error(errorMessage.value)
    }

    stopStream()
    status.value = 'running'
    errorMessage.value = ''
    framesResult.value = []
    info.value = null
    progress.value = cloneProgress()
    lastSeed.value = seed.value

    if (selectedModel.value) {
      await updateOptions({ sd_model_checkpoint: selectedModel.value })
    }

    const payload: Record<string, unknown> = {
      __strict_version: 1,
      img2vid_prompt: prompt.value,
      img2vid_neg_prompt: negativePrompt.value,
      img2vid_width: width.value,
      img2vid_height: height.value,
      img2vid_num_frames: frames.value,
      img2vid_fps: fps.value,
      img2vid_steps: steps.value,
      img2vid_cfg_scale: cfgScale.value,
      img2vid_seed: seed.value,
      img2vid_sampler: sampler.value,
      img2vid_scheduler: scheduler.value,
      img2vid_init_image: initImageData.value,
      img2vid_styles: styles.value,
      video_filename_prefix: filenamePrefix.value,
      video_format: videoFormat.value,
      video_pix_fmt: videoPixFormat.value,
      video_crf: videoCrf.value,
      video_loop_count: videoLoopCount.value,
      video_pingpong: videoPingpong.value,
      video_save_metadata: videoSaveMetadata.value,
      video_save_output: videoSaveOutput.value,
      video_trim_to_audio: videoTrimToAudio.value,
      video_interpolation: {
        enabled: rifeEnabled.value,
        model: rifeModel.value,
        times: rifeTimes.value,
      },
      wan_high: {
        sampler: hiSampler.value,
        scheduler: hiScheduler.value,
        steps: hiSteps.value,
        cfg_scale: hiCfgScale.value,
        seed: hiSeed.value,
        model_dir: hiModelDir.value || undefined,
        lora_path: hiLoraPath.value || undefined,
        lora_weight: hiLoraWeight.value,
      },
      wan_low: {
        sampler: loSampler.value,
        scheduler: loScheduler.value,
        steps: loSteps.value,
        cfg_scale: loCfgScale.value,
        seed: loSeed.value,
        model_dir: loModelDir.value || undefined,
        lora_path: loLoraPath.value || undefined,
        lora_weight: loLoraWeight.value,
      },
    }

    try {
      const { task_id } = await startImg2Vid(payload)
      taskId.value = task_id
      progress.value.stage = 'submitted'
      unsubscribe = subscribeTask(task_id, handleTaskEvent)
    } catch (error) {
      status.value = 'error'
      errorMessage.value = error instanceof Error ? error.message : String(error)
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
        framesResult.value = event.images
        info.value = event.info
        status.value = 'done'
        if (event.info && typeof event.info === 'object') {
          const val = Number((event.info as Record<string, unknown>).seed)
          if (!Number.isNaN(val)) lastSeed.value = val
        }
        break
      case 'error':
        status.value = 'error'
        errorMessage.value = event.message
        stopStream()
        break
      case 'end':
        if (status.value !== 'error') status.value = 'done'
        stopStream()
        break
      default:
        break
    }
  }

  const isRunning = computed(() => status.value === 'running')

  function randomizeSeed(): void {
    if (seed.value !== -1) lastSeed.value = seed.value
    seed.value = -1
  }

  function reuseSeed(): void {
    if (lastSeed.value !== null) seed.value = lastSeed.value
  }

  function randomizeSeedHi(): void { if (hiSeed.value !== -1) lastSeedHi.value = hiSeed.value; hiSeed.value = -1 }
  function randomizeSeedLo(): void { if (loSeed.value !== -1) lastSeedLo.value = loSeed.value; loSeed.value = -1 }
  function reuseSeedHi(): void { if (lastSeedHi.value !== null) hiSeed.value = lastSeedHi.value }
  function reuseSeedLo(): void { if (lastSeedLo.value !== null) loSeed.value = lastSeedLo.value }

  return {
    prompt,
    negativePrompt,
    width,
    height,
    frames,
    fps,
    steps,
    cfgScale,
    seed,
    sampler,
    scheduler,
    hiSampler,
    hiScheduler,
    hiSteps,
    hiCfgScale,
    hiSeed,
    loSampler,
    loScheduler,
    loSteps,
    loCfgScale,
    loSeed,
    hiModelDir,
    loModelDir,
    hiLoraPath,
    hiLoraWeight,
    loLoraPath,
    loLoraWeight,
    styles,
    filenamePrefix,
    videoFormat,
    videoPixFormat,
    videoCrf,
    videoLoopCount,
    videoPingpong,
    videoSaveMetadata,
    videoSaveOutput,
    videoTrimToAudio,
    rifeEnabled,
    rifeModel,
    rifeTimes,
    models,
    samplers,
    schedulers,
    selectedModel,
    initImageData,
    initImageName,
    status,
    errorMessage,
    taskId,
    progress,
    framesResult,
    info,
    init,
    loadModels,
    loadSamplers,
    loadSchedulers,
    setInitImage,
    clearInitImage,
    generate,
    stopStream,
    isRunning,
    toDataUrl,
    setModel,
    randomizeSeed,
    reuseSeed,
    randomizeSeedHi,
    randomizeSeedLo,
    reuseSeedHi,
    reuseSeedLo,
  }
})
