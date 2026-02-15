/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Img2img generation store (init image + model/sampler/scheduler selection + SSE task updates).
Holds img2img form state, loads model/sampler/scheduler lists, starts `/api/img2img` jobs, and consumes task events to update progress and
gallery state.

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): Store status enum (`idle`/`running`/`error`/`done`).
- `ProgressState` (interface): Progress payload shape tracked during generation.
- `DEFAULT_PROGRESS` (constant): Default progress state for resets.
- `useImg2ImgStore` (store): Pinia store for img2img; owns form state, init-image selection, API calls, SSE subscription, and progress/gallery.
- `readFileAsDataURL` (function): Reads a File into a data URL for preview/upload.
- `readImageDimensions` (function): Extracts width/height from a data URL image.
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { ModelInfo, SamplerInfo, SchedulerInfo, GeneratedImage, TaskEvent } from '../api/types'
import { fetchModels, fetchSamplers, fetchSchedulers, startImg2Img, subscribeTask } from '../api/client'
import { useQuicksettingsStore } from './quicksettings'

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

export const useImg2ImgStore = defineStore('img2img', () => {
  const prompt = ref('')
  const negativePrompt = ref('')
  const steps = ref(20)
  const cfgScale = ref(7)
  const width = ref(512)
  const height = ref(512)
  const denoiseStrength = ref(0.5)
  const seed = ref(-1)
  const batchSize = ref(1)
  const batchCount = ref(1)
  const styles = ref<string[]>([])

  const models = ref<ModelInfo[]>([])
  const samplers = ref<SamplerInfo[]>([])
  const schedulers = ref<SchedulerInfo[]>([])

  const selectedModel = ref<string>('')
  const selectedSampler = ref<string>('')
  const selectedScheduler = ref<string>('')

  const initImageData = ref<string>('')
  const initImageName = ref<string>('')
  const initImagePreview = computed(() => initImageData.value || '')
  const hasInitImage = computed(() => Boolean(initImageData.value))

  const status = ref<Status>('idle')
  const errorMessage = ref('')
  const taskId = ref<string>('')
  const progress = ref<ProgressState>({ ...DEFAULT_PROGRESS })
  const gallery = ref<GeneratedImage[]>([])
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
    }
  }

  function updateModel(title: string): void {
    selectedModel.value = title
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
  }

  function resetProgress(): void {
    progress.value = { ...DEFAULT_PROGRESS }
  }

  async function setInitImage(file: File): Promise<void> {
    initImageName.value = file.name
    const dataUrl = await readFileAsDataURL(file)
    initImageData.value = dataUrl
    const dims = await readImageDimensions(dataUrl)
    width.value = dims.width
    height.value = dims.height
  }

  async function clearInitImage(): Promise<void> {
    initImageData.value = ''
    initImageName.value = ''
  }

  function buildPayload(): Record<string, unknown> {
    if (!initImageData.value) {
      throw new Error('Please select an initial image.')
    }
    if (!selectedModel.value) {
      throw new Error('Please select a model.')
    }

    const qs = useQuicksettingsStore()
    return {
      model: selectedModel.value,
      device: qs.currentDevice,
      settings_revision: qs.getSettingsRevision(),
      img2img_init_image: initImageData.value,
      img2img_prompt: prompt.value,
      img2img_neg_prompt: negativePrompt.value,
      img2img_styles: styles.value,
      img2img_batch_count: batchCount.value,
      img2img_batch_size: batchSize.value,
      img2img_cfg_scale: cfgScale.value,
      img2img_distilled_cfg_scale: 3.5,
      img2img_image_cfg_scale: 1.0,
      img2img_height: height.value,
      img2img_width: width.value,
      img2img_steps: steps.value,
      img2img_sampling: selectedSampler.value,
      img2img_scheduler: selectedScheduler.value,
      img2img_seed: seed.value,
      img2img_denoising_strength: denoiseStrength.value,
    }
  }

  async function generate(): Promise<void> {
    stopStream()
    status.value = 'running'
    errorMessage.value = ''
    gallery.value = []
    info.value = null
    resetProgress()
    lastSeed.value = seed.value

    if (!initImageData.value) {
      status.value = 'error'
      errorMessage.value = 'Initial image required.'
      throw new Error(errorMessage.value)
    }

    const payload = buildPayload()
    try {
      const { task_id } = await startImg2Img(payload)
      taskId.value = task_id
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
        if (status.value !== 'error') {
          status.value = 'done'
        }
        stopStream()
        break
      default:
        break
    }
  }

  const isRunning = computed(() => status.value === 'running')

  const lastSeed = ref<number | null>(null)

  function randomizeSeed(): void {
    if (seed.value !== -1) lastSeed.value = seed.value
    seed.value = -1
  }

  function reuseSeed(): void {
    if (lastSeed.value !== null) seed.value = lastSeed.value
  }

  return {
    prompt,
    negativePrompt,
    steps,
    cfgScale,
    width,
    height,
    denoiseStrength,
    seed,
    batchSize,
    batchCount,
    styles,
    models,
    samplers,
    schedulers,
    selectedModel,
    selectedSampler,
    selectedScheduler,
    initImageData,
    initImageName,
    initImagePreview,
    hasInitImage,
    status,
    errorMessage,
    taskId,
    progress,
    gallery,
    info,
    init,
    loadModels,
    loadSamplers,
    loadSchedulers,
    updateModel,
    setSampler,
    setScheduler,
    setInitImage,
    clearInitImage,
    generate,
    stopStream,
    isRunning,
    randomizeSeed,
    reuseSeed,
  }
})

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(reader.result as string)
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

function readImageDimensions(dataUrl: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.width, height: img.height })
    img.onerror = (err) => reject(err)
    img.src = dataUrl
  })
}
