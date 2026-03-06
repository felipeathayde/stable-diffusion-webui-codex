/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend-driven XYZ sweep store for image tabs.
Builds parameter grid combos, enqueues jobs, starts txt2img tasks (including required `settings_revision`), streams task events, and supports stop modes/cancellation while collecting
per-cell results. Hires upscaler values are stable ids (`latent:*` / `spandrel:*`) for hires-fix wiring; hires tile prefs (fallback/min_tile) are propagated from the shared upscalers store.
Preflight now fails loud when VAE selection is empty before queuing XYZ requests.

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): XYZ sweep lifecycle status (`idle`/`running`/`stopped`/`error`/`done`).
- `StopMode` (type): Stop behavior for a running sweep (`immediate` vs `after_current`).
- `XyzJob` (interface): Internal job record for each cell (payload/task id/status/result/error).
- `useXyzStore` (store): Pinia store for XYZ sweeps; builds combos, runs jobs, subscribes to task SSE, and writes results into cells.
- `enabled`/`xEnabled`/`yEnabled`/`zEnabled` (store refs): Master and per-axis toggles used by the embedded XYZ card + Run integration.
- `XyzStore` (type): Convenience return type alias for `useXyzStore`.
*/

// tags: xyz, store, sweeps
import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'

import { cancelTask, startTxt2Img, subscribeTask } from '../api/client'
import { buildTxt2ImgPayload } from '../api/payloads'
import type { Txt2ImgRequest } from '../api/payloads'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { AXIS_OPTIONS, buildCombos, labelOf, parseAxisValues, type AxisParam, type AxisValue, type XyzCell } from '../utils/xyz'
import { useModelTabsStore, type ImageBaseParams } from './model_tabs'
import { useEngineCapabilitiesStore } from './engine_capabilities'
import { useQuicksettingsStore } from './quicksettings'
import { useUpscalersStore } from './upscalers'
import { fallbackSamplingDefaultsForTabFamily, resolveImageRequestEngineId, type TabFamily } from '../utils/engine_taxonomy'

type Status = 'idle' | 'running' | 'stopped' | 'error' | 'done'
type StopMode = 'immediate' | 'after_current'

interface XyzJob {
  id: string
  combo: { x: AxisValue; y: AxisValue | null; z: AxisValue | null }
  payload: Txt2ImgRequest
  status: 'queued' | 'running' | 'done' | 'error' | 'stopped'
  taskId?: string
  image?: GeneratedImage
  info?: unknown
  error?: string
}

export const useXyzStore = defineStore('xyz', () => {
  const xParam = ref<AxisParam>('cfg')
  const yParam = ref<AxisParam>('steps')
  const zParam = ref<AxisParam>('sampler')
  const enabled = ref(false)
  const xEnabled = ref(true)
  const yEnabled = ref(true)
  const zEnabled = ref(true)

  const xValuesText = ref('6, 7, 8')
  const yValuesText = ref('20, 28')
  const zValuesText = ref('')

  const status = ref<Status>('idle')
  const errorMessage = ref('')
  const stopRequested = ref(false)
  const stopMode = ref<StopMode>('immediate')

  const progress = reactive({ total: 0, completed: 0, current: '' })
  const cells = ref<XyzCell[]>([])
  const jobs = ref<XyzJob[]>([])
  const activeTaskId = ref<string | null>(null)

  let unsubscribe: (() => void) | null = null

  const axisKind = (param: AxisParam): 'text' | 'number' => {
    return AXIS_OPTIONS.find((o) => o.id === param)?.kind ?? 'text'
  }

  const xParsedValues = computed<AxisValue[]>(() => parseAxisValues(xValuesText.value, axisKind(xParam.value)))
  const yParsedValues = computed<AxisValue[]>(() => parseAxisValues(yValuesText.value, axisKind(yParam.value)))
  const zParsedValues = computed<AxisValue[]>(() => parseAxisValues(zValuesText.value, axisKind(zParam.value)))

  const xValues = computed<AxisValue[]>(() => (xEnabled.value ? xParsedValues.value : ['(base)']))
  const yValues = computed<AxisValue[]>(() => (yEnabled.value ? yParsedValues.value : []))
  const zValues = computed<AxisValue[]>(() => (zEnabled.value ? zParsedValues.value : []))

  const combos = computed(() => buildCombos(xValues.value, yValues.value, zValues.value))

  const groupedByZ = computed(() => {
    const groups = new Map<string, XyzCell[]>()
    for (const cell of cells.value) {
      const key = labelOf(cell.z)
      const arr = groups.get(key) ?? []
      arr.push(cell)
      groups.set(key, arr)
    }
    return Array.from(groups.entries()).map(([label, rows]) => ({ label, rows }))
  })

  async function stop(mode: StopMode = 'immediate'): Promise<void> {
    stopRequested.value = true
    stopMode.value = mode
    if (unsubscribe) {
      unsubscribe()
      unsubscribe = null
    }
    const taskId = activeTaskId.value
    if (taskId && mode === 'immediate') {
      try { await cancelTask(taskId, 'immediate') } catch (err) { console.warn('[xyz] cancel failed', err) }
    }
    if (status.value === 'running') status.value = 'stopped'
  }

  function resetProgress(): void {
    progress.total = 0
    progress.completed = 0
    progress.current = ''
  }

  function resetStopState(): void {
    stopRequested.value = false
    stopMode.value = 'immediate'
  }

  function buildBaseForm(): any {
    const tabs = useModelTabsStore()
    const quick = useQuicksettingsStore()
    const caps = useEngineCapabilitiesStore()
    const activeTab = tabs.activeTab
    const params = activeTab?.params as ImageBaseParams | undefined
    const tabFamily = (activeTab?.type || 'sdxl') as TabFamily
    const engineKey = resolveImageRequestEngineId(tabFamily, false)
    const checkpoint = String((params as any)?.checkpoint || '').trim()
    const modelLabel = checkpoint || quick.currentModel
    const resolvedModelInfo = quick.resolveModelInfo(modelLabel)
    const guidanceMode = engineKey === 'flux2'
      ? (() => {
          const variant = quick.resolveFlux2CheckpointVariant(resolvedModelInfo ?? checkpoint)
          if (!variant) {
            throw new Error('Unsupported FLUX.2 checkpoint variant. Only Klein 4B/base-4B is supported.')
          }
          return variant === 'base' ? 'cfg' : 'distilled_cfg'
        })()
      : undefined
    const resolvedModelSha = quick.resolveModelSha(modelLabel)
    const fallbackSampling = fallbackSamplingDefaultsForTabFamily(tabFamily)
    const samplingDefaults = caps.resolveSamplingDefaults(engineKey, {
      fallbackSampler: fallbackSampling.sampler,
      fallbackScheduler: fallbackSampling.scheduler,
    })
    const sampler =
      (typeof params?.sampler === 'string' && params.sampler.trim())
        ? params.sampler
        : samplingDefaults.sampler
    const scheduler =
      (typeof params?.scheduler === 'string' && params.scheduler.trim())
        ? params.scheduler
        : samplingDefaults.scheduler
    
    return {
      prompt: params?.prompt ?? '',
      negativePrompt: params?.negativePrompt ?? '',
      width: params?.width ?? 1024,
      height: params?.height ?? 1024,
      steps: params?.steps ?? 30,
      guidanceScale: params?.cfgScale ?? 7,
      sampler,
      scheduler,
      seed: params?.seed ?? -1,
      batchSize: 1,
      batchCount: 1,
      styles: [],
      device: quick.currentDevice,
      settingsRevision: quick.getSettingsRevision(),
      engine: engineKey,
      model: resolvedModelSha || modelLabel,
      guidanceMode,
    }
  }

  function applyAxis(form: any, param: AxisParam, value: AxisValue): void {
    switch (param) {
      case 'prompt':
        form.prompt = String(value)
        break
      case 'negative':
        form.negativePrompt = String(value)
        break
      case 'cfg':
        form.guidanceScale = Number(value)
        break
      case 'steps':
        form.steps = Number(value)
        break
      case 'sampler':
        form.sampler = String(value)
        break
      case 'scheduler':
        form.scheduler = String(value)
        break
      case 'seed':
        form.seed = Number(value)
        break
      case 'width':
        form.width = Number(value)
        break
      case 'height':
        form.height = Number(value)
        break
      case 'hires_scale':
        form.hires = form.hires || { enabled: true, scale: 2.0, denoise: 0.4, steps: 0, resizeX: 0, resizeY: 0, upscaler: 'latent:bicubic-aa', tile: { tile: 256, overlap: 16 } }
        form.hires.enabled = true
        form.hires.scale = Number(value)
        break
      case 'hires_steps':
        form.hires = form.hires || { enabled: true, scale: 2.0, denoise: 0.4, steps: 0, resizeX: 0, resizeY: 0, upscaler: 'latent:bicubic-aa', tile: { tile: 256, overlap: 16 } }
        form.hires.enabled = true
        form.hires.steps = Number(value)
        break
      case 'refiner_model':
        form.refiner = form.refiner || { enabled: true, swapAtStep: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
        form.refiner.enabled = true
        form.refiner.model = String(value)
        break
      case 'refiner_steps':
        form.refiner = form.refiner || { enabled: true, swapAtStep: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
        form.refiner.enabled = true
        form.refiner.swapAtStep = Math.max(1, Math.trunc(Number(value)))
        break
      case 'refiner_cfg':
        form.refiner = form.refiner || { enabled: true, swapAtStep: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
        form.refiner.enabled = true
        form.refiner.cfg = Number(value)
        break
      default:
        break
    }
  }

  async function awaitResult(taskId: string): Promise<{ images: GeneratedImage[]; info?: unknown }> {
    return new Promise((resolve, reject) => {
      let result: { images: GeneratedImage[]; info?: unknown } | null = null
      unsubscribe = subscribeTask(
        taskId,
        (event: TaskEvent) => {
          if (event.type === 'result') {
            result = { images: event.images, info: event.info }
          }
          if (event.type === 'error') {
            reject(new Error(event.message ?? 'Task failed'))
          }
          if (event.type === 'end') {
            if (result) resolve(result)
            else reject(new Error('Task ended without result'))
          }
        },
        (err) => reject(err instanceof Error ? err : new Error(String(err)))
      )
    })
  }

  async function run(): Promise<void> {
    const tabs = useModelTabsStore()
    const quick = useQuicksettingsStore()
    const activeTab = tabs.activeTab
    const params = activeTab?.params as ImageBaseParams | undefined

    if (!enabled.value) {
      errorMessage.value = 'Enable XYZ before running.'
      status.value = 'error'
      return
    }
    if (!xEnabled.value && !yEnabled.value && !zEnabled.value) {
      errorMessage.value = 'Enable at least one axis before running XYZ.'
      status.value = 'error'
      return
    }
    if (xEnabled.value && !xParsedValues.value.length) {
      errorMessage.value = 'X axis needs at least one value while enabled.'
      status.value = 'error'
      return
    }
    if (yEnabled.value && !yParsedValues.value.length) {
      errorMessage.value = 'Y axis needs at least one value while enabled.'
      status.value = 'error'
      return
    }
    if (zEnabled.value && !zParsedValues.value.length) {
      errorMessage.value = 'Z axis needs at least one value while enabled.'
      status.value = 'error'
      return
    }
    const samplerAxisEnabled =
      (xEnabled.value && xParam.value === 'sampler')
      || (yEnabled.value && yParam.value === 'sampler')
      || (zEnabled.value && zParam.value === 'sampler')
    const schedulerAxisEnabled =
      (xEnabled.value && xParam.value === 'scheduler')
      || (yEnabled.value && yParam.value === 'scheduler')
      || (zEnabled.value && zParam.value === 'scheduler')
    if (samplerAxisEnabled && schedulerAxisEnabled) {
      errorMessage.value = 'Sampler and Scheduler axes cannot be varied together in the same XYZ run.'
      status.value = 'error'
      return
    }
    if (!params?.prompt?.trim()) {
      errorMessage.value = 'Prompt must not be empty before running XYZ.'
      status.value = 'error'
      return
    }
    try {
      quick.requireVaeSelection()
    } catch (error) {
      errorMessage.value = error instanceof Error ? error.message : String(error)
      status.value = 'error'
      return
    }

    errorMessage.value = ''
    resetStopState()
    status.value = 'running'
    resetProgress()

    const comboList = combos.value
    progress.total = comboList.length
    progress.completed = 0
    cells.value = comboList.map((combo) => ({ x: combo.x, y: combo.y, z: combo.z, status: 'queued' }))
    jobs.value = []
    const upscalers = useUpscalersStore()
    const hiresFallbackOnOom = Boolean(upscalers.fallbackOnOom)
    const hiresMinTile = Number(upscalers.minTile)

    // Pre-build job queue with payload snapshots
    for (const combo of comboList) {
      const form = buildBaseForm()
      if (xEnabled.value) applyAxis(form, xParam.value, combo.x)
      if (combo.y !== null && yEnabled.value) applyAxis(form, yParam.value, combo.y)
      if (combo.z !== null && zEnabled.value) applyAxis(form, zParam.value, combo.z)
      try {
        const payload = buildTxt2ImgPayload(form, { hiresFallbackOnOom, hiresMinTile })
        jobs.value.push({
          id: `job-${jobs.value.length + 1}`,
          combo: { x: combo.x, y: combo.y, z: combo.z },
          payload,
          status: 'queued',
        })
      } catch (err) {
        errorMessage.value = err instanceof Error ? err.message : String(err)
        status.value = 'error'
        return
      }
    }

    for (let idx = 0; idx < jobs.value.length; idx++) {
      const job = jobs.value[idx]
      const cell = cells.value[idx]
      if (!job || !cell) continue

      if (stopRequested.value && stopMode.value === 'after_current') {
        job.status = 'stopped'
        cell.status = 'stopped'
        continue
      }
      if (stopRequested.value && stopMode.value === 'immediate') {
        job.status = 'stopped'
        cell.status = 'stopped'
        break
      }

      job.status = 'running'
      cell.status = 'running'
      const currentParts: string[] = []
      if (xEnabled.value) currentParts.push(labelOf(job.combo.x))
      if (yEnabled.value) currentParts.push(labelOf(job.combo.y))
      if (zEnabled.value) currentParts.push(labelOf(job.combo.z))
      progress.current = currentParts.join(' / ') || 'base'

      try {
        const { task_id } = await startTxt2Img(job.payload)
        job.taskId = task_id
        activeTaskId.value = task_id
        const result = await awaitResult(task_id)
        job.status = 'done'
        cell.status = 'done'
        job.image = result.images?.[0]
        cell.image = result.images?.[0]
        job.info = result.info
        cell.info = result.info
        progress.completed += 1
      } catch (err) {
        job.status = stopRequested.value ? 'stopped' : 'error'
        cell.status = job.status
        const msg = err instanceof Error ? err.message : String(err)
        job.error = msg
        cell.error = msg
        errorMessage.value = msg
        status.value = stopRequested.value ? 'stopped' : 'error'
        if (!stopRequested.value || stopMode.value === 'immediate') {
          break
        }
      } finally {
        activeTaskId.value = null
        if (unsubscribe) {
          unsubscribe()
          unsubscribe = null
        }
      }
    }

    if (status.value === 'running') {
      status.value = stopRequested.value ? 'stopped' : 'done'
    }
  }

  return {
    xParam,
    yParam,
    zParam,
    enabled,
    xEnabled,
    yEnabled,
    zEnabled,
    xValuesText,
    yValuesText,
    zValuesText,
    xValues,
    yValues,
    zValues,
    combos,
    groupedByZ,
    status,
    errorMessage,
    progress,
    cells,
    stopRequested,
    stopMode,
    run,
    stop,
  }
})

export type XyzStore = ReturnType<typeof useXyzStore>
