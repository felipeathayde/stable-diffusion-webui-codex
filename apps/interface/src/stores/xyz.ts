// tags: xyz, store, sweeps
import { defineStore } from 'pinia'
import { computed, reactive, ref } from 'vue'

import { cancelTask, startTxt2Img, subscribeTask, updateOptions } from '../api/client'
import { buildTxt2ImgPayload } from '../api/payloads'
import type { Txt2ImgRequest } from '../api/payloads'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { AXIS_OPTIONS, buildCombos, labelOf, parseAxisValues, type AxisParam, type AxisValue, type XyzCell } from '../utils/xyz'
import { useSdxlStore } from './sdxl'
import { useQuicksettingsStore } from './quicksettings'

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

  const xValues = computed<AxisValue[]>(() => parseAxisValues(xValuesText.value, axisKind(xParam.value)))
  const yValues = computed<AxisValue[]>(() => parseAxisValues(yValuesText.value, axisKind(yParam.value)))
  const zValues = computed<AxisValue[]>(() => parseAxisValues(zValuesText.value, axisKind(zParam.value)))

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

  function buildBaseForm(): any {
    const sdxl = useSdxlStore()
    const quick = useQuicksettingsStore()
    return {
      prompt: sdxl.prompt,
      negativePrompt: sdxl.negativePrompt,
      width: sdxl.width,
      height: sdxl.height,
      steps: sdxl.steps,
      guidanceScale: sdxl.cfgScale,
      sampler: sdxl.selectedSampler || 'automatic',
      scheduler: sdxl.selectedScheduler || 'automatic',
      seed: sdxl.seed,
      batchSize: sdxl.batchSize,
      batchCount: sdxl.batchCount,
      styles: [],
      device: quick.currentDevice,
      engine: 'sdxl',
      model: sdxl.selectedModel,
      highres: { ...sdxl.highres },
      refiner: { ...sdxl.refiner },
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
      case 'highres_scale':
        form.highres = form.highres || { enabled: true, scale: 1.0, denoise: 0.4, steps: 0, resizeX: 0, resizeY: 0, upscaler: 'Use same upscaler' }
        form.highres.enabled = true
        form.highres.scale = Number(value)
        break
      case 'highres_steps':
        form.highres = form.highres || { enabled: true, scale: 1.0, denoise: 0.4, steps: 0, resizeX: 0, resizeY: 0, upscaler: 'Use same upscaler' }
        form.highres.enabled = true
        form.highres.steps = Number(value)
        break
      case 'refiner_model':
        form.refiner = form.refiner || { enabled: true, steps: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
        form.refiner.enabled = true
        form.refiner.model = String(value)
        break
      case 'refiner_steps':
        form.refiner = form.refiner || { enabled: true, steps: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
        form.refiner.enabled = true
        form.refiner.steps = Number(value)
        break
      case 'refiner_cfg':
        form.refiner = form.refiner || { enabled: true, steps: 10, cfg: form.guidanceScale ?? 7, seed: -1 }
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
    const sdxl = useSdxlStore()
    if (!xValues.value.length) {
      errorMessage.value = 'X axis needs at least one value.'
      status.value = 'error'
      return
    }
    if (!sdxl.prompt.trim()) {
      errorMessage.value = 'Prompt must not be empty before running XYZ.'
      status.value = 'error'
      return
    }

    errorMessage.value = ''
    stopRequested.value = false
    stopMode.value = 'immediate'
    status.value = 'running'
    resetProgress()

    const comboList = combos.value
    progress.total = comboList.length
    progress.completed = 0
    cells.value = comboList.map((combo) => ({ x: combo.x, y: combo.y, z: combo.z, status: 'queued' }))
    jobs.value = []

    // Ensure engine/model are set before firing many tasks
    if (sdxl.selectedModel) {
      try {
        await updateOptions({ codex_engine: 'sdxl', sd_model_checkpoint: sdxl.selectedModel })
      } catch (err) {
        errorMessage.value = err instanceof Error ? err.message : String(err)
        status.value = 'error'
        return
      }
    }

    // Pre-build job queue with payload snapshots
    for (const combo of comboList) {
      const form = buildBaseForm()
      applyAxis(form, xParam.value, combo.x)
      if (combo.y !== null) applyAxis(form, yParam.value, combo.y)
      if (combo.z !== null) applyAxis(form, zParam.value, combo.z)
      try {
        const payload = buildTxt2ImgPayload(form)
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
      progress.current = `${labelOf(job.combo.x)} / ${labelOf(job.combo.y)} / ${labelOf(job.combo.z)}`

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
