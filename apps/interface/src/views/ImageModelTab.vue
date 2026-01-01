<template>
  <section v-if="tab" class="panels">
    <!-- Left column: Prompt + Parameters -->
    <div class="panel-stack" ref="leftStack">
      <PromptCard
        v-model:prompt="promptText"
        v-model:negative="negativeText"
        :defaultShowNegative="defaultShowNegative"
        :supportsNegative="supportsNegative"
        :allowNegativeToggle="supportsNegative"
        :enableAssets="enableAssets"
        :enableStyles="enableStyles"
        :toolbarLabel="toolbarLabel"
        :fieldsId="`image-modeltab-prompt-${tabId}`"
      >
        <div v-if="isRunning" class="panel-progress">
          <p><strong>Stage:</strong> {{ progress.stage }}</p>
          <p v-if="progressPercent !== null">Progress: {{ progressPercent.toFixed(1) }}%</p>
          <p v-if="progress.totalSteps && progress.step !== null">
            Step {{ progress.step }} / {{ progress.totalSteps }}
          </p>
          <p v-if="progress.etaSeconds !== null" class="caption">ETA ~ {{ progress.etaSeconds.toFixed(0) }}s</p>
        </div>
        <div v-if="errorMessage" class="panel-error">
          {{ errorMessage }}
        </div>

        <div v-if="supportsImg2Img" class="panel-section">
          <label class="switch-label">
            <input type="checkbox" :checked="params.useInitImage" :disabled="isRunning" @change="onInitToggle" />
            <span>Use Initial Image (img2img)</span>
          </label>

          <div v-if="params.useInitImage">
            <InitialImageCard
              label="Initial Image"
              :src="params.initImageData"
              :has-image="Boolean(params.initImageData)"
              :disabled="isRunning"
              @set="onInitFileSet"
              @clear="clearInit"
            >
              <template #footer>
                <p v-if="params.initImageName" class="caption">{{ params.initImageName }}</p>
              </template>
            </InitialImageCard>

            <SliderField
              label="Denoise"
              :modelValue="params.denoiseStrength"
              :min="0"
              :max="1"
              :step="0.01"
              :inputStep="0.05"
              inputClass="cdx-input-w-xs"
              :disabled="isRunning"
              @update:modelValue="(v) => setParams({ denoiseStrength: clampFloat(v, 0, 1) })"
            />
          </div>
        </div>
      </PromptCard>

      <div class="panel">
        <div class="panel-header">
          Generation Parameters
          <div class="toolbar">
            <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="loadProfile">Load profile</button>
            <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="saveProfile">Save profile</button>
          </div>
        </div>
        <div class="panel-body">
          <BasicParametersCard
            :samplers="filteredSamplers"
            :schedulers="filteredSchedulers"
            :sampler="params.sampler"
            :scheduler="params.scheduler"
            :steps="params.steps"
            :width="params.width"
            :height="params.height"
            :cfg-scale="params.cfgScale"
            :seed="params.seed"
            :resolutionPresets="resolutionPresets"
            :show-cfg="true"
            :cfg-label="cfgLabel"
            :show-init-image-dims="params.useInitImage && Boolean(params.initImageData)"
            :disabled="isRunning"
            @update:sampler="(v: string) => setParams({ sampler: v })"
            @update:scheduler="(v: string) => setParams({ scheduler: v })"
            @update:steps="(v: number) => setParams({ steps: Math.max(1, Math.trunc(v)) })"
            @update:width="(v: number) => setParams({ width: Math.max(64, Math.trunc(v)) })"
            @update:height="(v: number) => setParams({ height: Math.max(64, Math.trunc(v)) })"
            @update:cfgScale="(v: number) => setParams({ cfgScale: v })"
            @update:seed="(v: number) => setParams({ seed: Math.trunc(v) })"
            @random-seed="randomizeSeed"
            @reuse-seed="reuseSeed"
            @sync-init-image-dims="syncInitImageDims"
          />

          <HighresSettingsCard
            v-if="showHighres"
            :enabled="params.highres.enabled"
            :denoise="params.highres.denoise"
            :scale="params.highres.scale"
            :steps="params.highres.steps"
            :upscaler="params.highres.upscaler"
            :base-width="params.width"
            :base-height="params.height"
            :refinerEnabled="params.highres.refiner?.enabled"
            :refinerSteps="params.highres.refiner?.steps"
            :refinerCfg="params.highres.refiner?.cfg"
            :refinerSeed="params.highres.refiner?.seed"
            :refinerModel="params.highres.refiner?.model"
            :refinerVae="params.highres.refiner?.vae"
            @update:enabled="(v: boolean) => setHighres({ enabled: v })"
            @update:denoise="(v: number) => setHighres({ denoise: clampFloat(v, 0, 1) })"
            @update:scale="(v: number) => setHighres({ scale: v })"
            @update:steps="(v: number) => setHighres({ steps: Math.max(0, Math.trunc(v)) })"
            @update:upscaler="(v: string) => setHighres({ upscaler: v })"
            @update:refinerEnabled="(v: boolean) => setHighresRefiner({ enabled: v })"
            @update:refinerSteps="(v: number) => setHighresRefiner({ steps: Math.max(0, Math.trunc(v)) })"
            @update:refinerCfg="(v: number) => setHighresRefiner({ cfg: v })"
            @update:refinerSeed="(v: number) => setHighresRefiner({ seed: Math.trunc(v) })"
            @update:refinerModel="(v: string) => setHighresRefiner({ model: v })"
            @update:refinerVae="(v: string) => setHighresRefiner({ vae: v })"
          />

          <RefinerSettingsCard
            v-if="showGlobalRefiner"
            :enabled="params.refiner.enabled"
            :steps="params.refiner.steps"
            :cfg="params.refiner.cfg"
            :seed="params.refiner.seed"
            :model="params.refiner.model"
            :vae="params.refiner.vae"
            @update:enabled="(v: boolean) => setRefiner({ enabled: v })"
            @update:steps="(v: number) => setRefiner({ steps: Math.max(0, Math.trunc(v)) })"
            @update:cfg="(v: number) => setRefiner({ cfg: v })"
            @update:seed="(v: number) => setRefiner({ seed: Math.trunc(v) })"
            @update:model="(v: string) => setRefiner({ model: v })"
            @update:vae="(v: string) => setRefiner({ vae: v })"
          />
        </div>
      </div>
    </div>

    <!-- Right column: Run + Results -->
    <div class="panel-stack">
      <RunCard
        :generateDisabled="isRunning"
        :isRunning="isRunning"
        :showBatchControls="true"
        :batchCount="params.batchCount"
        :batchSize="params.batchSize"
        :disabled="isRunning"
        @generate="generate"
        @update:batchCount="(v: number) => setParams({ batchCount: Math.max(1, Math.trunc(v)) })"
        @update:batchSize="(v: number) => setParams({ batchSize: Math.max(1, Math.trunc(v)) })"
      >
        <div v-if="copyNotice" class="caption">{{ copyNotice }}</div>
        <RunSummaryChips :text="runSummary" />
      </RunCard>

      <ResultsCard :showGenerate="false" headerClass="three-cols" headerRightClass="results-actions">
        <template #header-right>
          <div class="gentime-display" v-if="gentimeSeconds !== null">
            <span class="caption">Time: {{ gentimeSeconds.toFixed(2) }}s</span>
          </div>
          <button class="btn btn-sm btn-secondary" type="button" :disabled="workflowBusy" @click="sendToWorkflows">
            {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
          </button>
          <button class="btn btn-sm btn-outline" type="button" @click="copyCurrentParams">Copy params</button>
        </template>

        <div class="gen-card mb-3">
          <div class="row-split">
            <span class="label-muted">History</span>
          </div>
          <div v-if="history.length" class="cdx-history-list">
            <div v-for="item in history" :key="item.taskId" :class="['cdx-history-item', { 'is-selected': item.taskId === selectedTaskId }]">
              <div class="cdx-history-meta">
                <div class="cdx-history-title">{{ formatHistoryTitle(item) }}</div>
                <div class="cdx-history-sub">{{ item.summary }}</div>
                <div v-if="item.promptPreview" class="cdx-history-sub">{{ item.promptPreview }}</div>
                <div v-if="item.status !== 'completed'" class="caption">Status: {{ item.status }}</div>
                <div v-if="item.errorMessage" class="caption">Error: {{ item.errorMessage }}</div>
              </div>
              <div class="cdx-history-actions">
                <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning || historyLoadingTaskId === item.taskId" @click="loadHistory(item.taskId)">
                  {{ historyLoadingTaskId === item.taskId ? 'Loading…' : 'Load' }}
                </button>
                <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="applyHistory(item)">Apply</button>
                <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="copyHistoryParams(item)">Copy</button>
              </div>
            </div>
          </div>
          <div v-else class="caption">No runs yet.</div>

          <div class="cdx-history-actions mt-2">
            <button class="btn btn-sm btn-ghost" type="button" :disabled="!history.length || isRunning" @click="clearHistory">Clear history</button>
          </div>
        </div>

        <ResultViewer
          mode="image"
          :images="images"
          :width="params.width"
          :height="params.height"
          emptyText="No images yet. Generate to see results here."
          :style="previewStyle"
        >
          <template #image-actions="{ image, index }">
            <button
              v-if="supportsImg2Img"
              class="gallery-action"
              type="button"
              title="Send to Img2Img"
              @click="sendToImg2Img(image)"
            >
              Send to Img2Img
            </button>
            <button class="gallery-action" type="button" title="Download Image" @click="download(image, index)">
              Download
            </button>
          </template>
        </ResultViewer>
      </ResultsCard>

      <div class="panel" v-if="info">
        <div class="panel-header">Generation Info</div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ formatJson(info) }}</pre>
        </div>
      </div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { fetchSamplers, fetchSchedulers } from '../api/client'
import type { GeneratedImage, SamplerInfo, SchedulerInfo } from '../api/types'
import { formatJson, useResultsCard } from '../composables/useResultsCard'
import { useGeneration, type ImageRunHistoryItem } from '../composables/useGeneration'
import { useModelTabsStore, type ImageBaseParams } from '../stores/model_tabs'
import { getEngineConfig, getEngineDefaults, type EngineType } from '../stores/engine_config'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import { useWorkflowsStore } from '../stores/workflows'
import BasicParametersCard from '../components/BasicParametersCard.vue'
import HighresSettingsCard from '../components/HighresSettingsCard.vue'
import InitialImageCard from '../components/InitialImageCard.vue'
import PromptCard from '../components/prompt/PromptCard.vue'
import RefinerSettingsCard from '../components/RefinerSettingsCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import SliderField from '../components/ui/SliderField.vue'

const props = defineProps<{ tabId: string; type: EngineType }>()
const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const workflows = useWorkflowsStore()

// Use unified generation composable
const {
  generate,
  stopStream,
  gallery,
  progress,
  errorMessage,
  isRunning,
  lastSeed,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  tab,
  info,
  gentimeMs,
  loadHistory,
  clearHistory,
} = useGeneration(props.tabId)

const leftStack = ref<HTMLElement | null>(null)
const previewStyle = ref<Record<string, string>>({})
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(async () => {
  if (!store.tabs.length) store.load()
  void engineCaps.init()
  const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers
  syncPreviewHeight()
  window.addEventListener('resize', syncPreviewHeight)
})

onBeforeUnmount(() => {
  stopStream()
  window.removeEventListener('resize', syncPreviewHeight)
})

const workflowBusy = ref(false)
const { notice: copyNotice, toast, copyJson } = useResultsCard()

const params = computed<ImageBaseParams>(() => (tab.value?.params as any) as ImageBaseParams)
const engineConfig = computed(() => getEngineConfig(props.type))
const engineSurface = computed(() => engineCaps.get(props.type))

const supportsNegative = computed(() => engineConfig.value.capabilities.usesNegativePrompt)
const supportsImg2Img = computed(() => {
  const surf = engineSurface.value
  if (surf) return Boolean(surf.supports_img2img)
  return engineConfig.value.capabilities.tasks.includes('img2img')
})

const enableAssets = computed(() => true)
const enableStyles = computed(() => true)
const toolbarLabel = computed(() => (props.type === 'zimage' ? 'Z Image Turbo' : ''))

const cfgLabel = computed(() => (engineConfig.value.capabilities.usesDistilledCfg ? 'Distilled CFG' : 'CFG'))
const defaultShowNegative = computed(() => props.type === 'sdxl' && supportsNegative.value)

const showHighres = computed(() => {
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_highres
})

const showGlobalRefiner = computed(() => {
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_refiner
})

const filteredSamplers = computed(() => {
  const allowed = engineSurface.value?.samplers as string[] | null | undefined
  if (!allowed || allowed.length === 0) return samplers.value
  return samplers.value.filter(s => allowed.includes(s.name))
})

const filteredSchedulers = computed(() => {
  const allowed = engineSurface.value?.schedulers as string[] | null | undefined
  if (!allowed || allowed.length === 0) return schedulers.value
  return schedulers.value.filter(s => allowed.includes(s.name))
})

const promptText = computed({
  get: () => params.value.prompt,
  set: (value: string) => setParams({ prompt: value }),
})

const negativeText = computed({
  get: () => params.value.negativePrompt,
  set: (value: string) => {
    if (!supportsNegative.value) return
    setParams({ negativePrompt: value })
  },
})

watch(supportsImg2Img, (supported) => {
  if (supported) return
  if (!params.value.useInitImage) return
  setParams({ useInitImage: false, initImageData: '', initImageName: '' })
}, { immediate: true })

watch(showHighres, (show) => {
  if (show) return
  if (!params.value.highres.enabled && !params.value.highres.refiner?.enabled) return
  setHighres({
    enabled: false,
    refiner: { ...(params.value.highres.refiner || {}), enabled: false },
  } as any)
})

watch(showGlobalRefiner, (show) => {
  if (show) return
  if (!params.value.refiner.enabled) return
  setRefiner({ enabled: false })
})

watch([supportsImg2Img, showHighres, showGlobalRefiner, () => params.value.useInitImage], () => {
  void nextTick(syncPreviewHeight)
})

const images = computed(() => gallery.value)

const gentimeSeconds = computed(() => {
  if (gentimeMs.value == null) return null
  return gentimeMs.value / 1000
})

const progressPercent = computed(() => {
  if (progress.value.percent !== null) return progress.value.percent
  if (!progress.value.totalSteps || progress.value.step === null) return null
  return (progress.value.step / progress.value.totalSteps) * 100
})

const resolutionPresets = computed((): [number, number][] => {
  if (props.type === 'sd15') return [[512, 512], [512, 768], [768, 512]]
  return [[1024, 1024], [1152, 896], [1216, 832], [1344, 768]]
})

const runSummary = computed(() => {
  const sampler = params.value.sampler || 'automatic'
  const scheduler = params.value.scheduler || 'automatic'
  const seedLabel = params.value.seed === -1 ? 'seed random' : `seed ${params.value.seed}`
  return `${params.value.width}×${params.value.height} px · ${params.value.steps} steps · ${cfgLabel.value} ${params.value.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel} · batch ${params.value.batchCount}×${params.value.batchSize}`
})

async function sendToWorkflows(): Promise<void> {
  if (!tab.value) return
  workflowBusy.value = true
  try {
    await workflows.createSnapshot({
      name: `${tab.value.title} — ${new Date().toLocaleString()}`,
      source_tab_id: tab.value.id,
      type: tab.value.type,
      engine_semantics: tab.value.type === 'wan' ? 'wan22' : tab.value.type,
      params_snapshot: tab.value.params as Record<string, unknown>,
    })
    toast('Snapshot saved to Workflows.')
  } catch (e) {
    toast(e instanceof Error ? e.message : String(e))
  } finally {
    workflowBusy.value = false
  }
}

async function copyCurrentParams(): Promise<void> {
  if (!tab.value) return
  await copyJson(tab.value.params, 'Copied params.')
}

async function copyHistoryParams(item: ImageRunHistoryItem): Promise<void> {
  await copyJson(item.paramsSnapshot, 'Copied history params.')
}

function applyHistory(item: ImageRunHistoryItem): void {
  const snap = item.paramsSnapshot as Partial<ImageBaseParams>
  setParams({
    ...(snap as any),
    useInitImage: false,
    initImageData: '',
    initImageName: '',
  })
  toast('Applied history params.')
}

function formatHistoryTitle(item: { mode: string; createdAtMs: number; taskId: string }): string {
  const dt = new Date(item.createdAtMs || Date.now())
  const hh = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const label = item.mode === 'img2img' ? 'Img2Img' : 'Txt2Img'
  return `${label} · ${hh}`
}

function profileStorageKeyFor(type: EngineType): string {
  if (type === 'flux') return 'codex.flux.profile'
  if (type === 'sdxl') return 'codex.sdxl.profile.v1'
  if (type === 'zimage') return 'codex.zimage.profile'
  if (type === 'sd15') return 'codex.sd15.profile.v1'
  return `codex.${type}.profile.v1`
}

function loadProfile(): void {
  const key = profileStorageKeyFor(props.type)
  try {
    const raw = localStorage.getItem(key)
    if (!raw) {
      toast('No saved profile found.')
      return
    }

    const snapshot = JSON.parse(raw) as Record<string, unknown>
    const next: Partial<ImageBaseParams> = {}

    const numberOrNull = (value: unknown): number | null => {
      const n = Number(value)
      return Number.isFinite(n) ? n : null
    }

    if (typeof snapshot.prompt === 'string') next.prompt = snapshot.prompt
    if (supportsNegative.value && typeof snapshot.negativePrompt === 'string') next.negativePrompt = snapshot.negativePrompt
    const steps = numberOrNull(snapshot.steps); if (steps !== null) next.steps = Math.max(1, Math.trunc(steps))
    const cfgScale = numberOrNull(snapshot.cfgScale); if (cfgScale !== null) next.cfgScale = cfgScale
    const width = numberOrNull(snapshot.width); if (width !== null) next.width = Math.max(64, Math.trunc(width))
    const height = numberOrNull(snapshot.height); if (height !== null) next.height = Math.max(64, Math.trunc(height))
    const seed = numberOrNull(snapshot.seed); if (seed !== null) next.seed = Math.trunc(seed)
    const batchSize = numberOrNull(snapshot.batchSize); if (batchSize !== null) next.batchSize = Math.max(1, Math.trunc(batchSize))
    const batchCount = numberOrNull(snapshot.batchCount); if (batchCount !== null) next.batchCount = Math.max(1, Math.trunc(batchCount))

    const selectedModel = typeof snapshot.selectedModel === 'string' ? snapshot.selectedModel : ''
    const selectedSampler = typeof snapshot.selectedSampler === 'string' ? snapshot.selectedSampler : ''
    const selectedScheduler = typeof snapshot.selectedScheduler === 'string' ? snapshot.selectedScheduler : ''

    if (selectedModel) next.checkpoint = selectedModel
    if (selectedSampler) next.sampler = selectedSampler
    if (selectedScheduler) next.scheduler = selectedScheduler

    setParams(next)
    toast('Loaded saved profile.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function saveProfile(): void {
  const key = profileStorageKeyFor(props.type)
  try {
    const snapshot = {
      prompt: params.value.prompt,
      negativePrompt: supportsNegative.value ? params.value.negativePrompt : '',
      steps: params.value.steps,
      cfgScale: params.value.cfgScale,
      width: params.value.width,
      height: params.value.height,
      seed: params.value.seed,
      batchSize: params.value.batchSize,
      batchCount: params.value.batchCount,
      selectedModel: params.value.checkpoint,
      selectedSampler: params.value.sampler,
      selectedScheduler: params.value.scheduler,
    }
    localStorage.setItem(key, JSON.stringify(snapshot))
    toast('Profile saved.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function setParams(patch: Partial<ImageBaseParams>): void {
  if (!tab.value) return
  store.updateParams(props.tabId, patch as any)
}

function setHighres(patch: Record<string, unknown>): void {
  setParams({ highres: { ...(params.value.highres as any), ...patch } as any })
}

function setHighresRefiner(patch: Record<string, unknown>): void {
  const nextRefiner = { ...((params.value.highres as any).refiner || {}), ...patch }
  setHighres({ refiner: nextRefiner })
}

function setRefiner(patch: Record<string, unknown>): void {
  setParams({ refiner: { ...(params.value.refiner as any), ...patch } as any })
}

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

const _KONTEXT_DEFAULT_STEPS = 28
const _KONTEXT_DEFAULT_DISTILLED_CFG = 2.5
const _INIT_IMAGE_DIM_MIN = 64
const _INIT_IMAGE_DIM_MAX = 8192
const _INIT_IMAGE_DIM_STEP = 8

function snapInitImageDim(value: number): number {
  const clamped = Math.max(_INIT_IMAGE_DIM_MIN, Math.min(_INIT_IMAGE_DIM_MAX, Math.trunc(value)))
  const snapped = Math.round(clamped / _INIT_IMAGE_DIM_STEP) * _INIT_IMAGE_DIM_STEP
  return Math.max(_INIT_IMAGE_DIM_MIN, Math.min(_INIT_IMAGE_DIM_MAX, snapped))
}

function onInitToggle(e: Event): void {
  const checked = (e.target as HTMLInputElement).checked
  const wasEnabled = Boolean(params.value.useInitImage)
  setParams({ useInitImage: checked })
  if (!checked) {
    setParams({ initImageData: '', initImageName: '' })
    return
  }
  if (!wasEnabled) maybeApplyKontextDefaults()
}

async function onInitFileSet(file: File): Promise<void> {
  const wasEnabled = Boolean(params.value.useInitImage)
  const dataUrl = await readFileAsDataURL(file)
  const patch: Partial<ImageBaseParams> = { initImageData: dataUrl, initImageName: file.name, useInitImage: true }
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    patch.width = snapInitImageDim(width)
    patch.height = snapInitImageDim(height)
  } catch {
    // ignore: keep current dims
  }
  setParams(patch)
  if (!wasEnabled) maybeApplyKontextDefaults()
}

function clearInit(): void { setParams({ initImageData: '', initImageName: '' }) }

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function randomizeSeed(): void {
  setParams({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) setParams({ seed: lastSeed.value })
}

function download(image: GeneratedImage, index: number): void {
  const link = document.createElement('a')
  link.href = toDataUrl(image)
  link.download = `${props.type}_${index + 1}.png`
  link.click()
}

async function sendToImg2Img(image: GeneratedImage): Promise<void> {
  if (!supportsImg2Img.value) return
  const wasEnabled = Boolean(params.value.useInitImage)
  const dataUrl = toDataUrl(image)
  const patch: Partial<ImageBaseParams> = { useInitImage: true, initImageData: dataUrl, initImageName: `from_${props.type}.png` }
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    patch.width = snapInitImageDim(width)
    patch.height = snapInitImageDim(height)
  } catch {
    // ignore
  }
  setParams(patch)
  if (!wasEnabled) maybeApplyKontextDefaults()
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file)
  })
}

function readImageDimensions(src: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth || img.width, height: img.naturalHeight || img.height })
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = src
  })
}

async function syncInitImageDims(): Promise<void> {
  const src = String(params.value.initImageData || '')
  if (!src) return
  try {
    const { width, height } = await readImageDimensions(src)
    setParams({ width: snapInitImageDim(width), height: snapInitImageDim(height) })
  } catch {
    // ignore
  }
}

function maybeApplyKontextDefaults(): void {
  if (props.type !== 'flux') return
  const defaults = getEngineDefaults('flux')
  const defaultCfg = defaults.distilledCfg ?? defaults.cfg
  // Only apply when user hasn't customized away from the Flux defaults.
  if (params.value.steps === defaults.steps) setParams({ steps: _KONTEXT_DEFAULT_STEPS })
  if (params.value.cfgScale === defaultCfg) setParams({ cfgScale: _KONTEXT_DEFAULT_DISTILLED_CFG })
}

function syncPreviewHeight(): void {
  const el = leftStack.value
  if (!el) return
  const h = el.getBoundingClientRect().height
  previewStyle.value = { minHeight: `${Math.max(300, Math.floor(h))}px` }
}

defineExpose({ generate })
</script>
