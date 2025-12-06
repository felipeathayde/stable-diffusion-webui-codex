<template>
  <section class="panels">
    <div class="panel-stack" ref="leftStack">
      <div class="panel">
        <div class="panel-header"><span>Prompt</span>
          <div class="toolbar prompt-toolbar">
            <button class="btn btn-sm btn-secondary" type="button" @click="showCkpt = true">Checkpoints</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showTI = true">Textual Inversion</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showLora = true">LoRA</button>
            <label class="label-muted styles-label">Styles</label>
            <input class="ui-input styles-input" list="style-list" v-model="styleName" placeholder="Filter styles" />
            <datalist id="style-list">
              <option v-for="s in styleNames" :key="s" :value="s" />
            </datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="showStyle = true">New</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyStyle(styleName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <PromptFields v-model:prompt="store.prompt" v-model:negative="store.negativePrompt" :hide-negative="true" />
          <div v-if="store.isRunning" class="panel-progress">
            <p><strong>Stage:</strong> {{ store.progress.stage }}</p>
            <p v-if="progressPercent !== null">Progress: {{ progressPercent.toFixed(1) }}%</p>
            <p v-if="store.progress.totalSteps > 0">
              Step {{ store.progress.step }} / {{ store.progress.totalSteps }}
            </p>
          </div>
          <div v-if="store.status === 'error'" class="panel-error">
            {{ store.errorMessage }}
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <GenerationSettingsCard
            :samplers="store.samplers"
            :schedulers="store.schedulers"
            :sampler="store.selectedSampler"
            :scheduler="store.selectedScheduler"
            :steps="store.steps"
            :width="store.width"
            :height="store.height"
            :cfg-scale="store.cfgScale"
            :seed="store.seed"
            :batch-size="store.batchSize"
            :batch-count="store.batchCount"
            @update:sampler="setSampler"
            @update:scheduler="setScheduler"
            @update:steps="(v:number)=>store.steps=v"
            @update:width="(v:number)=>store.width=v"
            @update:height="(v:number)=>store.height=v"
            @update:cfgScale="(v:number)=>store.cfgScale=v"
            @update:seed="(v:number)=>store.seed=v"
            @update:batchSize="(v:number)=>store.batchSize=v"
            @update:batchCount="(v:number)=>store.batchCount=v"
            @random-seed="store.randomizeSeed"
            @reuse-seed="store.reuseSeed"
          />
          <HighresSettingsCard
            v-if="showHighres"
            v-model:enabled="store.highres.enabled"
            v-model:denoise="store.highres.denoise"
            v-model:scale="store.highres.scale"
            v-model:steps="store.highres.steps"
            v-model:upscaler="store.highres.upscaler"
            :base-width="store.width"
            :base-height="store.height"
            v-model:refinerEnabled="store.highres.refiner.enabled"
            v-model:refinerSteps="store.highres.refiner.steps"
            v-model:refinerCfg="store.highres.refiner.cfg"
            v-model:refinerSeed="store.highres.refiner.seed"
            v-model:refinerModel="store.highres.refiner.model"
            v-model:refinerVae="store.highres.refiner.vae"
          />
          <RefinerSettingsCard
            v-if="showGlobalRefiner"
            v-model:enabled="store.refiner.enabled"
            v-model:steps="store.refiner.steps"
            v-model:cfg="store.refiner.cfg"
            v-model:seed="store.refiner.seed"
            v-model:model="store.refiner.model"
            v-model:vae="store.refiner.vae"
          />
          <div class="toolbar">
            <div class="qs-actions">
              <button class="btn btn-sm btn-outline" type="button" v-for="p in resolutionPresets" :key="p[0] + 'x' + p[1]" @click="applyResolutionPreset(p[0], p[1])">{{ p[0] }}×{{ p[1] }}</button>
            </div>
            <span class="caption">Aspect ratio: {{ aspectLabel }}</span>
          </div>
          <div class="toolbar">
            <button class="btn btn-sm btn-secondary" type="button" :disabled="store.isRunning" @click="store.saveProfile()">Save Profile</button>
            <span class="caption" v-if="store.profileMessage">{{ store.profileMessage }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header three-cols results-sticky"><span>Results</span>
          <div class="header-center">
            <button class="btn btn-md btn-primary results-generate" :disabled="store.isRunning" @click="onGenerate">
              Generate
            </button>
          </div>
          <div class="header-right results-actions">
            <input class="ui-input" :list="'flux-preset-list'" v-model="presetName" placeholder="Preset" />
            <datalist id="flux-preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="savePreset(presetName)">Save</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyPreset(presetName)">Apply</button>
            <div class="gentime-display" v-if="gentimeSeconds !== null">
              <span class="caption">Time: {{ gentimeSeconds.toFixed(2) }}s</span>
            </div>
          </div>
        </div>
        <div class="panel-body" :style="previewStyle">
          <ResultViewer mode="image" :images="store.gallery" :width="store.width" :height="store.height" emptyText="No images yet. Generate to see results here.">
            <template #image-actions="{ image, index }">
              <button class="gallery-action" type="button" @click="sendToImg2Img(image)" title="Send to Img2Img">Send to Img2Img</button>
              <button class="gallery-action" type="button" @click="sendToInpaint(image)" title="Send to Inpaint">Send to Inpaint</button>
              <button class="gallery-action" type="button" @click="download(image, index)" title="Download Image">Download</button>
            </template>
          </ResultViewer>
        </div>
      </div>

      <div class="panel" v-if="store.info">
        <div class="panel-header">Generation Info</div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ asJson(store.info) }}</pre>
        </div>
      </div>
    </div>

    <CheckpointModal v-model="showCkpt" />
    <LoraModal v-model="showLora" @insert="onInsertToken" />
    <TextualInversionModal v-model="showTI" @insert="onInsertToken" />
    <StyleEditorModal v-model="showStyle" @saved="onStyleSaved" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useFluxStore } from '../stores/flux'
import { useImg2ImgStore } from '../stores/img2img'
import { useInpaintStore } from '../stores/inpaint'
import CheckpointModal from '../components/modals/CheckpointModal.vue'
import LoraModal from '../components/modals/LoraModal.vue'
import TextualInversionModal from '../components/modals/TextualInversionModal.vue'
import PromptFields from '../components/prompt/PromptFields.vue'
import GenerationSettingsCard from '../components/GenerationSettingsCard.vue'
import HighresSettingsCard from '../components/HighresSettingsCard.vue'
import RefinerSettingsCard from '../components/RefinerSettingsCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import StyleEditorModal from '../components/modals/StyleEditorModal.vue'
import { usePresetsStore } from '../stores/presets'
import { useStylesStore } from '../stores/styles'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import type { GeneratedImage } from '../api/types'

const store = useFluxStore()
const img2img = useImg2ImgStore()
const inpaint = useInpaintStore()
const router = useRouter()
const presetsStore = usePresetsStore()
const stylesStore = useStylesStore()
const quicksettings = useQuicksettingsStore()
const engineCaps = useEngineCapabilitiesStore()

const showCkpt = ref(false)
const showLora = ref(false)
const showTI = ref(false)
const showStyle = ref(false)
const styleName = ref('')
const presetName = ref('')
const leftStack = ref<HTMLElement | null>(null)
const previewStyle = ref<Record<string, string>>({})
const gentimeSeconds = computed(() => {
  if (store.gentimeMs == null) return null
  return store.gentimeMs / 1000
})
const progressPercent = computed(() => {
  if (!store.progress.totalSteps) return null
  const step = store.progress.step ?? 0
  return (step / store.progress.totalSteps) * 100
})

onMounted(() => {
  void store.init()
  syncPreviewHeight()
  window.addEventListener('resize', syncPreviewHeight)
  void engineCaps.init()
})

onBeforeUnmount(() => {
  store.stopStream()
  window.removeEventListener('resize', syncPreviewHeight)
})

type PromptInsertPayload = string | { token: string; target?: 'positive' | 'negative' }

function onInsertToken(payload: PromptInsertPayload): void {
  const token = typeof payload === 'string' ? payload : payload.token
  if (!token) return
  // Flux does not use a separate negative/uncond path; always append to the main prompt.
  store.prompt = (store.prompt ? store.prompt + ' ' : '') + token
}

function onGenerate(event: Event): void {
  event.preventDefault()
  void store.generate()
}

function onModelInputChange(event: Event): void {
  const value = (event.target as HTMLInputElement).value.trim()
  if (value && value !== store.selectedModel) {
    void store.updateModel(value)
  }
}

function setSampler(value: string): void {
  store.selectedSampler = value
}

function setScheduler(value: string): void {
  store.selectedScheduler = value
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

function download(image: GeneratedImage, index: number): void {
  const link = document.createElement('a')
  link.href = toDataUrl(image)
  link.download = `flux_${index + 1}.png`
  link.click()
}

function sendToImg2Img(image: GeneratedImage): void {
  try {
    const bytes = atob(image.data)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const file = new File([buf], 'from_flux.png', { type: `image/${image.format}` })
    void img2img.setInitImage(file)
    router.push('/img2img')
  } catch (e) {
    console.error('Failed to send to Img2Img', e)
  }
}

function sendToInpaint(image: GeneratedImage): void {
  try {
    const bytes = atob(image.data)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const file = new File([buf], 'from_flux.png', { type: `image/${image.format}` })
    void inpaint.setInitImage(file)
    router.push({ path: '/img2img', query: { tab: 'inpaint' } })
  } catch (e) {
    console.error('Failed to send to Inpaint', e)
  }
}

const presetNames = computed(() => presetsStore.names('flux'))
const styleNames = computed(() => stylesStore.names())
const semanticEngine = computed(() => quicksettings.currentEngine || 'flux')
const engineSurface = computed(() => engineCaps.get(semanticEngine.value))
const showHighres = computed(() => {
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_highres
})
const showGlobalRefiner = computed(() => {
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_refiner
})

function snapshotParams(): Record<string, unknown> {
  return {
    prompt: store.prompt,
    negative: store.negativePrompt,
    steps: store.steps,
    width: store.width,
    height: store.height,
    cfgScale: store.cfgScale,
    batchSize: store.batchSize,
    batchCount: store.batchCount,
    sampler: store.selectedSampler,
    scheduler: store.selectedScheduler,
  }
}
function applyParams(v: Record<string, unknown>): void {
  if ('prompt' in v) store.prompt = String(v.prompt)
  if ('negative' in v) store.negativePrompt = String(v.negative)
  if ('steps' in v) store.steps = Number(v.steps)
  if ('width' in v) store.width = Number(v.width)
  if ('height' in v) store.height = Number(v.height)
  if ('cfgScale' in v) store.cfgScale = Number(v.cfgScale)
  if ('batchSize' in v) store.batchSize = Number(v.batchSize)
  if ('batchCount' in v) store.batchCount = Number(v.batchCount)
  if ('sampler' in v) store.selectedSampler = String(v.sampler)
  if ('scheduler' in v) store.selectedScheduler = String(v.scheduler)
}
function savePreset(name: string): void { presetsStore.upsert('flux', name, snapshotParams()) }
function applyPreset(name: string): void { const v = presetsStore.get('flux', name); if (v) applyParams(v) }
function applyStyle(name: string): void {
  const d = stylesStore.get(name)
  if (!d) return
  if (d.prompt) store.prompt += (store.prompt ? ' ' : '') + d.prompt
  if (d.negative) store.negativePrompt += (store.negativePrompt ? ' ' : '') + d.negative
}
function onStyleSaved(): void { /* reactive */ }

function syncPreviewHeight(): void {
  const el = leftStack.value
  if (!el) return
  const h = el.getBoundingClientRect().height
  previewStyle.value = { minHeight: `${Math.max(300, Math.floor(h))}px` }
}

function asJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch (error) {
    return String(value)
  }
}

const resolutionPresets = computed((): [number, number][] => [
  [1024, 1024],
  [1152, 896],
  [1216, 832],
  [1344, 768],
])
const aspectLabel = computed(() => store.aspectLabel)
function applyResolutionPreset(w: number, h: number): void { store.width = w; store.height = h }

</script>
