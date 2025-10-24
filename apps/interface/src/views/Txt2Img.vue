<template>
  <section class="panels">
    
    <!-- Left column: Prompt + Parameters (like A1111) -->
    <div class="panel-stack" ref="leftStack">
      <div class="panel">
        <div class="panel-header"><span>Prompt</span>
          <div class="toolbar prompt-toolbar">
            <button class="btn btn-sm btn-secondary" type="button" @click="showCkpt=true">Checkpoints</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showTI=true">Textual Inversion</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showLora=true">LoRA</button>
            <!-- Styles inline to tweak prompts -->
            <label class="label-muted styles-label">Styles</label>
            <input class="ui-input styles-input" list="style-list" v-model="styleName" placeholder="Filter styles" />
            <datalist id="style-list">
              <option v-for="s in styleNames" :key="s" :value="s" />
            </datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="showStyle=true">New</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyStyle(styleName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <PromptFields v-model:prompt="store.prompt" v-model:negative="store.negativePrompt" />
          <!-- generation controls moved to Results header -->
          <div v-if="store.isRunning" class="panel-progress">
            <p><strong>Stage:</strong> {{ store.progress.stage }}</p>
            <p v-if="store.progress.percent !== null">Progress: {{ store.progress.percent?.toFixed(1) }}%</p>
            <p v-if="store.progress.step !== null && store.progress.totalSteps !== null">
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
            @update:sampler="store.setSampler"
            @update:scheduler="store.setScheduler"
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
          <div class="toolbar">
            <div class="qs-actions">
              <button class="btn btn-sm btn-outline" type="button" v-for="p in resolutionPresets" :key="p[0]+'x'+p[1]" @click="applyResolutionPreset(p[0], p[1])">{{ p[0] }}×{{ p[1] }}</button>
            </div>
            <span class="caption">Aspect ratio: {{ aspectLabel }}</span>
          </div>
          
        </div>
      </div>
    </div>

    <!-- Right column: Preview/Gallery + Info (like A1111) -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header three-cols results-sticky"><span>Results</span>
          <div class="header-center"><button class="btn btn-md btn-primary results-generate" :disabled="store.isRunning" @click="onGenerate">Generate</button></div>
          <div class="header-right results-actions">
            <input class="ui-input" :list="'preset-list'" v-model="presetName" placeholder="Preset" />
            <datalist id="preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="savePreset(presetName)">Save</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyPreset(presetName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <ResultViewer mode="image" :images="store.gallery" :width="store.width" :height="store.height" emptyText="No images yet. Generate to see results here." :style="previewStyle">
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
    <!-- Modals -->
    <CheckpointModal v-model="showCkpt" />
    <LoraModal v-model="showLora" @insert="onInsertToken" />
    <TextualInversionModal v-model="showTI" @insert="onInsertToken" />
    <StyleEditorModal v-model="showStyle" @saved="onStyleSaved" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useTxt2ImgStore } from '../stores/txt2img'
import { useImg2ImgStore } from '../stores/img2img'
import { useInpaintStore } from '../stores/inpaint'
import CheckpointModal from '../components/modals/CheckpointModal.vue'
import LoraModal from '../components/modals/LoraModal.vue'
import TextualInversionModal from '../components/modals/TextualInversionModal.vue'
import PromptFields from '../components/prompt/PromptFields.vue'
import GenerationSettingsCard from '../components/GenerationSettingsCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import StyleEditorModal from '../components/modals/StyleEditorModal.vue'
import { usePresetsStore } from '../stores/presets'
import { useStylesStore } from '../stores/styles'
import type { GeneratedImage } from '../api/types'

const store = useTxt2ImgStore()
const img2img = useImg2ImgStore()
const inpaint = useInpaintStore()
const router = useRouter()
const presetsStore = usePresetsStore()
const stylesStore = useStylesStore()
// Subtabs removed; actions via header toolbar
const showCkpt = ref(false)
const showLora = ref(false)
const showTI = ref(false)
const showStyle = ref(false)
const styleName = ref('')
const presetName = ref('')
const leftStack = ref<HTMLElement | null>(null)
const previewStyle = ref<Record<string,string>>({})

onMounted(() => {
  void store.init()
  syncPreviewHeight()
  window.addEventListener('resize', syncPreviewHeight)
})

onBeforeUnmount(() => {
  store.stopStream()
  window.removeEventListener('resize', syncPreviewHeight)
})

function onModelChange(event: Event): void {
  const target = event.target as HTMLSelectElement
  void store.updateModel(target.value)
}

function onSamplerChange(event: Event): void {
  const target = event.target as HTMLSelectElement
  store.setSampler(target.value)
}

function onSchedulerChange(event: Event): void {
  const target = event.target as HTMLSelectElement
  store.setScheduler(target.value)
}

// clear removed per new design

function onGenerate(event: Event): void {
  event.preventDefault()
  void store.generate()
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

function download(image: GeneratedImage, index: number): void {
  const link = document.createElement('a')
  link.href = toDataUrl(image)
  link.download = `txt2img_${index + 1}.png`
  link.click()
}

function sendToImg2Img(image: GeneratedImage): void {
  try {
    const bytes = atob(image.data)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const file = new File([buf], 'from_txt2img.png', { type: `image/${image.format}` })
    void img2img.setInitImage(file)
    router.push('/img2img')
  } catch (e) {
    console.error('Failed to send to Img2Img', e)
  }
}

function onInsertToken(token: string): void {
  store.prompt = (store.prompt ? store.prompt + ' ' : '') + token
}

function sendToInpaint(image: GeneratedImage): void {
  try {
    const bytes = atob(image.data)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const file = new File([buf], 'from_txt2img.png', { type: `image/${image.format}` })
    void inpaint.setInitImage(file)
    router.push({ path: '/img2img', query: { tab: 'inpaint' } })
  } catch (e) {
    console.error('Failed to send to Inpaint', e)
  }
}

// Presets and Styles integration
const presetNames = computed(() => presetsStore.names('txt2img'))
const styleNames = computed(() => stylesStore.names())
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
function savePreset(name: string): void { presetsStore.upsert('txt2img', name, snapshotParams()) }
function applyPreset(name: string): void { const v = presetsStore.get('txt2img', name); if (v) applyParams(v) }
function applyStyle(name: string): void { const d = stylesStore.get(name); if (!d) return; if (d.prompt) store.prompt += (store.prompt? ' ' : '') + d.prompt; if (d.negative) store.negativePrompt += (store.negativePrompt? ' ' : '') + d.negative }
function onStyleSaved(name: string): void { /* styles list reactive */ }

// Preview height sync with left blocks
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

const aspectLabel = computed(() => {
  const g = (a:number,b:number)=>b?g(b,a%b):a
  const d = g(store.width, store.height) || 1
  return `${store.width/d}:${store.height/d}`
})
const isXL = computed(() => (store.selectedModel || '').toLowerCase().includes('xl'))
const resolutionPresets = computed((): [number,number][] => {
  return isXL.value ? [ [1024,1024], [1152,896], [1216,832] ] : [ [512,512], [512,768], [768,512] ]
})
function applyResolutionPreset(w:number,h:number): void { store.width = w; store.height = h }
function swapWH(): void { const w = store.width; store.width = store.height; store.height = w }

function applyAspect(ratio: number): void {
  if (!Number.isFinite(ratio) || ratio <= 0) return
  if (ratio >= 1) {
    const newHeight = Math.max(64, Math.round((store.width / ratio) / 64) * 64)
    store.height = newHeight
  } else {
    const newWidth = Math.max(64, Math.round((store.height * ratio) / 64) * 64)
    store.width = newWidth
  }
}
</script>
