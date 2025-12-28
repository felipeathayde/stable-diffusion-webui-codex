<template>
  <section class="panels">
    
    <!-- Left column: Prompt + Parameters (Codex layout) -->
    <div class="panel-stack" ref="leftStack">
      <PromptCard v-model:prompt="store.prompt" v-model:negative="store.negativePrompt">
        <!-- generation controls moved to Run card header -->
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
      </PromptCard>

      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <BasicParametersCard
            :samplers="store.samplers"
            :schedulers="store.schedulers"
            :sampler="store.selectedSampler"
            :scheduler="store.selectedScheduler"
            :steps="store.steps"
            :width="store.width"
            :height="store.height"
            :cfg-scale="store.cfgScale"
            :seed="store.seed"
            :resolutionPresets="resolutionPresets"
            @update:sampler="store.setSampler"
            @update:scheduler="store.setScheduler"
            @update:steps="(v:number)=>store.steps=v"
            @update:width="(v:number)=>store.width=v"
            @update:height="(v:number)=>store.height=v"
            @update:cfgScale="(v:number)=>store.cfgScale=v"
            @update:seed="(v:number)=>store.seed=v"
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
        </div>
      </div>
    </div>

    <!-- Right column: Preview/Gallery + Info (Codex layout) -->
    <div class="panel-stack">
      <RunCard
        :generateDisabled="store.isRunning"
        :isRunning="store.isRunning"
        :batchCount="store.batchCount"
        :batchSize="store.batchSize"
        :disabled="store.isRunning"
        @generate="onGenerate"
        @update:batchCount="(v:number)=>store.batchCount=v"
        @update:batchSize="(v:number)=>store.batchSize=v"
      >
        <RunSummaryChips :text="runSummary" />
      </RunCard>

      <ResultsCard :showGenerate="false" headerClass="three-cols" headerRightClass="results-actions">
        <template #header-right>
          <input class="ui-input" :list="'preset-list'" v-model="presetName" placeholder="Preset" />
          <datalist id="preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
          <button class="btn btn-sm btn-secondary" type="button" @click="savePreset(presetName)">Save</button>
          <button class="btn btn-sm btn-outline" type="button" @click="applyPreset(presetName)">Apply</button>
        </template>

        <ResultViewer mode="image" :images="store.gallery" :width="store.width" :height="store.height" emptyText="No images yet. Generate to see results here." :style="previewStyle">
          <template #image-actions="{ image, index }">
            <button class="gallery-action" type="button" @click="sendToImg2Img(image)" title="Send to Img2Img">Send to Img2Img</button>
            <button class="gallery-action" type="button" @click="sendToInpaint(image)" title="Send to Inpaint">Send to Inpaint</button>
            <button class="gallery-action" type="button" @click="download(image, index)" title="Download Image">Download</button>
          </template>
        </ResultViewer>
      </ResultsCard>

      <div class="panel" v-if="store.info">
        <div class="panel-header">Generation Info</div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ formatJson(store.info) }}</pre>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
	import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
	import { useTxt2ImgStore } from '../stores/txt2img'
	import PromptCard from '../components/prompt/PromptCard.vue'
	import BasicParametersCard from '../components/BasicParametersCard.vue'
	import HighresSettingsCard from '../components/HighresSettingsCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import { formatJson } from '../composables/useResultsCard'
	import { usePresetsStore } from '../stores/presets'
	import { useQuicksettingsStore } from '../stores/quicksettings'
	import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
	import { useModelTabNavigation } from '../composables/useModelTabNavigation'
	import type { GeneratedImage } from '../api/types'

	const store = useTxt2ImgStore()
	const presetsStore = usePresetsStore()
	const quicksettings = useQuicksettingsStore()
	const engineCaps = useEngineCapabilitiesStore()
	const { openModelTab } = useModelTabNavigation()
// Subtabs removed; actions via header toolbar
const presetName = ref('')
const leftStack = ref<HTMLElement | null>(null)
const previewStyle = ref<Record<string,string>>({})

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

function onGenerate(): void {
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

	async function sendToImg2Img(image: GeneratedImage): Promise<void> {
	  try {
	    await openModelTab('sd15', { initImage: { dataUrl: toDataUrl(image), name: 'from_txt2img.png' } })
	  } catch (e) {
	    console.error('Failed to open Img2Img tab', e)
	  }
	}
	
	async function sendToInpaint(image: GeneratedImage): Promise<void> {
	  try {
	    await openModelTab('sd15', { initImage: { dataUrl: toDataUrl(image), name: 'from_txt2img_inpaint.png' } })
	  } catch (e) {
	    console.error('Failed to open Inpaint tab', e)
	  }
	}

// Presets and Styles integration
const presetNames = computed(() => presetsStore.names('txt2img'))
const semanticEngine = computed(() => quicksettings.currentEngine || 'sd15')
const engineSurface = computed(() => engineCaps.get(semanticEngine.value))
const showHighres = computed(() => {
  const surf = engineSurface.value
  // Default to visible when capabilities are unavailable.
  if (!surf) return true
  return surf.supports_highres
})
const showGlobalRefiner = computed(() => {
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_refiner
})
const runSummary = computed(() => {
  const sampler = store.selectedSampler || 'automatic'
  const scheduler = store.selectedScheduler || 'automatic'
  const seedLabel = store.seed === -1 ? 'seed random' : `seed ${store.seed}`
  return `${store.width}×${store.height} px · ${store.steps} steps · CFG ${store.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel} · batch ${store.batchCount}×${store.batchSize}`
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
function savePreset(name: string): void { presetsStore.upsert('txt2img', name, snapshotParams()) }
function applyPreset(name: string): void { const v = presetsStore.get('txt2img', name); if (v) applyParams(v) }

// Preview height sync with left blocks
function syncPreviewHeight(): void {
  const el = leftStack.value
  if (!el) return
  const h = el.getBoundingClientRect().height
  previewStyle.value = { minHeight: `${Math.max(300, Math.floor(h))}px` }
}

const isXL = computed(() => (store.selectedModel || '').toLowerCase().includes('xl'))
const resolutionPresets = computed((): [number,number][] => {
  return isXL.value ? [ [1024,1024], [1152,896], [1216,832] ] : [ [512,512], [512,768], [768,512] ]
})
</script>
