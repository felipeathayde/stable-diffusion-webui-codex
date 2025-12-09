<template>
  <section class="panels">
    <div class="panel-stack" ref="leftStack">
      <div class="panel">
        <div class="panel-header"><span>Prompt</span>
          <div class="toolbar prompt-toolbar">
            <label class="label-muted styles-label">Z Image Turbo</label>
          </div>
        </div>
        <div class="panel-body">
          <PromptFields v-model:prompt="store.prompt" v-model:negative="store.negativePrompt" :hide-negative="false" />
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
            :show-cfg="true"
            cfg-label="Distilled CFG"
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
            <div class="gentime-display" v-if="gentimeSeconds !== null">
              <span class="caption">Time: {{ gentimeSeconds.toFixed(2) }}s</span>
            </div>
          </div>
        </div>
        <div class="panel-body" :style="previewStyle">
          <ResultViewer mode="image" :images="store.gallery" :width="store.width" :height="store.height" emptyText="No images yet. Generate to see results here.">
            <template #image-actions="{ image, index }">
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
  </section>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, computed } from 'vue'
import { useZImageStore } from '../stores/zimage'
import PromptFields from '../components/prompt/PromptFields.vue'
import GenerationSettingsCard from '../components/GenerationSettingsCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import type { GeneratedImage } from '../api/types'

const store = useZImageStore()

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
})

onBeforeUnmount(() => {
  store.stopStream()
  window.removeEventListener('resize', syncPreviewHeight)
})

function onGenerate(event: Event): void {
  event.preventDefault()
  void store.generate()
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
  link.download = `zimage_${index + 1}.png`
  link.click()
}

function syncPreviewHeight(): void {
  const el = leftStack.value
  if (!el) return
  const h = el.getBoundingClientRect().height
  previewStyle.value = { minHeight: `${Math.max(300, Math.floor(h))}px` }
}

function asJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
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
