<template>
  <section class="panels">
    <div class="panel-stack" ref="leftStack">
      <PromptCard
        v-model:prompt="store.prompt"
        v-model:negative="store.negativePrompt"
        :enableAssets="false"
        :enableStyles="false"
        :supportsNegative="false"
        toolbarLabel="Z Image Turbo"
      >
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
      </PromptCard>

      <div class="panel">
        <div class="panel-header">
          Generation Parameters
          <div class="toolbar">
            <span v-if="store.profileMessage" class="caption">{{ store.profileMessage }}</span>
            <button class="btn btn-sm btn-secondary" type="button" :disabled="store.isRunning" @click="store.saveProfile()">Save Profile</button>
          </div>
        </div>
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
            :show-cfg="true"
            cfg-label="Distilled CFG"
            @update:sampler="setSampler"
            @update:scheduler="setScheduler"
            @update:steps="(v:number)=>store.steps=v"
            @update:width="(v:number)=>store.width=v"
            @update:height="(v:number)=>store.height=v"
            @update:cfgScale="(v:number)=>store.cfgScale=v"
            @update:seed="(v:number)=>store.seed=v"
            @random-seed="store.randomizeSeed"
            @reuse-seed="store.reuseSeed"
          />
        </div>
      </div>
    </div>

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
          <div class="gentime-display" v-if="gentimeSeconds !== null">
            <span class="caption">Time: {{ gentimeSeconds.toFixed(2) }}s</span>
          </div>
        </template>

        <div :style="previewStyle">
          <ResultViewer mode="image" :images="store.gallery" :width="store.width" :height="store.height" emptyText="No images yet. Generate to see results here.">
            <template #image-actions="{ image, index }">
              <button class="gallery-action" type="button" @click="download(image, index)" title="Download Image">Download</button>
            </template>
          </ResultViewer>
        </div>
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
import { useZImageStore } from '../stores/zimage'
import PromptCard from '../components/prompt/PromptCard.vue'
import BasicParametersCard from '../components/BasicParametersCard.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import { formatJson } from '../composables/useResultsCard'
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
const runSummary = computed(() => {
  const sampler = store.selectedSampler || 'automatic'
  const scheduler = store.selectedScheduler || 'automatic'
  const seedLabel = store.seed === -1 ? 'seed random' : `seed ${store.seed}`
  return `${store.width}×${store.height} px · ${store.steps} steps · CFG ${store.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel} · batch ${store.batchCount}×${store.batchSize}`
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

function onGenerate(): void {
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

const resolutionPresets = computed((): [number, number][] => [
  [1024, 1024],
  [1152, 896],
  [1216, 832],
  [1344, 768],
])
</script>
