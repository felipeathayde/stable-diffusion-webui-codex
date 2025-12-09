<template>
  <section v-if="tab">
    <!-- Prompt & Input -->
    <div class="panel">
      <div class="panel-header"><h3 class="h4">Prompt & Input</h3></div>
      <div class="panel-body">
        <div class="grid grid-2">
          <div>
            <label class="label">Prompt</label>
            <textarea class="ui-textarea" rows="3" :value="params.prompt" @input="setParams({ prompt: ($event.target as HTMLTextAreaElement).value })"></textarea>
            <label class="label" style="margin-top:.5rem">Negative</label>
            <textarea class="ui-textarea" rows="2" :value="params.negativePrompt" @input="setParams({ negativePrompt: ($event.target as HTMLTextAreaElement).value })"></textarea>
          </div>
          <div>
            <div class="grid grid-2">
              <div>
                <label class="label">Width</label>
                <input class="ui-input" type="number" min="64" step="8" :value="params.width" @change="setParams({ width: toInt($event, params.width) })" />
              </div>
              <div>
                <label class="label">Height</label>
                <input class="ui-input" type="number" min="64" step="8" :value="params.height" @change="setParams({ height: toInt($event, params.height) })" />
              </div>
            </div>
            <div class="panel-sub" style="margin-top:.5rem">
              <label class="switch-label">
                <input type="checkbox" :checked="params.useInitImage" @change="onInitToggle" />
                <span>Use Initial Image (img2img)</span>
              </label>
              <div v-if="params.useInitImage" class="grid grid-2" style="margin-top:.5rem">
                <div>
                  <label class="label">Image</label>
                  <input class="ui-input" type="file" accept="image/*" @change="onFile" />
                  <div v-if="params.initImageName" class="muted" style="margin-top:.25rem">{{ params.initImageName }}</div>
                  <button v-if="params.initImageData" class="btn btn-sm" type="button" style="margin-top:.5rem" @click="clearInit">Clear</button>
                </div>
                <div v-if="params.initImageData">
                  <label class="label">Preview</label>
                  <img :src="params.initImageData" alt="init" style="max-width:100%; border-radius:.25rem;" />
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Generation Parameters -->
    <div class="panel">
      <div class="panel-header"><h3 class="h4">Generation Parameters</h3></div>
      <div class="panel-body">
        <div class="grid grid-3">
          <div>
            <label class="label">Sampler</label>
            <select class="select-md" :value="params.sampler" @change="setParams({ sampler: ($event.target as HTMLSelectElement).value })">
              <option v-for="s in samplers" :key="s.name" :value="s.name">{{ s.name }}</option>
            </select>
          </div>
          <div>
            <label class="label">Scheduler</label>
            <select class="select-md" :value="params.scheduler" @change="setParams({ scheduler: ($event.target as HTMLSelectElement).value })">
              <option v-for="s in schedulers" :key="s.name" :value="s.name">{{ s.name }}</option>
            </select>
          </div>
          <div>
            <label class="label">Steps</label>
            <input class="ui-input" type="number" min="1" :value="params.steps" @change="setParams({ steps: toInt($event, params.steps) })" />
          </div>
        </div>
        <div class="grid grid-3" style="margin-top:.5rem">
          <div>
            <label class="label">CFG</label>
            <input class="ui-input" type="number" step="0.5" :value="params.cfgScale" @change="setParams({ cfgScale: toFloat($event, params.cfgScale) })" />
          </div>
          <div>
            <label class="label">Seed</label>
            <div class="grid grid-3">
              <input class="ui-input" type="number" :value="params.seed" @change="setParams({ seed: toInt($event, params.seed) })" />
              <button class="btn btn-sm" type="button" @click="randomizeSeed">Random</button>
              <button class="btn btn-sm" type="button" @click="reuseSeed" :disabled="lastSeed === null">Reuse</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Results -->
    <div class="panel">
      <div class="panel-header sticky">
        <div class="grid grid-2">
          <h3 class="h4">Results</h3>
          <div style="text-align:right">
            <button class="btn btn-primary" type="button" :disabled="isRunning" @click="generate">{{ isRunning ? 'Running…' : 'Generate' }}</button>
          </div>
        </div>
      </div>
      <div class="panel-body">
        <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
        <div v-if="images.length" class="results-grid">
          <img v-for="(img, i) in images" :key="i" :src="toDataUrl(img)" :alt="`img-${i}`" />
        </div>
        <div v-else class="muted">No results yet.</div>
      </div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref } from 'vue'
import { useModelTabsStore, type ImageBaseParams } from '../stores/model_tabs'
import type { SamplerInfo, SchedulerInfo, GeneratedImage } from '../api/types'
import { fetchSamplers, fetchSchedulers } from '../api/client'
import { useGeneration } from '../composables/useGeneration'
import type { EngineType } from '../stores/engine_config'

const props = defineProps<{ tabId: string; type: EngineType }>()
const store = useModelTabsStore()

// Use unified generation composable
const {
  generate,
  status,
  gallery,
  errorMessage,
  isRunning,
  lastSeed,
  tab,
  params: genParams,
} = useGeneration(props.tabId)

const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(async () => {
  if (!store.tabs.length) store.load()
  const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers
})

const params = computed<ImageBaseParams>(() => (tab.value?.params as any) as ImageBaseParams)
const images = computed(() => gallery.value)

function setParams(patch: Partial<ImageBaseParams>): void {
  if (!tab.value) return
  store.updateParams(props.tabId, { ...(tab.value.params as any), ...patch })
}

function toInt(e: Event, fallback: number): number { const v = Number((e.target as HTMLInputElement).value); return Number.isFinite(v) ? Math.trunc(v) : fallback }
function toFloat(e: Event, fallback: number): number { const v = Number((e.target as HTMLInputElement).value); return Number.isFinite(v) ? v : fallback }

function onInitToggle(e: Event): void {
  setParams({ useInitImage: (e.target as HTMLInputElement).checked })
  if (!(e.target as HTMLInputElement).checked) setParams({ initImageData: '', initImageName: '' })
}

async function onFile(e: Event): Promise<void> {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  const dataUrl = await readFileAsDataURL(file)
  setParams({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function clearInit(): void { setParams({ initImageData: '', initImageName: '' }) }

function toDataUrl(img: GeneratedImage): string { return `data:image/${img.format};base64,${img.data}` }

function randomizeSeed(): void {
  if (params.value.seed !== -1 && lastSeed.value === null) {
    // Store current seed before randomizing
  }
  setParams({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) setParams({ seed: lastSeed.value })
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file)
  })
}

defineExpose({ generate })
</script>

