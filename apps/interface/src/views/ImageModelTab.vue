<template>
  <section v-if="tab">
    <!-- Prompt & Input -->
    <div class="panel">
      <div class="panel-header">Prompt & Input</div>
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
      <div class="panel-header">Generation Parameters</div>
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
    <RunCard
      :isRunning="isRunning"
      :generateDisabled="isRunning"
      :showBatchControls="false"
      @generate="generate"
    >
      <div v-if="copyNotice" class="caption">{{ copyNotice }}</div>
      <RunSummaryChips :text="runSummary" />
    </RunCard>

    <ResultsCard
      :showGenerate="false"
      headerClass="three-cols"
      headerRightClass="results-header-actions"
    >
      <template #header-right>
        <button class="btn btn-sm btn-outline" type="button" :disabled="workflowBusy" @click="sendToWorkflows">
          {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
        </button>
        <button class="btn btn-sm btn-outline" type="button" @click="copyCurrentParams">Copy params</button>
      </template>

      <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
      <div v-if="images.length" class="results-grid">
        <img v-for="(img, i) in images" :key="i" :src="toDataUrl(img)" :alt="`img-${i}`" />
      </div>
      <div v-else class="muted">No results yet.</div>
    </ResultsCard>
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
import { useWorkflowsStore } from '../stores/workflows'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import { useResultsCard } from '../composables/useResultsCard'

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
const runSummary = computed(() => {
  const sampler = params.value.sampler || 'automatic'
  const scheduler = params.value.scheduler || 'automatic'
  const seedLabel = params.value.seed === -1 ? 'seed random' : `seed ${params.value.seed}`
  return `${params.value.width}×${params.value.height} px · ${params.value.steps} steps · CFG ${params.value.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel}`
})

const workflows = useWorkflowsStore()
const workflowBusy = ref(false)

const { notice: copyNotice, toast, copyJson } = useResultsCard()

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
