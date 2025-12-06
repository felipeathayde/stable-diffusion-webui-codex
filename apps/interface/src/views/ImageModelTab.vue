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
import { useModelTabsStore, type BaseTab, type ImageBaseParams } from '../stores/model_tabs'
import type { SamplerInfo, SchedulerInfo, GeneratedImage, TaskEvent } from '../api/types'
import { fetchSamplers, fetchSchedulers, startTxt2Img, startImg2Img, subscribeTask } from '../api/client'
import { buildTxt2ImgPayload, deriveFluxTextEncoderOverrideFromLabels, formatZodError } from '../api/payloads'
import type { Txt2ImgRequest } from '../api/payloads'
import { useQuicksettingsStore } from '../stores/quicksettings'

const props = defineProps<{ tabId: string; type: 'sd15' | 'sdxl' | 'flux' }>()
const store = useModelTabsStore()

const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])
const quicksettings = useQuicksettingsStore()

onMounted(async () => {
  if (!store.tabs.length) store.load()
  const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers
})

const tab = computed<BaseTab | null>(() => store.tabs.find(t => t.id === props.tabId) || null)
const params = computed<ImageBaseParams>(() => (tab.value?.params as any) as ImageBaseParams)

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

// Generate
type Status = 'idle' | 'running' | 'error' | 'done'
const status = ref<Status>('idle')
const errorMessage = ref('')
const images = ref<GeneratedImage[]>([])
let unsubscribe: (() => void) | null = null
const lastSeed = ref<number | null>(null)

function stopStream(): void { if (unsubscribe) { unsubscribe(); unsubscribe = null } }
const isRunning = computed(() => status.value === 'running')
function toDataUrl(img: GeneratedImage): string { return `data:image/${img.format};base64,${img.data}` }

async function generate(): Promise<void> {
  if (!tab.value) return
  stopStream(); status.value = 'running'; errorMessage.value = ''; images.value = []
  const p = params.value
  try {
    const quick = quicksettings
    const modelRef = typeof p.model === 'string' && p.model.length > 0 ? p.model : undefined
    if (p.useInitImage && p.initImageData) {
      const shared = {
        width: p.width,
        height: p.height,
        steps: p.steps,
        cfg_scale: p.cfgScale,
        seed: p.seed,
        sampler: p.sampler,
        scheduler: p.scheduler,
      }
      const payload: Record<string, unknown> = {
        __strict_version: 1,
        codex_device: quick.currentDevice,
        img2img_prompt: p.prompt,
        img2img_neg_prompt: p.negativePrompt,
        img2img_init_image: p.initImageData,
        ...Object.fromEntries(Object.entries(shared).map(([k, v]) => [`img2img_${k}`, v])),
        engine: props.type,
        codex_engine: props.type,
      }
      if (modelRef) {
        payload.model = modelRef
        payload.sd_model_checkpoint = modelRef
      }
      const { task_id } = await startImg2Img(payload)
      unsubscribe = subscribeTask(task_id, onTaskEvent)
    } else {
      let payload: Txt2ImgRequest
      try {
        const teOverride = props.type === 'flux'
          ? deriveFluxTextEncoderOverrideFromLabels(quick.currentTextEncoders)
          : undefined
        payload = buildTxt2ImgPayload({
          prompt: p.prompt,
          negativePrompt: p.negativePrompt,
          width: p.width,
          height: p.height,
          steps: p.steps,
          guidanceScale: p.cfgScale,
          sampler: p.sampler || 'automatic',
          scheduler: p.scheduler || 'automatic',
          seed: p.seed,
          batchSize: 1,
          batchCount: 1,
          styles: [],
          device: quick.currentDevice,
          engine: props.type,
          model: modelRef,
          textEncoderOverride: teOverride,
        })
      } catch (error) {
        status.value = 'error'
        errorMessage.value = formatZodError(error)
        return
      }
      const { task_id } = await startTxt2Img(payload)
      unsubscribe = subscribeTask(task_id, onTaskEvent)
    }
  } catch (err) {
    status.value = 'error'
    errorMessage.value = err instanceof Error ? err.message : String(err)
  }
}

function onTaskEvent(event: TaskEvent): void {
  switch (event.type) {
    case 'result':
      images.value = event.images
      status.value = 'done'
      stopStream()
      break
    case 'error':
      status.value = 'error'
      errorMessage.value = event.message
      stopStream()
      break
    default: break
  }
}

function randomizeSeed(): void {
  if (params.value.seed !== -1) lastSeed.value = params.value.seed
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
