<template>
  <section v-if="tab" class="panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header"><span>Prompt</span></div>
        <div class="panel-body">
          <PromptFields v-model:prompt="videoPrompt" v-model:negative="videoNegative" />

          <div class="gen-card">
            <div class="wan22-toggle-head">
              <span class="label-muted">Initial Image (img2vid)</span>
              <label class="wan22-toggle">
                <input type="checkbox" :disabled="isRunning" :checked="video.useInitImage" @change="onInitToggle" />
                <span>Enable</span>
              </label>
            </div>
            <div v-if="video.useInitImage" class="mt-2">
              <InitialImageCard
                label="Image"
                :disabled="isRunning"
                :src="video.initImageData"
                :hasImage="Boolean(video.initImageData)"
                @set="onInitImageFile"
                @clear="clearInit"
              >
                <template #footer>
                  <div v-if="video.initImageName" class="caption mt-1">{{ video.initImageName }}</div>
                </template>
              </InitialImageCard>
            </div>
            <p v-else class="caption">Disabled (txt2vid).</p>
          </div>

          <div v-if="errorMessage" class="panel-error">{{ errorMessage }}</div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <div class="gen-card">
            <div class="wan22-toggle-head">
              <span class="label-muted">WAN Runtime & Assets</span>
            </div>
            <div class="wan22-grid">
              <div>
                <label class="label-muted">Model Format</label>
                <select class="select-md" :disabled="isRunning" :value="wanFormat" @change="onFormatChange">
                  <option value="auto">Auto</option>
                  <option value="diffusers">Diffusers</option>
                  <option value="gguf">GGUF</option>
                </select>
                <p class="caption mt-1">Assets are managed by QuickSettings (header).</p>
              </div>
              <div>
                <label class="label-muted">High model dir</label>
                <div class="caption break-words">{{ high.modelDir || 'Unset (set in QuickSettings)' }}</div>
              </div>
              <div>
                <label class="label-muted">Low model dir</label>
                <div class="caption break-words">{{ low.modelDir || 'Unset (set in QuickSettings)' }}</div>
              </div>
            </div>
            <div class="wan22-grid">
              <div>
                <label class="label-muted">Text Encoder</label>
                <div class="caption break-words">{{ assets.textEncoder || 'Built-in / paths.json default' }}</div>
              </div>
              <div>
                <label class="label-muted">VAE</label>
                <div class="caption break-words">{{ assets.vae || 'Built-in / paths.json default' }}</div>
              </div>
              <div>
                <label class="label-muted">Metadata Dir</label>
                <div class="caption break-words">{{ assets.metadata || 'Built-in / vendored HF metadata' }}</div>
              </div>
            </div>
            <div class="panel-error" v-if="!high.modelDir && !low.modelDir">
              WAN model directory is empty. Set WAN High/Low model dirs in QuickSettings.
            </div>
          </div>

          <div class="gen-card">
            <div class="wan22-toggle-head">
              <span class="label-muted">Video</span>
            </div>
            <div class="wan22-grid">
              <div>
                <label class="label-muted">Width</label>
                <input class="ui-input" type="number" min="64" step="8" :disabled="isRunning" :value="video.width" @change="setVideo({ width: toInt($event, video.width) })" />
              </div>
              <div>
                <label class="label-muted">Height</label>
                <input class="ui-input" type="number" min="64" step="8" :disabled="isRunning" :value="video.height" @change="setVideo({ height: toInt($event, video.height) })" />
              </div>
            </div>
            <VideoSettingsCard
              :frames="video.frames"
              :fps="video.fps"
              @update:frames="(v:number)=>setVideo({ frames: v })"
              @update:fps="(v:number)=>setVideo({ fps: v })"
            />
          </div>

          <WanStagePanel
            title="High Noise"
            :stage="high"
            :samplers="samplers"
            :schedulers="schedulers"
            :disabled="isRunning"
            @update:stage="setHigh"
          />

          <WanStagePanel
            title="Low Noise"
            :stage="low"
            :samplers="samplers"
            :schedulers="schedulers"
            :disabled="isRunning"
            @update:stage="setLow"
          />

          <WanVideoOutputPanel :video="video" :disabled="isRunning" @update:video="setVideo" />
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Workflows</div>
        <div class="panel-body">
          <div class="gen-card">
            <div class="grid grid-2" style="align-items:center">
              <div>
                <button class="btn btn-secondary" type="button" :disabled="workflowBusy" @click="sendToWorkflows">Send to Workflows</button>
                <RouterLink class="btn btn-ghost" to="/workflows" style="margin-left:.5rem">Open</RouterLink>
              </div>
              <div style="text-align:right" class="caption">Saves a snapshot of this tab’s params.</div>
            </div>
            <div v-if="workflowOk" class="caption mt-1">{{ workflowOk }}</div>
            <div v-if="workflowErr" class="panel-error mt-2">{{ workflowErr }}</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right column: Results -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header three-cols results-sticky"><span>Results</span>
          <div class="header-center"><button class="btn btn-md btn-primary results-generate" :disabled="isRunning" @click="generate">{{ isRunning ? 'Running…' : 'Generate' }}</button></div>
        </div>
        <div class="panel-body">
          <div v-if="isRunning" class="panel-progress">
            <p><strong>Stage:</strong> {{ progress.stage }}</p>
            <p v-if="progress.percent !== null">Progress: {{ progress.percent.toFixed(1) }}%</p>
            <p v-if="progress.step !== null && progress.totalSteps !== null">
              Step {{ progress.step }} / {{ progress.totalSteps }}
            </p>
          </div>
          <ResultViewer mode="video" :frames="framesResult" :toDataUrl="toDataUrl" emptyText="No results yet." />
        </div>
      </div>

      <div class="panel" v-if="info">
        <div class="panel-header">Generation Info</div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ asJson(info) }}</pre>
        </div>
      </div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref } from 'vue'
import { useModelTabsStore, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import type { SamplerInfo, SchedulerInfo, GeneratedImage } from '../api/types'
import { fetchSamplers, fetchSchedulers } from '../api/client'
import ResultViewer from '../components/ResultViewer.vue'
import InitialImageCard from '../components/InitialImageCard.vue'
import VideoSettingsCard from '../components/VideoSettingsCard.vue'
import PromptFields from '../components/prompt/PromptFields.vue'
import WanStagePanel from '../components/wan/WanStagePanel.vue'
import WanVideoOutputPanel from '../components/wan/WanVideoOutputPanel.vue'
import { useVideoGeneration } from '../composables/useVideoGeneration'
import { createWorkflow } from '../api/client'

const props = defineProps<{ tabId: string }>()
const store = useModelTabsStore()

// Load option lists
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(async () => {
  if (!store.tabs.length) store.load()
  const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers
})

const tab = computed(() => store.tabs.find(t => t.id === props.tabId) || null)

function defaultStage(): WanStageParams {
  return { modelDir: '', sampler: '', scheduler: '', steps: 30, cfgScale: 7, seed: -1, lightning: false, loraEnabled: false, loraPath: '', loraWeight: 1.0 }
}
function defaultVideo(): WanVideoParams {
  return {
    prompt: '', negativePrompt: '', width: 768, height: 432, fps: 24, frames: 16,
    useInitImage: false, initImageData: '', initImageName: '',
    filenamePrefix: 'wan22', format: 'video/h264-mp4', pixFmt: 'yuv420p', crf: 15,
    loopCount: 0, pingpong: false, trimToAudio: false, saveMetadata: true, saveOutput: true,
    rifeEnabled: false, rifeModel: '', rifeTimes: 2,
  }
}

const video = computed<WanVideoParams>(() => ((tab.value?.params as any)?.video as WanVideoParams) || defaultVideo())
const high = computed<WanStageParams>(() => ((tab.value?.params as any)?.high as WanStageParams) || defaultStage())
const low = computed<WanStageParams>(() => ((tab.value?.params as any)?.low as WanStageParams) || defaultStage())
const wanFormat = computed<string>(() => (tab.value?.params as any)?.modelFormat || 'auto')

interface WanAssetsParams { metadata: string; textEncoder: string; vae: string }
function defaultAssets(): WanAssetsParams { return { metadata: '', textEncoder: '', vae: '' } }

const assets = computed<WanAssetsParams>(() => ((tab.value?.params as any)?.assets as WanAssetsParams) || defaultAssets())

function setVideo(patch: Partial<WanVideoParams>): void {
  if (!tab.value) return
  const current = (tab.value.params as any).video as WanVideoParams
  store.updateParams(props.tabId, { video: { ...current, ...patch } })
}
function setHigh(patch: Partial<WanStageParams>): void {
  if (!tab.value) return
  const current = (tab.value.params as any).high as WanStageParams
  store.updateParams(props.tabId, { high: { ...current, ...patch } })
}
function setLow(patch: Partial<WanStageParams>): void {
  if (!tab.value) return
  const current = (tab.value.params as any).low as WanStageParams
  store.updateParams(props.tabId, { low: { ...current, ...patch } })
}

const videoPrompt = computed({
  get: () => video.value.prompt,
  set: (value: string) => setVideo({ prompt: value }),
})

const videoNegative = computed({
  get: () => video.value.negativePrompt,
  set: (value: string) => setVideo({ negativePrompt: value }),
})

function onFormatChange(e: Event): void {
  if (!tab.value) return
  store.updateParams(props.tabId, { modelFormat: (e.target as HTMLSelectElement).value })
}

function toInt(e: Event, fallback: number): number { const v = Number((e.target as HTMLInputElement).value); return Number.isFinite(v) ? Math.trunc(v) : fallback }

function onInitToggle(e: Event): void {
  setVideo({ useInitImage: (e.target as HTMLInputElement).checked })
  if (!(e.target as HTMLInputElement).checked) setVideo({ initImageData: '', initImageName: '' })
}

async function onInitImageFile(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  setVideo({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function clearInit(): void { setVideo({ initImageData: '', initImageName: '' }) }

// Generation wiring (composable)
const {
  generate,
  isRunning,
  progress,
  frames: framesResult,
  info,
  errorMessage,
} = useVideoGeneration(props.tabId)

const workflowBusy = ref(false)
const workflowOk = ref('')
const workflowErr = ref('')

async function sendToWorkflows(): Promise<void> {
  if (!tab.value) return
  workflowOk.value = ''
  workflowErr.value = ''
  workflowBusy.value = true
  try {
    await createWorkflow({
      name: `${tab.value.title} — ${new Date().toLocaleString()}`,
      source_tab_id: tab.value.id,
      type: tab.value.type,
      engine_semantics: tab.value.type === 'wan' ? 'wan22' : tab.value.type,
      params_snapshot: tab.value.params as Record<string, unknown>,
    })
    workflowOk.value = 'Workflow saved.'
  } catch (e) {
    workflowErr.value = e instanceof Error ? e.message : String(e)
  } finally {
    workflowBusy.value = false
  }
}

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function asJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch (error) {
    return String(value)
  }
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

defineExpose({ generate })
</script>
