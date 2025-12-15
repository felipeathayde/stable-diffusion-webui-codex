<template>
  <section v-if="tab" class="panels wan-panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header"><span>Prompt</span></div>
        <div class="panel-body">
          <PromptFields v-model:prompt="videoPrompt" v-model:negative="videoNegative" />

          <div class="gen-card">
            <div class="wan22-toggle-head">
              <span class="label-muted">Mode</span>
            </div>
            <div class="subtabs">
              <button class="subtab" type="button" :disabled="isRunning" :class="{ active: mode === 'txt2vid' }" @click="setInputMode('txt2vid')">Text (txt2vid)</button>
              <button class="subtab" type="button" :disabled="isRunning" :class="{ active: mode === 'img2vid' }" @click="setInputMode('img2vid')">Image (img2vid)</button>
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
            <p v-else class="caption">Text mode: no initial image.</p>
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
                <div class="caption break-words">{{ assets.textEncoder || 'Built-in (default)' }}</div>
              </div>
              <div>
                <label class="label-muted">VAE</label>
                <div class="caption break-words">{{ assets.vae || 'Built-in (default)' }}</div>
              </div>
              <div>
                <label class="label-muted">Metadata Dir</label>
                <div class="caption break-words">{{ assets.metadata || 'Built-in (bundled metadata)' }}</div>
              </div>
            </div>
            <p v-if="!high.modelDir && !low.modelDir" class="caption">Tip: set WAN High/Low model dirs in QuickSettings.</p>
          </div>

          <div class="gen-card">
            <div class="wan22-toggle-head">
              <span class="label-muted">Video</span>
            </div>
            <div class="wan22-grid">
              <div>
                <label class="label-muted">Width (px)</label>
                <input class="ui-input" type="number" min="64" step="8" :disabled="isRunning" :value="video.width" @change="onWidthChange" />
              </div>
              <div>
                <label class="label-muted">Height (px)</label>
                <input class="ui-input" type="number" min="64" step="8" :disabled="isRunning" :value="video.height" @change="onHeightChange" />
              </div>
              <div>
                <label class="label-muted">Aspect</label>
                <select class="select-md" :disabled="isRunning" :value="aspectMode" @change="onAspectModeChange">
                  <option value="free">Free</option>
                  <option value="current">Lock current</option>
                  <option value="16:9">16:9</option>
                  <option value="1:1">1:1</option>
                  <option value="9:16">9:16</option>
                  <option value="4:3">4:3</option>
                  <option value="3:4">3:4</option>
                </select>
                <p v-if="aspectMode !== 'free'" class="caption mt-1">Keeps ratio while editing width/height.</p>
              </div>
            </div>
            <VideoSettingsCard
              :frames="video.frames"
              :fps="video.fps"
              @update:frames="(v:number)=>setVideo({ frames: v })"
              @update:fps="(v:number)=>setVideo({ fps: v })"
            />
          </div>

          <details class="accordion" open>
            <summary>High Noise</summary>
            <div class="accordion-body">
              <WanStagePanel
                title="High Noise"
                embedded
                :stage="high"
                :samplers="samplers"
                :schedulers="schedulers"
                :disabled="isRunning"
                @update:stage="setHigh"
              />
              <div class="wan-callout-actions">
                <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="copyHighToLow">Copy High → Low</button>
              </div>
            </div>
          </details>

          <details class="accordion">
            <summary>Low Noise</summary>
            <div class="accordion-body">
              <WanStagePanel
                title="Low Noise"
                embedded
                :stage="low"
                :samplers="samplers"
                :schedulers="schedulers"
                :disabled="isRunning"
                @update:stage="setLow"
              />
            </div>
          </details>

          <WanVideoOutputPanel :video="video" :disabled="isRunning" @update:video="setVideo" />
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Workflows</div>
        <div class="panel-body">
          <div class="gen-card">
            <div class="grid grid-cols-2 items-center gap-3">
              <div>
                <button class="btn btn-secondary" type="button" :disabled="workflowBusy" @click="sendToWorkflows">Send to Workflows</button>
                <RouterLink class="btn btn-ghost ml-2" to="/workflows">Open</RouterLink>
              </div>
              <div class="caption text-right">Saves a snapshot of this tab’s params.</div>
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
          <div class="header-center"><button class="btn btn-md btn-primary results-generate" :disabled="isRunning || !canGenerate" :title="!canGenerate ? blockedReason : ''" @click="generate">{{ isRunning ? 'Running…' : 'Generate' }}</button></div>
          <div class="header-right">
            <div class="wan-header-actions">
              <button
                v-if="isRunning"
                class="btn btn-sm btn-secondary"
                type="button"
                :disabled="queue.length >= queueMax || !canGenerate"
                :title="queue.length >= queueMax ? `Queue full (max ${queueMax}).` : (!canGenerate ? blockedReason : '')"
                @click="queueNext"
              >
                Queue ({{ queue.length }}/{{ queueMax }})
              </button>
              <button v-else-if="history.length" class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="reuseLast">
                Reuse last
              </button>
              <button class="btn btn-sm btn-outline" type="button" :disabled="workflowBusy" @click="sendToWorkflows">
                {{ workflowBusy ? 'Saving…' : 'Snapshot' }}
              </button>
              <button v-if="isRunning" class="btn btn-sm btn-secondary" type="button" :disabled="cancelRequested" @click="cancel()">
                {{ cancelRequested ? 'Cancelling…' : 'Cancel' }}
              </button>
            </div>
          </div>
        </div>
        <div class="panel-body">
          <div v-if="!isRunning && !canGenerate" class="panel-progress">
            <div class="wan-callout-actions">
              <span>{{ blockedReason }}</span>
              <button v-if="needsWanModels" class="btn btn-sm btn-secondary" type="button" @click="focusWanModelsQuicksettings">Fix in QuickSettings</button>
            </div>
          </div>
          <div v-if="copyNotice" class="caption">{{ copyNotice }}</div>
          <div v-if="isRunning" class="panel-progress">
            <p><strong>Stage:</strong> {{ progress.stage }}</p>
            <p v-if="progress.percent !== null">Progress: {{ progress.percent.toFixed(1) }}%</p>
            <progress v-if="progress.percent !== null" class="wan-progress" :value="progress.percent" max="100"></progress>
            <p v-if="progress.step !== null && progress.totalSteps !== null">
              Step {{ progress.step }} / {{ progress.totalSteps }}
            </p>
            <p v-if="progress.etaSeconds !== null" class="caption">ETA ~ {{ progress.etaSeconds.toFixed(0) }}s</p>
            <div v-if="queue.length" class="wan-callout-actions mt-2">
              <span class="caption">Queued: {{ queue.length }} / {{ queueMax }}</span>
              <button class="btn btn-sm btn-ghost" type="button" @click="clearQueue">Clear queue</button>
            </div>
          </div>
          <ResultViewer mode="video" :frames="framesResult" :toDataUrl="toDataUrl" emptyText="No results yet.">
            <template #empty>
              <div class="wan-results-empty">
                <div class="wan-empty-title">No results yet</div>
                <ol class="wan-empty-steps">
                  <li v-if="needsWanModels">Set WAN High/Low models in QuickSettings.</li>
                  <li>Pick a mode (Text or Image) and write your prompt.</li>
                  <li>Adjust resolution/stages as needed.</li>
                  <li>Click Generate.</li>
                </ol>
                <div class="wan-callout-actions">
                  <button v-if="needsWanModels" class="btn btn-sm btn-secondary" type="button" @click="focusWanModelsQuicksettings">Open QuickSettings</button>
                  <button class="btn btn-sm btn-primary" type="button" :disabled="isRunning || !canGenerate" @click="generate">Generate</button>
                </div>
                <p v-if="!canGenerate" class="caption mt-2">{{ blockedReason }}</p>
              </div>
            </template>
          </ResultViewer>
        </div>
      </div>

      <div class="panel" v-if="info">
        <div class="panel-header three-cols"><span>Generation Info</span><div class="header-center"></div>
          <div class="header-right">
            <div class="wan-header-actions">
              <button class="btn btn-sm btn-outline" type="button" @click="copyInfo">Copy info</button>
            </div>
          </div>
        </div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ asJson(info) }}</pre>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header three-cols"><span>Run Summary</span><div class="header-center"></div>
          <div class="header-right">
            <div class="wan-header-actions">
              <button class="btn btn-sm btn-outline" type="button" @click="copyCurrentParams">Copy params</button>
            </div>
          </div>
        </div>
        <div class="panel-body">
          <div class="wan-summary-grid">
            <div>
              <label class="label-muted">Mode</label>
              <div class="caption">{{ mode }}</div>
            </div>
            <div>
              <label class="label-muted">Resolution</label>
              <div class="caption">{{ video.width }}×{{ video.height }} px</div>
            </div>
            <div>
              <label class="label-muted">Timing</label>
              <div class="caption">{{ video.frames }} frames @ {{ video.fps }} fps (~ {{ durationLabel }}s)</div>
            </div>
            <div>
              <label class="label-muted">Format</label>
              <div class="caption">{{ wanFormat }}</div>
            </div>
            <div>
              <label class="label-muted">High</label>
              <div class="caption">{{ high.steps }} steps · CFG {{ high.cfgScale }}</div>
            </div>
            <div>
              <label class="label-muted">Low</label>
              <div class="caption">{{ low.steps }} steps · CFG {{ low.cfgScale }}</div>
            </div>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">History</div>
        <div class="panel-body">
          <div v-if="history.length" class="wan-history-list">
            <div v-for="item in history" :key="item.taskId" :class="['wan-history-item', { 'is-selected': item.taskId === selectedTaskId }]">
              <div class="wan-history-meta">
                <div class="wan-history-title">{{ formatHistoryTitle(item) }}</div>
                <div class="wan-history-sub">{{ item.summary }}</div>
                <div v-if="item.promptPreview" class="wan-history-sub">{{ item.promptPreview }}</div>
                <div v-if="item.status !== 'completed'" class="caption">Status: {{ item.status }}</div>
                <div v-if="item.errorMessage" class="caption">Error: {{ item.errorMessage }}</div>
              </div>
              <div class="wan-history-actions">
                <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning || historyLoadingTaskId === item.taskId" @click="loadHistory(item.taskId)">
                  {{ historyLoadingTaskId === item.taskId ? 'Loading…' : 'View' }}
                </button>
                <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="applyHistory(item)">Apply</button>
                <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="copyHistoryParams(item)">Copy</button>
              </div>
            </div>
          </div>
          <div v-else class="caption">No runs yet.</div>

          <details v-if="diffText" class="accordion">
            <summary>Diff vs previous run</summary>
            <div class="accordion-body">
              <pre class="text-xs break-words">{{ diffText }}</pre>
            </div>
          </details>
          <div class="wan-callout-actions mt-2">
            <button class="btn btn-sm btn-ghost" type="button" :disabled="!history.length || isRunning" @click="clearHistory">Clear history</button>
          </div>
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
import { useVideoGeneration, type VideoRunHistoryItem } from '../composables/useVideoGeneration'
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

function copyHighToLow(): void {
  setLow({
    sampler: high.value.sampler,
    scheduler: high.value.scheduler,
    steps: high.value.steps,
    cfgScale: high.value.cfgScale,
    seed: high.value.seed,
    lightning: high.value.lightning,
    loraEnabled: high.value.loraEnabled,
    loraPath: high.value.loraPath,
    loraWeight: high.value.loraWeight,
  })
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

async function onInitImageFile(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  setVideo({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function clearInit(): void { setVideo({ initImageData: '', initImageName: '' }) }

// Generation wiring (composable)
const {
  generate,
  isRunning,
  canGenerate,
  blockedReason,
  cancel,
  cancelRequested,
  progress,
  frames: framesResult,
  info,
  errorMessage,
  mode,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  loadHistory,
  clearHistory,
  queue,
  queueMax,
  enqueue,
  clearQueue,
} = useVideoGeneration(props.tabId)

const needsWanModels = computed(() => !high.value.modelDir && !low.value.modelDir)
const copyNotice = ref('')
let copyTimer: number | null = null

function toast(message: string): void {
  copyNotice.value = message
  if (copyTimer) window.clearTimeout(copyTimer)
  copyTimer = window.setTimeout(() => {
    copyNotice.value = ''
    copyTimer = null
  }, 2000)
}

function focusWanModelsQuicksettings(): void {
  const el = document.getElementById('qs-wan-high') as HTMLElement | null
  el?.focus()
}

function setInputMode(next: 'txt2vid' | 'img2vid'): void {
  if (next === 'txt2vid') {
    setVideo({ useInitImage: false, initImageData: '', initImageName: '' })
    return
  }
  setVideo({ useInitImage: true })
}

const durationLabel = computed(() => {
  const fps = Number(video.value.fps) || 0
  const frames = Number(video.value.frames) || 0
  if (fps <= 0) return '0.00'
  return (frames / fps).toFixed(2)
})

function buildCurrentSnapshot(): Record<string, unknown> {
  return {
    mode: video.value.useInitImage ? 'img2vid' : 'txt2vid',
    initImageName: video.value.initImageName || '',
    prompt: String(video.value.prompt || ''),
    negativePrompt: String(video.value.negativePrompt || ''),
    width: video.value.width,
    height: video.value.height,
    frames: video.value.frames,
    fps: video.value.fps,
    format: String(wanFormat.value || 'auto'),
    assets: {
      metadata: String(assets.value.metadata || ''),
      textEncoder: String(assets.value.textEncoder || ''),
      vae: String(assets.value.vae || ''),
    },
    high: {
      modelDir: high.value.modelDir,
      sampler: high.value.sampler,
      scheduler: high.value.scheduler,
      steps: high.value.steps,
      cfgScale: high.value.cfgScale,
      seed: high.value.seed,
      lightning: high.value.lightning,
      loraEnabled: high.value.loraEnabled,
      loraPath: high.value.loraPath,
      loraWeight: high.value.loraWeight,
    },
    low: {
      modelDir: low.value.modelDir,
      sampler: low.value.sampler,
      scheduler: low.value.scheduler,
      steps: low.value.steps,
      cfgScale: low.value.cfgScale,
      seed: low.value.seed,
      lightning: low.value.lightning,
      loraEnabled: low.value.loraEnabled,
      loraPath: low.value.loraPath,
      loraWeight: low.value.loraWeight,
    },
    output: {
      filenamePrefix: video.value.filenamePrefix,
      format: video.value.format,
      pixFmt: video.value.pixFmt,
      crf: video.value.crf,
      loopCount: video.value.loopCount,
      pingpong: video.value.pingpong,
      trimToAudio: video.value.trimToAudio,
      saveMetadata: video.value.saveMetadata,
      saveOutput: video.value.saveOutput,
    },
    interpolation: {
      enabled: video.value.rifeEnabled,
      model: video.value.rifeModel,
      times: video.value.rifeTimes,
    },
  }
}

async function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text)
    return
  }
  const textarea = document.createElement('textarea')
  textarea.value = text
  textarea.style.position = 'fixed'
  textarea.style.opacity = '0'
  document.body.appendChild(textarea)
  textarea.select()
  document.execCommand('copy')
  textarea.remove()
}

async function copyCurrentParams(): Promise<void> {
  try {
    await copyToClipboard(asJson(buildCurrentSnapshot()))
    toast('Copied current params JSON.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

async function copyInfo(): Promise<void> {
  try {
    await copyToClipboard(asJson(info.value))
    toast('Copied info JSON.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

async function copyHistoryParams(item: VideoRunHistoryItem): Promise<void> {
  try {
    await copyToClipboard(asJson(item.paramsSnapshot))
    toast('Copied history params JSON.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

async function queueNext(): Promise<void> {
  try {
    await enqueue()
    toast('Queued next run.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

function applyHistory(item: VideoRunHistoryItem): void {
  const snap = (item.paramsSnapshot || {}) as any

  const nextMode: 'txt2vid' | 'img2vid' = snap.mode === 'img2vid' ? 'img2vid' : 'txt2vid'
  if (nextMode === 'txt2vid') {
    setVideo({ useInitImage: false })
  } else {
    setVideo({ useInitImage: true })
  }

  const output = snap.output || {}
  const interpolation = snap.interpolation || {}

  setVideo({
    prompt: String(snap.prompt || ''),
    negativePrompt: String(snap.negativePrompt || ''),
    width: Number(snap.width) || video.value.width,
    height: Number(snap.height) || video.value.height,
    frames: Number(snap.frames) || video.value.frames,
    fps: Number(snap.fps) || video.value.fps,
    filenamePrefix: String(output.filenamePrefix || video.value.filenamePrefix),
    format: String(output.format || video.value.format),
    pixFmt: String(output.pixFmt || video.value.pixFmt),
    crf: Number.isFinite(output.crf) ? Number(output.crf) : video.value.crf,
    loopCount: Number.isFinite(output.loopCount) ? Number(output.loopCount) : video.value.loopCount,
    pingpong: Boolean(output.pingpong),
    trimToAudio: Boolean(output.trimToAudio),
    saveMetadata: Boolean(output.saveMetadata),
    saveOutput: Boolean(output.saveOutput),
    rifeEnabled: Boolean(interpolation.enabled),
    rifeModel: String(interpolation.model || ''),
    rifeTimes: Number.isFinite(interpolation.times) ? Number(interpolation.times) : video.value.rifeTimes,
  })

  const fmt = String(snap.format || 'auto')
  if (fmt) store.updateParams(props.tabId, { modelFormat: fmt })

  const snapAssets = snap.assets || {}
  if (snapAssets && typeof snapAssets === 'object') {
    store.updateParams(props.tabId, { assets: { ...assets.value, ...snapAssets } })
  }

  const hi = snap.high || {}
  setHigh({
    modelDir: String(hi.modelDir || ''),
    sampler: String(hi.sampler || ''),
    scheduler: String(hi.scheduler || ''),
    steps: Number(hi.steps) || high.value.steps,
    cfgScale: Number(hi.cfgScale) || high.value.cfgScale,
    seed: Number.isFinite(hi.seed) ? Number(hi.seed) : high.value.seed,
    lightning: Boolean(hi.lightning),
    loraEnabled: Boolean(hi.loraEnabled),
    loraPath: String(hi.loraPath || ''),
    loraWeight: Number.isFinite(hi.loraWeight) ? Number(hi.loraWeight) : high.value.loraWeight,
  })

  const lo = snap.low || {}
  setLow({
    modelDir: String(lo.modelDir || ''),
    sampler: String(lo.sampler || ''),
    scheduler: String(lo.scheduler || ''),
    steps: Number(lo.steps) || low.value.steps,
    cfgScale: Number(lo.cfgScale) || low.value.cfgScale,
    seed: Number.isFinite(lo.seed) ? Number(lo.seed) : low.value.seed,
    lightning: Boolean(lo.lightning),
    loraEnabled: Boolean(lo.loraEnabled),
    loraPath: String(lo.loraPath || ''),
    loraWeight: Number.isFinite(lo.loraWeight) ? Number(lo.loraWeight) : low.value.loraWeight,
  })

  toast('Applied params from history.')
}

function reuseLast(): void {
  if (!history.value.length) return
  applyHistory(history.value[0] as VideoRunHistoryItem)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function formatDiffValue(value: unknown): string {
  if (typeof value === 'string') {
    const v = value.length > 160 ? value.slice(0, 160) + '…' : value
    return JSON.stringify(v)
  }
  if (typeof value === 'number' || typeof value === 'boolean' || value === null || value === undefined) {
    return String(value)
  }
  try {
    const raw = JSON.stringify(value)
    if (raw.length > 180) return raw.slice(0, 180) + '…'
    return raw
  } catch {
    return String(value)
  }
}

function diffObjects(before: unknown, after: unknown, prefix = '', out: Array<{ path: string; before: unknown; after: unknown }> = []): Array<{ path: string; before: unknown; after: unknown }> {
  if (out.length > 80) return out
  if (before === after) return out

  const aObj = isRecord(before)
  const bObj = isRecord(after)
  if (aObj && bObj) {
    const keys = new Set([...Object.keys(before), ...Object.keys(after)])
    for (const k of keys) {
      const nextPrefix = prefix ? `${prefix}.${k}` : k
      diffObjects((before as any)[k], (after as any)[k], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  if (Array.isArray(before) && Array.isArray(after)) {
    const max = Math.max(before.length, after.length)
    for (let i = 0; i < max; i++) {
      const nextPrefix = `${prefix}[${i}]`
      diffObjects(before[i], after[i], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  out.push({ path: prefix || '(root)', before, after })
  return out
}

const selectedHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const id = String(selectedTaskId.value || '')
  if (!id) return null
  return (history.value as VideoRunHistoryItem[]).find((h) => h.taskId === id) || null
})

const previousHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const selected = selectedHistoryItem.value
  if (!selected) return null
  const idx = (history.value as VideoRunHistoryItem[]).findIndex((h) => h.taskId === selected.taskId)
  if (idx < 0) return null
  return (history.value as VideoRunHistoryItem[])[idx + 1] || null
})

const diffText = computed(() => {
  const selected = selectedHistoryItem.value
  const prev = previousHistoryItem.value
  if (!selected || !prev) return ''

  const rows = diffObjects(prev.paramsSnapshot, selected.paramsSnapshot)
  if (!rows.length) return ''

  return rows
    .map((r) => `${r.path}: ${formatDiffValue(r.before)} → ${formatDiffValue(r.after)}`)
    .join('\n')
})

type AspectMode = 'free' | 'current' | '16:9' | '1:1' | '9:16' | '4:3' | '3:4'
const aspectMode = ref<AspectMode>('free')
const aspectRatio = ref<number | null>(null)

function snapDim(value: number): number {
  const step = 8
  const min = 64
  const v = Number.isFinite(value) ? value : min
  return Math.max(min, Math.round(v / step) * step)
}

function ratioForMode(mode: AspectMode): number | null {
  if (mode === 'current') {
    const w = Number(video.value.width) || 0
    const h = Number(video.value.height) || 0
    return h > 0 ? w / h : null
  }
  if (mode === '16:9') return 16 / 9
  if (mode === '1:1') return 1
  if (mode === '9:16') return 9 / 16
  if (mode === '4:3') return 4 / 3
  if (mode === '3:4') return 3 / 4
  return null
}

function onAspectModeChange(e: Event): void {
  const mode = String((e.target as HTMLSelectElement).value || 'free') as AspectMode
  aspectMode.value = mode
  if (mode === 'free') {
    aspectRatio.value = null
    return
  }
  const ratio = ratioForMode(mode)
  aspectRatio.value = ratio
  if (!ratio) return

  // For fixed presets, snap the current size into the chosen ratio (preserve width).
  if (mode !== 'current') {
    const w = snapDim(Number(video.value.width) || 64)
    const h = snapDim(w / ratio)
    setVideo({ width: w, height: h })
  }
}

function onWidthChange(e: Event): void {
  const nextW = snapDim(toInt(e, video.value.width))
  const r = aspectRatio.value
  if (r && r > 0) {
    const nextH = snapDim(nextW / r)
    setVideo({ width: nextW, height: nextH })
    return
  }
  setVideo({ width: nextW })
}

function onHeightChange(e: Event): void {
  const nextH = snapDim(toInt(e, video.value.height))
  const r = aspectRatio.value
  if (r && r > 0) {
    const nextW = snapDim(nextH * r)
    setVideo({ width: nextW, height: nextH })
    return
  }
  setVideo({ height: nextH })
}

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

function formatHistoryTitle(item: { mode: string; createdAtMs: number; taskId: string }): string {
  const d = new Date(item.createdAtMs)
  const ts = Number.isFinite(item.createdAtMs) ? d.toLocaleString() : ''
  return `${item.mode} · ${ts} · ${item.taskId}`
}

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
