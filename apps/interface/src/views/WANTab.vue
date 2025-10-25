<template>
  <section v-if="tab" class="panels">
    <div class="panel-stack" ref="leftStack">
    <!-- Prompt & Input Panel -->
    <div class="panel">
      <div class="panel-header">
        <h3 class="h4">Prompt & Input</h3>
      </div>
      <div class="panel-body">
        <div class="grid grid-2">
          <div>
            <label class="label">Prompt</label>
            <textarea class="ui-textarea" rows="3" :value="video.prompt" @input="setVideo({ prompt: ($event.target as HTMLTextAreaElement).value })"></textarea>
            <label class="label" style="margin-top:.5rem">Negative</label>
            <textarea class="ui-textarea" rows="2" :value="video.negativePrompt" @input="setVideo({ negativePrompt: ($event.target as HTMLTextAreaElement).value })"></textarea>
          </div>
          <div>
            <div class="grid grid-2">
              <div>
                <label class="label">Width</label>
                <input class="ui-input" type="number" min="64" step="8" :value="video.width" @change="setVideo({ width: toInt($event, video.width) })" />
              </div>
              <div>
                <label class="label">Height</label>
                <input class="ui-input" type="number" min="64" step="8" :value="video.height" @change="setVideo({ height: toInt($event, video.height) })" />
              </div>
            </div>
            <VideoSettingsCard
              :frames="video.frames"
              :fps="video.fps"
              @update:frames="(v:number)=>setVideo({ frames: v })"
              @update:fps="(v:number)=>setVideo({ fps: v })"
            />
            <div class="muted" style="margin-top:.25rem" v-if="durationSec > 0">~ {{ durationSec.toFixed(2) }}s</div>
          </div>
        </div>

        <div class="panel-sub" style="margin-top: .75rem">
          <label class="switch-label">
            <input type="checkbox" :checked="video.useInitImage" @change="onInitToggle" />
            <span>Use Initial Image (img2vid)</span>
          </label>
          <div v-if="video.useInitImage" class="grid grid-2" style="margin-top:.5rem">
            <div>
              <label class="label">Image</label>
              <input class="ui-input" type="file" accept="image/*" @change="onFile" />
              <div v-if="video.initImageName" class="muted" style="margin-top:.25rem">{{ video.initImageName }}</div>
              <button v-if="video.initImageData" class="btn btn-sm" type="button" style="margin-top:.5rem" @click="clearInit">Clear</button>
            </div>
            <div v-if="video.initImageData">
              <label class="label">Preview</label>
              <img :src="video.initImageData" alt="init" style="max-width:100%; border-radius:.25rem;" />
            </div>
          </div>
        </div>

        <div class="panel-sub" style="margin-top:.75rem">
          <h4 class="h6" style="margin:0 0 .25rem 0">WAN Assets</h4>
          <div class="grid grid-3">
            <div>
              <label class="label" for="wanMeta">Metadata Dir (input list)</label>
              <input id="wanMeta" class="ui-input" list="dl-meta-wan" v-model="assets.metadata" placeholder="apps/server/backend/huggingface/<org>/<repo>" autocomplete="off" autocapitalize="off" spellcheck="false" />
              <datalist id="dl-meta-wan">
                <option v-for="m in inv.metadata" :key="m.path" :value="m.name">{{ m.name }}</option>
              </datalist>
            </div>
            <div>
              <label class="label" for="wanTE">Text Encoder (input list)</label>
              <input id="wanTE" class="ui-input" list="dl-te-wan" v-model="assets.textEncoder" placeholder="models/text-encoder/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
              <datalist id="dl-te-wan">
                <option v-for="t in inv.textEncoders" :key="t.path" :value="t.name">{{ t.name }}</option>
              </datalist>
            </div>
            <div>
              <label class="label" for="wanVAE">VAE (input list)</label>
              <input id="wanVAE" class="ui-input" list="dl-vaes-wan" v-model="assets.vae" placeholder="models/VAE/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
              <datalist id="dl-vaes-wan">
                <option v-for="v in inv.vaes" :key="v.path" :value="v.name">{{ v.name }}</option>
              </datalist>
            </div>
          </div>
          <p class="muted" style="margin-top:.25rem">Select filenames; the app resolves to full paths at submit.</p>
        </div>
      </div>
    </div>

    <!-- High Noise Panel (subset; full set in F2 completion) -->
    <div class="panel">
      <div class="panel-header"><h3 class="h4">High Noise</h3></div>
      <div class="panel-body">
        <div class="grid grid-3">
          <div>
            <label class="label">Sampler</label>
            <select class="select-md" :value="high.sampler" @change="setHigh({ sampler: ($event.target as HTMLSelectElement).value })">
              <option v-for="s in samplers" :key="s.name" :value="s.name">{{ s.name }}</option>
            </select>
          </div>
          <div>
            <label class="label">Scheduler</label>
            <select class="select-md" :value="high.scheduler" @change="setHigh({ scheduler: ($event.target as HTMLSelectElement).value })">
              <option v-for="s in schedulers" :key="s.name" :value="s.name">{{ s.name }}</option>
            </select>
          </div>
          <div>
            <label class="label">Steps</label>
            <input class="ui-input" type="number" min="1" :value="high.steps" @change="setHigh({ steps: toInt($event, high.steps) })" />
          </div>
        </div>
        <div class="grid grid-3" style="margin-top:.5rem">
          <div>
            <label class="label">CFG</label>
            <input class="ui-input" type="number" step="0.5" :value="high.cfgScale" @change="setHigh({ cfgScale: toFloat($event, high.cfgScale) })" />
          </div>
          <div>
            <label class="label">Seed</label>
            <div class="grid grid-3">
              <input class="ui-input" type="number" :value="high.seed" @change="setHigh({ seed: toInt($event, high.seed) })" />
              <button class="btn btn-sm" type="button" @click="randomizeSeedHigh">Random</button>
              <button class="btn btn-sm" type="button" @click="reuseSeedHigh" :disabled="lastSeedHi === null">Reuse</button>
            </div>
          </div>
          <div>
            <label class="label">Model Dir</label>
            <input class="ui-input" type="text" :value="high.modelDir" @change="setHigh({ modelDir: ($event.target as HTMLInputElement).value })" placeholder="/path/to/high" />
          </div>
        </div>
        <div class="grid grid-3" style="margin-top:.5rem">
          <div>
            <label class="label">Model Format</label>
            <select class="select-md" :value="wanFormat" @change="onFormatChange">
              <option value="auto">Auto</option>
              <option value="diffusers">Diffusers</option>
              <option value="gguf">GGUF</option>
            </select>
          </div>
        </div>
        <div class="panel-sub" style="margin-top:.5rem">
          <label class="switch-label">
            <input type="checkbox" :checked="high.lightning" @change="onLightningHigh($event)" />
            <span>Lightning</span>
          </label>
          <label class="switch-label" style="margin-left:1rem">
            <input type="checkbox" :checked="high.loraEnabled" @change="setHigh({ loraEnabled: ($event.target as HTMLInputElement).checked })" />
            <span>Use LoRA</span>
          </label>
          <div v-if="high.loraEnabled" class="grid grid-2" style="margin-top:.5rem">
            <div>
              <label class="label">LoRA Path</label>
              <input class="ui-input" type="text" :value="high.loraPath" @change="setHigh({ loraPath: ($event.target as HTMLInputElement).value })" />
            </div>
            <div>
              <label class="label">Weight</label>
              <input class="ui-input" type="number" step="0.05" :value="high.loraWeight" @change="setHigh({ loraWeight: toFloat($event, high.loraWeight) })" />
            </div>
          </div>
        </div>
        <div class="error" v-if="!high.modelDir" style="margin-top:.5rem">High: model directory is empty.</div>
        <div class="error" v-if="high.loraEnabled && !high.loraPath" style="margin-top:.25rem">High: LoRA enabled but path is empty.</div>
      </div>
    </div>

    <!-- Low Noise (same subset) -->
    <div class="panel">
      <div class="panel-header"><h3 class="h4">Low Noise</h3></div>
      <div class="panel-body">
        <div class="grid grid-3">
          <div>
            <label class="label">Sampler</label>
            <select class="select-md" :value="low.sampler" @change="setLow({ sampler: ($event.target as HTMLSelectElement).value })">
              <option v-for="s in samplers" :key="s.name" :value="s.name">{{ s.name }}</option>
            </select>
          </div>
          <div>
            <label class="label">Scheduler</label>
            <select class="select-md" :value="low.scheduler" @change="setLow({ scheduler: ($event.target as HTMLSelectElement).value })">
              <option v-for="s in schedulers" :key="s.name" :value="s.name">{{ s.name }}</option>
            </select>
          </div>
          <div>
            <label class="label">Steps</label>
            <input class="ui-input" type="number" min="1" :value="low.steps" @change="setLow({ steps: toInt($event, low.steps) })" />
          </div>
        </div>
        <div class="grid grid-3" style="margin-top:.5rem">
          <div>
            <label class="label">CFG</label>
            <input class="ui-input" type="number" step="0.5" :value="low.cfgScale" @change="setLow({ cfgScale: toFloat($event, low.cfgScale) })" />
          </div>
          <div>
            <label class="label">Seed</label>
            <div class="grid grid-3">
              <input class="ui-input" type="number" :value="low.seed" @change="setLow({ seed: toInt($event, low.seed) })" />
              <button class="btn btn-sm" type="button" @click="randomizeSeedLow">Random</button>
              <button class="btn btn-sm" type="button" @click="reuseSeedLow" :disabled="lastSeedLow === null">Reuse</button>
            </div>
          </div>
          <div>
            <label class="label">Model Dir</label>
            <input class="ui-input" type="text" :value="low.modelDir" @change="setLow({ modelDir: ($event.target as HTMLInputElement).value })" placeholder="/path/to/low" />
          </div>
        </div>
        <div class="panel-sub" style="margin-top:.5rem">
          <label class="switch-label">
            <input type="checkbox" :checked="low.lightning" @change="onLightningLow($event)" />
            <span>Lightning</span>
          </label>
          <label class="switch-label" style="margin-left:1rem">
            <input type="checkbox" :checked="low.loraEnabled" @change="setLow({ loraEnabled: ($event.target as HTMLInputElement).checked })" />
            <span>Use LoRA</span>
          </label>
          <div v-if="low.loraEnabled" class="grid grid-2" style="margin-top:.5rem">
            <div>
              <label class="label">LoRA Path</label>
              <input class="ui-input" type="text" :value="low.loraPath" @change="setLow({ loraPath: ($event.target as HTMLInputElement).value })" />
            </div>
            <div>
              <label class="label">Weight</label>
              <input class="ui-input" type="number" step="0.05" :value="low.loraWeight" @change="setLow({ loraWeight: toFloat($event, low.loraWeight) })" />
            </div>
          </div>
        </div>
        <div class="error" v-if="!low.modelDir" style="margin-top:.5rem">Low: model directory is empty.</div>
        <div class="error" v-if="low.loraEnabled && !low.loraPath" style="margin-top:.25rem">Low: LoRA enabled but path is empty.</div>
      </div>
    </div>

    <!-- Video Export (subset) -->
    <div class="panel">
      <div class="panel-header"><h3 class="h4">Video Output</h3></div>
      <div class="panel-body">
        <div class="grid grid-3">
          <div>
            <label class="label">Filename Prefix</label>
            <input class="ui-input" type="text" :value="video.filenamePrefix" @change="setVideo({ filenamePrefix: ($event.target as HTMLInputElement).value })" />
          </div>
          <div>
            <label class="label">Format</label>
            <select class="select-md" :value="video.format" @change="setVideo({ format: ($event.target as HTMLSelectElement).value })">
              <option value="video/h264-mp4">H.264 MP4</option>
              <option value="video/h265-mp4">H.265 MP4</option>
              <option value="video/webm">WebM</option>
              <option value="image/gif">GIF</option>
            </select>
          </div>
          <div>
            <label class="label">CRF</label>
            <input class="ui-input" type="number" min="0" max="51" :value="video.crf" @change="setVideo({ crf: toInt($event, video.crf) })" />
          </div>
        </div>
        <div class="grid grid-3" style="margin-top:.5rem">
          <div>
            <label class="label">Pixel Format</label>
            <select class="select-md" :value="video.pixFmt" @change="setVideo({ pixFmt: ($event.target as HTMLSelectElement).value })">
              <option value="yuv420p">yuv420p</option>
              <option value="yuv444p">yuv444p</option>
              <option value="yuv422p">yuv422p</option>
            </select>
          </div>
          <div>
            <label class="label">Loop Count</label>
            <input class="ui-input" type="number" min="0" :value="video.loopCount" @change="setVideo({ loopCount: toInt($event, video.loopCount) })" />
          </div>
          <div class="grid grid-2">
            <label class="switch-label" style="margin-top:1.75rem">
              <input type="checkbox" :checked="video.pingpong" @change="setVideo({ pingpong: ($event.target as HTMLInputElement).checked })" />
              <span>Ping-pong</span>
            </label>
            <label class="switch-label" style="margin-top:1.75rem">
              <input type="checkbox" :checked="video.saveOutput" @change="setVideo({ saveOutput: ($event.target as HTMLInputElement).checked })" />
              <span>Save output</span>
            </label>
          </div>
          <div class="grid grid-2" style="margin-top:.5rem">
            <label class="switch-label">
              <input type="checkbox" :checked="video.saveMetadata" @change="setVideo({ saveMetadata: ($event.target as HTMLInputElement).checked })" />
              <span>Save metadata</span>
            </label>
            <label class="switch-label">
              <input type="checkbox" :checked="video.trimToAudio" @change="setVideo({ trimToAudio: ($event.target as HTMLInputElement).checked })" />
              <span>Trim to audio</span>
            </label>
          </div>
          <div class="panel-sub" style="margin-top:.75rem">
            <label class="switch-label">
              <input type="checkbox" :checked="video.rifeEnabled" @change="setVideo({ rifeEnabled: ($event.target as HTMLInputElement).checked })" />
              <span>Enable Interpolation (RIFE)</span>
            </label>
            <div v-if="video.rifeEnabled" class="grid grid-3" style="margin-top:.5rem">
              <div>
                <label class="label">Model</label>
                <input class="ui-input" type="text" :value="video.rifeModel" @change="setVideo({ rifeModel: ($event.target as HTMLInputElement).value })" />
              </div>
              <div>
                <label class="label">Times</label>
                <input class="ui-input" type="number" min="1" :value="video.rifeTimes" @change="setVideo({ rifeTimes: toInt($event, video.rifeTimes) })" />
              </div>
            </div>
          </div>
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
          <div v-if="errorMessage" class="error">{{ errorMessage }}</div>
          <ResultViewer mode="video" :frames="framesResult" :toDataUrl="toDataUrl" emptyText="No results yet." />
        </div>
      </div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, reactive } from 'vue'
import { useModelTabsStore, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import type { SamplerInfo, SchedulerInfo, GeneratedImage, TaskEvent } from '../api/types'
import { fetchSamplers, fetchSchedulers, startTxt2Vid, startImg2Vid, subscribeTask, fetchModelInventory } from '../api/client'
import ResultViewer from '../components/ResultViewer.vue'
import VideoSettingsCard from '../components/VideoSettingsCard.vue'

const props = defineProps<{ tabId: string }>()
const store = useModelTabsStore()

// Load option lists
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(async () => {
  if (!store.tabs.length) store.load()
  const [samp, sched, inv] = await Promise.all([fetchSamplers(), fetchSchedulers(), fetchModelInventory()])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers
  // Load inventory for WAN assets
  invState.inv = {
    vaes: (inv.vaes || []).map((v:any)=>({ name:v.name, path:v.path })),
    textEncoders: (inv.text_encoders || []).map((t:any)=>({ name:t.name, path:t.path })),
    metadata: (inv.metadata || []).map((m:any)=>({ name:m.name, path:m.path })),
  }
  invState.maps.vae = Object.fromEntries(invState.inv.vaes.map((x)=>[x.name,x.path]))
  invState.maps.te = Object.fromEntries(invState.inv.textEncoders.map((x)=>[x.name,x.path]))
  invState.maps.meta = Object.fromEntries(invState.inv.metadata.map((x)=>[x.name,x.path]))
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

// WAN assets (local state)
const invState = reactive({
  inv: { vaes: [] as Array<{name:string;path:string}>, textEncoders: [] as Array<{name:string;path:string}>, metadata: [] as Array<{name:string;path:string}> },
  maps: { vae: {} as Record<string,string>, te: {} as Record<string,string>, meta: {} as Record<string,string> },
})
const inv = invState.inv
const maps = invState.maps
const assets = reactive({ metadata: '', textEncoder: '', vae: '' })

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

function onFormatChange(e: Event): void {
  if (!tab.value) return
  store.updateParams(props.tabId, { modelFormat: (e.target as HTMLSelectElement).value })
}

function toInt(e: Event, fallback: number): number { const v = Number((e.target as HTMLInputElement).value); return Number.isFinite(v) ? Math.trunc(v) : fallback }
function toFloat(e: Event, fallback: number): number { const v = Number((e.target as HTMLInputElement).value); return Number.isFinite(v) ? v : fallback }

// Duration helper
const durationSec = computed(() => {
  const f = Number(video.value?.frames || 0)
  const r = Number(video.value?.fps || 0)
  return r > 0 ? f / r : 0
})

function onInitToggle(e: Event): void {
  setVideo({ useInitImage: (e.target as HTMLInputElement).checked })
  if (!(e.target as HTMLInputElement).checked) setVideo({ initImageData: '', initImageName: '' })
}

async function onFile(e: Event): Promise<void> {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file) return
  const dataUrl = await readFileAsDataURL(file)
  setVideo({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function clearInit(): void { setVideo({ initImageData: '', initImageName: '' }) }

// Generation wiring (local state)
type Status = 'idle' | 'running' | 'error' | 'done'
const status = ref<Status>('idle')
const errorMessage = ref('')
const framesResult = ref<GeneratedImage[]>([])
let unsubscribe: (() => void) | null = null

// Local last seeds for reuse
const lastSeedHi = ref<number | null>(null)
const lastSeedLow = ref<number | null>(null)

function stopStream(): void { if (unsubscribe) { unsubscribe(); unsubscribe = null } }
const isRunning = computed(() => status.value === 'running')

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function randomizeSeedHigh(): void {
  if (high.value.seed !== -1) lastSeedHi.value = high.value.seed
  setHigh({ seed: -1 })
}
function reuseSeedHigh(): void {
  if (lastSeedHi.value !== null) setHigh({ seed: lastSeedHi.value })
}
function randomizeSeedLow(): void {
  if (low.value.seed !== -1) lastSeedLow.value = low.value.seed
  setLow({ seed: -1 })
}
function reuseSeedLow(): void {
  if (lastSeedLow.value !== null) setLow({ seed: lastSeedLow.value })
}

async function generate(): Promise<void> {
  if (!tab.value) return
  stopStream()
  status.value = 'running'
  errorMessage.value = ''
  framesResult.value = []

  const v = video.value
  const hi = high.value
  const lo = low.value
  // Resolve asset names to absolute paths
  const resolve = (val: string, map: Record<string,string>) => (map && map[val]) ? map[val] : val
  const metaDir = resolve(assets.metadata, maps.meta)
  const tePath = resolve(assets.textEncoder, maps.te)
  const vaePath = resolve(assets.vae, maps.vae)
  const extras = {
    video_filename_prefix: v.filenamePrefix,
    video_format: v.format,
    video_pix_fmt: v.pixFmt,
    video_crf: v.crf,
    video_loop_count: v.loopCount,
    video_pingpong: v.pingpong,
    video_save_metadata: v.saveMetadata,
    video_save_output: v.saveOutput,
    video_trim_to_audio: v.trimToAudio,
    wan_high: {
      sampler: hi.sampler, scheduler: hi.scheduler, steps: hi.steps, cfg_scale: hi.cfgScale, seed: hi.seed,
      model_dir: hi.modelDir || undefined,
      lora_path: hi.loraEnabled ? hi.loraPath || undefined : undefined,
      lora_weight: hi.loraEnabled ? hi.loraWeight : undefined,
    },
    wan_low: {
      sampler: lo.sampler, scheduler: lo.scheduler, steps: lo.steps, cfg_scale: lo.cfgScale, seed: lo.seed,
      model_dir: lo.modelDir || undefined,
      lora_path: lo.loraEnabled ? lo.loraPath || undefined : undefined,
      lora_weight: lo.loraEnabled ? lo.loraWeight : undefined,
    },
    wan_format: wanFormat.value,
    wan_metadata_dir: metaDir || undefined,
    wan_text_encoder_path: tePath || undefined,
    wan_vae_path: vaePath || undefined,
  } as Record<string, unknown>

  try {
    if (v.useInitImage && v.initImageData) {
      const payload = {
        __strict_version: 1,
        img2vid_prompt: v.prompt,
        img2vid_neg_prompt: v.negativePrompt,
        img2vid_width: v.width,
        img2vid_height: v.height,
        img2vid_num_frames: v.frames,
        img2vid_fps: v.fps,
        img2vid_init_image: v.initImageData,
        ...extras,
        video_interpolation: { enabled: v.rifeEnabled, model: v.rifeModel, times: v.rifeTimes },
      }
      const { task_id } = await startImg2Vid(payload)
      unsubscribe = subscribeTask(task_id, onTaskEvent)
    } else {
      const payload = {
        __strict_version: 1,
        txt2vid_prompt: v.prompt,
        txt2vid_neg_prompt: v.negativePrompt,
        txt2vid_width: v.width,
        txt2vid_height: v.height,
        txt2vid_num_frames: v.frames,
        txt2vid_fps: v.fps,
        ...extras,
        video_interpolation: { enabled: v.rifeEnabled, model: v.rifeModel, times: v.rifeTimes },
      }
      const { task_id } = await startTxt2Vid(payload)
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
      framesResult.value = event.images
      status.value = 'done'
      stopStream()
      break
    case 'error':
      status.value = 'error'
      errorMessage.value = event.message
      stopStream()
      break
    default:
      break
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

function onLightningHigh(e: Event): void {
  const enabled = (e.target as HTMLInputElement).checked
  // Do not force steps; only toggle lightning flag
  setHigh({ lightning: enabled })
}

function onLightningLow(e: Event): void {
  const enabled = (e.target as HTMLInputElement).checked
  // Do not force steps; only toggle lightning flag
  setLow({ lightning: enabled })
}

defineExpose({ generate })
</script>
