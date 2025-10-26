<template>
  <section class="panels test-view">
    <div class="panel-stack">
      <div class="panel test-view-panel">
        <div class="panel-header">
          <h3 class="h4">WAN GGUF Harness</h3>
        </div>
        <div class="panel-body">
          <div class="test-section">
            <h4 class="h5">Global Overrides</h4>
            <p class="muted test-sublabel">Provide explicit paths when you want to override the default WAN assets.</p>
            <div class="test-grid test-grid-two">
              <div class="field-stack">
                <label class="label" for="vaeDir">VAE (input list)</label>
                <input id="vaeDir" class="ui-input" list="dl-vaes" v-model="state.vaeDir" placeholder="/models/VAE/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
                <datalist id="dl-vaes">
                  <option v-for="opt in options.vaes" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
                </datalist>
              </div>
              <div class="field-stack">
                <label class="label" for="textEncoderDir">Text Encoder (input list)</label>
                <input id="textEncoderDir" class="ui-input" list="dl-te" v-model="state.textEncoderDir" placeholder="/models/text-encoder/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
                <datalist id="dl-te">
                  <option v-for="opt in options.textEncoders" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
                </datalist>
              </div>
              <div class="field-stack">
                <label class="label" for="metadataDir">Metadata Dir (input list)</label>
                <input id="metadataDir" class="ui-input" list="dl-meta" v-model="state.metadataDir" placeholder="apps/server/backend/huggingface/<org>/<repo>" autocomplete="off" autocapitalize="off" spellcheck="false" />
                <datalist id="dl-meta">
                  <option v-for="opt in options.metadata" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
                </datalist>
              </div>
            </div>
          </div>

          <div class="test-section">
            <h4 class="h5">Prompt & Dimensions</h4>
            <div class="test-grid test-grid-two">
              <div class="field-stack">
                <label class="label" for="prompt">Prompt</label>
                <textarea id="prompt" class="ui-textarea" rows="3" v-model="state.prompt"></textarea>
                <label class="label" for="negative">Negative</label>
                <textarea id="negative" class="ui-textarea" rows="2" v-model="state.negative"></textarea>
              </div>
              <div class="field-stack">
                <div class="test-grid test-grid-two">
                  <div>
                    <label class="label" for="width">Width</label>
                    <input id="width" class="ui-input" type="number" min="64" step="8" v-model.number="state.width" />
                  </div>
                  <div>
                    <label class="label" for="height">Height</label>
                    <input id="height" class="ui-input" type="number" min="64" step="8" v-model.number="state.height" />
                  </div>
                </div>
                <div class="test-grid test-grid-two">
                  <div>
                    <label class="label" for="frames">Frames</label>
                    <input id="frames" class="ui-input" type="number" min="1" v-model.number="state.frames" />
                  </div>
                  <div>
                    <label class="label" for="fps">FPS</label>
                    <input id="fps" class="ui-input" type="number" min="1" v-model.number="state.fps" />
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="test-section">
            <h4 class="h5">Input Mode</h4>
            <label class="switch-label test-init-toggle">
              <input type="checkbox" v-model="state.useInitImage" />
              <span>Use Initial Image (img2vid)</span>
            </label>
            <div v-if="state.useInitImage" class="test-init-upload">
              <label class="label" for="initImage">Initial Image</label>
              <input id="initImage" class="ui-input" type="file" accept="image/*" @change="onFile" />
              <div v-if="state.initImageName" class="test-init-meta">
                {{ state.initImageName }} — resized to {{ state.width }}×{{ state.height }}
              </div>
              <div v-if="state.initImageData" class="test-init-preview">
                <img :src="state.initImageData" alt="Initial preview" class="test-init-thumb" />
              </div>
            </div>
          </div>

          <div class="test-section">
            <h4 class="h5">High Stage</h4>
            <div class="test-grid test-grid-three">
              <div class="field-stack">
                <label class="label" for="highModel">High Model (.gguf)</label>
                <select id="highModel" class="select-md" v-model="state.high.modelDir">
                  <option value="">— Select —</option>
                  <option v-for="opt in options.wanHigh" :key="opt.path" :value="opt.path">{{ opt.name }}</option>
                </select>
              </div>
              <div>
                <label class="label" for="highSteps">Steps</label>
                <input id="highSteps" class="ui-input" type="number" min="1" v-model.number="state.high.steps" />
              </div>
              <div>
                <label class="label" for="highCfg">CFG</label>
                <input id="highCfg" class="ui-input" type="number" step="0.5" v-model.number="state.high.cfgScale" />
              </div>
            </div>
            <div class="test-grid test-grid-lora">
              <div class="field-stack">
                <label class="switch-label">
                  <input type="checkbox" v-model="state.high.useLora" />
                  <span>Use Auxiliary LoRA (High)</span>
                </label>
                <div v-if="state.high.useLora">
                  <label class="label" for="highLora">LoRA (input list)</label>
                  <input id="highLora" class="ui-input" list="dl-lora" v-model="state.high.loraPath" placeholder="/models/Lora/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
                  <datalist id="dl-lora">
                    <option v-for="opt in options.loras" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
                  </datalist>
                </div>
              </div>
              <div>
                <label class="label" for="highLoraWeight">LoRA Weight</label>
                <input id="highLoraWeight" class="ui-input" type="number" step="0.05" v-model.number="state.high.loraWeight" />
              </div>
            </div>
          </div>

          <div class="test-section">
            <h4 class="h5">Low Stage</h4>
            <div class="test-grid test-grid-three">
              <div class="field-stack">
                <label class="label" for="lowModel">Low Model (.gguf)</label>
                <select id="lowModel" class="select-md" v-model="state.low.modelDir">
                  <option value="">— Select —</option>
                  <option v-for="opt in options.wanLow" :key="opt.path" :value="opt.path">{{ opt.name }}</option>
                </select>
              </div>
              <div>
                <label class="label" for="lowSteps">Steps</label>
                <input id="lowSteps" class="ui-input" type="number" min="1" v-model.number="state.low.steps" />
              </div>
              <div>
                <label class="label" for="lowCfg">CFG</label>
                <input id="lowCfg" class="ui-input" type="number" step="0.5" v-model.number="state.low.cfgScale" />
              </div>
            </div>
            <div class="test-grid test-grid-lora">
              <div class="field-stack">
                <label class="switch-label">
                  <input type="checkbox" v-model="state.low.useLora" />
                  <span>Use Auxiliary LoRA (Low)</span>
                </label>
                <div v-if="state.low.useLora">
                  <label class="label" for="lowLora">LoRA (input list)</label>
                  <input id="lowLora" class="ui-input" list="dl-lora" v-model="state.low.loraPath" placeholder="/models/Lora/*.safetensors" />
                </div>
              </div>
              <div>
                <label class="label" for="lowLoraWeight">LoRA Weight</label>
                <input id="lowLoraWeight" class="ui-input" type="number" step="0.05" v-model.number="state.low.loraWeight" />
              </div>
            </div>
          </div>

          <div class="test-section">
            <h4 class="h5">Execution</h4>
            <div class="test-grid test-grid-three">
              <div>
                <label class="label" for="sampler">Sampler</label>
                <input id="sampler" class="ui-input" type="text" v-model="state.sampler" placeholder="Euler" />
              </div>
              <div>
                <label class="label" for="scheduler">Scheduler</label>
                <input id="scheduler" class="ui-input" type="text" v-model="state.scheduler" placeholder="Automatic" />
              </div>
              <div>
                <label class="label" for="format">Format</label>
                <select id="format" class="select-md" v-model="state.wanFormat">
                  <option value="gguf">GGUF</option>
                  <option value="diffusers">Diffusers</option>
                  <option value="auto">Auto</option>
                </select>
              </div>
          </div>
            <div class="test-grid test-grid-three">
              <div>
                <label class="label" for="seed">Seed</label>
                <input id="seed" class="ui-input" type="number" v-model.number="state.seed" placeholder="-1 (random)" />
              </div>
              <div>
                <label class="label" for="offloadLevel">Offload Level</label>
                <select id="offloadLevel" class="select-md" v-model.number="state.offloadLevel">
                  <option :value="0">0 — Off</option>
                  <option :value="1">1 — Light</option>
                  <option :value="2">2 — Balanced</option>
                  <option :value="3">3 — Aggressive</option>
                </select>
              </div>
              <div>
                <label class="label" for="teKernel">Text Encoder Kernel</label>
                <select id="teKernel" class="select-md" v-model="state.teKernel">
                  <option value="auto">Auto</option>
                  <option value="hf-cpu">HF on CPU</option>
                  <option value="hf-cuda">HF on CUDA</option>
                  <option value="cuda-fp8">CUDA FP8 (experimental)</option>
                </select>
                <div v-if="state.teKernel === 'cuda-fp8'" class="mt-2">
                  <label class="switch-label">
                    <input type="checkbox" v-model="state.teKernelRequire" />
                    <span>Require TE CUDA kernel</span>
                  </label>
                </div>
              </div>
            </div>
          </div>

          <div class="toolbar test-toolbar">
            <button class="btn btn-primary" type="button" :disabled="isRunning" @click="generate">{{ isRunning ? 'Running…' : 'Generate' }}</button>
            <span class="muted test-status" v-if="errorMessage">{{ errorMessage }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <div class="panel test-results-panel">
        <div class="panel-header"><span>Results</span></div>
        <div class="panel-body">
          <ResultViewer mode="video" :frames="framesResult" :toDataUrl="toDataUrl" emptyText="No results yet." />
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { reactive, ref } from 'vue'
import { startImg2Vid, startTxt2Vid, subscribeTask, fetchModelInventory } from '../api/client'
import ResultViewer from '../components/ResultViewer.vue'
import type { GeneratedImage, TaskEvent } from '../api/types'

const state = reactive({
  prompt: '',
  negative: '',
  width: 512,
  height: 512,
  frames: 90,
  fps: 15,
  useInitImage: true,
  initImageData: '',
  initImageName: '',
  vaeDir: '',
  textEncoderDir: '',
  tokenizerDir: '',
  metadataDir: '',
  high: {
    modelDir: '',
    steps: 2, cfgScale: 3,
    loraPath: '', loraWeight: 1.0,
  },
  low: {
    modelDir: '',
    steps: 2, cfgScale: 3,
    loraPath: '', loraWeight: 1.0,
  },
  sampler: 'Euler', scheduler: 'Simple', wanFormat: 'gguf',
  offloadLevel: 3,
  teKernel: 'auto',
  teKernelRequire: true,
  seed: -1,
})

const isRunning = ref(false)
const errorMessage = ref('')
const framesResult = ref<GeneratedImage[]>([])
let unsubscribe: (() => void) | null = null

const options = reactive({
  vaes: [] as Array<{ name: string; path: string }>,
  textEncoders: [] as Array<{ name: string; path: string }>,
  metadata: [] as Array<{ name: string; path: string }>,
  loras: [] as Array<{ name: string; path: string }>,
  wanHigh: [] as Array<{ name: string; path: string }>,
  wanLow: [] as Array<{ name: string; path: string }>,
})

const maps = reactive({
  vae: {} as Record<string, string>,
  te: {} as Record<string, string>,
  meta: {} as Record<string, string>,
  lora: {} as Record<string, string>,
  wanHigh: {} as Record<string, string>,
  wanLow: {} as Record<string, string>,
})

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function stopStream(): void { if (unsubscribe) { unsubscribe(); unsubscribe = null } }

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

// Load inventory for selects on mount (immediately invoked)
(async () => {
  try {
    const inv = await fetchModelInventory()
    options.vaes = (inv.vaes || []).map((v: any) => ({ name: v.name, path: v.path }))
    options.textEncoders = (inv.text_encoders || []).map((t: any) => ({ name: t.name, path: t.path }))
    options.metadata = (inv.metadata || []).map((m: any) => ({ name: m.name, path: m.path }))
    options.loras = (inv.loras || []).map((l: any) => ({ name: l.name, path: l.path }))
    const gguf = inv.wan22?.gguf || []
    // List all .gguf files under models/wan22 without stage filtering
    options.wanHigh = gguf.map((e: any) => ({ name: e.name, path: e.path }))
    options.wanLow = gguf.map((e: any) => ({ name: e.name, path: e.path }))
    // Build name->path maps for resolution at submit
    maps.vae = Object.fromEntries(options.vaes.map((x) => [x.name, x.path]))
    maps.te = Object.fromEntries(options.textEncoders.map((x) => [x.name, x.path]))
    maps.meta = Object.fromEntries(options.metadata.map((x) => [x.name, x.path]))
    maps.lora = Object.fromEntries(options.loras.map((x) => [x.name, x.path]))
    maps.wanHigh = Object.fromEntries(options.wanHigh.map((x) => [x.name, x.path]))
    maps.wanLow = Object.fromEntries(options.wanLow.map((x) => [x.name, x.path]))
  } catch (err) {
    console.error('[inventory] failed to load', err)
  }
})()

async function onFile(e: Event): Promise<void> {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  const raw = await readFileAsDataURL(file)
  // Resize to match pipeline size (state.width/state.height) before sending
  try {
    state.initImageData = await resizeDataUrl(raw, state.width, state.height)
  } catch {
    // fallback to raw if browser resize fails
    state.initImageData = raw
  }
  state.initImageName = file.name
}

async function generate(): Promise<void> {
  stopStream()
  isRunning.value = true
  errorMessage.value = ''
  framesResult.value = []
  try {
    const resolve = (val: string, map: Record<string, string>) => (map && map[val]) ? map[val] : val
    const highModel = resolve(state.high.modelDir, maps.wanHigh)
    const lowModel = resolve(state.low.modelDir, maps.wanLow)
    const vaePath = resolve(state.vaeDir, maps.vae)
    const tePath = resolve(state.textEncoderDir, maps.te)
    const metaDir = resolve(state.metadataDir, maps.meta)

    const extras: Record<string, unknown> = {
      wan_high: { sampler: state.sampler, scheduler: state.scheduler, steps: state.high.steps, cfg_scale: state.high.cfgScale, model_dir: highModel || undefined },
      wan_low: { sampler: state.sampler, scheduler: state.scheduler, steps: state.low.steps, cfg_scale: state.low.cfgScale, model_dir: lowModel || undefined },
      wan_format: state.wanFormat,
      wan_vae_path: vaePath || undefined,
      wan_text_encoder_path: tePath || undefined,
      wan_metadata_dir: metaDir || undefined,
    }
    // Offload/kernel controls
    ;(extras as any).gguf_offload_level = state.offloadLevel
    if (state.teKernel === 'hf-cpu') {
      ;(extras as any).gguf_te_device = 'cpu'
    } else if (state.teKernel === 'hf-cuda') {
      ;(extras as any).gguf_te_device = 'cuda'
    } else if (state.teKernel === 'cuda-fp8') {
      ;(extras as any).gguf_te_impl = 'cuda_fp8'
      ;(extras as any).gguf_te_kernel_required = !!state.teKernelRequire
      ;(extras as any).gguf_te_device = 'cuda'
    }
    if (state.high.useLora && state.high.loraPath) { (extras.wan_high as any).lora_path = resolve(state.high.loraPath, maps.lora); (extras.wan_high as any).lora_weight = state.high.loraWeight }
    if (state.low.useLora && state.low.loraPath) { (extras.wan_low as any).lora_path = resolve(state.low.loraPath, maps.lora); (extras.wan_low as any).lora_weight = state.low.loraWeight }
    if (state.useInitImage && state.initImageData) {
      const payload = {
        __strict_version: 1,
        img2vid_prompt: state.prompt,
        img2vid_neg_prompt: state.negative,
        img2vid_width: state.width,
        img2vid_height: state.height,
        img2vid_num_frames: state.frames,
        img2vid_fps: state.fps,
        img2vid_seed: state.seed,
        img2vid_init_image: state.initImageData,
        ...extras,
      }
      const { task_id } = await startImg2Vid(payload)
      unsubscribe = subscribeTask(task_id, onTaskEvent)
    } else {
      const payload = {
        __strict_version: 1,
        txt2vid_prompt: state.prompt,
        txt2vid_neg_prompt: state.negative,
        txt2vid_width: state.width,
        txt2vid_height: state.height,
        txt2vid_num_frames: state.frames,
        txt2vid_fps: state.fps,
        txt2vid_seed: state.seed,
        ...extras,
      }
      const { task_id } = await startTxt2Vid(payload)
      unsubscribe = subscribeTask(task_id, onTaskEvent)
    }
  } catch (e) {
    isRunning.value = false
    errorMessage.value = e instanceof Error ? e.message : String(e)
  }
}

function resizeDataUrl(src: string, width: number, height: number): Promise<string> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      try {
        const canvas = document.createElement('canvas')
        canvas.width = Math.max(1, Math.floor(width))
        canvas.height = Math.max(1, Math.floor(height))
        const ctx = canvas.getContext('2d')
        if (!ctx) { reject(new Error('no 2d context')); return }
        ctx.imageSmoothingEnabled = true
        ctx.imageSmoothingQuality = 'high'
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height)
        const out = canvas.toDataURL('image/png')
        resolve(out)
      } catch (err) {
        reject(err as Error)
      }
    }
    img.onerror = () => reject(new Error('image load failed'))
    img.src = src
  })
}

function onTaskEvent(ev: TaskEvent): void {
  switch (ev.type) {
    case 'result':
      framesResult.value = ev.images
      isRunning.value = false
      stopStream()
      break
    case 'error':
      errorMessage.value = ev.message
      isRunning.value = false
      stopStream()
      break
    default:
      break
  }
}

</script>
