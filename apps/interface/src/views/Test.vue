<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Internal/test view for WAN GGUF harness (manual request builder + SSE streaming preview).
Used for debugging WAN video generation, asset selection, and streaming task events without going through the full Model Tabs UX.

Symbols (top-level; keep in sync; no ghosts):
- `Test` (component): WAN GGUF harness UI; builds payloads, starts tasks, subscribes to SSE events, and renders previews/results.
- `toDataUrl` (function): Converts a `GeneratedImage` payload into a data URL for rendering.
- `stopStream` (function): Unsubscribes from the active task SSE stream (if any).
- `readFileAsDataURL` (function): Reads a `File` into a data URL (used for init image upload).
- `onFile` (function): Handles file input change events and stores image data (async).
- `generate` (function): Submits the current request payload and starts streaming task events (async).
- `resizeDataUrl` (function): Resizes a data URL image to the given dimensions (used for previews).
- `onTaskEvent` (function): Handles task SSE events (status/progress/result/error/end) and updates UI state.
- `saveProfile` (function): Persists current UI settings/profile locally for reuse.
-->

<template>
  <section class="panels test-view">
    <div class="panel-stack">
      <div class="panel test-view-panel">
        <div class="panel-header">WAN GGUF Harness</div>
        <div class="panel-body">
          <div class="test-section">
            <h4 class="h5">Global Overrides</h4>
            <p class="muted test-sublabel">Provide explicit sha256 values (64-hex) for WAN assets.</p>
            <div class="test-grid test-grid-two">
              <div class="field-stack">
                <label class="label" for="vaeSha">VAE sha256 (input list)</label>
                <input id="vaeSha" class="ui-input" list="dl-vaes" v-model="state.vaeSha" placeholder="64-hex sha256" autocomplete="off" autocapitalize="off" spellcheck="false" />
                <datalist id="dl-vaes">
                  <option v-for="opt in options.vaes" :key="opt.sha256" :value="opt.name">{{ opt.name }}</option>
                </datalist>
              </div>
              <div class="field-stack">
                <label class="label" for="textEncoderSha">Text Encoder sha256 (input list)</label>
                <input id="textEncoderSha" class="ui-input" list="dl-te" v-model="state.textEncoderSha" placeholder="64-hex sha256" autocomplete="off" autocapitalize="off" spellcheck="false" />
                <datalist id="dl-te">
                  <option v-for="opt in options.textEncoders" :key="opt.sha256" :value="opt.name">{{ opt.name }}</option>
                </datalist>
              </div>
              <div class="field-stack">
                <label class="label" for="metadataDir">Metadata Dir (input list)</label>
                <input id="metadataDir" class="ui-input" list="dl-meta" v-model="state.metadataDir" placeholder="apps/backend/huggingface/<org>/<repo>" autocomplete="off" autocapitalize="off" spellcheck="false" />
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
                <select id="highModel" class="select-md" v-model="state.high.modelSha">
                  <option value="">— Select —</option>
                  <option v-for="opt in options.wanHigh" :key="opt.sha256" :value="opt.sha256">{{ opt.name }}</option>
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
	                  <input id="highLora" class="ui-input" list="dl-lora" v-model="state.high.loraSha" placeholder="64-hex sha256" autocomplete="off" autocapitalize="off" spellcheck="false" />
	                  <datalist id="dl-lora">
	                    <option v-for="opt in options.loras" :key="opt.sha256" :value="opt.name">{{ opt.name }}</option>
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
                <select id="lowModel" class="select-md" v-model="state.low.modelSha">
                  <option value="">— Select —</option>
                  <option v-for="opt in options.wanLow" :key="opt.sha256" :value="opt.sha256">{{ opt.name }}</option>
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
	                  <input id="lowLora" class="ui-input" list="dl-lora" v-model="state.low.loraSha" placeholder="64-hex sha256" />
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
                <input id="sampler" class="ui-input" type="text" v-model="state.sampler" placeholder="euler" />
              </div>
              <div>
                <label class="label" for="scheduler">Scheduler</label>
                <input id="scheduler" class="ui-input" type="text" v-model="state.scheduler" placeholder="simple" />
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
              <!-- Offload level and Text Encoder kernel controls removed; backend reads env vars only -->
            </div>
          </div>

          <div class="toolbar test-toolbar">
            <button class="btn btn-primary" type="button" :disabled="isRunning" @click="generate">{{ isRunning ? 'Running…' : 'Generate' }}</button>
            <button class="btn" type="button" :disabled="isRunning" @click="saveProfile">Save Profile</button>
            <span class="muted test-status" v-if="errorMessage">{{ errorMessage }}</span>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <div class="panel test-results-panel">
        <div class="panel-header">Results</div>
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
  vaeSha: '',
  textEncoderSha: '',
  tokenizerDir: '',
  metadataDir: '',
  high: {
    modelSha: '',
    steps: 2, cfgScale: 3,
    useLora: false,
    loraSha: '', loraWeight: 1.0,
  },
  low: {
    modelSha: '',
    steps: 2, cfgScale: 3,
    useLora: false,
    loraSha: '', loraWeight: 1.0,
  },
  sampler: 'euler', scheduler: 'simple', wanFormat: 'gguf',
  // Offload level and TE kernel are controlled via env vars only
  seed: -1,
})

const isRunning = ref(false)
const errorMessage = ref('')
const framesResult = ref<GeneratedImage[]>([])
let unsubscribe: (() => void) | null = null

const options = reactive({
  vaes: [] as Array<{ name: string; sha256: string }>,
  textEncoders: [] as Array<{ name: string; sha256: string }>,
  metadata: [] as Array<{ name: string; path: string }>,
  loras: [] as Array<{ name: string; sha256: string }>,
  wanHigh: [] as Array<{ name: string; sha256: string }>,
  wanLow: [] as Array<{ name: string; sha256: string }>,
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
    options.vaes = (inv.vaes || [])
      .filter((v: any) => typeof v?.sha256 === 'string' && /^[0-9a-f]{64}$/i.test(String(v.sha256)))
      .map((v: any) => ({ name: String(v.name || ''), sha256: String(v.sha256 || '').toLowerCase() }))
    options.textEncoders = (inv.text_encoders || [])
      .filter((t: any) => typeof t?.sha256 === 'string' && /^[0-9a-f]{64}$/i.test(String(t.sha256)))
      .map((t: any) => ({ name: String(t.name || ''), sha256: String(t.sha256 || '').toLowerCase() }))
    options.metadata = (inv.metadata || []).map((m: any) => ({ name: m.name, path: m.path }))
    options.loras = (inv.loras || [])
      .filter((l: any) => typeof l?.sha256 === 'string' && /^[0-9a-f]{64}$/i.test(String(l.sha256)))
      .map((l: any) => ({ name: String(l.name || ''), sha256: String(l.sha256 || '').toLowerCase() }))
    const gguf = inv.wan22?.gguf || []
    // List all .gguf files under models/wan22 without stage filtering
    options.wanHigh = gguf
      .filter((e: any) => typeof e?.sha256 === 'string' && /^[0-9a-f]{64}$/i.test(String(e.sha256)))
      .map((e: any) => ({ name: String(e.name || ''), sha256: String(e.sha256 || '').toLowerCase() }))
    options.wanLow = options.wanHigh
    // Build name->path maps for resolution at submit
    maps.vae = Object.fromEntries(options.vaes.map((x) => [x.name, x.sha256]))
    maps.te = Object.fromEntries(options.textEncoders.map((x) => [x.name, x.sha256]))
    maps.meta = Object.fromEntries(options.metadata.map((x) => [x.name, x.path]))
    maps.lora = Object.fromEntries(options.loras.map((x) => [x.name, x.sha256]))
    maps.wanHigh = Object.fromEntries(options.wanHigh.map((x) => [x.name, x.sha256]))
    maps.wanLow = maps.wanHigh
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
    const shaRe = /^[0-9a-f]{64}$/i
    const resolveSha = (val: string, map: Record<string, string>) => {
      const raw = String(val || '').trim()
      if (!raw) return ''
      if (shaRe.test(raw)) return raw.toLowerCase()
      const mapped = (map && map[raw]) ? map[raw] : raw
      if (!shaRe.test(mapped)) throw new Error(`expected sha256 (64 hex), got: ${mapped}`)
      return mapped.toLowerCase()
    }
    const resolvePath = (val: string, map: Record<string, string>) => (map && map[val]) ? map[val] : val

    const highSha = resolveSha(state.high.modelSha, maps.wanHigh)
    const lowSha = resolveSha(state.low.modelSha, maps.wanLow)
    const vaeSha = resolveSha(state.vaeSha, maps.vae)
    const teSha = resolveSha(state.textEncoderSha, maps.te)
    const metaDir = resolvePath(state.metadataDir, maps.meta)

    const extras: Record<string, unknown> = {
      wan_high: { sampler: state.sampler, scheduler: state.scheduler, steps: state.high.steps, cfg_scale: state.high.cfgScale, model_sha: highSha || undefined },
      wan_low: { sampler: state.sampler, scheduler: state.scheduler, steps: state.low.steps, cfg_scale: state.low.cfgScale, model_sha: lowSha || undefined },
      wan_vae_sha: vaeSha || undefined,
      wan_tenc_sha: teSha || undefined,
      wan_metadata_dir: metaDir || undefined,
    }
    // Offload/kernel controls removed (payload/options-driven; no env overrides).
    if (state.high.useLora && state.high.loraSha) { (extras.wan_high as any).lora_sha = resolveSha(state.high.loraSha, maps.lora); (extras.wan_high as any).lora_weight = state.high.loraWeight }
    if (state.low.useLora && state.low.loraSha) { (extras.wan_low as any).lora_sha = resolveSha(state.low.loraSha, maps.lora); (extras.wan_low as any).lora_weight = state.low.loraWeight }
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
      const quick = (await import('../stores/quicksettings')).useQuicksettingsStore()
      const { task_id } = await startImg2Vid({ codex_device: quick.currentDevice, ...payload })
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
      const quick = (await import('../stores/quicksettings')).useQuicksettingsStore()
      const { task_id } = await startTxt2Vid({ codex_device: quick.currentDevice, ...payload })
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

const STORAGE_KEY = 'codex.test.profile.v1'

// Load defaults from localStorage on first mount
;(function loadDefaults() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return
    const snap = JSON.parse(raw)
    const assign = (k: string, v: any) => { if (v !== undefined && v !== null && (k in state)) (state as any)[k] = v }
    assign('prompt', snap.prompt)
    assign('negative', snap.negative)
    assign('width', Number(snap.width))
    assign('height', Number(snap.height))
    assign('frames', Number(snap.frames))
    assign('fps', Number(snap.fps))
    assign('useInitImage', !!snap.useInitImage)
    assign('vaeSha', snap.vaeSha)
    assign('textEncoderSha', snap.textEncoderSha)
    assign('tokenizerDir', snap.tokenizerDir)
    assign('metadataDir', snap.metadataDir)
    // high/low blocks
    if (snap.high) {
      state.high.modelSha = snap.high.modelSha ?? state.high.modelSha
      state.high.steps = Number(snap.high.steps ?? state.high.steps)
      state.high.cfgScale = Number(snap.high.cfgScale ?? state.high.cfgScale)
      state.high.useLora = !!snap.high.useLora
      state.high.loraSha = snap.high.loraSha ?? ''
      state.high.loraWeight = Number(snap.high.loraWeight ?? 1.0)
    }
    if (snap.low) {
      state.low.modelSha = snap.low.modelSha ?? state.low.modelSha
      state.low.steps = Number(snap.low.steps ?? state.low.steps)
      state.low.cfgScale = Number(snap.low.cfgScale ?? state.low.cfgScale)
      state.low.useLora = !!snap.low.useLora
      state.low.loraSha = snap.low.loraSha ?? ''
      state.low.loraWeight = Number(snap.low.loraWeight ?? 1.0)
    }
    assign('sampler', snap.sampler)
    assign('scheduler', snap.scheduler)
    assign('wanFormat', snap.wanFormat)
    assign('seed', Number(snap.seed))
  } catch (e) {
    console.warn('[test] failed to load defaults', e)
  }
})()

function saveProfile(): void {
  try {
    const snap = {
      // note: we intentionally do not persist initImageData to localStorage due to size limits
      prompt: state.prompt,
      negative: state.negative,
      width: state.width,
      height: state.height,
      frames: state.frames,
      fps: state.fps,
      useInitImage: state.useInitImage,
      vaeSha: state.vaeSha,
      textEncoderSha: state.textEncoderSha,
      tokenizerDir: state.tokenizerDir,
      metadataDir: state.metadataDir,
      high: { ...state.high },
      low: { ...state.low },
      sampler: state.sampler,
      scheduler: state.scheduler,
      wanFormat: state.wanFormat,
      seed: state.seed,
    }
    // sanitize big fields
    delete (snap as any).initImageData
    delete (snap as any).initImageName
    localStorage.setItem(STORAGE_KEY, JSON.stringify(snap))
    errorMessage.value = 'Saved defaults for Test tab'
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
  }
}

</script>
