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
                <label class="label" for="vaeDir">VAE Path (.safetensors)</label>
                <input id="vaeDir" class="ui-input" type="text" v-model="state.vaeDir" placeholder="C:\\...\\vae.safetensors" />
              </div>
              <div class="field-stack">
                <label class="label" for="textEncoderDir">Text Encoder Path (.safetensors)</label>
                <input id="textEncoderDir" class="ui-input" type="text" v-model="state.textEncoderDir" placeholder="C:\\...\\text_encoder.safetensors" />
              </div>
              <div class="field-stack">
                <label class="label" for="tokenizerDir">Tokenizer Dir (optional)</label>
                <input id="tokenizerDir" class="ui-input" type="text" v-model="state.tokenizerDir" placeholder="C:\\...\\tokenizer" />
                <span class="muted test-sublabel">Leave blank to auto-fetch.</span>
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
                <label class="label" for="highModel">Model Dir</label>
                <input id="highModel" class="ui-input" type="text" v-model="state.high.modelDir" placeholder="C:\\...HighNoise..." />
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
                <label class="label" for="highLora">LoRA Path</label>
                <input id="highLora" class="ui-input" type="text" v-model="state.high.loraPath" placeholder="C:\\...\\lora.safetensors" />
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
                <label class="label" for="lowModel">Model Dir</label>
                <input id="lowModel" class="ui-input" type="text" v-model="state.low.modelDir" placeholder="C:\\...LowNoise..." />
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
                <label class="label" for="lowLora">LoRA Path</label>
                <input id="lowLora" class="ui-input" type="text" v-model="state.low.loraPath" placeholder="C:\\...\\lora.safetensors" />
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
import { startImg2Vid, startTxt2Vid, subscribeTask } from '../api/client'
import ResultViewer from '../components/ResultViewer.vue'
import type { GeneratedImage, TaskEvent } from '../api/types'

const state = reactive({
  prompt: '',
  negative: '',
  width: 768,
  height: 432,
  frames: 16,
  fps: 24,
  useInitImage: true,
  initImageData: '',
  initImageName: '',
  // Defaults (Windows paths provided by you)
  vaeDir: 'C:\\\u005cUsers\\\u005clucas\\\u005cOneDrive\\\u005cDocumentos\\\u005cstable-diffusion-webui-codex\\\u005cmodels\\\u005ccodex\\\u005cWan2.1_VAE.safetensors',
  textEncoderDir: 'C:\\\u005cUsers\\\u005clucas\\\u005cOneDrive\\\u005cDocumentos\\\u005cstable-diffusion-webui-codex\\\u005cmodels\\\u005ccodex\\\u005cumt5_xxl_fp8_e4m3fn_scaled.safetensors',
  tokenizerDir: '',
  high: {
    modelDir: 'C:\\\u005cUsers\\\u005clucas\\\u005cOneDrive\\\u005cDocumentos\\\u005cstable-diffusion-webui-codex\\\u005cmodels\\\u005ccodex\\\u005cWan2.2-I2V-A14B-HighNoise-Q2_K.gguf',
    steps: 4, cfgScale: 7,
    loraPath: 'C:\\\u005cUsers\\\u005clucas\\\u005cOneDrive\\\u005cDocumentos\\\u005cstable-diffusion-webui-codex\\\u005cmodels\\\u005ccodex\\\u005chigh_noise_model_lora.safetensors', loraWeight: 1.0,
  },
  low: {
    modelDir: 'C:\\\u005cUsers\\\u005clucas\\\u005cOneDrive\\\u005cDocumentos\\\u005cstable-diffusion-webui-codex\\\u005cmodels\\\u005ccodex\\\u005cWan2.2-I2V-A14B-LowNoise-Q2_K.gguf',
    steps: 4, cfgScale: 7,
    loraPath: 'C:\\\u005cUsers\\\u005clucas\\\u005cOneDrive\\\u005cDocumentos\\\u005cstable-diffusion-webui-codex\\\u005cmodels\\\u005ccodex\\\u005clow_noise_model_lora.safetensors', loraWeight: 1.0,
  },
  sampler: 'Euler', scheduler: 'Simple', wanFormat: 'gguf',
  seed: -1,
})

const isRunning = ref(false)
const errorMessage = ref('')
const framesResult = ref<GeneratedImage[]>([])
let unsubscribe: (() => void) | null = null

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
    const extras: Record<string, unknown> = {
      wan_high: { sampler: state.sampler, scheduler: state.scheduler, steps: state.high.steps, cfg_scale: state.high.cfgScale, model_dir: state.high.modelDir || undefined },
      wan_low: { sampler: state.sampler, scheduler: state.scheduler, steps: state.low.steps, cfg_scale: state.low.cfgScale, model_dir: state.low.modelDir || undefined },
      wan_format: state.wanFormat,
      wan_vae_dir: state.vaeDir || undefined,
      wan_text_encoder_dir: state.textEncoderDir || undefined,
      wan_tokenizer_dir: state.tokenizerDir || undefined,
    }
    if (state.high.loraPath) { (extras.wan_high as any).lora_path = state.high.loraPath; (extras.wan_high as any).lora_weight = state.high.loraWeight }
    if (state.low.loraPath) { (extras.wan_low as any).lora_path = state.low.loraPath; (extras.wan_low as any).lora_weight = state.low.loraWeight }
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
