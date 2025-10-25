<template>
  <div class="panel wan22-panel">
    <div class="panel-header">WAN 2.2 Settings</div>
    <div class="panel-body wan22-body">
      <div class="panel-section wan22-section">
        <h3 class="label-muted">WAN22 Models</h3>
        <div class="wan22-grid">
          <div>
            <label class="label-muted" for="wanHigh">High Model (.gguf)</label>
            <input id="wanHigh" class="ui-input" list="dl-wan-high" v-model="store.hiModelDir" placeholder="/models/wan22/*High*.gguf" autocomplete="off" autocapitalize="off" spellcheck="false" />
            <datalist id="dl-wan-high">
              <option v-for="opt in options.wanHigh" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
            </datalist>
          </div>
          <div>
            <label class="label-muted" for="wanLow">Low Model (.gguf)</label>
            <input id="wanLow" class="ui-input" list="dl-wan-low" v-model="store.loModelDir" placeholder="/models/wan22/*Low*.gguf" autocomplete="off" autocapitalize="off" spellcheck="false" />
            <datalist id="dl-wan-low">
              <option v-for="opt in options.wanLow" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
            </datalist>
          </div>
        </div>
        <div class="wan22-grid">
          <div>
            <label class="switch-label">
              <input type="checkbox" v-model="store.hiUseLora" />
              <span>Use Auxiliary LoRA (High)</span>
            </label>
            <div v-if="store.hiUseLora">
              <label class="label-muted" for="wanHiLora">LoRA (input list)</label>
              <input id="wanHiLora" class="ui-input" list="dl-lora" v-model="store.hiLoraPath" placeholder="/models/Lora/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
            </div>
          </div>
          <div>
            <label class="switch-label">
              <input type="checkbox" v-model="store.loUseLora" />
              <span>Use Auxiliary LoRA (Low)</span>
            </label>
            <div v-if="store.loUseLora">
              <label class="label-muted" for="wanLoLora">LoRA (input list)</label>
              <input id="wanLoLora" class="ui-input" list="dl-lora" v-model="store.loLoraPath" placeholder="/models/Lora/*.safetensors" autocomplete="off" autocapitalize="off" spellcheck="false" />
            </div>
          </div>
        </div>
        <datalist id="dl-lora">
          <option v-for="opt in options.loras" :key="opt.path" :value="opt.name">{{ opt.name }}</option>
        </datalist>
      </div>

      <div class="panel-section wan22-section">
        <h3 class="label-muted">Video Output</h3>
        <div class="wan22-grid">
          <div>
            <label class="label-muted">Filename Prefix</label>
            <input class="ui-input" type="text" v-model="store.filenamePrefix" placeholder="wan22" />
          </div>
          <div>
            <label class="label-muted">Format</label>
            <select class="select-md" v-model="store.videoFormat">
              <option v-for="opt in formatOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </div>
          <div>
            <label class="label-muted">Pixel Format</label>
            <select class="select-md" v-model="store.videoPixFormat">
              <option v-for="opt in pixFmtOptions" :key="opt" :value="opt">{{ opt }}</option>
            </select>
          </div>
          <div>
            <label class="label-muted">CRF</label>
            <input class="ui-input" type="number" min="0" max="51" v-model.number="store.videoCrf" />
          </div>
          <div>
            <label class="label-muted">Loop Count</label>
            <input class="ui-input" type="number" min="0" v-model.number="store.videoLoopCount" />
          </div>
        </div>
        <div class="wan22-toggle-row">
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoPingpong" />
            Ping-pong playback
          </label>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoSaveMetadata" />
            Save metadata
          </label>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoSaveOutput" />
            Save output file
          </label>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.videoTrimToAudio" />
            Trim to audio
          </label>
        </div>
      </div>

      <div class="panel-section wan22-section">
        <div class="wan22-toggle-head">
          <h3 class="label-muted">Interpolation (RIFE)</h3>
          <label class="wan22-toggle">
            <input type="checkbox" v-model="store.rifeEnabled" />
            Enable
          </label>
        </div>
        <div v-if="store.rifeEnabled" class="wan22-grid">
          <div>
            <label class="label-muted">Model</label>
            <input class="ui-input" type="text" v-model="store.rifeModel" />
          </div>
          <div>
            <label class="label-muted">Times</label>
            <input class="ui-input" type="number" min="1" max="8" step="1" v-model.number="store.rifeTimes" />
          </div>
        </div>
        <p v-else class="caption">Interpolation disabled.</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { fetchModelInventory } from '../api/client'

const props = defineProps<{ store: any }>()
const store = props.store

const formatOptions = ['video/h264-mp4', 'video/h265-mp4', 'video/gif']
const pixFmtOptions = ['yuv420p', 'yuv444p', 'yuv422p']

const options = reactive({
  wanHigh: [] as Array<{ name: string; path: string }>,
  wanLow: [] as Array<{ name: string; path: string }>,
  loras: [] as Array<{ name: string; path: string }>,
})

const maps = reactive({
  wanHigh: {} as Record<string, string>,
  wanLow: {} as Record<string, string>,
  lora: {} as Record<string, string>,
})

// Load inventory for datalists on mount
;(async () => {
  try {
    const inv = await fetchModelInventory()
    const gguf = inv.wan22?.gguf || []
    options.wanHigh = gguf.filter((e: any) => e.stage === 'high').map((e: any) => ({ name: e.name, path: e.path }))
    options.wanLow = gguf.filter((e: any) => e.stage === 'low').map((e: any) => ({ name: e.name, path: e.path }))
    options.loras = (inv.loras || []).map((x: any) => ({ name: x.name, path: x.path }))
    maps.wanHigh = Object.fromEntries(options.wanHigh.map((x) => [x.name, x.path]))
    maps.wanLow = Object.fromEntries(options.wanLow.map((x) => [x.name, x.path]))
    maps.lora = Object.fromEntries(options.loras.map((x) => [x.name, x.path]))
  } catch (err) {
    console.error('[wan22-panel] inventory load failed', err)
  }
})()

// Resolve names to paths before use; panel does not submit, but expose helpers if parent needs them
function resolveName(val: string, map: Record<string, string>): string { return (map && map[val]) ? map[val] : val }
// Optionally, parents can call resolve on store fields before sending to backend
store.__resolveWan22Paths = () => {
  store.hiModelDir = resolveName(store.hiModelDir, maps.wanHigh)
  store.loModelDir = resolveName(store.loModelDir, maps.wanLow)
  if (!store.hiUseLora) store.hiLoraPath = ''
  if (!store.loUseLora) store.loLoraPath = ''
  if (store.hiLoraPath) store.hiLoraPath = resolveName(store.hiLoraPath, maps.lora)
  if (store.loLoraPath) store.loLoraPath = resolveName(store.loLoraPath, maps.lora)
}
</script>
