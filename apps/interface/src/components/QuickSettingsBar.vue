<template>
  <section class="quicksettings">
    <div class="quicksettings-group">
      <label class="label-muted" title="Presets for UI + defaults; does not change the underlying engine semantics">Model UI</label>
      <div class="qs-row">
        <select class="select-md" :value="selectedPreset" @change="onPresetChange">
          <option v-for="name in presetChoices" :key="name" :value="name">{{ name }}</option>
        </select>
      </div>
    </div>
    <div class="quicksettings-group">
      <label class="label-muted">Mode</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentMode" @change="onModeChange">
          <option v-for="m in store.modeChoices" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
    </div>
    <div class="quicksettings-group" v-if="!hideCheckpoint">
      <label class="label-muted">Checkpoint</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentModel" @change="onModelChange">
          <option v-for="model in store.models" :key="model.title" :value="model.title">
            {{ model.title }}
          </option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">VAE</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentVae" @change="onVaeChange">
          <option v-for="v in store.vaeChoices" :key="v" :value="v">{{ v }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Text Encoder</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentTextEncoders[0] ?? ''" @change="onTextEncoderChange">
          <option value="">Automatic</option>
          <option v-for="te in store.textEncoderChoices" :key="te" :value="te">{{ te }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Diffusion in Low Bits</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentUnetDtype" @change="onUnetDtypeChange">
          <option v-for="opt in store.unetDtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">GPU Weights (MB)</label>
      <div class="qs-row">
        <input class="ui-input" type="number" :min="0" :max="store.gpuTotalMb" :value="store.gpuWeightsMb" @change="onGpuWeightsChange" />
      </div>
  </div>

    <div class="quicksettings-group">
      <label class="label-muted">Attention Backend</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentAttention" @change="onAttentionChange">
          <option v-for="opt in store.attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>

    <!-- Right-most refresh button spanning to the end -->
    <div class="quicksettings-group quicksettings-right">
      <label class="label-muted">Models</label>
      <div class="qs-row">
        <button class="btn btn-sm btn-secondary" type="button" @click="refreshAll" title="Refresh checkpoint, VAE and text encoder lists">Refresh</button>
      </div>
    </div>
  
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useRoute } from 'vue-router'
import { useUiBlocksStore } from '../stores/ui_blocks'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const selectedPreset = ref('')
const uiBlocks = useUiBlocksStore()

onMounted(() => {
  void store.init()
  void presets.init(currentTab())
})

watch(() => route.path, async () => {
  await presets.init(currentTab())
  selectedPreset.value = ''
})

function currentTab(): 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' {
  const p = route.path
  if (p.startsWith('/img2img')) return 'img2img'
  if (p.startsWith('/txt2vid')) return 'txt2vid'
  if (p.startsWith('/img2vid')) return 'img2vid'
  return 'txt2img'
}

const presetChoices = computed(() => presets.namesFor(currentTab()))
const hideCheckpoint = computed(() => {
  const p = route.path
  // In model tabs (/models), the tab manages model dirs (e.g., WAN 2.2); hide checkpoint there.
  if (p.startsWith('/models')) return true
  const isVideo = p.startsWith('/txt2vid') || p.startsWith('/img2vid')
  return isVideo && uiBlocks.semanticEngine === 'wan22'
})

async function onPresetChange(event: Event): Promise<void> {
  const title = (event.target as HTMLSelectElement).value
  selectedPreset.value = title
  await presets.applyByTitle(title, currentTab())
  // Refresh quicksettings to reflect applied checkpoint/options
  await store.init()
}

function onModelChange(event: Event): void {
  void store.setModel((event.target as HTMLSelectElement).value)
}

// sampler/scheduler/seed handlers removed from quicksettings

function onModeChange(event: Event): void {
  void store.setMode((event.target as HTMLSelectElement).value)
}

async function refreshAll(): Promise<void> { await store.init() }

function onVaeChange(event: Event): void {
  void store.setVae((event.target as HTMLSelectElement).value)
}

function onTextEncoderChange(event: Event): void {
  const select = event.target as HTMLSelectElement
  const value = select.value
  // Backend ainda espera array; enviamos [] para Automático (vazio) ou [value]
  const payload = value ? [value] : []
  void store.setTextEncoders(payload)
}

function onUnetDtypeChange(event: Event): void {
  void store.setUnetDtype((event.target as HTMLSelectElement).value)
}

function onGpuWeightsChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  void store.setGpuWeightsMb(Number.isFinite(v) ? v : store.gpuWeightsMb)
}

function onAttentionChange(event: Event): void {
  void store.setAttentionBackend((event.target as HTMLSelectElement).value)
}

// random seed button removed from quicksettings
</script>
