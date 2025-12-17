<template>
    <div class="quicksettings-group qs-group-wan-mode">
      <label class="label-muted">Mode</label>
      <div class="qs-row">
        <select id="qs-wan-mode" class="select-md" :value="mode" @change="$emit('update:mode', ($event.target as HTMLSelectElement).value)">
          <option value="txt2vid">Text (txt2vid)</option>
          <option value="img2vid">Image (img2vid)</option>
          <option value="vid2vid">Video (vid2vid)</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-format">
      <label class="label-muted">Format</label>
      <div class="qs-row">
        <select class="select-md" :value="modelFormat" @change="$emit('update:modelFormat', ($event.target as HTMLSelectElement).value)">
          <option value="auto">Auto</option>
          <option value="diffusers">Diffusers</option>
          <option value="gguf">GGUF</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-high">
      <label class="label-muted">WAN High model</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select id="qs-wan-high" class="select-md" :value="highModel" @change="$emit('update:highModel', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="m in highChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseHigh')">+</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-low">
      <label class="label-muted">WAN Low model</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select id="qs-wan-low" class="select-md" :value="lowModel" @change="$emit('update:lowModel', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="m in lowChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseLow')">+</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-assets">
      <label class="label-muted">WAN Assets</label>
      <div class="qs-row">
        <button class="btn btn-secondary qs-overrides-btn" type="button" :title="assetsSummary" @click="showAssetsModal = true">Assets…</button>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-guided">
      <label class="label-muted">Guide</label>
      <div class="qs-row">
        <button class="btn btn-secondary qs-overrides-btn" type="button" @click="$emit('guidedGen')">Guided gen</button>
      </div>
    </div>

    <div class="quicksettings-group qs-group-unet-dtype">
      <label class="label-muted">Diffusion in Low Bits</label>
      <div class="qs-row">
        <select class="select-md" :value="unetDtype" @change="$emit('update:unetDtype', ($event.target as HTMLSelectElement).value)">
          <option v-for="opt in unetDtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group qs-group-perf qs-group-perf-vram">
      <label class="label-muted">GPU VRAM (MB)</label>
      <div class="qs-row">
        <input
          class="ui-input"
          type="number"
          :min="0"
          :max="gpuTotalMb"
          :value="gpuWeightsMb"
          @change="$emit('update:gpuWeightsMb', Number(($event.target as HTMLInputElement).value))"
        />
      </div>
    </div>

    <div class="quicksettings-group qs-group-attention">
      <label class="label-muted">Attention Backend</label>
      <div class="qs-row">
        <select class="select-md" :value="attentionBackend" @change="$emit('update:attentionBackend', ($event.target as HTMLSelectElement).value)">
          <option v-for="opt in attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group qs-group-overrides">
      <label class="label-muted">Overrides</label>
      <div class="qs-row">
        <button class="btn btn-secondary qs-overrides-btn" type="button" @click="$emit('openOverrides')">
          Set overrides
        </button>
      </div>
    </div>

    <QuickSettingsWanAssetsModal
      v-model="showAssetsModal"
      :metadata-dir="metadataDir"
      :metadata-choices="metadataChoices"
      :text-encoder="textEncoder"
      :text-encoder-choices="textEncoderChoices"
      :vae="vae"
      :vae-choices="vaeChoices"
      @update:metadataDir="$emit('update:metadataDir', $event)"
      @update:textEncoder="$emit('update:textEncoder', $event)"
      @update:vae="$emit('update:vae', $event)"
      @browseMetadata="$emit('browseMetadata')"
      @browseTe="$emit('browseTe')"
      @browseVae="$emit('browseVae')"
    />
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import QuickSettingsWanAssetsModal from '../modals/QuickSettingsWanAssetsModal.vue'

const props = defineProps<{
  mode: string
  modelFormat: string
  highModel: string
  highChoices: string[]
  lowModel: string
  lowChoices: string[]
  metadataDir: string
  metadataChoices: string[]
  textEncoder: string
  textEncoderChoices: string[]
  vae: string
  vaeChoices: string[]
  unetDtype: string
  unetDtypeChoices: string[]
  gpuWeightsMb: number
  gpuTotalMb: number
  attentionBackend: string
  attentionChoices: Array<{ value: string; label: string }>
}>()

const builtInLabel = 'Built-in'
const showAssetsModal = ref(false)

function dirLabel(path: string): string {
  const norm = path.replace(/\\/g, '/')
  if (!norm) return ''
  const idx = norm.lastIndexOf('/')
  return idx >= 0 ? norm.slice(idx + 1) || norm : norm
}

function encoderLabel(value: string): string {
  const norm = String(value || '').replace(/\\/g, '/')
  if (!norm) return ''
  if (!norm.includes('/')) return norm
  const [family, ...rest] = norm.split('/').filter(Boolean)
  if (!family || rest.length === 0) return norm
  const tail = rest[rest.length - 1] || rest[0]
  // For file labels like wan22//abs/path/to/file.safetensors, show wan22/file.safetensors.
  return `${family}/${tail}`
}

const assetsSummary = computed(() => {
  const meta = props.metadataDir ? dirLabel(props.metadataDir) : builtInLabel
  const te = props.textEncoder ? encoderLabel(props.textEncoder) : builtInLabel
  const v = props.vae ? dirLabel(props.vae) : builtInLabel
  return `Metadata: ${meta} · TE: ${te} · VAE: ${v}`
})
</script>
