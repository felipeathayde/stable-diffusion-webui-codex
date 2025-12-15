<template>
    <div class="quicksettings-group qs-group-wan-high">
      <label class="label-muted">WAN High model</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="highModel" @change="$emit('update:highModel', ($event.target as HTMLSelectElement).value)">
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
          <select class="select-md" :value="lowModel" @change="$emit('update:lowModel', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="m in lowChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseLow')">+</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-metadata">
      <label class="label-muted">WAN Metadata</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="metadataDir" @change="$emit('update:metadataDir', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="m in metadataChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseMetadata')">+</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-text-encoder">
      <label class="label-muted">WAN Text Encoder</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ encoderLabel(te) }}</option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseTe')">+</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group qs-group-wan-vae">
      <label class="label-muted">WAN VAE</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="v in vaeChoices" :key="v" :value="v">{{ dirLabel(v) }}</option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseVae')">+</button>
        </div>
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
</template>

<script setup lang="ts">
const props = defineProps<{
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
</script>
