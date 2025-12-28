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

  <div class="quicksettings-group qs-group-wan-lightx2v">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', lightx2v ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="lightx2v"
        title="Enable LightX2V runtime"
        @click="$emit('update:lightx2v', !lightx2v)"
      >
        LightX2V
      </button>
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
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseHigh')">+</button>
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
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseLow')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-metadata">
    <label class="label-muted">WAN Metadata</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-metadata" class="select-md" :value="metadataDir" @change="$emit('update:metadataDir', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="m in metadataChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
        </select>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseMetadata')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-text-encoder">
    <label class="label-muted">WAN Text Encoder</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-text-encoder" class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ encoderLabel(te) }}</option>
        </select>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseTe')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-vae">
    <label class="label-muted">WAN VAE</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select id="qs-wan-vae" class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ builtInLabel }}</option>
          <option v-for="v in vaeChoices" :key="v" :value="v">{{ dirLabel(v) }}</option>
        </select>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseVae')">+</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  mode: string
  lightx2v: boolean
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
