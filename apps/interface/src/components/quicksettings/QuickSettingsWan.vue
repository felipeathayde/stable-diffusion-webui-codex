<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN22-specific quicksettings selectors.
Renders WAN model-mode selection (t2v/i2v/v2v + size), LightX2V toggle, stage model dirs (high/low), text encoder, and VAE selectors with “Browse…” actions.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsWan` (component): WAN quicksettings row used by the main quicksettings bar.
- `dirLabel` (function): Produces compact directory/file labels from absolute paths.
- `encoderLabel` (function): Produces compact `family/basename` labels for WAN text encoder values.
-->

<template>
  <div class="quicksettings-group qs-group-wan-mode">
    <label class="label-muted">Mode</label>
    <div class="qs-row">
      <select id="qs-wan-mode" class="select-md" :value="mode" @change="$emit('update:mode', ($event.target as HTMLSelectElement).value)">
        <option value="i2v_14b">I2V 14B</option>
        <option value="t2v_14b">T2V 14B</option>
        <option value="i2v_5b">I2V 5B</option>
        <option value="t2v_5b">T2V 5B</option>
        <option value="v2v_14b">V2V 14B</option>
      </select>
    </div>
  </div>

  <div class="quicksettings-group qs-group-wan-refresh">
    <label class="label-muted">Lists</label>
    <div class="qs-row">
      <button class="btn qs-btn-secondary qs-refresh-btn" type="button" title="Refresh lists" @click="$emit('refresh')">Refresh</button>
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
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!highModel"
          title="Show model metadata"
          aria-label="Show model metadata"
          @click="$emit('showMetadata', { kind: 'wan_high_model', value: highModel })"
        >
          i
        </button>
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
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!lowModel"
          title="Show model metadata"
          aria-label="Show model metadata"
          @click="$emit('showMetadata', { kind: 'wan_low_model', value: lowModel })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseLow')">+</button>
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
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!textEncoder"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: 'wan_text_encoder', value: textEncoder })"
        >
          i
        </button>
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
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!vae"
          title="Show VAE metadata"
          aria-label="Show VAE metadata"
          @click="$emit('showMetadata', { kind: 'wan_vae', value: vae })"
        >
          i
        </button>
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
  textEncoder: string
  textEncoderChoices: string[]
  vae: string
  vaeChoices: string[]
}>()

defineEmits<{
  (e: 'update:mode', value: string): void
  (e: 'update:lightx2v', value: boolean): void
  (e: 'update:highModel', value: string): void
  (e: 'update:lowModel', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'browseHigh'): void
  (e: 'browseLow'): void
  (e: 'browseTe'): void
  (e: 'browseVae'): void
  (e: 'refresh'): void
  (e: 'showMetadata', payload: { kind: 'wan_high_model' | 'wan_low_model' | 'wan_text_encoder' | 'wan_vae'; value: string }): void
}>()

const builtInLabel = 'Select…'

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
