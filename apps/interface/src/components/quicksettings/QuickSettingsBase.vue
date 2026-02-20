<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Base quicksettings selectors for diffusion tabs (SD15/SDXL).
Renders Checkpoint/VAE selectors (plus optional Text Encoder) for model tabs, emitting updates and “add path” actions to the parent quicksettings bar.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsBase` (component): Base quicksettings selectors for diffusion model tabs.
- `isVaeSentinel` (function): Returns whether a VAE value is a sentinel selection (built-in/none) without metadata.
- `vaeLabel` (function): Formats canonical VAE sentinel values for dropdown display.
- `textEncoderLabel` (function): Builds a compact `family/basename` label for text encoder dropdown values.
-->

<template>
  <div class="quicksettings-group qs-group-checkpoint">
    <label class="label-muted">Checkpoint</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
          <option v-if="checkpoints.length === 0" value="">No models found</option>
          <option v-for="model in checkpoints" :key="model" :value="model">
            {{ model }}
          </option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!checkpoint"
          title="Show checkpoint metadata"
          aria-label="Show checkpoint metadata"
          @click="$emit('showMetadata', { kind: 'checkpoint', value: checkpoint })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addCheckpointPath')">+</button>
      </div>
    </div>
  </div>

    <div class="quicksettings-group qs-group-vae">
      <label class="label-muted">VAE</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
            <option v-for="v in vaeChoices" :key="v" :value="v">
              {{ vaeLabel(v) }}
            </option>
          </select>
          <button
            class="btn qs-btn-outline qs-inline-btn qs-info-btn"
            type="button"
            :disabled="!vae || isVaeSentinel(vae)"
            title="Show VAE metadata"
            aria-label="Show VAE metadata"
            @click="$emit('showMetadata', { kind: 'vae', value: vae })"
          >
            i
          </button>
          <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addVaePath')">+</button>
        </div>
      </div>
    </div>

    <div v-if="showTextEncoder !== false" class="quicksettings-group qs-group-text-encoder">
      <label class="label-muted">Text Encoder</label>
      <div class="qs-row">
        <select class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ textEncoderAutomaticLabel }}</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ textEncoderLabel(te) }}</option>
        </select>
      </div>
    </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  checkpoint: string
  checkpoints: string[]
  vae: string
  vaeChoices: string[]
  textEncoder: string
  textEncoderChoices: any
  textEncoderAutomaticLabel?: string
  showTextEncoder?: boolean
}>()

defineEmits<{
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'showMetadata', payload: { kind: 'checkpoint' | 'vae' | 'text_encoder'; value: string }): void
}>()

const textEncoderAutomaticLabel = props.textEncoderAutomaticLabel ?? 'Built-in'

function isVaeSentinel(value: string): boolean {
  const normalized = String(value || '').trim().toLowerCase()
  return normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in' || normalized === 'none'
}

function vaeLabel(value: string): string {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in') return 'Built-in'
  if (normalized === 'none') return 'None'
  return value
}

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const [family, ...rest] = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (!family || rest.length === 0) return value
  const basename = rest[rest.length - 1] || rest[0]
  return `${family}/${basename}`
}
</script>
