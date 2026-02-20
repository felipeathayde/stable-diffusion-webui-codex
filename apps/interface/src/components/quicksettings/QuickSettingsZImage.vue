<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image quicksettings selectors (model + variant + assets).
Renders Z-Image model, Turbo/Base toggle, VAE, and Qwen3 text encoder selectors for Z-Image model tabs, and exposes an
optional `modeToggles` slot immediately after Turbo for parent-owned mode actions (IMG2IMG/INPAINT).

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsZImage` (component): Z-Image quicksettings row used by the main quicksettings bar.
- `truncatePath` (function): Truncates absolute paths for compact dropdown labels.
- `isVaeSentinel` (function): Returns whether a VAE value is a sentinel selection (built-in/none) without metadata.
- `textEncoderLabel` (function): Builds a compact `family/basename` label for text encoder values.
-->

<template>
  <!-- Z Image-specific quicksettings row -->
  <div class="quicksettings-group qs-group-checkpoint">
    <label class="label-muted">Model</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
          <option v-if="checkpoints.length === 0" value="">No models found</option>
          <option v-for="model in checkpoints" :key="model" :value="model">{{ truncatePath(model) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!checkpoint"
          title="Show model metadata"
          aria-label="Show model metadata"
          @click="$emit('showMetadata', { kind: 'checkpoint', value: checkpoint })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addCheckpointPath')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-zimage-turbo">
    <div class="qs-row">
      <button
        :class="['btn', 'qs-toggle-btn', turbo ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="turbo"
        :disabled="turboLocked"
        :title="turboLocked ? 'Turbo variant is fixed by model metadata' : 'Toggle Turbo variant'"
        @click="$emit('update:turbo', !turbo)"
      >
        Turbo
      </button>
    </div>
  </div>

  <slot name="modeToggles" />

  <div class="quicksettings-group qs-group-vae">
    <label class="label-muted">VAE</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
          <option value="">Select VAE</option>
          <option v-for="v in vaeChoices" :key="v" :value="v">{{ truncatePath(v) }}</option>
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

  <div class="quicksettings-group qs-group-text-encoder">
    <label class="label-muted">Text Encoder (Qwen3)</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
          <option value="">Select Text Encoder</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ textEncoderLabel(te) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!textEncoder"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: 'text_encoder', value: textEncoder })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addTencPath')">+</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  checkpoint: string
  checkpoints: string[]
  turbo: boolean
  turboLocked: boolean
  vae: string
  vaeChoices: string[]
  textEncoder: string
  textEncoderChoices: string[]
}>()

defineEmits<{
  (e: 'update:checkpoint', value: string): void
  (e: 'update:turbo', value: boolean): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'addTencPath'): void
  (e: 'showMetadata', payload: { kind: 'checkpoint' | 'vae' | 'text_encoder'; value: string }): void
}>()

function truncatePath(path: string, maxLen = 40): string {
  if (!path || path.length <= maxLen) return path
  const parts = path.replace(/\\/g, '/').split('/')
  const name = parts[parts.length - 1] || path
  return name.length > maxLen ? `...${name.slice(-maxLen)}` : name
}

function isVaeSentinel(value: string): boolean {
  const normalized = String(value || '').trim().toLowerCase()
  return normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in' || normalized === 'none'
}

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const parts = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (parts.length < 2) return value
  return `${parts[0]}/${parts[parts.length - 1]}`
}
</script>
