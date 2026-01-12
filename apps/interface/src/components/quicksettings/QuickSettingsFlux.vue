<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: FLUX.1-specific quicksettings selectors.
Renders checkpoint/VAE selectors plus a dual text-encoder selection row (CLIP + T5) for FLUX.1 model tabs.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsFlux` (component): FLUX.1 quicksettings for checkpoint/VAE and dual text encoders.
- `truncatePath` (function): Truncates absolute paths for compact dropdown labels.
- `textEncoderLabel` (function): Builds a compact `family/basename` label for text encoder values.
-->

<template>
  <!-- FLUX.1-specific quicksettings row -->
  <div class="quicksettings-group qs-group-checkpoint">
    <label class="label-muted">Checkpoint</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
          <option v-for="model in checkpoints" :key="model" :value="model">{{ truncatePath(model) }}</option>
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
          <option value="">Select VAE</option>
          <option v-for="v in vaeChoices" :key="v" :value="v">{{ truncatePath(v) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!vae"
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

  <div class="quicksettings-group qs-group-flux1-tenc">
    <label class="label-muted">Text Encoders</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="textEncoderPrimary" @change="$emit('update:textEncoderPrimary', ($event.target as HTMLSelectElement).value)">
          <option value="">Select CLIP</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ textEncoderLabel(te) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!textEncoderPrimary"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: 'text_encoder_primary', value: textEncoderPrimary })"
        >
          i
        </button>
        <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addTencPath')">+</button>
      </div>
      <div class="qs-pair">
        <select class="select-md" :value="textEncoderSecondary" @change="$emit('update:textEncoderSecondary', ($event.target as HTMLSelectElement).value)">
          <option value="">Select T5</option>
          <option v-for="te in textEncoderChoices" :key="`sec-${te}`" :value="te">{{ textEncoderLabel(te) }}</option>
        </select>
        <button
          class="btn qs-btn-outline qs-inline-btn qs-info-btn"
          type="button"
          :disabled="!textEncoderSecondary"
          title="Show text encoder metadata"
          aria-label="Show text encoder metadata"
          @click="$emit('showMetadata', { kind: 'text_encoder_secondary', value: textEncoderSecondary })"
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
  vae: string
  vaeChoices: string[]
  textEncoderPrimary: string
  textEncoderSecondary: string
  textEncoderChoices: string[]
}>()

defineEmits<{
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoderPrimary', value: string): void
  (e: 'update:textEncoderSecondary', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'addTencPath'): void
  (e: 'showMetadata', payload: { kind: 'checkpoint' | 'vae' | 'text_encoder_primary' | 'text_encoder_secondary'; value: string }): void
}>()

function truncatePath(path: string, maxLen = 40): string {
  if (!path || path.length <= maxLen) return path
  const parts = path.replace(/\\/g, '/').split('/')
  const name = parts[parts.length - 1] || path
  return name.length > maxLen ? `...${name.slice(-maxLen)}` : name
}

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const parts = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (parts.length < 2) return value
  return `${parts[0]}/${parts[parts.length - 1]}`
}
</script>
