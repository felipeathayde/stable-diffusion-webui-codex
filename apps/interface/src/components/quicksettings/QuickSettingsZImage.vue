<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Z-Image quicksettings selectors (model + assets).
Renders Z-Image model, VAE, and Qwen3 text encoder selectors for Z-Image model tabs.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsZImage` (component): Z-Image quicksettings row used by the main quicksettings bar.
- `truncatePath` (function): Truncates absolute paths for compact dropdown labels.
- `textEncoderLabel` (function): Builds a compact `family/basename` label for text encoder values.
-->

<template>
  <!-- Z Image-specific quicksettings row -->
  <div class="quicksettings-group qs-group-checkpoint">
    <label class="label-muted">Model</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
          <option v-for="model in checkpoints" :key="model" :value="model">{{ truncatePath(model) }}</option>
        </select>
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
  textEncoder: string
  textEncoderChoices: string[]
}>()

defineEmits<{
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'addTencPath'): void
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
