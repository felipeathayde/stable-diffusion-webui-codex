<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN stage LoRA selector + weight field.
Renders a LoRA dropdown (from WAN LoRA inventory choices) and an optional weight input, emitting updates to the parent stage panel.

Symbols (top-level; keep in sync; no ghosts):
- `WanStageLoraField` (component): Stage-level LoRA path + weight inputs for WAN generation.
- `onPathChange` (function): Emits an updated LoRA path selection.
- `onWeightChange` (function): Emits an updated LoRA weight value.
-->

<template>
  <div class="gc-row">
    <div class="gc-col gc-col--wide field">
      <label class="label-muted">LoRA (wan22-loras)</label>
      <select class="select-md" :disabled="disabled" :value="loraPath" @change="onPathChange">
        <option value="">None</option>
        <option v-for="opt in choices" :key="opt.path" :value="opt.path">{{ opt.name }}</option>
      </select>
    </div>
    <div v-if="loraPath" class="gc-col field">
      <label class="label-muted">LoRA weight</label>
      <input class="ui-input" type="number" step="0.05" :disabled="disabled" :value="loraWeight" @change="onWeightChange" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  loraPath: string
  loraWeight: number
  choices: Array<{ name: string; path: string }>
  disabled?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:loraPath', value: string): void
  (e: 'update:loraWeight', value: number): void
}>()

const disabled = computed(() => props.disabled === true)

function onPathChange(event: Event): void {
  emit('update:loraPath', (event.target as HTMLSelectElement).value)
}

function onWeightChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(v)) return
  emit('update:loraWeight', v)
}
</script>
