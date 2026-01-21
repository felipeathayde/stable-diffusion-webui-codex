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
- `WanStageLoraField` (component): Stage-level LoRA sha + weight inputs for WAN generation.
- `onShaChange` (function): Emits an updated LoRA sha selection.
- `onWeightChange` (function): Emits an updated LoRA weight value.
-->

<template>
  <div class="gc-row">
    <div class="gc-col gc-col--wide field">
      <label class="label-muted">LoRA (wan22-loras)</label>
      <select class="select-md" :disabled="disabled" :value="loraSha" @change="onShaChange">
        <option value="">None</option>
        <option v-for="opt in choices" :key="opt.sha256" :value="opt.sha256">{{ opt.name }}</option>
      </select>
    </div>
    <div v-if="loraSha" class="gc-col field">
      <label class="label-muted">LoRA weight</label>
      <input class="ui-input" type="number" step="0.05" :disabled="disabled" :value="loraWeight" @change="onWeightChange" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  loraSha: string
  loraWeight: number
  choices: Array<{ name: string; sha256: string }>
  disabled?: boolean
}>()

const emit = defineEmits<{
  (e: 'update:loraSha', value: string): void
  (e: 'update:loraWeight', value: number): void
}>()

const disabled = computed(() => props.disabled === true)

function onShaChange(event: Event): void {
  emit('update:loraSha', (event.target as HTMLSelectElement).value)
}

function onWeightChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(v)) return
  emit('update:loraWeight', v)
}
</script>
