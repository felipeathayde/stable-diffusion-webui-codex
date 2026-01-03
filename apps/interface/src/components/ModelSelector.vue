<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared checkpoint/model selector dropdown.
Renders a model list dropdown and emits updates (and a `change` event) when a model is selected.

Symbols (top-level; keep in sync; no ghosts):
- `ModelSelector` (component): Model selector dropdown component.
- `onChange` (function): Emits `update:modelValue` and `change` for the selected model title.
-->

<template>
  <div class="form-field">
    <label class="label">{{ labelText }}</label>
    <select class="select-md" :value="modelValue" @change="onChange">
      <option value="" disabled>Select model</option>
      <option v-for="m in models" :key="m.title" :value="m.title">
        {{ m.title }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ModelInfo } from '../api/types'

const props = defineProps<{
  models: ModelInfo[]
  modelValue: string
  label?: string
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
  change: (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Model Checkpoint')

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
  emit('change', value)
}
</script>
