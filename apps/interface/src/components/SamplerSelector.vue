<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Sampler dropdown selector.
Renders a sampler dropdown from the supported sampler list and emits the selected sampler name.

Symbols (top-level; keep in sync; no ghosts):
- `SamplerSelector` (component): Sampler selector component.
- `onChange` (function): Emits `update:modelValue` for the selected sampler.
-->

<template>
  <div class="form-field">
    <label class="label-muted">{{ labelText }}</label>
    <select class="select-md" :disabled="disabled" :value="modelValue" @change="onChange">
      <option v-if="allowEmpty" value="">{{ emptyLabelText }}</option>
      <option v-for="s in samplers" :key="s.name" :value="s.name">
        {{ s.label ?? s.name }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SamplerInfo } from '../api/types'

const props = defineProps<{
  samplers: SamplerInfo[]
  modelValue: string
  label?: string
  allowEmpty?: boolean
  emptyLabel?: string
  disabled?: boolean
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Sampler')
const allowEmpty = computed(() => props.allowEmpty === true)
const emptyLabelText = computed(() => props.emptyLabel ?? 'Select')
const disabled = computed(() => props.disabled === true)

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
}
</script>
