<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Scheduler dropdown selector.
Renders a scheduler dropdown from the supported scheduler list and emits the selected scheduler name.

Symbols (top-level; keep in sync; no ghosts):
- `SchedulerSelector` (component): Scheduler selector component.
- `onChange` (function): Emits `update:modelValue` for the selected scheduler.
-->

<template>
  <div class="form-field">
    <label class="label-muted">{{ labelText }}</label>
    <select class="select-md" :disabled="disabled" :value="modelValue" @change="onChange">
      <option v-if="allowEmpty" value="">{{ emptyLabelText }}</option>
      <option v-for="s in schedulers" :key="s.name" :value="s.name">
        {{ s.label }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SchedulerInfo } from '../api/types'

const props = defineProps<{
  schedulers: SchedulerInfo[]
  modelValue: string
  label?: string
  allowEmpty?: boolean
  emptyLabel?: string
  disabled?: boolean
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Scheduler')
const allowEmpty = computed(() => props.allowEmpty === true)
const emptyLabelText = computed(() => props.emptyLabel ?? 'Automatic')
const disabled = computed(() => props.disabled === true)

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
}
</script>
