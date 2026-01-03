<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Slider field with label and numeric input.
Composes a range slider with a numeric input (default `NumberStepperInput`) and emits updates for reactive settings and generation parameters.

Symbols (top-level; keep in sync; no ghosts):
- `SliderField` (component): Slider + numeric input field that emits `update:modelValue`.
- `onRangeInput` (function): Range slider input handler that emits a numeric value.
-->

<template>
  <div class="cdx-slider-field">
    <div class="cdx-slider-field__head">
      <label class="cdx-slider-field__label">{{ label }}</label>
      <div class="cdx-slider-field__right">
        <slot name="right">
          <NumberStepperInput
            :modelValue="modelValue"
            :min="min"
            :max="max"
            :step="numberStep"
            :nudgeStep="nudgeStep"
            :disabled="disabled"
            :showButtons="showButtons"
            :updateOnInput="numberUpdateOnInput"
            :size="numberSize"
            :inputClass="inputClass"
            @update:modelValue="(v) => emit('update:modelValue', v)"
          />
        </slot>
      </div>
    </div>
    <input
      class="slider cdx-slider-field__slider"
      type="range"
      :min="minAttr"
      :max="maxAttr"
      :step="stepAttr"
      :disabled="disabled"
      :value="modelValue"
      @input="onRangeInput"
    />
    <div v-if="$slots.below" class="cdx-slider-field__below">
      <slot name="below" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import NumberStepperInput from './NumberStepperInput.vue'

const props = withDefaults(defineProps<{
  label: string
  modelValue: number
  min?: number
  max?: number
  step?: number
  inputStep?: number
  nudgeStep?: number
  disabled?: boolean
  showButtons?: boolean
  numberUpdateOnInput?: boolean
  numberSize?: 'sm' | 'md'
  inputClass?: string
}>(), {
  step: 1,
  disabled: false,
  showButtons: true,
  numberUpdateOnInput: false,
  numberSize: 'sm',
  inputClass: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

const minAttr = computed(() => props.min ?? 0)
const maxAttr = computed(() => props.max ?? 100)
const stepAttr = computed(() => props.step ?? 1)
const numberStep = computed(() => props.inputStep ?? props.step ?? 1)

function onRangeInput(event: Event): void {
  const raw = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  emit('update:modelValue', raw)
}
</script>
