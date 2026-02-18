<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Slider field with label, optional hover tooltip, and numeric input.
Composes a range slider with a numeric input (default `NumberStepperInput`) and can render an elegant hover/focus tooltip near the label.

Symbols (top-level; keep in sync; no ghosts):
- `SliderField` (component): Slider + numeric input field that emits `update:modelValue`.
- `hasTooltip` (computed): Indicates whether tooltip content was provided for the label.
- `onRangeInput` (function): Range slider input handler that emits a numeric value.
-->

<template>
  <div class="cdx-slider-field">
    <div class="cdx-slider-field__head">
      <HoverTooltip
        v-if="hasTooltip"
        class="cdx-slider-field__label-tooltip"
        :title="tooltipTitle"
        :content="tooltip ?? ''"
      >
        <span class="cdx-slider-field__label-trigger">
          <span class="cdx-slider-field__label">{{ label }}</span>
          <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
        </span>
      </HoverTooltip>
      <span v-else class="cdx-slider-field__label">{{ label }}</span>
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
import HoverTooltip from './HoverTooltip.vue'

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
  tooltip?: string | readonly string[]
  tooltipTitle?: string
}>(), {
  step: 1,
  disabled: false,
  showButtons: true,
  numberUpdateOnInput: false,
  numberSize: 'sm',
  inputClass: '',
  tooltipTitle: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

const minAttr = computed(() => props.min ?? 0)
const maxAttr = computed(() => props.max ?? 100)
const stepAttr = computed(() => props.step ?? 1)
const numberStep = computed(() => props.inputStep ?? props.step ?? 1)
const hasTooltip = computed(() => {
  if (Array.isArray(props.tooltip)) return props.tooltip.some((line) => line.trim().length > 0)
  return typeof props.tooltip === 'string' && props.tooltip.trim().length > 0
})

function onRangeInput(event: Event): void {
  const raw = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  emit('update:modelValue', raw)
}
</script>
