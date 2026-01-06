<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Numeric input with optional +/- stepper controls.
Normalizes numeric input, clamps to min/max, snaps to step, and emits updates via `update:modelValue`.

Symbols (top-level; keep in sync; no ghosts):
- `NumberStepperInput` (component): Numeric input + optional stepper buttons with clamping/quantization.
- `commit` (function): Parses and commits an input value (clamp + quantize) to the model.
- `inc` (function): Increments the current value by `nudgeStep` and emits the result.
- `dec` (function): Decrements the current value by `nudgeStep` and emits the result.
-->

<template>
  <div class="cdx-stepper-input" :data-size="size">
    <input
      class="ui-input cdx-stepper-input__input"
      :class="[inputClass, { 'cdx-stepper-input__input--buttons': showButtons }]"
      type="number"
      :min="minAttr"
      :max="maxAttr"
      :step="stepAttr"
      :disabled="disabled"
      :value="modelValue"
      @input="onInput"
      @change="onChange"
    />
    <div v-if="showButtons" class="cdx-stepper-input__controls">
      <button class="cdx-stepper-input__btn" type="button" title="Increase" :disabled="disabled" @click="inc">+</button>
      <button class="cdx-stepper-input__btn" type="button" title="Decrease" :disabled="disabled" @click="dec">−</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: number
  min?: number
  max?: number
  step?: number
  nudgeStep?: number
  disabled?: boolean
  showButtons?: boolean
  updateOnInput?: boolean
  size?: 'sm' | 'md'
  inputClass?: string
}>(), {
  step: 1,
  disabled: false,
  showButtons: true,
  updateOnInput: false,
  size: 'sm',
  inputClass: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: number): void
}>()

const minAttr = computed(() => Number.isFinite(props.min) ? props.min : undefined)
const maxAttr = computed(() => Number.isFinite(props.max) ? props.max : undefined)
const stepAttr = computed(() => Number.isFinite(props.step) ? props.step : 1)
const nudgeStep = computed(() => Number.isFinite(props.nudgeStep) ? Number(props.nudgeStep) : Number(stepAttr.value))

function decimalsForStep(step: number): number {
  if (!Number.isFinite(step) || step <= 0) return 0
  const s = String(step)
  if (s.includes('e-')) {
    const [, exp] = s.split('e-')
    return Math.max(0, Math.trunc(Number(exp)))
  }
  const dot = s.indexOf('.')
  return dot === -1 ? 0 : Math.max(0, s.length - dot - 1)
}

function roundDecimals(value: number, decimals: number): number {
  if (!Number.isFinite(value)) return value
  if (decimals <= 0) return Math.round(value)
  return Number(value.toFixed(decimals))
}

function clamp(value: number): number {
  let v = value
  const min = minAttr.value
  const max = maxAttr.value
  if (min !== undefined) v = Math.max(min, v)
  if (max !== undefined) v = Math.min(max, v)
  return v
}

function quantize(value: number, step: number): number {
  if (!Number.isFinite(step) || step <= 0) return value
  const snapped = Math.round(value / step) * step
  return roundDecimals(snapped, decimalsForStep(step))
}

function commit(raw: string): void {
  const parsed = Number(raw)
  if (!Number.isFinite(parsed)) return

  const step = Number(stepAttr.value)
  const clamped = clamp(parsed)
  const next = quantize(clamped, step)
  emit('update:modelValue', next)
}

function onInput(e: Event): void {
  if (!props.updateOnInput) return
  commit((e.target as HTMLInputElement).value)
}

function onChange(e: Event): void {
  commit((e.target as HTMLInputElement).value)
}

function inc(): void {
  const base = Number.isFinite(props.modelValue) ? Number(props.modelValue) : 0
  const step = nudgeStep.value
  const next = quantize(clamp(base + step), Number(stepAttr.value))
  emit('update:modelValue', next)
}

function dec(): void {
  const base = Number.isFinite(props.modelValue) ? Number(props.modelValue) : 0
  const step = nudgeStep.value
  const next = quantize(clamp(base - step), Number(stepAttr.value))
  emit('update:modelValue', next)
}
</script>
