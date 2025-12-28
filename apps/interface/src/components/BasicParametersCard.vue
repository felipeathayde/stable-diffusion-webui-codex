<template>
  <div class="gen-card">
    <div class="gc-stack">
      <div class="gc-row">
        <SamplerSelector
          class="gc-col"
          :samplers="samplers"
          :modelValue="sampler"
          :label="samplerLabel"
          :allow-empty="allowEmptySampler"
          :emptyLabel="samplerEmptyLabel"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:sampler', v)"
        />
        <SchedulerSelector
          class="gc-col"
          :schedulers="schedulers"
          :modelValue="scheduler"
          :label="schedulerLabel"
          :allow-empty="allowEmptyScheduler"
          :emptyLabel="schedulerEmptyLabel"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:scheduler', v)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Steps"
          :modelValue="steps"
          :min="minSteps"
          :max="maxSteps"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:steps', clampInt(v, minSteps, maxSteps))"
        />
      </div>

      <div class="gc-row">
        <SliderField
          :class="['gc-col', { 'gc-col--wide': resolutionPresets.length === 0 }]"
          :label="widthLabel"
          :modelValue="width"
          :min="minWidth"
          :max="maxWidth"
          :step="widthStep"
          :inputStep="widthInputStep"
          :nudgeStep="widthInputStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:width', clampInt(v, minWidth, maxWidth))"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="width"
              :min="minWidth"
              :max="maxWidth"
              :step="widthInputStep"
              :nudgeStep="widthInputStep"
              inputClass="cdx-input-w-md"
              :disabled="disabled"
              @update:modelValue="(v) => emit('update:width', clampInt(v, minWidth, maxWidth))"
            />
            <button class="btn-swap" type="button" :disabled="disabled" title="Swap width/height" @click="swapWH">
              <span class="btn-swap-icon" aria-hidden="true">⇵</span>
            </button>
          </template>
        </SliderField>

        <SliderField
          :class="['gc-col', { 'gc-col--wide': resolutionPresets.length === 0 }]"
          :label="heightLabel"
          :modelValue="height"
          :min="minHeight"
          :max="maxHeight"
          :step="heightStep"
          :inputStep="heightInputStep"
          :nudgeStep="heightInputStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:height', clampInt(v, minHeight, maxHeight))"
        />

        <div v-if="resolutionPresets.length" class="gc-col gc-col--presets">
          <DimensionPresetsGrid :presets="resolutionPresets" :disabled="disabled" @apply="applyResolutionPreset" />
        </div>
      </div>
      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">{{ seedLabel }}</label>
          <div class="number-with-controls w-full">
            <input class="ui-input ui-input-sm pad-right" type="number" :disabled="disabled" :value="seed" @change="onSeedChange" />
            <div class="stepper">
              <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="emit('random-seed')">🎲</button>
              <button class="step-btn" type="button" :disabled="disabled" title="Reuse seed" @click="emit('reuse-seed')">↺</button>
            </div>
          </div>
        </div>

        <SliderField
          v-if="showCfg"
          class="gc-col gc-col--wide"
          :label="cfgLabel"
          :modelValue="cfgScale"
          :min="minCfg"
          :max="maxCfg"
          :step="cfgStep"
          :inputStep="cfgStep"
          :nudgeStep="cfgStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:cfgScale', clampFloat(v, minCfg, maxCfg))"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SamplerInfo, SchedulerInfo } from '../api/types'

import NumberStepperInput from './ui/NumberStepperInput.vue'
import DimensionPresetsGrid from './ui/DimensionPresetsGrid.vue'
import SliderField from './ui/SliderField.vue'
import SamplerSelector from './SamplerSelector.vue'
import SchedulerSelector from './SchedulerSelector.vue'

const props = withDefaults(defineProps<{
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  width: number
  height: number
  disabled?: boolean

  // Labels
  samplerLabel?: string
  schedulerLabel?: string
  seedLabel?: string
  cfgLabel?: string
  widthLabel?: string
  heightLabel?: string

  // Options
  allowEmptySampler?: boolean
  allowEmptyScheduler?: boolean
  samplerEmptyLabel?: string
  schedulerEmptyLabel?: string

  // Ranges / steps
  minSteps?: number
  maxSteps?: number
  showCfg?: boolean
  minCfg?: number
  maxCfg?: number
  cfgStep?: number
  minWidth?: number
  maxWidth?: number
  minHeight?: number
  maxHeight?: number
  widthStep?: number
  widthInputStep?: number
  heightStep?: number
  heightInputStep?: number

  resolutionPresets?: [number, number][]
}>(), {
  disabled: false,
  samplerLabel: 'Sampler',
  schedulerLabel: 'Scheduler',
  seedLabel: 'Seed',
  cfgLabel: 'CFG',
  widthLabel: 'Width',
  heightLabel: 'Height',
  allowEmptySampler: true,
  allowEmptyScheduler: true,
  samplerEmptyLabel: 'Automatic',
  schedulerEmptyLabel: 'Automatic',
  minSteps: 1,
  maxSteps: 150,
  showCfg: true,
  minCfg: 0,
  maxCfg: 30,
  cfgStep: 0.5,
  minWidth: 64,
  maxWidth: 2048,
  minHeight: 64,
  maxHeight: 2048,
  widthStep: 64,
  widthInputStep: 8,
  heightStep: 64,
  heightInputStep: 8,
  resolutionPresets: () => [],
})

const emit = defineEmits<{
  (e: 'update:sampler', value: string): void
  (e: 'update:scheduler', value: string): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfgScale', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'update:width', value: number): void
  (e: 'update:height', value: number): void
  (e: 'random-seed'): void
  (e: 'reuse-seed'): void
}>()

const showCfg = computed(() => props.showCfg !== false)

const minSteps = computed(() => Number.isFinite(props.minSteps) ? Math.trunc(Number(props.minSteps)) : 1)
const maxSteps = computed(() => Number.isFinite(props.maxSteps) ? Math.trunc(Number(props.maxSteps)) : 150)

const minCfg = computed(() => Number.isFinite(props.minCfg) ? Number(props.minCfg) : 0)
const maxCfg = computed(() => Number.isFinite(props.maxCfg) ? Number(props.maxCfg) : 30)
const cfgStep = computed(() => Number.isFinite(props.cfgStep) ? Number(props.cfgStep) : 0.5)

const minWidth = computed(() => Number.isFinite(props.minWidth) ? Math.trunc(Number(props.minWidth)) : 64)
const maxWidth = computed(() => Number.isFinite(props.maxWidth) ? Math.trunc(Number(props.maxWidth)) : 2048)
const minHeight = computed(() => Number.isFinite(props.minHeight) ? Math.trunc(Number(props.minHeight)) : 64)
const maxHeight = computed(() => Number.isFinite(props.maxHeight) ? Math.trunc(Number(props.maxHeight)) : 2048)

const widthStep = computed(() => Number.isFinite(props.widthStep) ? Math.trunc(Number(props.widthStep)) : 64)
const widthInputStep = computed(() => Number.isFinite(props.widthInputStep) ? Math.trunc(Number(props.widthInputStep)) : 8)
const heightStep = computed(() => Number.isFinite(props.heightStep) ? Math.trunc(Number(props.heightStep)) : 64)
const heightInputStep = computed(() => Number.isFinite(props.heightInputStep) ? Math.trunc(Number(props.heightInputStep)) : 8)

const resolutionPresets = computed(() => (Array.isArray(props.resolutionPresets) ? props.resolutionPresets : []))

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  const n = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, n))
}

function onSeedChange(event: Event): void {
  const raw = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  emit('update:seed', Math.trunc(raw))
}

function swapWH(): void {
  emit('update:width', clampInt(props.height, minWidth.value, maxWidth.value))
  emit('update:height', clampInt(props.width, minHeight.value, maxHeight.value))
}

function applyResolutionPreset(pair: [number, number]): void {
  const w = clampInt(pair[0], minWidth.value, maxWidth.value)
  const h = clampInt(pair[1], minHeight.value, maxHeight.value)
  emit('update:width', w)
  emit('update:height', h)
}
</script>
