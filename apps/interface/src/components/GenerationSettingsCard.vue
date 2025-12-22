<template>
  <div class="gen-card">
    <div class="gc-stack">
      <!-- Row 1: Sampler/Scheduler + Steps -->
      <div class="gc-row">
        <div class="gc-col gc-col--wide">
          <div class="two-up">
            <div class="field">
              <label class="label-muted">Sampler</label>
              <select class="select-md" :value="sampler" @change="onSamplerChange">
                <option v-for="s in samplers" :key="s.name" :value="s.name">{{ s.name }}</option>
              </select>
            </div>
            <div class="field">
              <label class="label-muted">Scheduler</label>
              <select class="select-md" :value="scheduler" @change="onSchedulerChange">
                <option v-for="s in schedulers" :key="s.name" :value="s.name">{{ s.label }}</option>
              </select>
            </div>
          </div>
        </div>
        <div class="gc-col gc-col--wide">
          <div class="field">
            <label class="label-muted">Sampling steps</label>
            <div class="row-inline">
              <input class="slider slider-grow" type="range" :min="minSteps" :max="maxSteps" :step="1" :value="steps" @input="onStepsChange" />
              <div class="number-with-controls ml-steps">
                <input class="ui-input ui-input-sm w-step" type="number" :min="minSteps" :max="maxSteps" step="1" :value="steps" @change="onStepsNumber" />
                <div class="stepper">
                  <button class="step-btn" type="button" title="Increase" @click="stepsInc">+</button>
                  <button class="step-btn" type="button" title="Decrease" @click="stepsDec">−</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Row 2: Width + Batches -->
      <div class="gc-row">
        <div class="gc-col gc-col--wide">
          <div class="field">
            <label class="label-muted">Width</label>
            <div class="row-inline">
              <div class="number-with-controls">
                <input class="ui-input ui-input-sm w-width pad-right" type="number" :min="minWidth" :max="maxWidth" step="8" :value="width" @change="onWidthChange" />
                <div class="stepper">
                  <button class="step-btn" type="button" title="Increase" @click="widthInc">+</button>
                  <button class="step-btn" type="button" title="Decrease" @click="widthDec">−</button>
                </div>
              </div>
              <input class="slider slider-grow" type="range" :min="minWidth" :max="maxWidth" step="64" :value="width" @input="onWidthRange" />
              <button class="btn-swap" type="button" title="Swap width/height" @click="swapWH">⇵</button>
            </div>
          </div>
        </div>
        <div class="gc-col gc-col--compact">
          <div class="right-row">
            <div class="field compact">
              <label class="label-muted">Batch count</label>
              <div class="number-with-controls">
                <input class="ui-input ui-input-sm w-batch pad-right" type="number" min="1" :value="batchCount" @change="onBatchCountChange" />
                <div class="stepper">
                  <button class="step-btn" type="button" title="Increase" @click="batchCountInc">+</button>
                  <button class="step-btn" type="button" title="Decrease" @click="batchCountDec">−</button>
                </div>
              </div>
            </div>
            <div class="field compact">
              <label class="label-muted">Batch size</label>
              <div class="number-with-controls">
                <input class="ui-input ui-input-sm w-batch pad-right" type="number" min="1" :value="batchSize" @change="onBatchSizeChange" />
                <div class="stepper">
                  <button class="step-btn" type="button" title="Increase" @click="batchSizeInc">+</button>
                  <button class="step-btn" type="button" title="Decrease" @click="batchSizeDec">−</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- Row 3: Height -->
      <div class="gc-row">
        <div class="gc-col gc-col--wide">
          <div class="field">
            <label class="label-muted">Height</label>
            <div class="row-inline">
              <div class="number-with-controls">
                <input class="ui-input ui-input-sm w-height pad-right" type="number" :min="minHeight" :max="maxHeight" step="8" :value="height" @change="onHeightChange" />
                <div class="stepper">
                  <button class="step-btn" type="button" title="Increase" @click="heightInc">+</button>
                  <button class="step-btn" type="button" title="Decrease" @click="heightDec">−</button>
                </div>
              </div>
              <input class="slider slider-grow" type="range" :min="minHeight" :max="maxHeight" step="64" :value="height" @input="onHeightRange" />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Seed full width at bottom -->
    <div class="gc-seed">
      <div class="gc-footer">
        <div v-if="showCfg" class="field cfg-field">
          <label class="label-muted">{{ cfgLabel }}</label>
          <div class="row-inline">
            <input class="slider slider-grow" type="range" min="0" max="30" step="0.5" :value="cfgScale" @input="onCfgRange" />
            <div class="number-with-controls">
              <input class="ui-input ui-input-sm w-cfg pad-right" type="number" :min="0" :max="30" step="0.5" :value="cfgScale" @change="onCfgChange" />
              <div class="stepper">
                <button class="step-btn" type="button" title="Increase" @click="cfgInc">+</button>
                <button class="step-btn" type="button" title="Decrease" @click="cfgDec">−</button>
              </div>
            </div>
          </div>
        </div>

        <div class="field seed-group">
          <label class="label-muted">Seed</label>
          <div class="number-with-controls w-full">
            <input class="ui-input pad-right" type="number" :value="seed" @change="onSeedChange" />
            <div class="stepper">
              <button class="step-btn" type="button" @click="emit('random-seed')" title="Random seed">🎲</button>
              <button class="step-btn" type="button" @click="emit('reuse-seed')" title="Reuse seed">↺</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SamplerInfo, SchedulerInfo } from '../api/types'

const aspectOptions = [
  // aspect options removed as per design decision
]

const props = defineProps<{
  sampler: string
  scheduler: string
  steps: number
  width: number
  height: number
  cfgScale: number
  seed: number
  batchSize: number
  batchCount: number
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  minSteps?: number
  maxSteps?: number
  minWidth?: number
  minHeight?: number
  maxWidth?: number
  maxHeight?: number
  // Conditional visibility
  showCfg?: boolean
  cfgLabel?: string
}>()

const showCfg = computed(() => props.showCfg ?? true)
const cfgLabel = computed(() => props.cfgLabel ?? 'CFG Scale')

const emit = defineEmits({
  'update:sampler': (value: string) => true,
  'update:scheduler': (value: string) => true,
  'update:steps': (value: number) => true,
  'update:width': (value: number) => true,
  'update:height': (value: number) => true,
  'update:cfgScale': (value: number) => true,
  'update:seed': (value: number) => true,
  'update:batchSize': (value: number) => true,
  'update:batchCount': (value: number) => true,
  'random-seed': () => true,
  'reuse-seed': () => true,
})

const minSteps = computed(() => props.minSteps ?? 1)
const maxSteps = computed(() => props.maxSteps ?? 150)
const minWidth = computed(() => props.minWidth ?? 64)
const minHeight = computed(() => props.minHeight ?? 64)
const maxWidth = computed(() => props.maxWidth ?? 2048)
const maxHeight = computed(() => props.maxHeight ?? 2048)

function onSamplerChange(event: Event): void {
  emit('update:sampler', (event.target as HTMLSelectElement).value)
}

function onSchedulerChange(event: Event): void {
  emit('update:scheduler', (event.target as HTMLSelectElement).value)
}

function onStepsChange(event: Event): void {
  emit('update:steps', Number((event.target as HTMLInputElement).value))
}
function onStepsNumber(event: Event): void {
  let v = Number((event.target as HTMLInputElement).value)
  if (Number.isNaN(v)) v = minSteps.value
  v = Math.max(minSteps.value, Math.min(maxSteps.value, v))
  emit('update:steps', v)
}
function stepsInc(): void {
  const v = Math.min(maxSteps.value, Number(props.steps) + 1)
  emit('update:steps', v)
}
function stepsDec(): void {
  const v = Math.max(minSteps.value, Number(props.steps) - 1)
  emit('update:steps', v)
}

function onWidthChange(event: Event): void {
  emit('update:width', Math.max(minWidth.value, Number((event.target as HTMLInputElement).value)))
}
function onWidthRange(event: Event): void {
  emit('update:width', Number((event.target as HTMLInputElement).value))
}
function widthInc(): void {
  const step = 8
  const v = Math.min(maxWidth.value, Number(props.width) + step)
  emit('update:width', v)
}
function widthDec(): void {
  const step = 8
  const v = Math.max(minWidth.value, Number(props.width) - step)
  emit('update:width', v)
}

function onHeightChange(event: Event): void {
  emit('update:height', Math.max(minHeight.value, Number((event.target as HTMLInputElement).value)))
}
function onHeightRange(event: Event): void {
  emit('update:height', Number((event.target as HTMLInputElement).value))
}
function heightInc(): void {
  const step = 8
  const v = Math.min(maxHeight.value, Number(props.height) + step)
  emit('update:height', v)
}
function heightDec(): void {
  const step = 8
  const v = Math.max(minHeight.value, Number(props.height) - step)
  emit('update:height', v)
}

function onCfgRange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:cfgScale', Number.isNaN(v) ? props.cfgScale : v)
}

function onCfgChange(event: Event): void {
  emit('update:cfgScale', Number((event.target as HTMLInputElement).value))
}
function cfgInc(): void {
  const step = 0.5
  const v = Math.min(30, Number(props.cfgScale) + step)
  emit('update:cfgScale', Number(v.toFixed(2)))
}
function cfgDec(): void {
  const step = 0.5
  const v = Math.max(0, Number(props.cfgScale) - step)
  emit('update:cfgScale', Number(v.toFixed(2)))
}
function swapWH(): void {
  emit('update:width', props.height)
  emit('update:height', props.width)
}

function onSeedChange(event: Event): void {
  emit('update:seed', Number((event.target as HTMLInputElement).value))
}

function onBatchSizeChange(event: Event): void {
  emit('update:batchSize', Math.max(1, Number((event.target as HTMLInputElement).value)))
}

function onBatchCountChange(event: Event): void {
  emit('update:batchCount', Math.max(1, Number((event.target as HTMLInputElement).value)))
}

function batchCountInc(): void {
  emit('update:batchCount', Math.max(1, Number(props.batchCount) + 1))
}
function batchCountDec(): void {
  emit('update:batchCount', Math.max(1, Number(props.batchCount) - 1))
}
function batchSizeInc(): void {
  emit('update:batchSize', Math.max(1, Number(props.batchSize) + 1))
}
function batchSizeDec(): void {
  emit('update:batchSize', Math.max(1, Number(props.batchSize) - 1))
}
</script>

<!-- styles moved to styles/components/generation-settings-card.css -->
