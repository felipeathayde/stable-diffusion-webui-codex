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
          <SliderField
            label="Sampling steps"
            :modelValue="steps"
            :min="minSteps"
            :max="maxSteps"
            :step="1"
            :inputStep="1"
            :nudgeStep="1"
            inputClass="cdx-input-w-md"
            @update:modelValue="(v) => emit('update:steps', v)"
          />
        </div>
      </div>

      <!-- Row 2: Width + Batches -->
      <div class="gc-row">
        <div class="gc-col gc-col--wide">
          <SliderField
            label="Width"
            :modelValue="width"
            :min="minWidth"
            :max="maxWidth"
            :step="64"
            :inputStep="8"
            :nudgeStep="8"
            inputClass="cdx-input-w-md"
            @update:modelValue="(v) => emit('update:width', v)"
          >
            <template #right>
              <NumberStepperInput
                :modelValue="width"
                :min="minWidth"
                :max="maxWidth"
                :step="8"
                :nudgeStep="8"
                inputClass="cdx-input-w-md"
                @update:modelValue="(v) => emit('update:width', v)"
              />
              <button class="btn-swap" type="button" title="Swap width/height" @click="swapWH">⇵</button>
            </template>
          </SliderField>
        </div>
        <div class="gc-col gc-col--compact">
          <div class="right-row">
            <div class="field compact">
              <label class="label-muted">Batch count</label>
              <div class="number-with-controls">
                <input class="ui-input ui-input-sm cdx-input-w-sm pad-right" type="number" min="1" :value="batchCount" @change="onBatchCountChange" />
                <div class="stepper">
                  <button class="step-btn" type="button" title="Increase" @click="batchCountInc">+</button>
                  <button class="step-btn" type="button" title="Decrease" @click="batchCountDec">−</button>
                </div>
              </div>
            </div>
            <div class="field compact">
              <label class="label-muted">Batch size</label>
              <div class="number-with-controls">
                <input class="ui-input ui-input-sm cdx-input-w-sm pad-right" type="number" min="1" :value="batchSize" @change="onBatchSizeChange" />
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
          <SliderField
            label="Height"
            :modelValue="height"
            :min="minHeight"
            :max="maxHeight"
            :step="64"
            :inputStep="8"
            :nudgeStep="8"
            inputClass="cdx-input-w-md"
            @update:modelValue="(v) => emit('update:height', v)"
          />
        </div>
      </div>
    </div>

    <!-- Seed full width at bottom -->
    <div class="gc-seed">
      <div class="gc-footer">
        <SliderField
          v-if="showCfg"
          class="cfg-field"
          :label="cfgLabel"
          :modelValue="cfgScale"
          :min="0"
          :max="30"
          :step="0.5"
          :inputStep="0.5"
          :nudgeStep="0.5"
          inputClass="cdx-input-w-md"
          @update:modelValue="(v) => emit('update:cfgScale', v)"
        />

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
import NumberStepperInput from './ui/NumberStepperInput.vue'
import SliderField from './ui/SliderField.vue'

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
