<template>
  <div class="cdx-stack">
    <BasicParametersCard
      :samplers="samplers"
      :schedulers="schedulers"
      :sampler="sampler"
      :scheduler="scheduler"
      :steps="steps"
      :cfgScale="cfgScale"
      :seed="seed"
      :width="width"
      :height="height"
      :disabled="disabled"
      :minSteps="minSteps"
      :maxSteps="maxSteps"
      :minWidth="minWidth"
      :maxWidth="maxWidth"
      :minHeight="minHeight"
      :maxHeight="maxHeight"
      :showCfg="showCfg"
      :cfgLabel="cfgLabel"
      @update:sampler="(v) => emit('update:sampler', v)"
      @update:scheduler="(v) => emit('update:scheduler', v)"
      @update:steps="(v) => emit('update:steps', v)"
      @update:cfgScale="(v) => emit('update:cfgScale', v)"
      @update:seed="(v) => emit('update:seed', v)"
      @update:width="(v) => emit('update:width', v)"
      @update:height="(v) => emit('update:height', v)"
      @random-seed="() => emit('random-seed')"
      @reuse-seed="() => emit('reuse-seed')"
    />

    <BatchSettingsCard
      :batchCount="batchCount"
      :batchSize="batchSize"
      :disabled="disabled"
      @update:batchCount="(v) => emit('update:batchCount', v)"
      @update:batchSize="(v) => emit('update:batchSize', v)"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SamplerInfo, SchedulerInfo } from '../api/types'

import BasicParametersCard from './BasicParametersCard.vue'
import BatchSettingsCard from './BatchSettingsCard.vue'

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
  disabled?: boolean
}>()

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

const showCfg = computed(() => props.showCfg ?? true)
const cfgLabel = computed(() => props.cfgLabel ?? 'CFG Scale')
const disabled = computed(() => props.disabled === true)

const minSteps = computed(() => props.minSteps ?? 1)
const maxSteps = computed(() => props.maxSteps ?? 150)
const minWidth = computed(() => props.minWidth ?? 64)
const minHeight = computed(() => props.minHeight ?? 64)
const maxWidth = computed(() => props.maxWidth ?? 2048)
const maxHeight = computed(() => props.maxHeight ?? 2048)

const {
  sampler,
  scheduler,
  steps,
  width,
  height,
  cfgScale,
  seed,
  batchSize,
  batchCount,
  samplers,
  schedulers,
} = props
</script>
