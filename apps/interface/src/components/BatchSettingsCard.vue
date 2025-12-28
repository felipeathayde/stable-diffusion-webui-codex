<template>
  <div class="gen-card">
    <div class="gc-row">
      <div class="gc-col gc-col--compact field">
        <label class="label-muted">Batch count</label>
        <NumberStepperInput
          :modelValue="batchCount"
          :min="minBatchCount"
          :max="maxBatchCount"
          :step="1"
          :nudgeStep="1"
          size="sm"
          inputClass="cdx-input-w-sm"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:batchCount', clampInt(v, minBatchCount, maxBatchCount))"
        />
      </div>
      <div class="gc-col gc-col--compact field">
        <label class="label-muted">Batch size</label>
        <NumberStepperInput
          :modelValue="batchSize"
          :min="minBatchSize"
          :max="maxBatchSize"
          :step="1"
          :nudgeStep="1"
          size="sm"
          inputClass="cdx-input-w-sm"
          :disabled="disabled"
          @update:modelValue="(v) => emit('update:batchSize', clampInt(v, minBatchSize, maxBatchSize))"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'

const props = withDefaults(defineProps<{
  batchCount: number
  batchSize: number
  disabled?: boolean
  minBatchCount?: number
  maxBatchCount?: number
  minBatchSize?: number
  maxBatchSize?: number
}>(), {
  disabled: false,
  minBatchCount: 1,
  maxBatchCount: 999,
  minBatchSize: 1,
  maxBatchSize: 999,
})

const emit = defineEmits<{
  (e: 'update:batchCount', value: number): void
  (e: 'update:batchSize', value: number): void
}>()

const minBatchCount = computed(() => Number.isFinite(props.minBatchCount) ? Math.trunc(Number(props.minBatchCount)) : 1)
const maxBatchCount = computed(() => Number.isFinite(props.maxBatchCount) ? Math.trunc(Number(props.maxBatchCount)) : 999)
const minBatchSize = computed(() => Number.isFinite(props.minBatchSize) ? Math.trunc(Number(props.minBatchSize)) : 1)
const maxBatchSize = computed(() => Number.isFinite(props.maxBatchSize) ? Math.trunc(Number(props.maxBatchSize)) : 999)

function clampInt(value: number, min: number, max: number): number {
  const n = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, n))
}
</script>

