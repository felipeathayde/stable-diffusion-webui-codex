<template>
  <div class="form-field">
    <label class="label">{{ labelText }}</label>
    <select class="select-md" :value="modelValue" @change="onChange">
      <option value="" disabled>Select model</option>
      <option v-for="m in models" :key="m.title" :value="m.title">
        {{ m.title }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { ModelInfo } from '../api/types'

const props = defineProps<{
  models: ModelInfo[]
  modelValue: string
  label?: string
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
  change: (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Model Checkpoint')

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
  emit('change', value)
}
</script>

