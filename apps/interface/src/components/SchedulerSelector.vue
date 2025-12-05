<template>
  <div class="form-field">
    <label class="label">{{ labelText }}</label>
    <select class="select-md" :value="modelValue" @change="onChange">
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
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Scheduler')

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
}
</script>

