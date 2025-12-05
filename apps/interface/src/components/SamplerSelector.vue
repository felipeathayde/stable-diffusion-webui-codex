<template>
  <div class="form-field">
    <label class="label">{{ labelText }}</label>
    <select class="select-md" :value="modelValue" @change="onChange">
      <option v-for="s in samplers" :key="s.name" :value="s.name">
        {{ s.label ?? s.name }}
      </option>
    </select>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { SamplerInfo } from '../api/types'

const props = defineProps<{
  samplers: SamplerInfo[]
  modelValue: string
  label?: string
}>()

const emit = defineEmits({
  'update:modelValue': (value: string) => true,
})

const labelText = computed(() => props.label ?? 'Sampler')

function onChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:modelValue', value)
}
</script>

