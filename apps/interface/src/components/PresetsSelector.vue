<template>
  <div class="form-field">
    <label class="label">{{ labelText }}</label>
    <div class="preset-row">
      <input
        class="ui-input"
        type="text"
        :list="datalistId"
        v-model="name"
        placeholder="Preset name"
      >
      <button class="btn btn-sm btn-outline" type="button" @click="apply">Apply</button>
    </div>
    <datalist :id="datalistId">
      <option v-for="p in names" :key="p" :value="p" />
    </datalist>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

const props = defineProps<{
  names: string[]
  label?: string
  modelValue?: string
  listId?: string
}>()

const emit = defineEmits({
  apply: (value: string) => true,
  'update:modelValue': (value: string) => true,
})

const name = ref(props.modelValue ?? '')

watch(() => props.modelValue, (v) => {
  if (typeof v === 'string' && v !== name.value) name.value = v
})

const labelText = computed(() => props.label ?? 'Presets')
const datalistId = computed(() => props.listId ?? 'preset-list')

function apply(): void {
  const v = name.value.trim()
  if (!v) return
  emit('update:modelValue', v)
  emit('apply', v)
}
</script>
