<template>
  <div class="cdx-res-presets" aria-label="Resolution presets">
    <button
      v-for="(p, i) in presets"
      :key="`${p[0]}x${p[1]}-${i}`"
      class="btn btn-sm btn-outline"
      type="button"
      :disabled="disabled"
      @click="emit('apply', p)"
    >
      {{ p[0] }}×{{ p[1] }}
    </button>
  </div>
</template>

<script setup lang="ts">
import { toRefs } from 'vue'

const props = withDefaults(defineProps<{
  presets: [number, number][]
  disabled?: boolean
}>(), {
  presets: () => [],
  disabled: false,
})

const emit = defineEmits<{
  (e: 'apply', value: [number, number]): void
}>()

const { disabled, presets } = toRefs(props)
</script>
