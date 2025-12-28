<template>
  <div class="caption cdx-chips-row">
    <span v-for="(chip, i) in chips" :key="i" class="chip">{{ chip }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  text: string
  separator?: string
}>(), {
  separator: '·',
})

const chips = computed(() => {
  return String(props.text ?? '')
    .split(props.separator)
    .map((s) => s.trim())
    .filter(Boolean)
})
</script>
