<template>
  <div class="prompt-box" :data-variant="variant">
    <div class="prompt-header">
      <span class="label-muted">{{ label }}</span>
    </div>
    <div class="prompt-editor-wrap">
      <span v-if="count > 0" class="ear-badge">{{ count }} chars</span>
      <PromptEditor v-model="inner" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import PromptEditor from './PromptEditor.vue'

const props = defineProps<{ label: string; modelValue: string; variant?: 'positive'|'negative' }>()
const emit = defineEmits<{ (e:'update:modelValue', v:string): void }>()

const inner = computed({
  get: () => props.modelValue || '',
  set: (v: string) => emit('update:modelValue', v),
})

const count = computed(() => inner.value.length)
</script>

<!-- styles moved to styles/components/prompt-box.css -->
