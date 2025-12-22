<template>
  <div class="prompt-box" :data-variant="variant">
    <div class="prompt-header">
      <span class="label-muted">{{ label }}</span>
    </div>
    <div class="prompt-editor-wrap">
      <span v-if="tokenCount > 0" class="ear-badge">{{ tokenCount }} tok</span>
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

function countTokens(text: string): number {
  const raw = String(text || '').trim()
  if (!raw) return 0
  return raw.split(/\s+/).filter(Boolean).length
}

const tokenCount = computed(() => countTokens(inner.value))
</script>

<!-- styles moved to styles/components/prompt-box.css -->
