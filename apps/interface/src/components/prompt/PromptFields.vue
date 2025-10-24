<template>
  <div class="prompt-fields">
    <div class="panel-section">
      <PromptBox label="Prompt" v-model="innerPrompt" variant="positive" />
    </div>
    <div class="panel-section">
      <PromptBox label="Negative Prompt" v-model="innerNegative" variant="negative" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import PromptBox from './PromptBox.vue'

const props = defineProps<{ prompt: string; negative: string }>()
const emit = defineEmits<{ 'update:prompt': [value: string]; 'update:negative': [value: string] }>()

const innerPrompt = computed({
  get: () => props.prompt,
  set: (value: string) => emit('update:prompt', value),
})

const innerNegative = computed({
  get: () => props.negative,
  set: (value: string) => emit('update:negative', value),
})

// counters moved inside PromptBox
</script>

<!-- styles moved to styles/components/prompt-fields.css -->
