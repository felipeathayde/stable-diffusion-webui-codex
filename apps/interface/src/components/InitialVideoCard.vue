<template>
  <div class="panel-section">
    <label class="label-muted">{{ label }}</label>
    <div class="init-picker">
      <div class="toolbar">
        <input class="ui-input" :disabled="disabled" type="file" :accept="accept" @change="onFile" />
        <button class="btn btn-sm btn-ghost" type="button" :disabled="disabled || !hasVideo" @click="$emit('clear')">Remove</button>
      </div>
      <div v-if="src" class="init-preview">
        <video :src="src" controls />
      </div>
      <p v-else class="caption">{{ placeholder }}</p>
      <slot name="footer" />
    </div>
  </div>
</template>

<script setup lang="ts">
const props = withDefaults(defineProps<{
  label?: string
  accept?: string
  src?: string
  hasVideo?: boolean
  disabled?: boolean
  placeholder?: string
}>(), {
  label: 'Input Video',
  accept: 'video/*',
  src: '',
  hasVideo: false,
  disabled: false,
  placeholder: 'Select a video to start.',
})

const emit = defineEmits<{ (e: 'set', file: File): void; (e: 'clear'): void }>()

function onFile(e: Event): void {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('set', file)
  input.value = ''
}
</script>

<!-- uses .init-picker styles from src/styles.css -->
