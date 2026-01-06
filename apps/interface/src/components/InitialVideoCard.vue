<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Initial video file picker for vid2vid workflows.
Provides a file input, preview, and remove action and emits the selected `File` back to the parent.

Symbols (top-level; keep in sync; no ghosts):
- `InitialVideoCard` (component): Initial video picker panel.
- `onFile` (function): Handles file selection and emits `set`/clears the input value.
-->

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
