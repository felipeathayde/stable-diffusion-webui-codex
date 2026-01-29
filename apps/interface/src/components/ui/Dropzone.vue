<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Small drag-and-drop file picker primitive.
Provides a keyboard-accessible dropzone that emits selected files and reports rejections without owning any upload/storage logic.

Symbols (top-level; keep in sync; no ghosts):
- `Dropzone` (component): Presentational dropzone that emits `select`/`rejected`.
- `openPicker` (function): Opens the native file picker.
- `acceptsFile` (function): Checks whether a file matches the `accept` patterns.
- `handleFiles` (function): Normalizes and emits selected files.
-->

<template>
  <div
    class="cdx-dropzone"
    :class="{ 'is-dragover': isDragover, 'is-disabled': props.disabled }"
    role="button"
    tabindex="0"
    @click="openPicker"
    @keydown.enter.prevent="openPicker"
    @keydown.space.prevent="openPicker"
    @dragenter.prevent="onDragEnter"
    @dragover.prevent="onDragOver"
    @dragleave.prevent="onDragLeave"
    @drop.prevent="onDrop"
  >
    <input
      ref="inputRef"
      class="cdx-dropzone__input"
      type="file"
      :accept="props.accept"
      :multiple="props.multiple"
      :disabled="props.disabled"
      @change="onInputChange"
    />

    <slot>
      <div class="cdx-dropzone__body">
        <div class="cdx-dropzone__title">{{ props.label }}</div>
        <div v-if="props.hint" class="cdx-dropzone__hint">{{ props.hint }}</div>
      </div>
    </slot>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

const props = withDefaults(defineProps<{
  accept?: string
  multiple?: boolean
  disabled?: boolean
  label?: string
  hint?: string
}>(), {
  accept: '',
  multiple: false,
  disabled: false,
  label: 'Drop a file here, or click to browse',
  hint: '',
})

const emit = defineEmits<{
  (e: 'select', files: File[]): void
  (e: 'rejected', payload: { reason: string; files: File[] }): void
}>()

const inputRef = ref<HTMLInputElement | null>(null)
const dragDepth = ref(0)

const acceptTokens = computed(() => {
  const raw = String(props.accept || '').trim()
  if (!raw) return []
  return raw.split(',').map(t => t.trim()).filter(Boolean)
})

const isDragover = computed(() => dragDepth.value > 0)

function openPicker(): void {
  if (props.disabled) return
  inputRef.value?.click()
}

function acceptsFile(file: File): boolean {
  const tokens = acceptTokens.value
  if (!tokens.length) return true

  const name = String(file.name || '')
  const lowerName = name.toLowerCase()
  const type = String(file.type || '').toLowerCase()

  for (const token of tokens) {
    const t = token.toLowerCase()
    if (t.startsWith('.')) {
      if (lowerName.endsWith(t)) return true
      continue
    }
    if (t.endsWith('/*')) {
      const prefix = t.slice(0, -1) // keep trailing slash
      if (type.startsWith(prefix)) return true
      continue
    }
    if (t.includes('/')) {
      if (type === t) return true
      continue
    }
  }

  return false
}

function handleFiles(files: File[]): void {
  if (props.disabled) return

  const accepted = files.filter(acceptsFile)
  if (accepted.length === 0) {
    emit('rejected', { reason: 'No files matched accept filter.', files })
    return
  }

  if (!props.multiple) {
    emit('select', [accepted[0]])
    return
  }
  emit('select', accepted)
}

function onInputChange(event: Event): void {
  const el = event.target as HTMLInputElement
  const list = el.files ? Array.from(el.files) : []
  // Reset so selecting the same file twice still triggers `change`.
  el.value = ''
  handleFiles(list)
}

function onDragEnter(): void {
  if (props.disabled) return
  dragDepth.value += 1
}

function onDragOver(): void {
  if (props.disabled) return
  // Keep `dragDepth > 0` while hovering within the zone.
  dragDepth.value = Math.max(1, dragDepth.value)
}

function onDragLeave(): void {
  if (props.disabled) return
  dragDepth.value = Math.max(0, dragDepth.value - 1)
}

function onDrop(event: DragEvent): void {
  if (props.disabled) return
  dragDepth.value = 0
  const list = event.dataTransfer?.files ? Array.from(event.dataTransfer.files) : []
  handleFiles(list)
}
</script>

