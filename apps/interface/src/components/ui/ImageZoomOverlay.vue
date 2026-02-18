<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Full-screen image zoom overlay with pan/zoom controls.
Provides a reusable overlay used by result previews and init-image previews, with shared close semantics for Escape and outside-click.

Symbols (top-level; keep in sync; no ghosts):
- `ImageZoomOverlay` (component): Full-screen image overlay with pan/zoom controls.
- `clearPanListeners` (function): Removes temporary pan mouse listeners from `window`.
- `close` (function): Closes the overlay and clears pan listeners.
- `adjustZoom` (function): Applies bounded zoom delta updates.
- `onOverlayWheel` (function): Handles wheel-based zoom updates.
- `onWindowKeydown` (function): Handles keyboard shortcuts (`Escape` closes).
-->

<template>
  <div v-if="isOpen" class="image-zoom-overlay" @click="close" @wheel.prevent="onOverlayWheel">
    <div class="image-zoom-main" @click.stop>
      <img
        :src="src"
        :alt="alt"
        :style="zoomStyle"
        @mousedown.prevent="onPanStart"
      />
    </div>
    <div class="image-zoom-toolbar" @click.stop>
      <div class="toolbar-group">
        <button class="btn btn-sm btn-outline" type="button" @click="resetView">Fit</button>
        <button class="btn btn-sm btn-outline" type="button" @click="setZoom(1)">1:1</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomIn">+</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomOut">-</button>
        <button class="btn btn-sm btn-secondary" type="button" @click="close">Close</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch } from 'vue'

const props = withDefaults(defineProps<{
  modelValue: boolean
  src: string
  alt?: string
}>(), {
  alt: 'Zoomed image',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
}>()

const isOpen = computed(() => Boolean(props.modelValue) && Boolean(props.src))
const src = computed(() => props.src)
const alt = computed(() => props.alt || 'Zoomed image')

const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
let panState: { startX: number; startY: number; originX: number; originY: number } | null = null

function clearPanListeners(): void {
  window.removeEventListener('mousemove', onPanMove)
  window.removeEventListener('mouseup', onPanEnd)
}

function close(): void {
  emit('update:modelValue', false)
  panState = null
  clearPanListeners()
}

watch(isOpen, (open) => {
  if (open) {
    zoom.value = 1
    offsetX.value = 0
    offsetY.value = 0
    window.addEventListener('keydown', onWindowKeydown)
    return
  }
  window.removeEventListener('keydown', onWindowKeydown)
  panState = null
  clearPanListeners()
}, { immediate: true })

function adjustZoom(delta: number): void {
  const next = zoom.value + delta
  zoom.value = Math.max(0.25, Math.min(8, next))
}

function zoomIn(): void {
  adjustZoom(0.25)
}

function zoomOut(): void {
  adjustZoom(-0.25)
}

function setZoom(value: number): void {
  zoom.value = value
}

const zoomStyle = computed(() => ({
  transform: `translate(${offsetX.value}px, ${offsetY.value}px) scale(${zoom.value})`,
}))

function resetView(): void {
  zoom.value = 1
  offsetX.value = 0
  offsetY.value = 0
}

function onPanStart(event: MouseEvent): void {
  panState = {
    startX: event.clientX,
    startY: event.clientY,
    originX: offsetX.value,
    originY: offsetY.value,
  }
  window.addEventListener('mousemove', onPanMove)
  window.addEventListener('mouseup', onPanEnd)
}

function onPanMove(event: MouseEvent): void {
  if (!panState) return
  const dx = event.clientX - panState.startX
  const dy = event.clientY - panState.startY
  offsetX.value = panState.originX + dx
  offsetY.value = panState.originY + dy
}

function onPanEnd(): void {
  panState = null
  clearPanListeners()
}

function onOverlayWheel(event: WheelEvent): void {
  if (!isOpen.value) return
  const delta = event.deltaY < 0 ? 0.25 : -0.25
  adjustZoom(delta)
}

function onWindowKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  if (!isOpen.value) return
  close()
}

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onWindowKeydown)
  panState = null
  clearPanListeners()
})
</script>
