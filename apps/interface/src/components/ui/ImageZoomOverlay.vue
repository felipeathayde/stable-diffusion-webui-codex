<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Full-screen image zoom overlay with pan/zoom controls.
Provides a reusable overlay used by result previews and init-image previews, with shared close semantics for Escape and outside-click,
plus true fit-to-viewport behavior for large images (without forcing a 25% minimum zoom on fit).

Symbols (top-level; keep in sync; no ghosts):
- `ImageZoomOverlay` (component): Full-screen image overlay with pan/zoom controls.
- `clearPanListeners` (function): Removes temporary pan mouse listeners from `window`.
- `close` (function): Closes the overlay and clears pan listeners.
- `computeFitZoom` (function): Computes the fit-to-viewport zoom with safe padding.
- `applyFitView` (function): Resets pan and applies fit zoom.
- `adjustZoom` (function): Applies bounded zoom delta updates.
- `onOverlayWheel` (function): Handles wheel-based zoom updates.
- `onWindowKeydown` (function): Handles keyboard shortcuts (`Escape` closes).
-->

<template>
  <div v-if="isOpen" class="image-zoom-overlay" @wheel.prevent="onOverlayWheel">
    <div ref="mainEl" class="image-zoom-main" @click="onMainClick">
      <img
        ref="imageEl"
        :src="src"
        :alt="alt"
        :style="zoomStyle"
        @click.stop
        @load="onImageLoad"
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
import { computed, nextTick, onBeforeUnmount, ref, watch } from 'vue'

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

const ZOOM_MIN = 0.25
const ZOOM_MAX = 8
const FIT_PADDING_PX = 24

const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
const mainEl = ref<HTMLElement | null>(null)
const imageEl = ref<HTMLImageElement | null>(null)
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
    window.addEventListener('keydown', onWindowKeydown)
    void nextTick(() => {
      applyFitView()
    })
    return
  }
  window.removeEventListener('keydown', onWindowKeydown)
  panState = null
  clearPanListeners()
}, { immediate: true })

function clampZoom(value: number): number {
  const minZoom = Math.min(ZOOM_MIN, computeFitZoom())
  if (!Number.isFinite(value)) return minZoom
  return Math.max(minZoom, Math.min(ZOOM_MAX, value))
}

function computeFitZoom(): number {
  const main = mainEl.value
  const image = imageEl.value
  if (!main || !image) return 1
  const naturalWidth = Number(image.naturalWidth)
  const naturalHeight = Number(image.naturalHeight)
  if (!Number.isFinite(naturalWidth) || !Number.isFinite(naturalHeight) || naturalWidth <= 0 || naturalHeight <= 0) return 1

  const availableWidth = Math.max(1, main.clientWidth - FIT_PADDING_PX * 2)
  const availableHeight = Math.max(1, main.clientHeight - FIT_PADDING_PX * 2)
  const fit = Math.min(availableWidth / naturalWidth, availableHeight / naturalHeight)
  if (!Number.isFinite(fit) || fit <= 0) return 1
  return Math.min(1, fit)
}

function applyFitView(): void {
  zoom.value = computeFitZoom()
  offsetX.value = 0
  offsetY.value = 0
}

function adjustZoom(delta: number): void {
  zoom.value = clampZoom(zoom.value + delta)
}

function zoomIn(): void {
  adjustZoom(0.25)
}

function zoomOut(): void {
  adjustZoom(-0.25)
}

function setZoom(value: number): void {
  zoom.value = clampZoom(value)
  offsetX.value = 0
  offsetY.value = 0
}

const zoomStyle = computed(() => ({
  transform: `translate(${offsetX.value}px, ${offsetY.value}px) scale(${zoom.value})`,
}))

function resetView(): void {
  applyFitView()
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

function onMainClick(event: MouseEvent): void {
  if (event.target !== event.currentTarget) return
  close()
}

function onImageLoad(): void {
  if (!isOpen.value) return
  applyFitView()
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
