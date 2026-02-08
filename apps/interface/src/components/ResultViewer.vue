<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Viewer for generated images/videos with zoom overlay.
Displays generated outputs and provides an overlay viewer for zoom/pan actions, wheel/keyboard zoom controls, and per-item controls
(including per-frame downloads when available).

Symbols (top-level; keep in sync; no ghosts):
- `ResultViewer` (component): Viewer component for generated outputs and overlay interactions.
- `adjustZoom` (function): Applies bounded zoom delta updates shared by button and wheel handlers.
- `onOverlayWheel` (function): Handles overlay mouse-wheel zoom in/out interactions.
- `onWindowKeydown` (function): Handles keyboard shortcuts for overlay interactions (`Escape` closes zoom).
-->

<template>
  <div class="viewer-card">
    <!-- Images gallery mode -->
    <template v-if="mode === 'image'">
      <template v-if="images && images.length">
        <div class="gallery-grid">
          <figure v-for="(img, index) in images" :key="index" class="gallery-figure">
            <img :src="imageUrl(img)" :alt="`Result ${index + 1}`" @click="openZoom(img)" />
            <div class="badge-ar" v-if="width && height">{{ width }}×{{ height }}</div>
            <div class="gallery-overlay">
              <div class="gallery-actions">
                <slot name="image-actions" :image="img" :index="index"></slot>
              </div>
            </div>
            <figcaption class="gallery-caption">Result {{ index + 1 }}</figcaption>
          </figure>
        </div>
      </template>
      <template v-else-if="previewImage">
        <div class="gallery-grid">
          <figure class="gallery-figure">
            <img :src="imageUrl(previewImage)" alt="Live preview" @click="openZoom(previewImage)" />
            <div class="badge-ar" v-if="width && height">{{ width }}×{{ height }}</div>
            <figcaption class="gallery-caption">{{ previewCaption || 'Live preview' }}</figcaption>
          </figure>
        </div>
      </template>
      <div v-else class="viewer-empty">
        <slot name="empty" :mode="mode" :emptyText="emptyText">{{ emptyText }}</slot>
      </div>
    </template>

    <!-- Frames mode (video) -->
    <template v-else>
      <template v-if="frames && frames.length">
        <div class="grid gap-3 md:grid-cols-2">
          <figure v-for="(frame, index) in frames" :key="index" class="group relative space-y-1">
            <img class="w-full rounded" :src="frameUrl(frame)" :alt="`Frame ${index + 1}`" />
            <div v-if="canDownloadFrame(frame)" class="absolute right-2 top-2 opacity-0 transition group-hover:opacity-100">
              <a
                class="btn btn-sm btn-outline"
                :href="frameUrl(frame)"
                :download="frameDownloadName(index, frame)"
                @click.stop
              >Download</a>
            </div>
          </figure>
        </div>
      </template>
      <div v-else class="viewer-empty">
        <slot name="empty" :mode="mode" :emptyText="emptyText">{{ emptyText }}</slot>
      </div>
    </template>

    <!-- Zoom overlay -->
    <div v-if="zoomedImage" class="image-zoom-overlay" @click.self="closeZoom" @wheel.prevent="onOverlayWheel">
      <div class="image-zoom-main">
        <img
          :src="imageUrl(zoomedImage)"
          :alt="'Zoomed result'"
          :style="zoomStyle"
          @mousedown.prevent="onPanStart"
        />
      </div>
      <div class="image-zoom-toolbar">
        <div class="toolbar-group">
          <button class="btn btn-sm btn-outline" type="button" @click.stop="resetView">Fit</button>
          <button class="btn btn-sm btn-outline" type="button" @click.stop="setZoom(1)">1:1</button>
          <button class="btn btn-sm btn-outline" type="button" @click.stop="zoomIn">+</button>
          <button class="btn btn-sm btn-outline" type="button" @click.stop="zoomOut">-</button>
          <button class="btn btn-sm btn-secondary" type="button" @click.stop="closeZoom">Close</button>
        </div>
        <span class="caption">Drag to pan · Scroll to zoom · Esc to close</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { GeneratedImage } from '../api/types'

const props = defineProps<{
  mode: 'image' | 'video'
  images?: GeneratedImage[]
  previewImage?: GeneratedImage | null
  previewCaption?: string
  isRunning?: boolean
  frames?: GeneratedImage[]
  toDataUrl?: (frame: GeneratedImage) => string
  emptyText?: string
  width?: number
  height?: number
}>()

const zoomedImage = ref<GeneratedImage | null>(null)
const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
let panState: { startX: number; startY: number; originX: number; originY: number } | null = null

function imageUrl(img: GeneratedImage): string {
  return `data:image/${img.format};base64,${img.data}`
}

function frameUrl(frame: GeneratedImage): string {
  if (typeof props.toDataUrl === 'function') return props.toDataUrl(frame)
  return imageUrl(frame)
}

function canDownloadFrame(_frame: GeneratedImage): boolean {
  return true
}

function frameDownloadName(index: number, frame: GeneratedImage): string {
  const ext = String(frame.format || '').trim().toLowerCase() || 'png'
  const num = String(index + 1).padStart(4, '0')
  return `frame_${num}.${ext}`
}

function openZoom(img: GeneratedImage): void {
  zoomedImage.value = img
  zoom.value = 1
  offsetX.value = 0
  offsetY.value = 0
}

function closeZoom(): void {
  zoomedImage.value = null
  panState = null
  window.removeEventListener('mousemove', onPanMove)
  window.removeEventListener('mouseup', onPanEnd)
}

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
  window.removeEventListener('mousemove', onPanMove)
  window.removeEventListener('mouseup', onPanEnd)
}

function onOverlayWheel(event: WheelEvent): void {
  if (!zoomedImage.value) return
  const delta = event.deltaY < 0 ? 0.25 : -0.25
  adjustZoom(delta)
}

function onWindowKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  if (!zoomedImage.value) return
  closeZoom()
}

onMounted(() => {
  window.addEventListener('keydown', onWindowKeydown)
})

onBeforeUnmount(() => {
  closeZoom()
  window.removeEventListener('keydown', onWindowKeydown)
})
</script>

<!-- styles moved to styles/components/result-viewer.css -->
