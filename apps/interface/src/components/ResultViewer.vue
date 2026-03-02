<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Viewer for generated images/videos with zoom overlay.
Displays generated outputs and provides an overlay viewer for zoom/pan actions, wheel/keyboard zoom controls, and per-item controls
(including per-frame downloads when available). Result media previews are rendered with `object-fit: contain` and a visual height cap
(`max-height: 30dvh`) plus proportional viewport-width cap (`max-width: min(100%, 42dvw)`) to avoid oversized cards while preserving
click-to-zoom full-screen behavior for images.

Symbols (top-level; keep in sync; no ghosts):
- `ResultViewer` (component): Viewer component for generated outputs and overlay interactions.
- `openZoom` (function): Opens the shared zoom overlay for a selected image.
- `closeZoom` (function): Closes the shared zoom overlay.
- `zoomOpen` (const): `v-model` bridge for the shared zoom overlay visibility.
- `zoomedImageSrc` (const): Derived data URL for the currently zoomed image.
-->

<template>
  <div class="viewer-card">
    <!-- Images gallery mode -->
    <template v-if="mode === 'image'">
      <template v-if="images && images.length">
        <div class="gallery-grid">
          <figure v-for="(img, index) in images" :key="index" class="gallery-figure">
            <img class="result-image" :src="imageUrl(img)" :alt="`Result ${index + 1}`" @click="openZoom(img)" />
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
            <img class="result-image" :src="imageUrl(previewImage)" alt="Live preview" @click="openZoom(previewImage)" />
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
            <img class="result-frame" :src="frameUrl(frame)" :alt="`Frame ${index + 1}`" />
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

    <ImageZoomOverlay v-model="zoomOpen" :src="zoomedImageSrc" alt="Zoomed result" />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { GeneratedImage } from '../api/types'
import ImageZoomOverlay from './ui/ImageZoomOverlay.vue'

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
}

function closeZoom(): void {
  zoomedImage.value = null
}

const zoomOpen = computed({
  get: () => Boolean(zoomedImage.value),
  set: (open: boolean) => {
    if (!open) closeZoom()
  },
})

const zoomedImageSrc = computed(() => (zoomedImage.value ? imageUrl(zoomedImage.value) : ''))
</script>

<!-- styles moved to styles/components/result-viewer.css -->
