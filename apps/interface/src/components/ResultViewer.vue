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
      <div v-else class="viewer-empty">{{ emptyText }}</div>
    </template>

    <!-- Frames mode (video) -->
    <template v-else>
      <template v-if="frames && frames.length">
        <div class="grid gap-3 md:grid-cols-2">
          <figure v-for="(frame, index) in frames" :key="index" class="space-y-1">
            <img class="w-full rounded" :src="frameUrl(frame)" :alt="`Frame ${index + 1}`" />
          </figure>
        </div>
      </template>
      <div v-else class="viewer-empty">{{ emptyText }}</div>
    </template>
  </div>

  <!-- Zoom overlay -->
  <div v-if="zoomedImage" class="modal-backdrop image-zoom-backdrop" @click.self="closeZoom">
    <div class="modal-panel image-zoom-panel">
      <div class="modal-header">
        <span>Image preview</span>
        <div class="image-zoom-controls">
          <button class="btn btn-sm btn-outline" type="button" @click.stop="zoomOut">-</button>
          <button class="btn btn-sm btn-outline" type="button" @click.stop="setZoom(1)">1:1</button>
          <button class="btn btn-sm btn-outline" type="button" @click.stop="zoomIn">+</button>
          <button class="btn btn-sm btn-secondary" type="button" @click.stop="closeZoom">Close</button>
        </div>
      </div>
      <div class="modal-body image-zoom-body">
        <div class="image-zoom-canvas">
          <img :src="imageUrl(zoomedImage)" :alt="'Zoomed result'" :style="zoomStyle" />
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import type { GeneratedImage } from '../api/types'

const props = defineProps<{
  mode: 'image' | 'video'
  images?: GeneratedImage[]
  frames?: unknown[]
  toDataUrl?: (frame: unknown) => string
  emptyText?: string
  width?: number
  height?: number
}>()

const zoomedImage = ref<GeneratedImage | null>(null)
const zoom = ref(1)

function imageUrl(img: GeneratedImage): string {
  return `data:image/${img.format};base64,${img.data}`
}

function frameUrl(frame: unknown): string {
  if (typeof props.toDataUrl === 'function') return props.toDataUrl(frame)
  return String(frame ?? '')
}

function openZoom(img: GeneratedImage): void {
  zoomedImage.value = img
  zoom.value = 1
}

function closeZoom(): void {
  zoomedImage.value = null
}

function zoomIn(): void {
  zoom.value = Math.min(zoom.value + 0.25, 8)
}

function zoomOut(): void {
  zoom.value = Math.max(zoom.value - 0.25, 0.25)
}

function setZoom(value: number): void {
  zoom.value = value
}

const zoomStyle = computed(() => ({
  transform: `scale(${zoom.value})`,
}))
</script>

<!-- styles moved to styles/components/result-viewer.css -->
