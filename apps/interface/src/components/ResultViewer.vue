<template>
  <div class="viewer-card">
    <!-- Images gallery mode -->
    <template v-if="mode === 'image'">
      <template v-if="images && images.length">
        <div class="gallery-grid">
          <figure v-for="(img, index) in images" :key="index" class="gallery-figure">
            <img :src="imageUrl(img)" :alt="`Result ${index + 1}`" />
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
</template>

<script setup lang="ts">
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

function imageUrl(img: GeneratedImage): string {
  return `data:image/${img.format};base64,${img.data}`
}

function frameUrl(frame: unknown): string {
  if (typeof props.toDataUrl === 'function') return props.toDataUrl(frame)
  return String(frame ?? '')
}
</script>

<!-- styles moved to styles/components/result-viewer.css -->
