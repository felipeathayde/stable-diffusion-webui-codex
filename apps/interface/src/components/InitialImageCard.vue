<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Initial image file picker for img2img-style workflows.
Provides a file input, preview, and remove action and emits the selected `File` back to the parent.
Supports optional pass-through WAN frame-guide config for zoom-overlay no-stretch projection metadata.

Symbols (top-level; keep in sync; no ghosts):
- `InitialImageCard` (component): Initial image picker panel.
- `zoomFrameGuide` (prop): Optional WAN frame-guide config forwarded to `ImageZoomOverlay`.
- `onFile` (function): Handles file-input selection and emits `set`.
- `onDropFiles` (function): Handles dropzone selection and emits `set`.
- `onPreviewClick` (function): Opens the zoom overlay for the current preview image.
-->

<template>
  <div class="panel-section">
    <label class="label-muted">{{ label }}</label>
    <div class="init-picker">
      <template v-if="dropzone">
        <Dropzone
          :accept="accept"
          :disabled="disabled"
          :label="placeholder"
          hint="Drop an image or click to browse."
          @select="onDropFiles"
          @rejected="onDropRejected"
        >
          <div class="init-dropzone-slot">
            <div
              v-if="src"
              :class="[
                'init-preview',
                thumbnail ? 'init-preview--thumb' : '',
                canZoom ? 'init-preview--clickable' : '',
              ]"
              @click.stop="onPreviewClick"
            >
              <img :src="src" alt="Initial" />
            </div>
            <p v-else class="caption">{{ placeholder }}</p>
          </div>
        </Dropzone>
        <div class="toolbar">
          <button class="btn btn-sm btn-ghost" type="button" :disabled="disabled || !hasImage" @click="$emit('clear')">Remove</button>
        </div>
      </template>
      <template v-else>
        <div class="toolbar">
          <input class="ui-input" :disabled="disabled" type="file" :accept="accept" @change="onFile" />
          <button class="btn btn-sm btn-ghost" type="button" :disabled="disabled || !hasImage" @click="$emit('clear')">Remove</button>
        </div>
        <div
          v-if="src"
          :class="[
            'init-preview',
            thumbnail ? 'init-preview--thumb' : '',
            canZoom ? 'init-preview--clickable' : '',
          ]"
          @click.stop="onPreviewClick"
        >
          <img :src="src" alt="Initial" />
        </div>
        <p v-else class="caption">{{ placeholder }}</p>
      </template>
      <slot name="footer" />
    </div>
    <ImageZoomOverlay
      v-model="zoomOpen"
      :src="src"
      alt="Initial image preview"
      :wanFrameGuide="zoomFrameGuide"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import Dropzone from './ui/Dropzone.vue'
import ImageZoomOverlay from './ui/ImageZoomOverlay.vue'
import type { WanImg2VidFrameGuideConfig } from '../utils/wan_img2vid_frame_projection'

const props = withDefaults(defineProps<{
  label?: string
  accept?: string
  src?: string
  hasImage?: boolean
  disabled?: boolean
  placeholder?: string
  dropzone?: boolean
  thumbnail?: boolean
  zoomable?: boolean
  zoomFrameGuide?: WanImg2VidFrameGuideConfig | null
}>(), {
  label: 'Initial Image',
  accept: 'image/*',
  src: '',
  hasImage: false,
  disabled: false,
  placeholder: 'Select an image to start.',
  dropzone: false,
  thumbnail: false,
  zoomable: false,
  zoomFrameGuide: null,
})

const emit = defineEmits<{
  (e: 'set', file: File): void
  (e: 'clear'): void
  (e: 'rejected', payload: { reason: string; files: File[] }): void
}>()
const zoomOpen = ref(false)
const canZoom = computed(() => Boolean(props.zoomable && props.src))

function onFile(e: Event): void {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (file) emit('set', file)
  input.value = ''
}

function onDropFiles(files: File[]): void {
  const file = files[0]
  if (file) emit('set', file)
}

function onDropRejected(payload: { reason: string; files: File[] }): void {
  emit('rejected', payload)
}

function onPreviewClick(): void {
  if (!canZoom.value) return
  zoomOpen.value = true
}
</script>

<!-- uses .init-picker styles from src/styles.css -->
