<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic init-image card for video workspaces.
Wraps the shared initial-image picker in neutral video-card chrome, surfaces the current init-image filename,
and optionally forwards WAN zoom-frame guide metadata without owning any family-specific runtime behavior.

Symbols (top-level; keep in sync; no ghosts):
- `VideoInitImageCard` (component): Generic img2vid init-image card.
-->

<template>
  <div class="gen-card cdx-video-card">
    <div class="cdx-video-card-header">
      <div class="cdx-video-card-header__left">
        <span class="cdx-video-card-header__title">{{ title }}</span>
      </div>
      <div v-if="$slots['header-actions']" class="cdx-video-card-header__right">
        <slot name="header-actions" />
      </div>
    </div>

    <div class="mt-2 cdx-video-card-body">
      <p v-if="subtitle" class="caption">{{ subtitle }}</p>
      <InitialImageCard
        :label="imageLabel"
        :src="initImageData"
        :has-image="hasInitImage"
        :disabled="disabled"
        :placeholder="placeholder"
        :dropzone="true"
        :zoomable="true"
        :zoomFrameGuide="zoomFrameGuide"
        @set="(file) => emit('set:initImage', file)"
        @clear="emit('clear:initImage')"
        @rejected="(payload) => emit('reject:initImage', payload)"
        @update:zoomFrameGuide="(value) => emit('update:zoomFrameGuide', value)"
      />
      <p v-if="initImageName" class="caption img2img-caption img2img-caption--init-name">{{ initImageName }}</p>
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import InitialImageCard from '../InitialImageCard.vue'
import type { WanImg2VidFrameGuideConfig } from '../../utils/wan_img2vid_frame_projection'

const props = withDefaults(defineProps<{
  title?: string
  subtitle?: string
  imageLabel?: string
  initImageData: string
  initImageName?: string
  disabled?: boolean
  placeholder?: string
  zoomFrameGuide?: WanImg2VidFrameGuideConfig | null
}>(), {
  title: 'Img2Vid Parameters',
  subtitle: 'Initial image',
  imageLabel: 'Image',
  initImageName: '',
  disabled: false,
  placeholder: 'Select an image to start.',
  zoomFrameGuide: null,
})

const emit = defineEmits<{
  (e: 'set:initImage', file: File): void
  (e: 'clear:initImage'): void
  (e: 'reject:initImage', payload: { reason: string; files: File[] }): void
  (e: 'update:zoomFrameGuide', value: WanImg2VidFrameGuideConfig): void
}>()

const hasInitImage = computed(() => Boolean(String(props.initImageData || '').trim()))
</script>
