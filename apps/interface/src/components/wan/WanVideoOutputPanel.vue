<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN video output options panel.
Configures export format/CRF/pixel format/loop options and output toggles (pingpong/save/metadata/trim + optional RIFE interpolation) for WAN video tasks.

Symbols (top-level; keep in sync; no ghosts):
- `WanVideoOutputPanel` (component): Video output settings panel used by WANTab.
- `updateVideo` (function): Emits a partial `video` patch (`update:video`).
- `toInt` (function): Parses an event target value as an integer (fallback on invalid).
-->

<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }]">
    <div v-if="!embedded" class="row-split">
      <span class="label-muted">Video Output</span>
    </div>

    <div class="gc-row">
      <div class="gc-col gc-col--wide">
        <label class="label-muted">Filename Prefix</label>
        <input
          class="ui-input"
          type="text"
          :disabled="disabled"
          :value="video.filenamePrefix"
          @change="updateVideo({ filenamePrefix: ($event.target as HTMLInputElement).value })"
        />
      </div>
      <div class="gc-col">
        <label class="label-muted">Format</label>
        <select class="select-md" :disabled="disabled" :value="video.format" @change="updateVideo({ format: ($event.target as HTMLSelectElement).value })">
          <option value="video/h264-mp4">H.264 MP4</option>
          <option value="video/h265-mp4">H.265 MP4</option>
          <option value="video/webm">WebM</option>
          <option value="image/gif">GIF</option>
        </select>
      </div>
      <div class="gc-col">
        <label class="label-muted">CRF</label>
        <input
          class="ui-input"
          type="number"
          min="0"
          max="51"
          :disabled="disabled"
          :value="video.crf"
          @change="updateVideo({ crf: toInt($event, video.crf) })"
        />
      </div>
    </div>

    <div class="gc-row">
      <div class="gc-col">
        <label class="label-muted">Pixel Format</label>
        <select class="select-md" :disabled="disabled" :value="video.pixFmt" @change="updateVideo({ pixFmt: ($event.target as HTMLSelectElement).value })">
          <option value="yuv420p">yuv420p</option>
          <option value="yuv444p">yuv444p</option>
          <option value="yuv422p">yuv422p</option>
        </select>
      </div>
      <div class="gc-col">
        <label class="label-muted">Loop Count</label>
        <input
          class="ui-input"
          type="number"
          min="0"
          :disabled="disabled"
          :value="video.loopCount"
          @change="updateVideo({ loopCount: toInt($event, video.loopCount) })"
        />
      </div>
    </div>

    <div class="cdx-form-row">
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.pingpong ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :disabled="disabled"
        :aria-pressed="video.pingpong"
        @click="updateVideo({ pingpong: !video.pingpong })"
      >
        Ping-pong
      </button>
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.saveOutput ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :disabled="disabled"
        :aria-pressed="video.saveOutput"
        @click="updateVideo({ saveOutput: !video.saveOutput })"
      >
        Save output
      </button>
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.saveMetadata ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :disabled="disabled"
        :aria-pressed="video.saveMetadata"
        @click="updateVideo({ saveMetadata: !video.saveMetadata })"
      >
        Save metadata
      </button>
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.trimToAudio ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :disabled="disabled"
        :aria-pressed="video.trimToAudio"
        @click="updateVideo({ trimToAudio: !video.trimToAudio })"
      >
        Trim to audio
      </button>
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.rifeEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :disabled="disabled"
        :aria-pressed="video.rifeEnabled"
        @click="updateVideo({ rifeEnabled: !video.rifeEnabled })"
      >
        Interpolation (RIFE)
      </button>
    </div>
    <div v-if="video.rifeEnabled" class="gc-row">
      <div class="gc-col">
        <label class="label-muted">Model</label>
        <input
          class="ui-input"
          type="text"
          :disabled="disabled"
          :value="video.rifeModel"
          @change="updateVideo({ rifeModel: ($event.target as HTMLInputElement).value })"
        />
      </div>
      <div class="gc-col">
        <label class="label-muted">Times</label>
        <input
          class="ui-input"
          type="number"
          min="1"
          :disabled="disabled"
          :value="video.rifeTimes"
          @change="updateVideo({ rifeTimes: toInt($event, video.rifeTimes) })"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { WanVideoParams } from '../../stores/model_tabs'

const props = withDefaults(defineProps<{
  video: WanVideoParams
  embedded?: boolean
  disabled?: boolean
}>(), {
  embedded: false,
  disabled: false,
})

const emit = defineEmits<{
  (e: 'update:video', patch: Partial<WanVideoParams>): void
}>()

function updateVideo(patch: Partial<WanVideoParams>): void {
  emit('update:video', patch)
}

function toInt(e: Event, fallback: number): number {
  const v = Number((e.target as HTMLInputElement).value)
  return Number.isFinite(v) ? Math.trunc(v) : fallback
}
</script>
