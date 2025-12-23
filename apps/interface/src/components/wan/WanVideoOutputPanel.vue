<template>
  <div class="gen-card">
    <div class="row-split">
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
      <label class="qs-switch qs-switch--sm">
        <input type="checkbox" :disabled="disabled" :checked="video.pingpong" @change="updateVideo({ pingpong: ($event.target as HTMLInputElement).checked })" />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="label-muted">Ping-pong</span>
      </label>
      <label class="qs-switch qs-switch--sm">
        <input type="checkbox" :disabled="disabled" :checked="video.saveOutput" @change="updateVideo({ saveOutput: ($event.target as HTMLInputElement).checked })" />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="label-muted">Save output</span>
      </label>
      <label class="qs-switch qs-switch--sm">
        <input type="checkbox" :disabled="disabled" :checked="video.saveMetadata" @change="updateVideo({ saveMetadata: ($event.target as HTMLInputElement).checked })" />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="label-muted">Save metadata</span>
      </label>
      <label class="qs-switch qs-switch--sm">
        <input type="checkbox" :disabled="disabled" :checked="video.trimToAudio" @change="updateVideo({ trimToAudio: ($event.target as HTMLInputElement).checked })" />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="label-muted">Trim to audio</span>
      </label>
    </div>

    <div class="row-split">
      <span class="label-muted">Interpolation (RIFE)</span>
      <label class="qs-switch qs-switch--sm">
        <input type="checkbox" :disabled="disabled" :checked="video.rifeEnabled" @change="updateVideo({ rifeEnabled: ($event.target as HTMLInputElement).checked })" />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="label-muted">Enable</span>
      </label>
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
    <p v-else class="caption">Interpolation disabled.</p>
  </div>
</template>

<script setup lang="ts">
import type { WanVideoParams } from '../../stores/model_tabs'

const props = withDefaults(defineProps<{
  video: WanVideoParams
  disabled?: boolean
}>(), {
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
