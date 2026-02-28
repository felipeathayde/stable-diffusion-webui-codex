<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN video output options panel.
Configures export format/pixel format/CRF/loop options, compact output toggles (`pingpong`, `returnFrames`), and RIFE interpolation target FPS (`0=off`) for WAN video tasks.
Also exposes optional SeedVR2 video upscaling controls (`video_upscaling`) inside a dedicated collapsible card with header toggle parity to Temporal Loom.

Symbols (top-level; keep in sync; no ghosts):
- `WanVideoOutputPanel` (component): Video output settings panel used by WANTab.
- `updateVideo` (function): Emits a partial `video` patch (`update:video`).
- `normalizeNonNegativeInteger` (function): Parses/clamps integer inputs to non-negative range with optional max.
- `normalizeInterpolationTargetFps` (function): Normalizes interpolation target FPS (`0` disables interpolation).
- `onLoopCountChange` (function): Applies normalized loop-count updates.
- `onCrfChange` (function): Applies normalized CRF updates.
- `onInterpolationTargetFpsChange` (function): Applies normalized interpolation target FPS updates.
- `onUpscalingResolutionChange` (function): Applies normalized upscaling target resolution updates.
- `onUpscalingMaxResolutionChange` (function): Applies normalized upscaling max-resolution updates.
- `onUpscalingBatchSizeChange` (function): Applies normalized upscaling batch-size updates (`4n+1`).
- `onUpscalingTemporalOverlapChange` (function): Applies normalized non-negative temporal-overlap updates.
- `onUpscalingPrependFramesChange` (function): Applies normalized non-negative prepend-frame updates.
- `onUpscalingInputNoiseScaleChange` (function): Applies normalized `[0,1]` input-noise updates.
- `onUpscalingLatentNoiseScaleChange` (function): Applies normalized `[0,1]` latent-noise updates.
- `interpolationCaption` (computed): User-facing interpolation summary (`Off`/`Target`) with effective backend output FPS.
- `upscalingCaption` (computed): User-facing upscaling summary.
-->

<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }]">
    <div v-if="!embedded" class="row-split">
      <span class="label-muted">Video Output</span>
    </div>

    <div class="gc-row">
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
        <label class="label-muted">Pixel Format</label>
        <select class="select-md" :disabled="disabled" :value="video.pixFmt" @change="updateVideo({ pixFmt: ($event.target as HTMLSelectElement).value })">
          <option value="yuv420p">yuv420p</option>
          <option value="yuv444p">yuv444p</option>
          <option value="yuv422p">yuv422p</option>
        </select>
      </div>
    </div>

    <div class="gc-row">
      <SliderField
        class="gc-col gc-col--compact"
        label="Loop Count"
        :modelValue="video.loopCount"
        :min="0"
        :max="32"
        :step="1"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="onLoopCountChange"
      />
      <SliderField
        class="gc-col gc-col--compact"
        label="CRF"
        :modelValue="video.crf"
        :min="0"
        :max="51"
        :step="1"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="onCrfChange"
      />
      <SliderField
        class="gc-col gc-col--compact"
        label="Interpolation FPS (RIFE)"
        :modelValue="video.interpolationFps"
        :min="0"
        :max="240"
        :step="1"
        :disabled="disabled"
        inputClass="cdx-input-w-md"
        @update:modelValue="onInterpolationTargetFpsChange"
      >
        <template #below>
          <span class="caption">{{ interpolationCaption }}</span>
        </template>
      </SliderField>
      <div class="gc-col gc-col--presets wan-video-output-toggle-row">
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
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.returnFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :disabled="disabled"
          :aria-pressed="video.returnFrames"
          @click="updateVideo({ returnFrames: !video.returnFrames })"
        >
          Return frames
        </button>
      </div>
    </div>

    <div class="gen-card refiner-card refiner-card--dense">
      <WanSubHeader
        title="Upscaling"
        :clickable="true"
        :disabled="disabled"
        :aria-pressed="video.upscalingEnabled"
        :aria-expanded="video.upscalingEnabled"
        @header-click="updateVideo({ upscalingEnabled: !video.upscalingEnabled })"
      >
        <button
          :class="[
            'btn',
            'qs-toggle-btn',
            'qs-toggle-btn--sm',
            video.upscalingEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off',
          ]"
          type="button"
          :disabled="disabled"
          :aria-pressed="video.upscalingEnabled"
          @click.stop="updateVideo({ upscalingEnabled: !video.upscalingEnabled })"
        >
          {{ video.upscalingEnabled ? 'Enabled' : 'Disabled' }}
        </button>
      </WanSubHeader>
      <div v-if="video.upscalingEnabled" class="param-blocks wan-temporal-controls">
        <div class="param-grid wan-temporal-row" data-cols="3">
          <div class="field">
            <label class="label-muted">Upscaling Model</label>
            <select
              class="select-md"
              :disabled="disabled"
              :value="video.upscalingModel"
              @change="updateVideo({ upscalingModel: ($event.target as HTMLSelectElement).value })"
            >
              <option value="seedvr2_ema_3b_fp16.safetensors">SeedVR2 EMA 3B FP16</option>
              <option value="seedvr2_ema_7b_fp16.safetensors">SeedVR2 EMA 7B FP16</option>
              <option value="seedvr2_ema_7b_sharp_fp16.safetensors">SeedVR2 EMA 7B Sharp FP16</option>
            </select>
          </div>
          <SliderField
            class="field"
            label="Upscale Resolution"
            :modelValue="video.upscalingResolution"
            :min="16"
            :max="4096"
            :step="16"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingResolutionChange"
          />
          <SliderField
            class="field"
            label="Max Resolution"
            :modelValue="video.upscalingMaxResolution"
            :min="0"
            :max="8192"
            :step="16"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingMaxResolutionChange"
          />
        </div>

        <div class="param-grid wan-temporal-row" data-cols="4">
          <SliderField
            class="field"
            label="Batch Size"
            :modelValue="video.upscalingBatchSize"
            :min="1"
            :max="129"
            :step="1"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingBatchSizeChange"
          />
          <SliderField
            class="field"
            label="Temporal Overlap"
            :modelValue="video.upscalingTemporalOverlap"
            :min="0"
            :max="128"
            :step="1"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingTemporalOverlapChange"
          />
          <SliderField
            class="field"
            label="Prepend Frames"
            :modelValue="video.upscalingPrependFrames"
            :min="0"
            :max="128"
            :step="1"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingPrependFramesChange"
          />
          <div class="field">
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.upscalingUniformBatchSize ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :disabled="disabled"
              :aria-pressed="video.upscalingUniformBatchSize"
              @click="updateVideo({ upscalingUniformBatchSize: !video.upscalingUniformBatchSize })"
            >
              Uniform Batch
            </button>
          </div>
        </div>

        <div class="param-grid wan-temporal-row" data-cols="3">
          <div class="field">
            <label class="label-muted">Color Correction</label>
            <select
              class="select-md"
              :disabled="disabled"
              :value="video.upscalingColorCorrection"
              @change="updateVideo({ upscalingColorCorrection: ($event.target as HTMLSelectElement).value as WanVideoParams['upscalingColorCorrection'] })"
            >
              <option value="lab">LAB</option>
              <option value="wavelet">Wavelet</option>
              <option value="wavelet_adaptive">Wavelet Adaptive</option>
              <option value="hsv">HSV</option>
              <option value="adain">AdaIN</option>
              <option value="none">None</option>
            </select>
          </div>
          <SliderField
            class="field"
            label="Input Noise"
            :modelValue="video.upscalingInputNoiseScale"
            :min="0"
            :max="1"
            :step="0.01"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingInputNoiseScaleChange"
          />
          <SliderField
            class="field"
            label="Latent Noise"
            :modelValue="video.upscalingLatentNoiseScale"
            :min="0"
            :max="1"
            :step="0.01"
            :disabled="disabled"
            inputClass="cdx-input-w-md"
            @update:modelValue="onUpscalingLatentNoiseScaleChange"
          />
        </div>
      </div>
      <div v-else class="caption">Upscaling is off.</div>
      <div v-if="video.upscalingEnabled" class="caption">{{ upscalingCaption }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import SliderField from '../ui/SliderField.vue'
import WanSubHeader from './WanSubHeader.vue'
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

function normalizeNonNegativeInteger(value: unknown, fallback: number, max?: number): number {
  const fallbackInt = Number.isFinite(fallback) ? Math.max(0, Math.trunc(fallback)) : 0
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallbackInt
  const parsed = Math.max(0, Math.trunc(numeric))
  if (typeof max === 'number' && Number.isFinite(max)) {
    return Math.min(Math.max(0, Math.trunc(max)), parsed)
  }
  return parsed
}

const MAX_INTERPOLATION_FPS = 240

function normalizeInterpolationTargetFps(value: unknown, fallback: number): number {
  const fallbackValue = Number.isFinite(fallback) ? Math.trunc(fallback) : 0
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return Math.max(0, Math.min(MAX_INTERPOLATION_FPS, fallbackValue))
  return Math.max(0, Math.min(MAX_INTERPOLATION_FPS, Math.trunc(numeric)))
}

function normalizeUnitInterval(value: unknown, fallback: number): number {
  const fallbackValue = Number.isFinite(Number(fallback)) ? Number(fallback) : 0
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return Math.min(1, Math.max(0, fallbackValue))
  return Math.min(1, Math.max(0, numeric))
}

function normalizeUpscalingBatchSize(value: unknown, fallback: number): number {
  const fallbackInt = Number.isFinite(Number(fallback)) ? Math.max(1, Math.trunc(Number(fallback))) : 5
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) return fallbackInt
  const intValue = Math.max(1, Math.trunc(numeric))
  const remainder = (intValue - 1) % 4
  if (remainder === 0) return intValue
  const down = intValue - remainder
  const up = down + 4
  if (down >= 1) {
    const downDistance = Math.abs(intValue - down)
    const upDistance = Math.abs(up - intValue)
    return downDistance <= upDistance ? down : up
  }
  return up
}

function onLoopCountChange(value: number): void {
  updateVideo({ loopCount: normalizeNonNegativeInteger(value, props.video.loopCount, 32) })
}

function onCrfChange(value: number): void {
  updateVideo({ crf: normalizeNonNegativeInteger(value, props.video.crf, 51) })
}

function onInterpolationTargetFpsChange(value: number): void {
  updateVideo({
    interpolationFps: normalizeInterpolationTargetFps(value, props.video.interpolationFps),
  })
}

function onUpscalingResolutionChange(value: number): void {
  updateVideo({ upscalingResolution: Math.max(16, Math.trunc(Number(value) || props.video.upscalingResolution)) })
}

function onUpscalingMaxResolutionChange(value: number): void {
  updateVideo({ upscalingMaxResolution: normalizeNonNegativeInteger(value, props.video.upscalingMaxResolution) })
}

function onUpscalingBatchSizeChange(value: number): void {
  updateVideo({ upscalingBatchSize: normalizeUpscalingBatchSize(value, props.video.upscalingBatchSize) })
}

function onUpscalingTemporalOverlapChange(value: number): void {
  updateVideo({ upscalingTemporalOverlap: normalizeNonNegativeInteger(value, props.video.upscalingTemporalOverlap) })
}

function onUpscalingPrependFramesChange(value: number): void {
  updateVideo({ upscalingPrependFrames: normalizeNonNegativeInteger(value, props.video.upscalingPrependFrames) })
}

function onUpscalingInputNoiseScaleChange(value: number): void {
  updateVideo({ upscalingInputNoiseScale: normalizeUnitInterval(value, props.video.upscalingInputNoiseScale) })
}

function onUpscalingLatentNoiseScaleChange(value: number): void {
  updateVideo({ upscalingLatentNoiseScale: normalizeUnitInterval(value, props.video.upscalingLatentNoiseScale) })
}

const interpolationCaption = computed<string>(() => {
  const targetFps = normalizeInterpolationTargetFps(
    props.video.interpolationFps,
    0,
  )
  const baseFps = normalizeNonNegativeInteger(props.video.fps, 0, MAX_INTERPOLATION_FPS)
  if (targetFps <= 0) {
    return baseFps > 0 ? `Off · Output: ${baseFps} fps` : 'Off'
  }
  if (baseFps <= 0) return `Target: ${targetFps} fps`
  if (targetFps <= baseFps) return `Disabled · Target (${targetFps} fps) <= base (${baseFps} fps)`
  const times = Math.max(2, Math.ceil(targetFps / baseFps))
  const outputFps = baseFps * times
  return `Target: ${targetFps} fps · Output: ${outputFps} fps`
})

const upscalingCaption = computed<string>(() => {
  if (!props.video.upscalingEnabled) return 'Upscaling is off.'
  return `Enabled · ${props.video.upscalingModel} · ${props.video.upscalingResolution}px target`
})
</script>
