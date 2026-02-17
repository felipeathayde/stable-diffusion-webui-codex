<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Video generation settings (frames + FPS).
Renders sliders for video frame count and FPS and derives an approximate duration label.

Symbols (top-level; keep in sync; no ghosts):
- `VideoSettingsCard` (component): Video settings card for video generation parameters.
- `durationLabel` (const): Computed duration label derived from frames/fps.
- `normalizeFrames` (function): Clamps/snap-normalizes frame count into the `4n+1` domain.
- `onFramesUpdate` (function): Emits normalized frame values for slider/input updates.
-->

<template>
  <div :class="['vid-card', { 'vid-card--embedded': embedded }]">
    <div class="vc-grid">
      <SliderField
        label="Frames"
        :modelValue="frames"
        :min="minFrames"
        :max="maxFrames"
        :step="4"
        :inputStep="1"
        :nudgeStep="4"
        inputClass="cdx-input-w-sm"
        @update:modelValue="onFramesUpdate"
      >
        <template #right>
          <NumberStepperInput
            :modelValue="frames"
            :min="minFrames"
            :max="maxFrames"
            :step="1"
            :nudgeStep="4"
            :inputClass="'cdx-input-w-sm'"
            @update:modelValue="onFramesUpdate"
          />
        </template>
        <template #below>
          <span class="caption">4n+1 · min {{ minFrames }} · max {{ maxFrames }}</span>
        </template>
      </SliderField>
      <SliderField
        label="FPS"
        :modelValue="fps"
        :min="minFps"
        :max="maxFps"
        :step="1"
        :inputStep="1"
        :nudgeStep="1"
        inputClass="cdx-input-w-sm"
        @update:modelValue="(v) => emit('update:fps', v)"
      >
        <template #below>
          <span class="caption vc-duration">~ {{ durationLabel }}</span>
        </template>
      </SliderField>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import SliderField from './ui/SliderField.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'

const props = withDefaults(defineProps<{
  frames: number
  fps: number
  embedded?: boolean
  minFrames?: number
  maxFrames?: number
  minFps?: number
  maxFps?: number
}>(), {
  embedded: false,
  minFrames: 9,
  maxFrames: 401,
  minFps: 8,
  maxFps: 60,
})

const emit = defineEmits({
  'update:frames': (v: number) => true,
  'update:fps': (v: number) => true,
})

const durationLabel = computed(() => {
  const f = Number(props.frames) || 0
  const fr = Number(props.fps) || 1
  const seconds = fr > 0 ? f / fr : 0
  return seconds.toFixed(2) + 's'
})

const minFrames = computed(() => props.minFrames)
const maxFrames = computed(() => props.maxFrames)
const minFps = computed(() => props.minFps)
const maxFps = computed(() => props.maxFps)

function normalizeFrames(rawValue: number): number {
  const min = Number(minFrames.value) || 1
  const max = Number(maxFrames.value) || min
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : min
  const clamped = Math.min(max, Math.max(min, numeric))

  if ((clamped - 1) % 4 === 0) return clamped

  const down = clamped - (((clamped - 1) % 4 + 4) % 4)
  const up = down + 4
  const downInRange = down >= min
  const upInRange = up <= max
  if (downInRange && upInRange) {
    const downDistance = Math.abs(clamped - down)
    const upDistance = Math.abs(up - clamped)
    return downDistance <= upDistance ? down : up
  }
  if (downInRange) return down
  if (upInRange) return up
  return min
}

function onFramesUpdate(value: number): void {
  emit('update:frames', normalizeFrames(value))
}
</script>

<!-- Uses shared styles (gen-card layout avoided to keep focus on video-only params) -->
