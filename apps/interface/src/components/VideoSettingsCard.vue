<template>
  <div class="vid-card">
    <div class="vc-grid">
      <div class="field">
        <label class="label-muted">Frames</label>
        <div class="row-inline">
          <div class="number-with-controls">
            <input class="ui-input ui-input-sm w-batch pad-right" type="number" :min="minFrames" :max="maxFrames" step="1" :value="frames" @change="onFramesNumber" />
            <div class="stepper">
              <button class="step-btn" type="button" title="Increase" @click="framesInc">+</button>
              <button class="step-btn" type="button" title="Decrease" @click="framesDec">−</button>
            </div>
          </div>
          <span class="caption">min {{ minFrames }} · max {{ maxFrames }}</span>
        </div>
        <input class="slider" type="range" :min="minFrames" :max="maxFrames" step="1" :value="frames" @input="onFramesRange" />
      </div>
      <div class="field">
        <label class="label-muted">FPS</label>
        <div class="row-inline">
          <div class="number-with-controls">
            <input class="ui-input ui-input-sm w-batch pad-right" type="number" :min="minFps" :max="maxFps" step="1" :value="fps" @change="onFpsNumber" />
            <div class="stepper">
              <button class="step-btn" type="button" title="Increase" @click="fpsInc">+</button>
              <button class="step-btn" type="button" title="Decrease" @click="fpsDec">−</button>
            </div>
          </div>
          <span class="caption vc-duration">~ {{ durationLabel }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  frames: number
  fps: number
  minFrames?: number
  maxFrames?: number
  minFps?: number
  maxFps?: number
}>(), {
  minFrames: 8,
  maxFrames: 64,
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

function onFramesRange(e: Event): void {
  emit('update:frames', Number((e.target as HTMLInputElement).value))
}

function onFramesNumber(e: Event): void {
  let v = Number((e.target as HTMLInputElement).value)
  if (Number.isNaN(v)) v = minFrames.value
  v = Math.max(minFrames.value, Math.min(maxFrames.value, v))
  emit('update:frames', v)
}

function framesInc(): void {
  const v = Math.min(maxFrames.value, Number(props.frames) + 1)
  emit('update:frames', v)
}
function framesDec(): void {
  const v = Math.max(minFrames.value, Number(props.frames) - 1)
  emit('update:frames', v)
}

function onFpsNumber(e: Event): void {
  let v = Number((e.target as HTMLInputElement).value)
  if (Number.isNaN(v)) v = minFps.value
  v = Math.max(minFps.value, Math.min(maxFps.value, v))
  emit('update:fps', v)
}

function fpsInc(): void {
  const v = Math.min(maxFps.value, Number(props.fps) + 1)
  emit('update:fps', v)
}
function fpsDec(): void {
  const v = Math.max(minFps.value, Number(props.fps) - 1)
  emit('update:fps', v)
}
</script>

<!-- Uses shared styles (gen-card layout avoided to keep focus on video-only params) -->
