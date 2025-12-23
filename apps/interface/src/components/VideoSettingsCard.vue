<template>
  <div :class="['vid-card', { 'vid-card--embedded': embedded }]">
    <div class="vc-grid">
      <SliderField
        label="Frames"
        :modelValue="frames"
        :min="minFrames"
        :max="maxFrames"
        :step="1"
        :inputStep="1"
        :nudgeStep="1"
        inputClass="cdx-input-w-sm"
        @update:modelValue="(v) => emit('update:frames', v)"
      >
        <template #below>
          <span class="caption">min {{ minFrames }} · max {{ maxFrames }}</span>
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
</script>

<!-- Uses shared styles (gen-card layout avoided to keep focus on video-only params) -->
