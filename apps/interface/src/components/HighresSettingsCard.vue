<template>
  <div class="highres-card">
    <div class="hr-header">
      <span class="label-muted">Highres (second pass)</span>
      <label class="hr-toggle">
        <input type="checkbox" :checked="enabled" @change="onToggle" />
        <span>{{ enabled ? 'Enabled' : 'Disabled' }}</span>
      </label>
    </div>
    <div class="hr-grid" :data-enabled="enabled ? '1' : '0'">
      <div class="hr-cell">
        <label class="label-muted">Scale</label>
        <div class="row-inline">
          <input
            class="slider slider-grow"
            type="range"
            min="1"
            max="4"
            step="0.1"
            :value="scale"
            :disabled="!enabled"
            @input="onScaleRange"
          />
          <input
            class="ui-input ui-input-sm hr-number"
            type="number"
            min="1"
            max="4"
            step="0.1"
            :value="scale"
            :disabled="!enabled"
            @change="onScaleNumber"
          />
        </div>
        <p class="hr-hint" v-if="targetWidth && targetHeight">Target ~ {{ targetWidth }}×{{ targetHeight }}</p>
      </div>
      <div class="hr-cell">
        <label class="label-muted">Denoise</label>
        <div class="row-inline">
          <input
            class="slider slider-grow"
            type="range"
            min="0"
            max="1"
            step="0.01"
            :value="denoise"
            :disabled="!enabled"
            @input="onDenoiseRange"
          />
          <input
            class="ui-input ui-input-sm hr-number"
            type="number"
            min="0"
            max="1"
            step="0.01"
            :value="denoise"
            :disabled="!enabled"
            @change="onDenoiseNumber"
          />
        </div>
      </div>
      <div class="hr-cell">
        <label class="label-muted">Hires steps</label>
        <input
          class="ui-input ui-input-sm"
          type="number"
          min="0"
          :value="steps"
          :disabled="!enabled"
          @change="onStepsChange"
        />
        <p class="hr-hint">0 = reuse base steps</p>
      </div>
      <div class="hr-cell">
        <label class="label-muted">Upscaler</label>
        <select class="select-md" :value="upscaler" :disabled="!enabled" @change="onUpscalerChange">
          <option v-for="opt in upscalerOptions" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: highres, settings, grid
import { computed } from 'vue'

const props = defineProps<{
  enabled: boolean
  denoise: number
  scale: number
  steps: number
  upscaler: string
  baseWidth?: number
  baseHeight?: number
}>()

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:denoise', value: number): void
  (e: 'update:scale', value: number): void
  (e: 'update:steps', value: number): void
  (e: 'update:upscaler', value: string): void
}>()

const upscalerOptions = ['Use same upscaler', 'Latent (nearest)', 'Latent (Lanczos)']

const targetWidth = computed(() => {
  if (!props.baseWidth || props.scale <= 1) return null
  return Math.round(props.baseWidth * props.scale)
})

const targetHeight = computed(() => {
  if (!props.baseHeight || props.scale <= 1) return null
  return Math.round(props.baseHeight * props.scale)
})

function onToggle(event: Event): void {
  emit('update:enabled', (event.target as HTMLInputElement).checked)
}

function clampScale(value: number): number {
  if (Number.isNaN(value)) return 1
  return Math.min(4, Math.max(1, value))
}

function onScaleRange(event: Event): void {
  emit('update:scale', clampScale(Number((event.target as HTMLInputElement).value)))
}

function onScaleNumber(event: Event): void {
  emit('update:scale', clampScale(Number((event.target as HTMLInputElement).value)))
}

function clampDenoise(value: number): number {
  if (Number.isNaN(value)) return 0.4
  return Math.min(1, Math.max(0, value))
}

function onDenoiseRange(event: Event): void {
  emit('update:denoise', clampDenoise(Number((event.target as HTMLInputElement).value)))
}

function onDenoiseNumber(event: Event): void {
  emit('update:denoise', clampDenoise(Number((event.target as HTMLInputElement).value)))
}

function onStepsChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:steps', Number.isNaN(v) || v < 0 ? 0 : v)
}

function onUpscalerChange(event: Event): void {
  emit('update:upscaler', (event.target as HTMLSelectElement).value)
}
</script>

<!-- styles in styles/components/highres-settings-card.css -->

