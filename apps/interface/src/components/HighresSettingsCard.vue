<template>
  <div class="highres-card">
    <div class="hr-header">
      <span class="label-muted">Highres (second pass)</span>
      <button class="hr-switch" type="button" @click="toggle">
        <span class="hr-switch-track" :data-on="enabled ? '1' : '0'">
          <span class="hr-switch-thumb" />
        </span>
      </button>
    </div>
    <div v-if="enabled" class="hr-grid">
      <div class="hr-cell">
        <SliderField
          label="Scale"
          :modelValue="scale"
          :min="1"
          :max="4"
          :step="0.1"
          :inputStep="0.1"
          :disabled="!enabled"
          :showButtons="false"
          inputClass="hr-number"
          @update:modelValue="(v) => emit('update:scale', v)"
        />
        <p class="hr-hint" v-if="targetWidth && targetHeight">Target ~ {{ targetWidth }}×{{ targetHeight }}</p>
      </div>
      <div class="hr-cell">
        <SliderField
          label="Denoise"
          :modelValue="denoise"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :disabled="!enabled"
          :showButtons="false"
          inputClass="hr-number"
          @update:modelValue="(v) => emit('update:denoise', v)"
        />
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
    <div v-if="enabled && showRefiner" class="hr-refiner">
      <RefinerSettingsCard
        label="Hires Refiner"
        :dense="true"
        v-model:enabled="refinerEnabled"
        v-model:steps="refinerSteps"
        v-model:cfg="refinerCfg"
        v-model:seed="refinerSeed"
        v-model:model="refinerModel"
        v-model:vae="refinerVae"
      />
      <p class="hr-hint">Runs after the hires pass; choose a refiner checkpoint and overrides here.</p>
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: highres, settings, grid
import { computed } from 'vue'
import RefinerSettingsCard from './RefinerSettingsCard.vue'
import SliderField from './ui/SliderField.vue'

const props = defineProps<{
  enabled: boolean
  denoise: number
  scale: number
  steps: number
  upscaler: string
  baseWidth?: number
  baseHeight?: number
  refinerEnabled?: boolean
  refinerSteps?: number
  refinerCfg?: number
  refinerSeed?: number
  refinerModel?: string
  refinerVae?: string
}>()

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:denoise', value: number): void
  (e: 'update:scale', value: number): void
  (e: 'update:steps', value: number): void
  (e: 'update:upscaler', value: string): void
  (e: 'update:refinerEnabled', value: boolean): void
  (e: 'update:refinerSteps', value: number): void
  (e: 'update:refinerCfg', value: number): void
  (e: 'update:refinerSeed', value: number): void
  (e: 'update:refinerModel', value: string): void
  (e: 'update:refinerVae', value: string): void
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

const showRefiner = computed(() => props.refinerEnabled !== undefined)
const refinerEnabled = computed({
  get: () => Boolean(props.refinerEnabled),
  set: (value: boolean) => emit('update:refinerEnabled', value),
})
const refinerSteps = computed({
  get: () => Number.isFinite(props.refinerSteps) ? Number(props.refinerSteps) : 0,
  set: (value: number) => emit('update:refinerSteps', value),
})
const refinerCfg = computed({
  get: () => Number.isFinite(props.refinerCfg) ? Number(props.refinerCfg) : 7,
  set: (value: number) => emit('update:refinerCfg', value),
})
const refinerSeed = computed({
  get: () => Number.isFinite(props.refinerSeed) ? Number(props.refinerSeed) : -1,
  set: (value: number) => emit('update:refinerSeed', value),
})
const refinerModel = computed({
  get: () => props.refinerModel ?? '',
  set: (value: string) => emit('update:refinerModel', value),
})
const refinerVae = computed({
  get: () => props.refinerVae ?? '',
  set: (value: string) => emit('update:refinerVae', value),
})

function toggle(): void {
  emit('update:enabled', !props.enabled)
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
