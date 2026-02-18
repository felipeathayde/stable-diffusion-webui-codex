<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational parameter card for image init/mask workflows.
Groups img2img controls (initial image + denoise) and optional inpaint controls (mask source + enforcement/fill + mask toggles/sliders),
including dropzone/thumb/zoom handling for init images and rejected-file pass-through emits for parent toasts.

Symbols (top-level; keep in sync; no ghosts):
- `Img2ImgInpaintParamsCard` (component): Presentational card for img2img/inpaint parameter controls.
- `onMaskEnforcementChange` (function): Emits raw mask enforcement select updates for parent-side normalization.
- `onInpaintingFillChange` (function): Emits raw masked-content numeric updates for parent-side normalization.
-->

<template>
  <div class="gen-card img2img-params-card">
    <div class="row-split">
      <span class="label-muted">Img2Img Parameters</span>
      <span class="caption">Initial image + denoise</span>
    </div>

    <InitialImageCard
      label="Initial Image"
      :src="initImageData"
      :has-image="Boolean(initImageData)"
      :disabled="disabled"
      :dropzone="true"
      :thumbnail="true"
      :zoomable="true"
      @set="(file) => emit('set:initImage', file)"
      @clear="() => emit('clear:initImage')"
      @rejected="(payload) => emit('reject:initImage', payload)"
    >
      <template #footer>
        <p v-if="initImageName" class="caption img2img-caption">{{ initImageName }}</p>
      </template>
    </InitialImageCard>

    <SliderField
      label="Denoise"
      :modelValue="denoiseStrength"
      :min="0"
      :max="1"
      :step="0.01"
      :inputStep="0.05"
      inputClass="cdx-input-w-xs"
      :disabled="disabled"
      @update:modelValue="(value) => emit('update:denoiseStrength', value)"
    />

    <div v-if="useMask" class="img2img-mask-stack">
      <div class="row-split">
        <span class="label-muted">Inpaint Parameters</span>
        <span class="caption">Mask controls</span>
      </div>

      <InitialImageCard
        label="Mask"
        accept="image/*"
        :src="maskImageData"
        :has-image="Boolean(maskImageData)"
        :disabled="disabled"
        placeholder="Select a mask image (RGBA/alpha supported)."
        @set="(file) => emit('set:maskImage', file)"
        @clear="() => emit('clear:maskImage')"
        @rejected="(payload) => emit('reject:maskImage', payload)"
      >
        <template #footer>
          <p v-if="maskImageName" class="caption img2img-caption">{{ maskImageName }}</p>
        </template>
      </InitialImageCard>

      <div class="img2img-mask-grid">
        <div class="field">
          <label class="label-muted">Enforcement</label>
          <select class="select-md" :disabled="disabled" :value="maskEnforcement" @change="onMaskEnforcementChange">
            <option value="post_blend">Forge-style (post-sample blend)</option>
            <option value="per_step_clamp">Clamp per step</option>
          </select>
        </div>

        <div class="field">
          <label class="label-muted">Masked content</label>
          <select class="select-md" :disabled="disabled" :value="inpaintingFill" @change="onInpaintingFillChange">
            <option :value="1">Original</option>
            <option :value="0">Fill</option>
            <option :value="2">Latent noise</option>
            <option :value="3">Latent nothing</option>
          </select>
        </div>
      </div>

      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', inpaintFullRes ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="inpaintFullRes"
        :disabled="disabled"
        @click="emit('toggle:inpaintFullRes')"
      >
        Inpaint area: Only masked (full-res)
      </button>

      <SliderField
        v-if="inpaintFullRes"
        label="Only masked padding"
        :modelValue="inpaintFullResPadding"
        :min="0"
        :max="256"
        :step="1"
        :inputStep="1"
        inputClass="cdx-input-w-xs"
        :disabled="disabled"
        @update:modelValue="(value) => emit('update:inpaintFullResPadding', value)"
      />

      <div class="img2img-toggle-row">
        <button
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', maskInvert ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :aria-pressed="maskInvert"
          :disabled="disabled"
          @click="emit('toggle:maskInvert')"
        >
          Invert mask
        </button>

        <button
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', maskRound ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :aria-pressed="maskRound"
          :disabled="disabled"
          @click="emit('toggle:maskRound')"
        >
          Round mask
        </button>
      </div>

      <SliderField
        label="Mask blur"
        :modelValue="maskBlur"
        :min="0"
        :max="64"
        :step="1"
        :inputStep="1"
        inputClass="cdx-input-w-xs"
        :disabled="disabled"
        @update:modelValue="(value) => emit('update:maskBlur', value)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import InitialImageCard from './InitialImageCard.vue'
import SliderField from './ui/SliderField.vue'

type MaskEnforcement = 'post_blend' | 'per_step_clamp'

withDefaults(defineProps<{
  disabled?: boolean
  initImageData?: string
  initImageName?: string
  denoiseStrength: number
  useMask: boolean
  maskImageData?: string
  maskImageName?: string
  maskEnforcement: MaskEnforcement
  inpaintingFill: number
  inpaintFullRes: boolean
  inpaintFullResPadding: number
  maskInvert: boolean
  maskRound: boolean
  maskBlur: number
}>(), {
  disabled: false,
  initImageData: '',
  initImageName: '',
  maskImageData: '',
  maskImageName: '',
  maskEnforcement: 'post_blend',
})

const emit = defineEmits<{
  (e: 'set:initImage', value: File): void
  (e: 'clear:initImage'): void
  (e: 'reject:initImage', payload: { reason: string; files: File[] }): void
  (e: 'update:denoiseStrength', value: number): void
  (e: 'set:maskImage', value: File): void
  (e: 'clear:maskImage'): void
  (e: 'reject:maskImage', payload: { reason: string; files: File[] }): void
  (e: 'update:maskEnforcement', value: string): void
  (e: 'update:inpaintingFill', value: number): void
  (e: 'toggle:inpaintFullRes'): void
  (e: 'update:inpaintFullResPadding', value: number): void
  (e: 'toggle:maskInvert'): void
  (e: 'toggle:maskRound'): void
  (e: 'update:maskBlur', value: number): void
}>()

function onMaskEnforcementChange(event: Event): void {
  emit('update:maskEnforcement', (event.target as HTMLSelectElement).value)
}

function onInpaintingFillChange(event: Event): void {
  emit('update:inpaintingFill', Number((event.target as HTMLSelectElement).value))
}
</script>

<!-- styles in styles/components/img2img-inpaint-params-card.css -->
