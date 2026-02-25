<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational parameter card for image init/mask workflows.
Groups img2img controls (initial image) and optional inpaint controls (canvas-mask tools + enforcement/fill + mask toggles/sliders),
including dropzone/thumb/zoom handling for init images, rejected-file pass-through emits for parent toasts, and optional
embedded/title/label overrides so non-image tabs can reuse the same card shell without duplicating UI logic.

Symbols (top-level; keep in sync; no ghosts):
- `Img2ImgInpaintParamsCard` (component): Presentational card for img2img/inpaint parameter controls.
- `onMaskEnforcementChange` (function): Emits raw mask enforcement select updates for parent-side normalization.
- `onInpaintingFillChange` (function): Emits raw masked-content numeric updates for parent-side normalization.
- `onMaskEditorApply` (function): Emits edited mask data URL produced by the inpaint mask editor overlay.
- `onMaskEditorExternalReset` (function): Forwards editor reset notices for parent-side toasts.
-->

<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }, 'img2img-params-card']">
    <WanSubHeader :title="sectionTitle">
      <template v-if="sectionSubtitle" #subtitle>
        <span class="caption">{{ sectionSubtitle }}</span>
      </template>
    </WanSubHeader>

    <InitialImageCard
      :label="initImageLabel"
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
        <div v-if="useMask" class="img2img-mask-inline-tools">
          <div class="img2img-mask-editor-actions">
            <button
              class="btn btn-sm btn-secondary"
              type="button"
              :disabled="disabled || !initImageData"
              @click="maskEditorOpen = true"
            >
              Edit mask
            </button>
            <button
              class="btn btn-sm btn-outline"
              type="button"
              :disabled="disabled || !maskImageData"
              @click="emit('clear:maskImage')"
            >
              Clear mask
            </button>
          </div>
          <p class="caption img2img-caption">
            {{ maskImageData ? (maskImageName || 'Mask ready') : 'No mask applied. Open the editor to draw or upload.' }}
          </p>
        </div>
      </template>
    </InitialImageCard>

    <div v-if="useMask" class="img2img-mask-stack">
      <WanSubHeader title="Inpaint Parameters">
        <template #subtitle>
          <span class="caption">Canvas mask tools + runtime parameters</span>
        </template>
      </WanSubHeader>

      <div class="gc-row img2img-mask-grid">
        <div class="field">
          <label class="label-muted">Enforcement</label>
          <select class="select-md" :disabled="disabled" :value="maskEnforcement" @change="onMaskEnforcementChange">
            <option value="per_step_clamp">Forge engine (per-step blend)</option>
            <option value="post_blend">Legacy post-sample blend</option>
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

    <InpaintMaskEditorOverlay
      v-model="maskEditorOpen"
      :init-image-data="initImageData"
      :initial-mask-data="maskImageData"
      :image-width="imageWidth"
      :image-height="imageHeight"
      @apply="onMaskEditorApply"
      @external-reset="onMaskEditorExternalReset"
    />
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import InitialImageCard from './InitialImageCard.vue'
import InpaintMaskEditorOverlay from './ui/InpaintMaskEditorOverlay.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

type MaskEnforcement = 'post_blend' | 'per_step_clamp'

withDefaults(defineProps<{
  disabled?: boolean
  embedded?: boolean
  sectionTitle?: string
  sectionSubtitle?: string
  initImageLabel?: string
  initImageData?: string
  initImageName?: string
  imageWidth: number
  imageHeight: number
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
  embedded: false,
  sectionTitle: 'Img2Img Parameters',
  sectionSubtitle: 'Initial image',
  initImageLabel: 'Initial Image',
  initImageData: '',
  initImageName: '',
  maskImageData: '',
  maskImageName: '',
  maskEnforcement: 'per_step_clamp',
})

const emit = defineEmits<{
  (e: 'set:initImage', value: File): void
  (e: 'clear:initImage'): void
  (e: 'reject:initImage', payload: { reason: string; files: File[] }): void
  (e: 'clear:maskImage'): void
  (e: 'apply:maskImageData', value: string): void
  (e: 'notice:maskEditorReset', message: string): void
  (e: 'update:maskEnforcement', value: string): void
  (e: 'update:inpaintingFill', value: number): void
  (e: 'toggle:inpaintFullRes'): void
  (e: 'update:inpaintFullResPadding', value: number): void
  (e: 'toggle:maskInvert'): void
  (e: 'toggle:maskRound'): void
  (e: 'update:maskBlur', value: number): void
}>()

const maskEditorOpen = ref(false)

function onMaskEnforcementChange(event: Event): void {
  emit('update:maskEnforcement', (event.target as HTMLSelectElement).value)
}

function onInpaintingFillChange(event: Event): void {
  emit('update:inpaintingFill', Number((event.target as HTMLSelectElement).value))
}

function onMaskEditorApply(maskDataUrl: string): void {
  emit('apply:maskImageData', maskDataUrl)
}

function onMaskEditorExternalReset(message: string): void {
  emit('notice:maskEditorReset', message)
}
</script>

<!-- styles in styles/components/img2img-inpaint-params-card.css -->
