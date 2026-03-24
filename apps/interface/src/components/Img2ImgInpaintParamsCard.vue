<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational parameter card for image init/mask workflows.
Groups img2img controls (initial image) and optional inpaint controls (canvas-mask tools + enforcement/fill + only-masked padding + mask blur + region splitting),
including dropzone/thumb/zoom handling for init images, rejected-file pass-through emits for parent toasts, and optional
embedded/title/label overrides so non-image tabs can reuse the same card shell without duplicating UI logic.
Supports optional pass-through WAN zoom frame-guide config for init-image overlays.
Init-image filename captions are centered in the footer area for clearer media identification.

Symbols (top-level; keep in sync; no ghosts):
- `Img2ImgInpaintParamsCard` (component): Presentational card for img2img/inpaint parameter controls.
- `INPAINT_PARAMETER_TOOLTIPS` (constant): Tooltip copy for inpaint select and slider controls.
- `zoomFrameGuide` (prop): Optional WAN frame-guide config forwarded to `InitialImageCard` zoom overlay.
- `onZoomFrameGuideUpdate` (function): Forwards zoom-overlay guide edits to parent WAN state.
- `onMaskEnforcementChange` (function): Emits raw mask enforcement select updates for parent-side normalization.
- `onInpaintingFillChange` (function): Emits raw masked-content numeric updates for parent-side normalization.
- `onMaskEditorApply` (function): Emits edited mask data URL produced by the inpaint mask editor overlay.
- `onInitPreviewClick` (function): Opens the mask editor when inpaint mode is active and an init image is present.
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
      :preview-click-action="useMask ? 'emit' : 'zoom'"
      :zoom-frame-guide="zoomFrameGuide"
      @set="(file) => emit('set:initImage', file)"
      @clear="() => emit('clear:initImage')"
      @rejected="(payload) => emit('reject:initImage', payload)"
      @preview-click="onInitPreviewClick(disabled, initImageData)"
      @update:zoom-frame-guide="onZoomFrameGuideUpdate"
    >
      <template #dropzone-actions>
        <div v-if="useMask" class="img2img-mask-editor-actions">
          <button
            class="btn btn-sm btn-outline"
            type="button"
            :disabled="disabled || !maskImageData"
            @click.stop.prevent="emit('clear:maskImage')"
          >
            Clear mask
          </button>
        </div>
      </template>
      <template #preview-overlay>
        <div
          v-if="maskImageData"
          class="img2img-mask-preview-overlay"
          :style="{ '--img2img-mask-src': `url('${maskImageData}')` }"
        />
      </template>
      <template #footer>
        <p v-if="initImageName" class="caption img2img-caption img2img-caption--init-name">{{ initImageName }}</p>
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
          <label class="label-muted">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              :title="INPAINT_PARAMETER_TOOLTIPS.enforcement.title"
              :content="INPAINT_PARAMETER_TOOLTIPS.enforcement.content"
            >
              <span class="cdx-slider-field__label-trigger">
                <span>Enforcement</span>
                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
              </span>
            </HoverTooltip>
          </label>
          <select class="select-md" :disabled="disabled" :value="maskEnforcement" @change="onMaskEnforcementChange">
            <option value="per_step_clamp">Per-step blend</option>
            <option value="post_blend">Post-sample blend</option>
          </select>
        </div>

        <div class="field">
          <label class="label-muted">
            <HoverTooltip
              class="cdx-slider-field__label-tooltip"
              :title="INPAINT_PARAMETER_TOOLTIPS.maskedContent.title"
              :content="INPAINT_PARAMETER_TOOLTIPS.maskedContent.content"
            >
              <span class="cdx-slider-field__label-trigger">
                <span>Masked content</span>
                <span class="cdx-slider-field__label-help" aria-hidden="true">?</span>
              </span>
            </HoverTooltip>
          </label>
          <select class="select-md" :disabled="disabled" :value="inpaintingFill" @change="onInpaintingFillChange">
            <option :value="1">Original</option>
            <option :value="0">Fill</option>
            <option :value="2">Latent noise</option>
            <option :value="3">Latent nothing</option>
          </select>
        </div>

        <div class="gc-col gc-col--presets img2img-mask-toggle-col">
          <button
            :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', maskRegionSplit ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
            type="button"
            :aria-pressed="maskRegionSplit"
            :disabled="disabled"
            @click="emit('toggle:maskRegionSplit')"
          >
            Split mask regions
          </button>
        </div>
      </div>

      <div class="gc-row img2img-mask-slider-row">
        <SliderField
          class="gc-col gc-col--wide"
          label="Only masked padding"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.onlyMaskedPadding.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.onlyMaskedPadding.title"
          :modelValue="inpaintFullResPadding"
          :min="0"
          :max="256"
          :step="1"
          :inputStep="1"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:inpaintFullResPadding', value)"
        />

        <SliderField
          class="gc-col gc-col--wide"
          label="Mask blur"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.maskBlur.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.maskBlur.title"
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
import HoverTooltip from './ui/HoverTooltip.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'
import type { WanImg2VidFrameGuideConfig } from '../utils/wan_img2vid_frame_projection'

type MaskEnforcement = 'post_blend' | 'per_step_clamp'

const INPAINT_PARAMETER_TOOLTIPS = {
  enforcement: {
    title: 'Enforcement',
    content: [
      'Controls when the preserved source image is reintroduced during masked denoise.',
      'Per-step blend: reapplies preserved content every step, usually keeping boundaries and structure tighter.',
      'Post-sample blend: lets sampling run freer, then blends preserved content back at the end, usually allowing larger interior changes.',
    ],
  },
  maskedContent: {
    title: 'Masked content',
    content: [
      'Chooses how the masked area is initialized before sampling.',
      'Original: starts from the current image crop.',
      'Fill: replaces the masked area with a blur-smear fill scaffold from surrounding pixels.',
      'Latent noise: injects fresh latent noise inside the mask for a stronger redraw.',
      'Latent nothing: zeros the masked latent, usually the most destructive reset.',
    ],
  },
  onlyMaskedPadding: {
    title: 'Only masked padding',
    content: [
      'Extra context around the masked crop used by the inpaint-only-masked pass.',
      'Increase: gives the model more surrounding context and can reduce seam pressure, but reprocesses a larger area.',
      'Decrease: keeps the edit tighter and faster, but can starve edge context and make seams harsher.',
    ],
  },
  maskBlur: {
    title: 'Mask blur',
    content: [
      'Softens the mask edge before the inpaint crop and conditioning bundle are prepared.',
      'Increase: widens the transition band and can hide hard cut lines, but lets the edit spill farther past the exact mask.',
      'Decrease: keeps the edit tighter to the painted mask, but edges can look harsher or more cut out.',
    ],
  },
} as const

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
  inpaintFullResPadding: number
  maskBlur: number
  maskRegionSplit?: boolean
  zoomFrameGuide?: WanImg2VidFrameGuideConfig | null
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
  maskRegionSplit: false,
  zoomFrameGuide: null,
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
  (e: 'update:inpaintFullResPadding', value: number): void
  (e: 'toggle:maskRegionSplit'): void
  (e: 'update:maskBlur', value: number): void
  (e: 'update:zoomFrameGuide', value: WanImg2VidFrameGuideConfig): void
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

function onInitPreviewClick(isDisabled: boolean, imageData: string): void {
  if (isDisabled || !imageData) return
  maskEditorOpen.value = true
}

function onMaskEditorExternalReset(message: string): void {
  emit('notice:maskEditorReset', message)
}

function onZoomFrameGuideUpdate(value: WanImg2VidFrameGuideConfig): void {
  emit('update:zoomFrameGuide', value)
}
</script>

<!-- styles in styles/components/img2img-inpaint-params-card.css -->
