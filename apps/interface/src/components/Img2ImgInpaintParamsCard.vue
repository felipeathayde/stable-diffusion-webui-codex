<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational parameter card for image init/mask workflows.
Groups img2img controls (initial image) and optional inpaint controls (canvas-mask tools + enforcement/fill + masked padding + mask blur + region splitting),
including dropzone/thumb/zoom handling for init images, rejected-file pass-through emits for parent toasts, and optional
embedded/title/label overrides so non-image tabs can reuse the same card shell without duplicating UI logic.
Supports optional pass-through WAN zoom frame-guide config for init-image overlays.
Saved inpaint masks can preview their hard mask, outward blur-spill range, and effective masked-region crop directly on the init-image thumbnail.
Init-image filename captions are centered in the footer area for clearer media identification.

Symbols (top-level; keep in sync; no ghosts):
- `Img2ImgInpaintParamsCard` (component): Presentational card for img2img/inpaint parameter controls.
- `INPAINT_PARAMETER_TOOLTIPS` (constant): Tooltip copy for inpaint select, slider, and split-toggle controls.
- `zoomFrameGuide` (prop): Optional WAN frame-guide config forwarded to `InitialImageCard` zoom overlay.
- `onZoomFrameGuideUpdate` (function): Forwards zoom-overlay guide edits to parent WAN state.
- `onMaskEnforcementChange` (function): Emits raw mask enforcement select updates for parent-side normalization.
- `onInpaintingFillChange` (function): Emits raw masked-content numeric updates for parent-side normalization.
- `onMaskEditorApply` (function): Emits edited mask data URL produced by the inpaint mask editor overlay.
- `onInitPreviewClick` (function): Opens the mask editor when inpaint mode is active and an init image is present.
- `onMaskEditorExternalReset` (function): Forwards editor reset notices for parent-side toasts.
- `loadMaskPreviewPlaneFromSource` (function): Decodes the saved mask PNG into a binary plane for preview geometry.
- `schedulePreviewMeasurement` (function): Measures the actual thumbnail box after layout so image-space crop geometry projects truthfully.
- `previewBlurSource` (ref): Cached RGBA data URL for the inline thumbnail blur-spill preview.
- `schedulePreviewBlurSourceRender` (function): Coalesces thumbnail blur-spill raster updates after mask/blur changes.
- `previewCropStyle` (computed): Projects the effective masked-region crop box into the contained thumbnail image area.
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
        <div v-if="maskImageData" ref="previewOverlayEl" class="img2img-mask-preview-layers">
          <img
            v-if="showPreviewBlur"
            class="img2img-mask-preview-spill"
            :src="previewBlurSource"
            alt=""
            aria-hidden="true"
          >
          <div class="img2img-mask-preview-overlay" :style="previewMaskStyle" />
          <div v-if="previewCropStyle" class="img2img-mask-preview-crop-box" :style="previewCropStyle" />
        </div>
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
          <HoverTooltip
            class="cdx-slider-field__label-tooltip"
            :title="INPAINT_PARAMETER_TOOLTIPS.splitMaskRegions.title"
            :content="INPAINT_PARAMETER_TOOLTIPS.splitMaskRegions.content"
            :wrapperFocusable="false"
          >
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', maskRegionSplit ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="maskRegionSplit"
              :disabled="disabled"
              @click="emit('toggle:maskRegionSplit')"
            >
              Split mask regions
            </button>
          </HoverTooltip>
        </div>
      </div>

      <div class="gc-row img2img-mask-slider-row">
        <SliderField
          class="gc-col gc-col--wide"
          label="Masked padding"
          :tooltip="INPAINT_PARAMETER_TOOLTIPS.maskedPadding.content"
          :tooltipTitle="INPAINT_PARAMETER_TOOLTIPS.maskedPadding.title"
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
      :processing-width="effectiveProcessingWidth"
      :processing-height="effectiveProcessingHeight"
      :mask-blur="maskBlur"
      :masked-padding="inpaintFullResPadding"
      @apply="onMaskEditorApply"
      @external-reset="onMaskEditorExternalReset"
      @update:maskBlur="(value) => emit('update:maskBlur', value)"
      @update:maskedPadding="(value) => emit('update:inpaintFullResPadding', value)"
    />
  </div>
</template>

<script setup lang="ts">
import { computed, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import InitialImageCard from './InitialImageCard.vue'
import InpaintMaskEditorOverlay from './ui/InpaintMaskEditorOverlay.vue'
import HoverTooltip from './ui/HoverTooltip.vue'
import SliderField from './ui/SliderField.vue'
import { rgbaToMaskPlane } from './ui/inpaint_mask_editor_engine'
import WanSubHeader from './wan/WanSubHeader.vue'
import {
  computeContainedImageRect,
  computeInpaintMaskBlurSpillAlphaPlane,
  computeInpaintMaskPreviewGeometry,
  projectImageRectToContainer,
  tintAlphaPlaneToRgba,
} from '../utils/inpaint_mask_preview'
import type { WanImg2VidFrameGuideConfig } from '../utils/wan_img2vid_frame_projection'

type MaskEnforcement = 'post_blend' | 'per_step_clamp'

const PREVIEW_BLUR_TINT = {
  red: 255,
  green: 178,
  blue: 68,
  opacity: 0.62,
} as const

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
      '[[Original:]] starts from the current image crop.',
      '[[Fill:]] replaces the masked area with a blur-smear fill scaffold from surrounding pixels.',
      '[[Latent noise:]] injects fresh latent noise inside the mask for a stronger redraw.',
      '[[Latent nothing:]] zeros the masked latent, usually the most destructive reset.',
    ],
  },
  splitMaskRegions: {
    title: 'Split mask regions',
    content: [
      'Runs disconnected mask islands as separate inpaint passes instead of one combined masked region.',
      '[[Enabled:]] isolated holes stay decoupled, which can reduce one region bleeding into another.',
      '[[Disabled:]] the full masked area is processed as one region.',
      'Requires batch size = 1, does not work with Invert mask, and unsupported engines still fail loud.',
    ],
  },
  maskedPadding: {
    title: 'Masked padding',
    content: [
      'Extra context around the masked crop used by the inpaint-only-masked pass.',
      '[[Increase:]] gives the model more surrounding context and can reduce seam pressure, but reprocesses a larger area.',
      '[[Decrease:]] keeps the edit tighter and faster, but can starve edge context and make seams harsher.',
    ],
  },
  maskBlur: {
    title: 'Mask blur',
    content: [
      'Softens the mask edge before the inpaint crop and conditioning bundle are prepared.',
      '[[Increase:]] widens the transition band and can hide hard cut lines, but lets the edit spill farther past the exact mask.',
      '[[Decrease:]] keeps the edit tighter to the painted mask, but edges can look harsher or more cut out.',
    ],
  },
} as const

const props = withDefaults(defineProps<{
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
  processingWidth?: number
  processingHeight?: number
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
const previewOverlayEl = ref<HTMLElement | null>(null)
const previewContainerWidth = ref(0)
const previewContainerHeight = ref(0)
const previewMaskPlane = ref<Uint8Array | null>(null)
const previewMaskDecodeToken = ref(0)
const previewBlurSource = ref('')
let previewResizeObserver: ResizeObserver | null = null
let previewMeasureRafId = 0
let previewBlurRenderRafId = 0
let previewBlurCanvas: HTMLCanvasElement | null = null

const effectiveProcessingWidth = computed(() => {
  const width = Number(props.processingWidth)
  if (Number.isFinite(width) && width > 0) return Math.trunc(width)
  return Math.max(1, Math.trunc(props.imageWidth))
})

const effectiveProcessingHeight = computed(() => {
  const height = Number(props.processingHeight)
  if (Number.isFinite(height) && height > 0) return Math.trunc(height)
  return Math.max(1, Math.trunc(props.imageHeight))
})

const previewGeometry = computed(() => {
  const maskPlane = previewMaskPlane.value
  if (!maskPlane) return null
  try {
    return computeInpaintMaskPreviewGeometry(maskPlane, {
      imageWidth: props.imageWidth,
      imageHeight: props.imageHeight,
      processingWidth: effectiveProcessingWidth.value,
      processingHeight: effectiveProcessingHeight.value,
      maskBlur: props.maskBlur,
      maskedPadding: props.inpaintFullResPadding,
    })
  } catch (error) {
    console.error('[Img2ImgInpaintParamsCard] Failed to compute inpaint preview geometry.', error)
    return null
  }
})

const containedPreviewRect = computed(() => {
  if (previewContainerWidth.value <= 0 || previewContainerHeight.value <= 0) return null
  try {
    return computeContainedImageRect(
      previewContainerWidth.value,
      previewContainerHeight.value,
      props.imageWidth,
      props.imageHeight,
    )
  } catch (error) {
    console.error('[Img2ImgInpaintParamsCard] Failed to project inpaint preview into thumbnail.', error)
    return null
  }
})

const previewMaskStyle = computed<CSSProperties>(() => ({
  '--img2img-mask-src': `url('${props.maskImageData}')`,
}))

const showPreviewBlur = computed(() => Boolean(previewBlurSource.value))

const previewCropStyle = computed<CSSProperties | null>(() => {
  const geometry = previewGeometry.value
  const containedRect = containedPreviewRect.value
  if (!geometry || !containedRect) return null
  const projected = projectImageRectToContainer(geometry.cropRegion, containedRect)
  return {
    left: `${projected.x1}px`,
    top: `${projected.y1}px`,
    width: `${projected.width}px`,
    height: `${projected.height}px`,
  }
})

watch(
  [() => props.maskImageData, () => props.imageWidth, () => props.imageHeight],
  ([sourceMask, imageWidth, imageHeight]) => {
    const token = (previewMaskDecodeToken.value += 1)
    const source = String(sourceMask || '').trim()
    if (!source || imageWidth <= 0 || imageHeight <= 0) {
      cancelPreviewBlurSourceRender()
      previewMaskPlane.value = null
      previewBlurSource.value = ''
      return
    }
    cancelPreviewBlurSourceRender()
    previewMaskPlane.value = null
    previewBlurSource.value = ''
    void loadMaskPreviewPlaneFromSource(source, imageWidth, imageHeight)
      .then((maskPlane) => {
        if (previewMaskDecodeToken.value !== token) return
        previewMaskPlane.value = maskPlane
      })
      .catch((error) => {
        if (previewMaskDecodeToken.value !== token) return
        previewMaskPlane.value = null
        console.error('[Img2ImgInpaintParamsCard] Failed to decode saved mask preview.', error)
      })
  },
  { immediate: true },
)

watch(
  [previewMaskPlane, () => props.maskBlur, () => props.imageWidth, () => props.imageHeight],
  () => {
    schedulePreviewBlurSourceRender()
  },
  { immediate: true },
)

watch(
  previewOverlayEl,
  (element) => {
    previewResizeObserver?.disconnect()
    previewResizeObserver = null
    cancelPreviewMeasurement()
    if (!element) {
      previewContainerWidth.value = 0
      previewContainerHeight.value = 0
      return
    }
    schedulePreviewMeasurement()
    if (typeof ResizeObserver !== 'undefined') {
      previewResizeObserver = new ResizeObserver(() => {
        schedulePreviewMeasurement()
      })
      previewResizeObserver.observe(element)
    }
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  previewResizeObserver?.disconnect()
  previewResizeObserver = null
  cancelPreviewMeasurement()
  cancelPreviewBlurSourceRender()
})

function cancelPreviewMeasurement(): void {
  if (previewMeasureRafId <= 0) return
  window.cancelAnimationFrame(previewMeasureRafId)
  previewMeasureRafId = 0
}

function cancelPreviewBlurSourceRender(): void {
  if (previewBlurRenderRafId <= 0) return
  window.cancelAnimationFrame(previewBlurRenderRafId)
  previewBlurRenderRafId = 0
}

function schedulePreviewMeasurement(): void {
  cancelPreviewMeasurement()
  previewMeasureRafId = window.requestAnimationFrame(() => {
    previewMeasureRafId = 0
    const element = previewOverlayEl.value
    if (!element) {
      previewContainerWidth.value = 0
      previewContainerHeight.value = 0
      return
    }
    const rect = element.getBoundingClientRect()
    previewContainerWidth.value = rect.width
    previewContainerHeight.value = rect.height
  })
}

function schedulePreviewBlurSourceRender(): void {
  cancelPreviewBlurSourceRender()
  previewBlurRenderRafId = window.requestAnimationFrame(() => {
    previewBlurRenderRafId = 0
    renderPreviewBlurSource()
  })
}

function getPreviewBlurCanvas(width: number, height: number): HTMLCanvasElement {
  if (!previewBlurCanvas) {
    previewBlurCanvas = document.createElement('canvas')
  }
  if (previewBlurCanvas.width !== width) previewBlurCanvas.width = width
  if (previewBlurCanvas.height !== height) previewBlurCanvas.height = height
  return previewBlurCanvas
}

function renderPreviewBlurSource(): void {
  const maskPlane = previewMaskPlane.value
  const imageWidth = Math.trunc(props.imageWidth)
  const imageHeight = Math.trunc(props.imageHeight)
  if (!maskPlane || imageWidth <= 0 || imageHeight <= 0) {
    previewBlurSource.value = ''
    return
  }

  try {
    const alphaPlane = computeInpaintMaskBlurSpillAlphaPlane(maskPlane, {
      imageWidth,
      imageHeight,
      maskBlur: props.maskBlur,
    })
    if (!alphaPlane) {
      previewBlurSource.value = ''
      return
    }

    const canvas = getPreviewBlurCanvas(imageWidth, imageHeight)
    const context = canvas.getContext('2d')
    if (!context) {
      throw new Error('Failed to create canvas context for inpaint blur thumbnail preview.')
    }
    context.clearRect(0, 0, imageWidth, imageHeight)
    const imageData = context.createImageData(imageWidth, imageHeight)
    imageData.data.set(tintAlphaPlaneToRgba(alphaPlane, imageWidth, imageHeight, PREVIEW_BLUR_TINT))
    context.putImageData(imageData, 0, 0)
    previewBlurSource.value = canvas.toDataURL('image/png')
  } catch (error) {
    previewBlurSource.value = ''
    console.error('[Img2ImgInpaintParamsCard] Failed to render inpaint blur preview source.', error)
  }
}

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

async function loadMaskPreviewPlaneFromSource(
  sourceMask: string,
  imageWidth: number,
  imageHeight: number,
): Promise<Uint8Array> {
  const maskImage = await loadMaskPreviewImage(sourceMask)
  const naturalWidth = maskImage.naturalWidth || maskImage.width
  const naturalHeight = maskImage.naturalHeight || maskImage.height
  if (naturalWidth !== imageWidth || naturalHeight !== imageHeight) {
    throw new Error(`Expected mask preview dimensions ${imageWidth}x${imageHeight}, got ${naturalWidth}x${naturalHeight}.`)
  }

  const canvas = document.createElement('canvas')
  canvas.width = imageWidth
  canvas.height = imageHeight
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Failed to create canvas context for saved mask preview decode.')
  }
  context.drawImage(maskImage, 0, 0, imageWidth, imageHeight)
  const rgba = context.getImageData(0, 0, imageWidth, imageHeight).data
  return rgbaToMaskPlane(rgba, imageWidth, imageHeight)
}

function loadMaskPreviewImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Failed to decode saved mask preview image.'))
    image.src = src
  })
}
</script>

<!-- styles in styles/components/img2img-inpaint-params-card.css -->
