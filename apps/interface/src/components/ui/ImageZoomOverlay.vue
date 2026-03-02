<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Full-screen image zoom overlay with pan/zoom controls and optional WAN frame-guide editing.
Provides a reusable overlay used by result previews and init-image previews, with close semantics for Escape/outside-click,
fit-to-viewport zoom behavior, and opt-in WAN guide editing (toggle, resize mode, target size, and drag crop offsets)
for no-stretch img2vid framing.

Symbols (top-level; keep in sync; no ghosts):
- `ImageZoomOverlay` (component): Full-screen image overlay with pan/zoom controls and optional WAN frame-guide editing.
- `close` (function): Closes the overlay and clears temporary listeners.
- `computeFitZoom` (function): Computes fit-to-viewport zoom with safe padding.
- `applyFitView` (function): Resets pan and applies fit zoom.
- `toggleFrameGuide` (function): Toggles WAN frame-guide visibility in the toolbar.
- `emitFrameGuideUpdate` (function): Normalizes and emits WAN frame-guide edits to parent components.
- `onGuideDragStart` (function): Starts drag tracking for WAN crop-guide movement.
- `onOverlayWheel` (function): Handles wheel zoom only on the main image region.
- `onWindowKeydown` (function): Handles keyboard shortcuts (`Escape` closes).
-->

<template>
  <div v-if="isOpen" class="image-zoom-overlay" @wheel="onOverlayWheel">
    <div ref="mainEl" class="image-zoom-main" @click="onMainClick">
      <div class="image-zoom-canvas" :style="zoomStyle" @click.stop>
        <img
          ref="imageEl"
          :src="src"
          :alt="alt"
          @load="onImageLoad"
          @mousedown.prevent="onPanStart"
        />
        <div
          v-if="frameGuideVisible"
          :class="['image-zoom-frame-guide', frameGuideDraggable ? 'image-zoom-frame-guide--draggable' : '']"
          :style="frameGuideRectStyle"
          aria-hidden="true"
          @mousedown.stop.prevent="onGuideDragStart"
        />
      </div>
    </div>
    <div :class="['image-zoom-toolbar', frameGuideConfigured ? 'image-zoom-toolbar--with-guide' : '']" @click.stop>
      <div class="toolbar-group">
        <button class="btn btn-sm btn-outline" type="button" @click="resetView">Fit</button>
        <button class="btn btn-sm btn-outline" type="button" @click="setZoom(1)">1:1</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomIn">+</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomOut">-</button>
        <button class="btn btn-sm btn-secondary" type="button" @click="close">Close</button>
      </div>
      <div v-if="frameGuideConfigured" class="toolbar-group image-zoom-toolbar__frame-guide">
        <button
          class="btn btn-sm btn-outline"
          type="button"
          :disabled="!frameProjection"
          @click="toggleFrameGuide"
        >
          {{ frameGuideVisible ? 'Guide: On' : 'Guide: Off' }}
        </button>

        <div class="image-zoom-guide-edit-row">
          <label class="label-muted" for="wan-guide-resize-mode">Resize</label>
          <select
            id="wan-guide-resize-mode"
            class="select-md"
            :value="frameGuideConfig?.resizeMode || 'auto'"
            @change="onResizeModeChange"
          >
            <option
              v-for="option in resizeModeOptions"
              :key="option.value"
              :value="option.value"
            >
              {{ option.label }}
            </option>
          </select>
        </div>

        <div class="image-zoom-guide-size-grid">
          <label class="label-muted" for="wan-guide-width">W</label>
          <div class="image-zoom-guide-size-input">
            <button class="btn btn-sm btn-outline" type="button" @click="nudgeGuideWidth(-1)">-</button>
            <input
              id="wan-guide-width"
              class="ui-input cdx-input-w-xs"
              type="number"
              :min="WAN_DIM_MIN"
              :max="WAN_DIM_MAX"
              :step="WAN_DIM_STEP"
              :value="frameGuideConfig?.targetWidth || WAN_DIM_MIN"
              @change="onGuideWidthChange"
            />
            <button class="btn btn-sm btn-outline" type="button" @click="nudgeGuideWidth(1)">+</button>
          </div>

          <label class="label-muted" for="wan-guide-height">H</label>
          <div class="image-zoom-guide-size-input">
            <button class="btn btn-sm btn-outline" type="button" @click="nudgeGuideHeight(-1)">-</button>
            <input
              id="wan-guide-height"
              class="ui-input cdx-input-w-xs"
              type="number"
              :min="WAN_DIM_MIN"
              :max="WAN_DIM_MAX"
              :step="WAN_DIM_STEP"
              :value="frameGuideConfig?.targetHeight || WAN_DIM_MIN"
              @change="onGuideHeightChange"
            />
            <button class="btn btn-sm btn-outline" type="button" @click="nudgeGuideHeight(1)">+</button>
          </div>
        </div>

        <div class="image-zoom-frame-meta">
          <div class="image-zoom-frame-meta__row"><span>Src</span><strong>{{ frameGuideSourceLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Scaled</span><strong>{{ frameGuideScaledLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Frame</span><strong>{{ frameGuideTargetLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Resize</span><strong>{{ frameGuideResizeLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Crop XY</span><strong>{{ frameGuideCropLabel }}</strong></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import {
  WAN_IMG2VID_RESIZE_MODE_LABELS,
  type WanImg2VidFrameGuideConfig,
  type WanImg2VidResizeMode,
  computeWanImg2VidFrameProjection,
  normalizeWanImg2VidResizeMode,
} from '../../utils/wan_img2vid_frame_projection'

const props = withDefaults(defineProps<{
  modelValue: boolean
  src: string
  alt?: string
  wanFrameGuide?: WanImg2VidFrameGuideConfig | null
}>(), {
  alt: 'Zoomed image',
  wanFrameGuide: null,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'update:wanFrameGuide', value: WanImg2VidFrameGuideConfig): void
}>()

const isOpen = computed(() => Boolean(props.modelValue) && Boolean(props.src))
const src = computed(() => props.src)
const alt = computed(() => props.alt || 'Zoomed image')

const ZOOM_MIN = 0.25
const ZOOM_MAX = 8
const FIT_PADDING_PX = 24
const WAN_DIM_MIN = 64
const WAN_DIM_MAX = 2048
const WAN_DIM_STEP = 16

const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
const mainEl = ref<HTMLElement | null>(null)
const imageEl = ref<HTMLImageElement | null>(null)
const sourceWidth = ref(0)
const sourceHeight = ref(0)
const showFrameGuide = ref(false)

let panState: { startX: number; startY: number; originX: number; originY: number } | null = null
let guideDragState: {
  startX: number
  startY: number
  originOffsetX: number
  originOffsetY: number
  slackX: number
  slackY: number
} | null = null

function requirePositiveInt(rawValue: number, label: string): number {
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric)) {
    throw new Error(`ImageZoomOverlay: ${label} must be finite (got ${String(rawValue)}).`)
  }
  const value = Math.trunc(numeric)
  if (value <= 0) {
    throw new Error(`ImageZoomOverlay: ${label} must be > 0 (got ${String(rawValue)}).`)
  }
  return value
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0.5
  return Math.max(0, Math.min(1, value))
}

function normalizeGuideOffset(rawValue: unknown, label: string): number {
  if (rawValue === undefined || rawValue === null || rawValue === '') return 0.5
  const numeric = Number(rawValue)
  if (!Number.isFinite(numeric) || numeric < 0 || numeric > 1) {
    throw new Error(`ImageZoomOverlay: ${label} must be a finite number in [0,1] (got ${String(rawValue)}).`)
  }
  return clamp01(numeric)
}

function snapGuideDim(rawValue: unknown): number {
  const numeric = Number(rawValue)
  const clamped = Number.isFinite(numeric)
    ? Math.max(WAN_DIM_MIN, Math.min(WAN_DIM_MAX, numeric))
    : WAN_DIM_MIN
  return Math.min(WAN_DIM_MAX, Math.max(WAN_DIM_MIN, Math.ceil(clamped / WAN_DIM_STEP) * WAN_DIM_STEP))
}

function normalizeFrameGuideConfig(rawGuide: WanImg2VidFrameGuideConfig): WanImg2VidFrameGuideConfig {
  return {
    targetWidth: snapGuideDim(requirePositiveInt(rawGuide.targetWidth, 'wanFrameGuide.targetWidth')),
    targetHeight: snapGuideDim(requirePositiveInt(rawGuide.targetHeight, 'wanFrameGuide.targetHeight')),
    resizeMode: normalizeWanImg2VidResizeMode(rawGuide.resizeMode, 'auto'),
    cropOffsetX: normalizeGuideOffset(rawGuide.cropOffsetX, 'wanFrameGuide.cropOffsetX'),
    cropOffsetY: normalizeGuideOffset(rawGuide.cropOffsetY, 'wanFrameGuide.cropOffsetY'),
  }
}

const frameGuideState = ref<WanImg2VidFrameGuideConfig | null>(null)

watch(
  () => props.wanFrameGuide,
  (guide) => {
    if (!guide) {
      frameGuideState.value = null
      showFrameGuide.value = false
      return
    }
    frameGuideState.value = normalizeFrameGuideConfig(guide)
  },
  { immediate: true, deep: true },
)

const frameGuideConfig = computed(() => frameGuideState.value)
const frameGuideConfigured = computed(() => frameGuideConfig.value !== null)

const frameProjection = computed(() => {
  const guide = frameGuideConfig.value
  if (!guide) return null
  if (sourceWidth.value <= 0 || sourceHeight.value <= 0) return null
  return computeWanImg2VidFrameProjection({
    sourceWidth: sourceWidth.value,
    sourceHeight: sourceHeight.value,
    frameWidth: guide.targetWidth,
    frameHeight: guide.targetHeight,
    resizeMode: guide.resizeMode,
    cropOffsetX: guide.cropOffsetX,
    cropOffsetY: guide.cropOffsetY,
  })
})

const frameGuideVisible = computed(() => Boolean(showFrameGuide.value && frameProjection.value))
const frameGuideDraggable = computed(() => {
  const projection = frameProjection.value
  if (!projection) return false
  return projection.slackX > 0 || projection.slackY > 0
})

const resizeModeOptions = computed(() => {
  return (Object.keys(WAN_IMG2VID_RESIZE_MODE_LABELS) as WanImg2VidResizeMode[]).map((mode) => ({
    value: mode,
    label: WAN_IMG2VID_RESIZE_MODE_LABELS[mode],
  }))
})

function formatDims(width: number, height: number): string {
  const w = Number.isFinite(width) ? Math.round(width) : 0
  const h = Number.isFinite(height) ? Math.round(height) : 0
  if (w <= 0 || h <= 0) return '—'
  return `${w}×${h}`
}

function formatOffset(value: number): string {
  if (!Number.isFinite(value)) return '0.500'
  return value.toFixed(3)
}

const frameGuideSourceLabel = computed(() => {
  const projection = frameProjection.value
  if (!projection) return '—'
  return formatDims(projection.sourceWidth, projection.sourceHeight)
})

const frameGuideScaledLabel = computed(() => {
  const projection = frameProjection.value
  if (!projection) return '—'
  return formatDims(projection.resizedWidth, projection.resizedHeight)
})

const frameGuideTargetLabel = computed(() => {
  const guide = frameGuideConfig.value
  if (!guide) return '—'
  return formatDims(guide.targetWidth, guide.targetHeight)
})

const frameGuideResizeLabel = computed(() => {
  const projection = frameProjection.value
  if (!projection) return WAN_IMG2VID_RESIZE_MODE_LABELS.auto
  const requested = WAN_IMG2VID_RESIZE_MODE_LABELS[projection.resizeMode]
  const resolved = WAN_IMG2VID_RESIZE_MODE_LABELS[projection.resolvedResizeMode]
  if (projection.resizeMode === 'auto') return `${requested} -> ${resolved}`
  return requested
})

const frameGuideCropLabel = computed(() => {
  const projection = frameProjection.value
  if (!projection) return '—'
  return `${formatOffset(projection.cropOffsetX)}, ${formatOffset(projection.cropOffsetY)}`
})

const frameGuideRectStyle = computed<CSSProperties>(() => {
  const projection = frameProjection.value
  if (!projection) return {}
  return {
    left: `${(projection.cropX / projection.sourceWidth) * 100}%`,
    top: `${(projection.cropY / projection.sourceHeight) * 100}%`,
    width: `${(projection.cropWidth / projection.sourceWidth) * 100}%`,
    height: `${(projection.cropHeight / projection.sourceHeight) * 100}%`,
  }
})

function clearPanListeners(): void {
  window.removeEventListener('mousemove', onPanMove)
  window.removeEventListener('mouseup', onPanEnd)
}

function clearGuideDragListeners(): void {
  window.removeEventListener('mousemove', onGuideDragMove)
  window.removeEventListener('mouseup', onGuideDragEnd)
}

function close(): void {
  emit('update:modelValue', false)
  panState = null
  guideDragState = null
  clearPanListeners()
  clearGuideDragListeners()
}

function emitFrameGuideUpdate(patch: Partial<WanImg2VidFrameGuideConfig>): void {
  const current = frameGuideConfig.value
  if (!current) return
  const next: WanImg2VidFrameGuideConfig = normalizeFrameGuideConfig({
    ...current,
    ...patch,
  })
  frameGuideState.value = next
  emit('update:wanFrameGuide', next)
}

watch(isOpen, (open) => {
  if (open) {
    showFrameGuide.value = false
    window.addEventListener('keydown', onWindowKeydown)
    void nextTick(() => {
      applyFitView()
    })
    return
  }
  window.removeEventListener('keydown', onWindowKeydown)
  panState = null
  guideDragState = null
  clearPanListeners()
  clearGuideDragListeners()
}, { immediate: true })

watch(src, () => {
  sourceWidth.value = 0
  sourceHeight.value = 0
  showFrameGuide.value = false
})

watch(frameProjection, (projection) => {
  if (!projection) {
    showFrameGuide.value = false
    guideDragState = null
    clearGuideDragListeners()
  }
})

function clampZoom(value: number): number {
  const minZoom = Math.min(ZOOM_MIN, computeFitZoom())
  if (!Number.isFinite(value)) return minZoom
  return Math.max(minZoom, Math.min(ZOOM_MAX, value))
}

function computeFitZoom(): number {
  const main = mainEl.value
  const image = imageEl.value
  if (!main || !image) return 1
  const naturalWidth = Number(image.naturalWidth)
  const naturalHeight = Number(image.naturalHeight)
  if (!Number.isFinite(naturalWidth) || !Number.isFinite(naturalHeight) || naturalWidth <= 0 || naturalHeight <= 0) return 1

  const availableWidth = Math.max(1, main.clientWidth - FIT_PADDING_PX * 2)
  const availableHeight = Math.max(1, main.clientHeight - FIT_PADDING_PX * 2)
  const fit = Math.min(availableWidth / naturalWidth, availableHeight / naturalHeight)
  if (!Number.isFinite(fit) || fit <= 0) return 1
  return Math.min(1, fit)
}

function applyFitView(): void {
  zoom.value = computeFitZoom()
  offsetX.value = 0
  offsetY.value = 0
}

function adjustZoom(delta: number): void {
  zoom.value = clampZoom(zoom.value + delta)
}

function zoomIn(): void {
  adjustZoom(0.25)
}

function zoomOut(): void {
  adjustZoom(-0.25)
}

function setZoom(value: number): void {
  zoom.value = clampZoom(value)
  offsetX.value = 0
  offsetY.value = 0
}

const zoomStyle = computed<CSSProperties>(() => ({
  position: 'relative',
  left: `${offsetX.value}px`,
  top: `${offsetY.value}px`,
  transform: `scale(${zoom.value})`,
}))

function resetView(): void {
  applyFitView()
}

function toggleFrameGuide(): void {
  if (!frameProjection.value) return
  showFrameGuide.value = !showFrameGuide.value
}

function onResizeModeChange(event: Event): void {
  const target = event.target as HTMLSelectElement
  emitFrameGuideUpdate({ resizeMode: normalizeWanImg2VidResizeMode(target.value, 'auto') })
}

function nudgeGuideWidth(direction: -1 | 1): void {
  const current = frameGuideConfig.value
  if (!current) return
  emitFrameGuideUpdate({ targetWidth: current.targetWidth + direction * WAN_DIM_STEP })
}

function nudgeGuideHeight(direction: -1 | 1): void {
  const current = frameGuideConfig.value
  if (!current) return
  emitFrameGuideUpdate({ targetHeight: current.targetHeight + direction * WAN_DIM_STEP })
}

function onGuideWidthChange(event: Event): void {
  const target = event.target as HTMLInputElement
  emitFrameGuideUpdate({ targetWidth: Number(target.value) })
}

function onGuideHeightChange(event: Event): void {
  const target = event.target as HTMLInputElement
  emitFrameGuideUpdate({ targetHeight: Number(target.value) })
}

function onPanStart(event: MouseEvent): void {
  panState = {
    startX: event.clientX,
    startY: event.clientY,
    originX: offsetX.value,
    originY: offsetY.value,
  }
  window.addEventListener('mousemove', onPanMove)
  window.addEventListener('mouseup', onPanEnd)
}

function onPanMove(event: MouseEvent): void {
  if (!panState) return
  const dx = event.clientX - panState.startX
  const dy = event.clientY - panState.startY
  offsetX.value = panState.originX + dx
  offsetY.value = panState.originY + dy
}

function onPanEnd(): void {
  panState = null
  clearPanListeners()
}

function onGuideDragStart(event: MouseEvent): void {
  const projection = frameProjection.value
  const guide = frameGuideConfig.value
  if (!projection || !guide || !frameGuideVisible.value) return
  if (projection.slackX <= 0 && projection.slackY <= 0) return
  guideDragState = {
    startX: event.clientX,
    startY: event.clientY,
    originOffsetX: clamp01(Number(guide.cropOffsetX)),
    originOffsetY: clamp01(Number(guide.cropOffsetY)),
    slackX: projection.slackX,
    slackY: projection.slackY,
  }
  window.addEventListener('mousemove', onGuideDragMove)
  window.addEventListener('mouseup', onGuideDragEnd)
}

function onGuideDragMove(event: MouseEvent): void {
  const drag = guideDragState
  if (!drag) return
  const effectiveZoom = Math.max(0.001, Number(zoom.value) || 1)
  const deltaSourceX = (event.clientX - drag.startX) / effectiveZoom
  const deltaSourceY = (event.clientY - drag.startY) / effectiveZoom
  const nextOffsetX = drag.slackX > 0
    ? clamp01(drag.originOffsetX + (deltaSourceX / drag.slackX))
    : drag.originOffsetX
  const nextOffsetY = drag.slackY > 0
    ? clamp01(drag.originOffsetY + (deltaSourceY / drag.slackY))
    : drag.originOffsetY
  emitFrameGuideUpdate({
    cropOffsetX: nextOffsetX,
    cropOffsetY: nextOffsetY,
  })
}

function onGuideDragEnd(): void {
  guideDragState = null
  clearGuideDragListeners()
}

function onOverlayWheel(event: WheelEvent): void {
  if (!isOpen.value) return
  const target = event.target as HTMLElement | null
  if (!target) return
  if (target.closest('.image-zoom-toolbar')) return
  if (!target.closest('.image-zoom-main') && !target.closest('.image-zoom-canvas')) return
  event.preventDefault()
  const delta = event.deltaY < 0 ? 0.25 : -0.25
  adjustZoom(delta)
}

function onMainClick(event: MouseEvent): void {
  if (event.target !== event.currentTarget) return
  close()
}

function onImageLoad(): void {
  if (!isOpen.value) return
  const image = imageEl.value
  sourceWidth.value = Number(image?.naturalWidth || 0)
  sourceHeight.value = Number(image?.naturalHeight || 0)
  applyFitView()
}

function onWindowKeydown(event: KeyboardEvent): void {
  if (event.key !== 'Escape') return
  if (!isOpen.value) return
  close()
}

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onWindowKeydown)
  panState = null
  guideDragState = null
  clearPanListeners()
  clearGuideDragListeners()
})
</script>
