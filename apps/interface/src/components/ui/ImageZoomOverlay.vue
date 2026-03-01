<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Full-screen image zoom overlay with pan/zoom controls and optional WAN frame projection guide.
Provides a reusable overlay used by result previews and init-image previews, with shared close semantics for Escape and outside-click,
true fit-to-viewport behavior for large images (without forcing a 25% minimum zoom on fit), and an opt-in no-stretch
`cover + center-crop` WAN frame guide (toggle + projection metadata) for init-image inspection.

Symbols (top-level; keep in sync; no ghosts):
- `ImageZoomOverlay` (component): Full-screen image overlay with pan/zoom controls and optional WAN frame guide.
- `clearPanListeners` (function): Removes temporary pan mouse listeners from `window`.
- `close` (function): Closes the overlay and clears pan listeners.
- `computeFitZoom` (function): Computes the fit-to-viewport zoom with safe padding.
- `applyFitView` (function): Resets pan and applies fit zoom.
- `adjustZoom` (function): Applies bounded zoom delta updates.
- `toggleFrameGuide` (function): Toggles the WAN frame guide visibility in the toolbar.
- `onOverlayWheel` (function): Handles wheel-based zoom updates.
- `onWindowKeydown` (function): Handles keyboard shortcuts (`Escape` closes).
-->

<template>
  <div v-if="isOpen" class="image-zoom-overlay" @wheel.prevent="onOverlayWheel">
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
          class="image-zoom-frame-guide"
          :style="frameGuideRectStyle"
          aria-hidden="true"
        />
      </div>
    </div>
    <div class="image-zoom-toolbar" @click.stop>
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
        <div class="image-zoom-frame-meta">
          <div class="image-zoom-frame-meta__row"><span>Src</span><strong>{{ frameGuideSourceLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Scaled</span><strong>{{ frameGuideScaledLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Frame</span><strong>{{ frameGuideTargetLabel }}</strong></div>
          <div class="image-zoom-frame-meta__row"><span>Policy</span><strong>{{ frameGuidePolicyLabel }}</strong></div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import {
  WAN_IMG2VID_FRAME_PROJECTION_POLICY_LABEL,
  type WanImg2VidFrameGuideConfig,
  type WanImg2VidFrameProjectionPolicy,
  computeWanImg2VidFrameProjection,
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
}>()

const isOpen = computed(() => Boolean(props.modelValue) && Boolean(props.src))
const src = computed(() => props.src)
const alt = computed(() => props.alt || 'Zoomed image')

const ZOOM_MIN = 0.25
const ZOOM_MAX = 8
const FIT_PADDING_PX = 24

const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)
const mainEl = ref<HTMLElement | null>(null)
const imageEl = ref<HTMLImageElement | null>(null)
const sourceWidth = ref(0)
const sourceHeight = ref(0)
const showFrameGuide = ref(false)
let panState: { startX: number; startY: number; originX: number; originY: number } | null = null

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

function resolveProjectionPolicy(rawPolicy: string | undefined): {
  policy: WanImg2VidFrameProjectionPolicy
  label: string
} {
  const raw = String(rawPolicy || '').trim()
  if (!raw || raw === 'cover_center_crop') {
    return {
      policy: 'cover_center_crop',
      label: WAN_IMG2VID_FRAME_PROJECTION_POLICY_LABEL,
    }
  }
  throw new Error(
    `ImageZoomOverlay: unsupported wanFrameGuide.policy '${raw}' (expected 'cover_center_crop').`,
  )
}

const frameGuideConfig = computed(() => {
  if (!props.wanFrameGuide) return null
  const resolvedPolicy = resolveProjectionPolicy(props.wanFrameGuide.policy)
  return {
    targetWidth: requirePositiveInt(props.wanFrameGuide.targetWidth, 'wanFrameGuide.targetWidth'),
    targetHeight: requirePositiveInt(props.wanFrameGuide.targetHeight, 'wanFrameGuide.targetHeight'),
    policy: resolvedPolicy.policy,
    policyLabel: resolvedPolicy.label,
  }
})

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
    policy: guide.policy,
  })
})

const frameGuideVisible = computed(() => Boolean(showFrameGuide.value && frameProjection.value))

function formatDims(width: number, height: number): string {
  const w = Number.isFinite(width) ? Math.round(width) : 0
  const h = Number.isFinite(height) ? Math.round(height) : 0
  if (w <= 0 || h <= 0) return '—'
  return `${w}×${h}`
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

const frameGuidePolicyLabel = computed(() => {
  const guide = frameGuideConfig.value
  if (!guide) return WAN_IMG2VID_FRAME_PROJECTION_POLICY_LABEL
  return guide.policyLabel
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

function close(): void {
  emit('update:modelValue', false)
  panState = null
  clearPanListeners()
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
  clearPanListeners()
}, { immediate: true })

watch(src, () => {
  sourceWidth.value = 0
  sourceHeight.value = 0
  showFrameGuide.value = false
})

watch(frameProjection, (projection) => {
  if (!projection) showFrameGuide.value = false
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

function onOverlayWheel(event: WheelEvent): void {
  if (!isOpen.value) return
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
  clearPanListeners()
})
</script>
