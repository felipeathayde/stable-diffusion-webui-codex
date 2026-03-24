<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Full-screen inpaint mask editor overlay for img2img/inpaint workflows.
Provides practical mask tools (brush, eraser, circle, polygon), shared blur/padding sliders, zoom/pan viewport controls,
undo/redo with deep history, mask upload import (auto-stretched to init-image dimensions), and apply/close semantics while
keeping state presentational (props/emits only).

Symbols (top-level; keep in sync; no ghosts):
- `InpaintMaskEditorOverlay` (component): Full-screen inpaint mask editor overlay.
- `loadMaskPlaneFromSources` (function): Loads/validates init+mask sources and initializes draft state.
- `renderMaskCanvas` (function): Renders current draft mask and active shape previews.
- `renderPreviewCanvas` (function): Renders blur halo and crop-region preview layers from the current working mask.
- `onStagePointerDown` (function): Handles drawing/panning pointer-down interactions.
- `onStagePointerMove` (function): Handles drawing/panning pointer-move interactions.
- `onStagePointerUp` (function): Commits or ends active interactions on pointer-up.
- `onMaskUploadInputChange` (function): Imports an uploaded mask image, stretches it to canvas dimensions, and commits it to history.
- `applyDraftMask` (function): Exports current draft mask and emits `apply`.
- `resetDraftFromSource` (function): Discards local draft and reloads source-derived baseline.
- `previewCropStyle` (ref): Pixel-space crop-box style for the current effective masked region.
-->

<template>
  <div
    v-if="isOpen"
    class="inpaint-mask-editor-overlay"
    @contextmenu.prevent
  >
    <div class="inpaint-mask-editor-main" @click="onMainClick">
      <div
        ref="stageEl"
        class="inpaint-mask-editor-stage"
        @wheel.prevent="onOverlayWheel"
        @pointerdown="onStagePointerDown"
        @pointermove="onStagePointerMove"
        @pointerup="onStagePointerUp"
        @pointercancel="onStagePointerCancel"
      >
        <div
          v-if="hasRenderableSource"
          ref="contentEl"
          class="inpaint-mask-editor-content"
          :style="contentTransformStyle"
          @click.stop
        >
          <img
            ref="initImageEl"
            class="inpaint-mask-editor-base-image"
            :src="initImageData"
            alt="Inpaint source"
            draggable="false"
            @load="onBaseImageLoad"
          >
          <canvas
            ref="previewCanvasEl"
            class="inpaint-mask-editor-preview-canvas"
            :width="imageWidth"
            :height="imageHeight"
          />
          <canvas
            ref="maskCanvasEl"
            class="inpaint-mask-editor-mask-canvas"
            :width="imageWidth"
            :height="imageHeight"
          />
          <div
            v-if="previewCropStyle"
            class="inpaint-mask-editor-crop-box"
            :style="previewCropStyle"
          />
          <div
            v-if="showBrushCursor"
            class="inpaint-mask-editor-cursor"
            :style="brushCursorStyle"
          />
        </div>
      </div>
    </div>

    <aside class="inpaint-mask-editor-toolbar" @click.stop>
      <div class="inpaint-mask-editor-toolbar__header">
        <strong>Mask Editor</strong>
        <span class="caption">{{ imageWidth }}×{{ imageHeight }}</span>
      </div>

      <div class="toolbar-group inpaint-mask-editor-toolbar__tools">
        <button
          v-for="entry in toolChoices"
          :key="entry.value"
          :class="['btn', 'btn-sm', activeTool === entry.value ? 'btn-secondary' : 'btn-outline']"
          type="button"
          @click="setTool(entry.value)"
        >
          {{ entry.label }}
        </button>
      </div>

      <div class="toolbar-group">
        <label class="label-muted" for="mask-editor-brush-size">Brush size</label>
        <div class="inpaint-mask-editor-toolbar__brush">
          <input
            id="mask-editor-brush-size"
            v-model.number="brushSize"
            class="slider"
            type="range"
            :min="1"
            :max="256"
            :step="1"
          >
          <input
            v-model.number="brushSize"
            class="ui-input cdx-input-w-xs"
            type="number"
            :min="1"
            :max="256"
            :step="1"
          >
        </div>
      </div>

      <div class="toolbar-group inpaint-mask-editor-toolbar__param-sliders">
        <SliderField
          label="Masked padding"
          :modelValue="maskedPadding"
          :min="0"
          :max="256"
          :step="1"
          :inputStep="1"
          inputClass="cdx-input-w-xs"
          :disabled="!engine || loadingSource"
          @update:modelValue="(value) => emit('update:maskedPadding', value)"
        />

        <SliderField
          label="Mask blur"
          :modelValue="maskBlur"
          :min="0"
          :max="64"
          :step="1"
          :inputStep="1"
          inputClass="cdx-input-w-xs"
          :disabled="!engine || loadingSource"
          @update:modelValue="(value) => emit('update:maskBlur', value)"
        />
      </div>

      <div class="toolbar-group inpaint-mask-editor-toolbar__actions">
        <button class="btn btn-sm btn-outline" type="button" @click="resetView">Fit</button>
        <button class="btn btn-sm btn-outline" type="button" @click="setZoom(1)">1:1</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomIn">+</button>
        <button class="btn btn-sm btn-outline" type="button" @click="zoomOut">-</button>
      </div>

      <div class="toolbar-group inpaint-mask-editor-toolbar__actions">
        <input
          ref="maskUploadInputEl"
          class="inpaint-mask-editor-upload-input"
          type="file"
          accept="image/*"
          @change="onMaskUploadInputChange"
        >
        <button class="btn btn-sm btn-outline" type="button" :disabled="!engine || loadingSource || uploadInFlight" @click="triggerMaskUpload">Upload mask</button>
        <button class="btn btn-sm btn-outline" type="button" :disabled="!canUndo" @click="undo">Undo</button>
        <button class="btn btn-sm btn-outline" type="button" :disabled="!canRedo" @click="redo">Redo</button>
        <button class="btn btn-sm btn-outline" type="button" @click="clearMask">Clear</button>
        <button class="btn btn-sm btn-destructive" type="button" @click="resetDraftFromSource">Reset draft</button>
      </div>

      <div class="inpaint-mask-editor-toolbar__shortcuts caption">
        <p>Polygon: Enter commit · Esc cancel · Backspace remove point.</p>
        <p>Undo/redo: Ctrl/Cmd+Z · Ctrl/Cmd+Shift+Z.</p>
        <p>Pan: right/middle drag. Close keeps draft.</p>
      </div>

      <div class="toolbar-group inpaint-mask-editor-toolbar__footer">
        <button class="btn btn-sm btn-secondary" type="button" @click="closeOverlay">Close</button>
        <button class="btn btn-sm btn-primary" type="button" :disabled="!engine" @click="applyDraftMask">Apply mask</button>
      </div>
    </aside>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, ref, watch, type CSSProperties } from 'vue'
import SliderField from './SliderField.vue'
import {
  InpaintMaskEditorEngine,
  MASK_VALUE_EMPTY,
  MASK_VALUE_FILLED,
  applyCircleToMask,
  applyPolygonToMask,
  applyStrokeToMask,
  maskPlaneToRgba,
  rgbaToMaskPlane,
  type MaskPoint,
} from './inpaint_mask_editor_engine'
import { computeInpaintMaskPreviewGeometry } from '../../utils/inpaint_mask_preview'

type ToolName = 'brush' | 'eraser' | 'circle' | 'polygon'

interface PanState {
  pointerId: number
  startClientX: number
  startClientY: number
  originOffsetX: number
  originOffsetY: number
}

interface CirclePreviewState {
  center: MaskPoint
  radius: number
}

const props = withDefaults(defineProps<{
  modelValue: boolean
  initImageData: string
  initialMaskData?: string
  imageWidth: number
  imageHeight: number
  processingWidth?: number
  processingHeight?: number
  maskBlur?: number
  maskedPadding?: number
}>(), {
  initialMaskData: '',
  processingWidth: 0,
  processingHeight: 0,
  maskBlur: 0,
  maskedPadding: 0,
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'apply', maskDataUrl: string): void
  (e: 'external-reset', message: string): void
  (e: 'update:maskBlur', value: number): void
  (e: 'update:maskedPadding', value: number): void
}>()

const ZOOM_MIN = 0.1
const ZOOM_MAX = 8
const FIT_PADDING_PX = 24
const TOOL_VALUE_BRUSH: ToolName = 'brush'
const TOOL_VALUE_ERASER: ToolName = 'eraser'
const TOOL_VALUE_CIRCLE: ToolName = 'circle'
const TOOL_VALUE_POLYGON: ToolName = 'polygon'

const toolChoices: Array<{ value: ToolName; label: string }> = [
  { value: TOOL_VALUE_BRUSH, label: 'Brush' },
  { value: TOOL_VALUE_ERASER, label: 'Eraser' },
  { value: TOOL_VALUE_CIRCLE, label: 'Circle' },
  { value: TOOL_VALUE_POLYGON, label: 'Polygon' },
]

const activeTool = ref<ToolName>(TOOL_VALUE_BRUSH)
const brushSize = ref<number>(32)

const stageEl = ref<HTMLElement | null>(null)
const contentEl = ref<HTMLElement | null>(null)
const initImageEl = ref<HTMLImageElement | null>(null)
const previewCanvasEl = ref<HTMLCanvasElement | null>(null)
const maskCanvasEl = ref<HTMLCanvasElement | null>(null)
const maskUploadInputEl = ref<HTMLInputElement | null>(null)

const zoom = ref(1)
const offsetX = ref(0)
const offsetY = ref(0)

const engine = ref<InpaintMaskEditorEngine | null>(null)
const baselineMask = ref<Uint8Array | null>(null)
const hasInitializedSource = ref(false)
const loadingSource = ref(false)
const loadedSourceFingerprint = ref('')

const canUndo = ref(false)
const canRedo = ref(false)
const uploadInFlight = ref(false)
const uploadRequestToken = ref(0)

const previewPointerPoint = ref<MaskPoint | null>(null)
const panState = ref<PanState | null>(null)
const activeDrawPointerId = ref<number | null>(null)
const strokePoints = ref<MaskPoint[] | null>(null)
const circlePreview = ref<CirclePreviewState | null>(null)
const polygonPoints = ref<MaskPoint[]>([])
const transientMask = ref<Uint8Array | null>(null)
const previewCropStyle = ref<CSSProperties | null>(null)
let previewBufferCanvas: HTMLCanvasElement | null = null

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

const isOpen = computed(() => Boolean(props.modelValue) && Boolean(props.initImageData))
const hasRenderableSource = computed(() => Boolean(props.initImageData) && props.imageWidth > 0 && props.imageHeight > 0)

const contentTransformStyle = computed<CSSProperties>(() => ({
  width: `${props.imageWidth}px`,
  height: `${props.imageHeight}px`,
  left: `${offsetX.value}px`,
  top: `${offsetY.value}px`,
  transform: `scale(${zoom.value})`,
}))

const showBrushCursor = computed(() => {
  if (!previewPointerPoint.value) return false
  if (!hasRenderableSource.value) return false
  return activeTool.value !== TOOL_VALUE_POLYGON
})

const brushCursorStyle = computed<CSSProperties>(() => {
  const point = previewPointerPoint.value
  if (!point) {
    return {
      width: '0px',
      height: '0px',
      left: '0px',
      top: '0px',
    }
  }
  const radius = resolveCursorRadius()
  return {
    width: `${radius * 2}px`,
    height: `${radius * 2}px`,
    left: `${point.x - radius}px`,
    top: `${point.y - radius}px`,
  }
})

watch(
  () => props.modelValue,
  (open) => {
    if (open) {
      window.addEventListener('keydown', onWindowKeydown)
      void ensureLoadedAndRender(false)
      return
    }
    window.removeEventListener('keydown', onWindowKeydown)
    resetInteractionState(false)
  },
  { immediate: true },
)

watch(
  [() => props.initImageData, () => props.initialMaskData, () => props.imageWidth, () => props.imageHeight],
  () => {
    if (!hasInitializedSource.value) return
    void ensureLoadedAndRender(true)
  },
)

watch(
  [() => props.maskBlur, () => props.maskedPadding, () => props.processingWidth, () => props.processingHeight],
  () => {
    if (!hasInitializedSource.value) return
    if (!isOpen.value) return
    renderMaskCanvas()
  },
)

watch(
  hasRenderableSource,
  (renderable) => {
    if (renderable) return
    if (!props.modelValue) return
    emit('external-reset', 'Mask editor closed: init image source is unavailable.')
    emit('update:modelValue', false)
  },
)

watch(
  () => brushSize.value,
  (value) => {
    const numeric = Number(value)
    if (!Number.isFinite(numeric)) {
      brushSize.value = 1
      return
    }
    const clamped = Math.max(1, Math.min(256, Math.trunc(numeric)))
    if (clamped !== value) brushSize.value = clamped
  },
)

async function ensureLoadedAndRender(notifyResetWhenOpen: boolean): Promise<void> {
  if (!hasRenderableSource.value) return
  if (loadingSource.value) return
  const nextFingerprint = buildSourceFingerprint()
  if (hasInitializedSource.value && loadedSourceFingerprint.value === nextFingerprint) {
    if (isOpen.value) {
      await nextTick()
      renderMaskCanvas()
    }
    return
  }
  loadingSource.value = true
  try {
    const maskPlane = await loadMaskPlaneFromSources()
    const hadExistingDraft = hasInitializedSource.value
    baselineMask.value = maskPlane
    loadedSourceFingerprint.value = nextFingerprint
    engine.value = new InpaintMaskEditorEngine({
      width: props.imageWidth,
      height: props.imageHeight,
      initialMask: maskPlane,
    })
    hasInitializedSource.value = true
    resetInteractionState(true)
    syncHistoryFlags()
    if (isOpen.value) {
      await nextTick()
      applyFitView()
      renderMaskCanvas()
    }
    if (notifyResetWhenOpen && hadExistingDraft && isOpen.value) {
      emit('external-reset', 'Mask editor draft reset: init image or mask source changed.')
    }
  } catch (error) {
    engine.value = null
    baselineMask.value = null
    hasInitializedSource.value = false
    loadedSourceFingerprint.value = ''
    resetInteractionState(true)
    syncHistoryFlags()
    if (isOpen.value) {
      const message = error instanceof Error ? error.message : String(error)
      emit('external-reset', `Mask editor unavailable: ${message}`)
      emit('update:modelValue', false)
    }
  } finally {
    loadingSource.value = false
  }
}

function buildSourceFingerprint(): string {
  return `${props.imageWidth}x${props.imageHeight}|${props.initImageData}|${props.initialMaskData}`
}

async function loadMaskPlaneFromSources(): Promise<Uint8Array> {
  const width = Math.trunc(props.imageWidth)
  const height = Math.trunc(props.imageHeight)
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Mask editor requires positive image dimensions.')
  }
  const sourceMask = String(props.initialMaskData || '').trim()
  if (!sourceMask) {
    return new Uint8Array(width * height)
  }

  const image = await loadImage(sourceMask)
  const naturalWidth = image.naturalWidth || image.width
  const naturalHeight = image.naturalHeight || image.height
  if (naturalWidth !== width || naturalHeight !== height) {
    throw new Error(`Mask dimensions mismatch: expected ${width}×${height}, got ${naturalWidth}×${naturalHeight}.`)
  }

  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Failed to create canvas context for mask decode.')
  }
  context.drawImage(image, 0, 0, width, height)
  const rgba = context.getImageData(0, 0, width, height).data
  return rgbaToMaskPlane(rgba, width, height)
}

function loadImage(src: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Failed to decode image source.'))
    image.src = src
  })
}

function getMaskContext(): CanvasRenderingContext2D | null {
  const canvas = maskCanvasEl.value
  if (!canvas) return null
  return canvas.getContext('2d', { willReadFrequently: true })
}

function getPreviewContext(): CanvasRenderingContext2D | null {
  const canvas = previewCanvasEl.value
  if (!canvas) return null
  return canvas.getContext('2d', { willReadFrequently: true })
}

function getPreviewBufferContext(width: number, height: number): CanvasRenderingContext2D | null {
  if (!previewBufferCanvas) {
    previewBufferCanvas = document.createElement('canvas')
  }
  if (previewBufferCanvas.width !== width) previewBufferCanvas.width = width
  if (previewBufferCanvas.height !== height) previewBufferCanvas.height = height
  return previewBufferCanvas.getContext('2d', { willReadFrequently: true })
}

function buildPolygonPreviewPoints(): MaskPoint[] | null {
  if (activeTool.value !== TOOL_VALUE_POLYGON) return null
  if (previewPointerPoint.value) {
    const previewPoints = [
      ...polygonPoints.value,
      normalizePoint(previewPointerPoint.value),
    ]
    return previewPoints.length >= 3 ? previewPoints : null
  }
  return polygonPoints.value.length >= 3 ? polygonPoints.value : null
}

function resolveWorkingMask(): Uint8Array {
  if (transientMask.value) return transientMask.value
  const current = engine.value?.currentMask
  const baseMask = current ? current : new Uint8Array(props.imageWidth * props.imageHeight)

  if (activeTool.value === TOOL_VALUE_CIRCLE && circlePreview.value) {
    const previewMask = new Uint8Array(baseMask)
    applyCircleToMask(
      previewMask,
      props.imageWidth,
      props.imageHeight,
      circlePreview.value.center,
      circlePreview.value.radius,
      MASK_VALUE_FILLED,
    )
    return previewMask
  }

  const polygonPreviewPoints = buildPolygonPreviewPoints()
  if (polygonPreviewPoints) {
    const previewMask = new Uint8Array(baseMask)
    applyPolygonToMask(
      previewMask,
      props.imageWidth,
      props.imageHeight,
      polygonPreviewPoints,
      MASK_VALUE_FILLED,
    )
    return previewMask
  }

  return baseMask
}

function renderMaskCanvas(): void {
  const width = props.imageWidth
  const height = props.imageHeight
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) return

  const context = getMaskContext()
  if (!context) return

  const mask = resolveWorkingMask()
  renderPreviewCanvas(mask)
  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    const baseIndex = pixel * 4
    if (mask[pixel] >= MASK_VALUE_FILLED) {
      rgba[baseIndex] = 255
      rgba[baseIndex + 1] = 56
      rgba[baseIndex + 2] = 56
      rgba[baseIndex + 3] = 138
      continue
    }
    rgba[baseIndex] = 0
    rgba[baseIndex + 1] = 0
    rgba[baseIndex + 2] = 0
    rgba[baseIndex + 3] = 0
  }

  context.clearRect(0, 0, width, height)
  context.putImageData(new ImageData(rgba, width, height), 0, 0)
  renderShapePreview(context)
}

function renderPreviewCanvas(mask: Uint8Array): void {
  const width = props.imageWidth
  const height = props.imageHeight
  const context = getPreviewContext()
  if (!context) return

  context.clearRect(0, 0, width, height)
  previewCropStyle.value = null

  let geometry = null
  try {
    geometry = computeInpaintMaskPreviewGeometry(mask, {
      imageWidth: width,
      imageHeight: height,
      processingWidth: effectiveProcessingWidth.value,
      processingHeight: effectiveProcessingHeight.value,
      maskBlur: props.maskBlur,
      maskedPadding: props.maskedPadding,
    })
  } catch (error) {
    console.error('[InpaintMaskEditorOverlay] Failed to compute mask preview geometry.', error)
    return
  }

  if (!geometry) return

  previewCropStyle.value = {
    left: `${geometry.cropRegion.x1}px`,
    top: `${geometry.cropRegion.y1}px`,
    width: `${geometry.cropRegion.width}px`,
    height: `${geometry.cropRegion.height}px`,
  }

  if (!Number.isFinite(props.maskBlur) || Number(props.maskBlur) <= 0) return

  const previewBufferContext = getPreviewBufferContext(width, height)
  if (!previewBufferContext || !previewBufferCanvas) return

  const rgba = new Uint8ClampedArray(width * height * 4)
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    if (mask[pixel] < MASK_VALUE_FILLED) continue
    const baseIndex = pixel * 4
    rgba[baseIndex] = 255
    rgba[baseIndex + 1] = 178
    rgba[baseIndex + 2] = 68
    rgba[baseIndex + 3] = 112
  }

  previewBufferContext.clearRect(0, 0, width, height)
  previewBufferContext.putImageData(new ImageData(rgba, width, height), 0, 0)

  context.save()
  context.filter = `blur(${Math.max(0, Number(props.maskBlur) || 0)}px)`
  context.drawImage(previewBufferCanvas, 0, 0)
  context.restore()
}

function renderShapePreview(context: CanvasRenderingContext2D): void {
  if (circlePreview.value && activeTool.value === TOOL_VALUE_CIRCLE) {
    const { center, radius } = circlePreview.value
    context.save()
    context.beginPath()
    context.arc(center.x, center.y, Math.max(0.5, radius), 0, Math.PI * 2)
    context.fillStyle = 'rgba(255,255,255,0.22)'
    context.strokeStyle = 'rgba(255,255,255,0.9)'
    context.lineWidth = 1
    context.fill()
    context.stroke()
    context.restore()
  }

  if (polygonPoints.value.length > 0 && activeTool.value === TOOL_VALUE_POLYGON) {
    context.save()
    context.strokeStyle = 'rgba(255,255,255,0.95)'
    context.lineWidth = 1
    context.fillStyle = 'rgba(255,255,255,0.2)'

    context.beginPath()
    context.moveTo(polygonPoints.value[0].x, polygonPoints.value[0].y)
    for (let index = 1; index < polygonPoints.value.length; index += 1) {
      context.lineTo(polygonPoints.value[index].x, polygonPoints.value[index].y)
    }
    if (previewPointerPoint.value) {
      context.lineTo(previewPointerPoint.value.x, previewPointerPoint.value.y)
    }
    if (polygonPoints.value.length >= 3) {
      context.closePath()
      context.fill()
    }
    context.stroke()

    for (const point of polygonPoints.value) {
      context.beginPath()
      context.arc(point.x, point.y, 2.6, 0, Math.PI * 2)
      context.fillStyle = 'rgba(255,255,255,0.95)'
      context.fill()
    }
    context.restore()
  }
}

function setTool(tool: ToolName): void {
  activeTool.value = tool
  if (tool !== TOOL_VALUE_POLYGON) {
    polygonPoints.value = []
  }
  renderMaskCanvas()
}

function resolveCursorRadius(): number {
  if (activeTool.value === TOOL_VALUE_CIRCLE && circlePreview.value) {
    return Math.max(0.5, circlePreview.value.radius)
  }
  return Math.max(0.5, Number(brushSize.value) / 2)
}

function normalizePoint(point: MaskPoint): MaskPoint {
  const width = Math.max(1, props.imageWidth)
  const height = Math.max(1, props.imageHeight)
  return {
    x: Math.max(0, Math.min(width - 0.0001, point.x)),
    y: Math.max(0, Math.min(height - 0.0001, point.y)),
  }
}

function eventToImagePoint(event: PointerEvent): MaskPoint | null {
  const content = contentEl.value
  if (!content) return null
  const rect = content.getBoundingClientRect()
  if (rect.width <= 0 || rect.height <= 0) return null

  const xRatio = (event.clientX - rect.left) / rect.width
  const yRatio = (event.clientY - rect.top) / rect.height
  return normalizePoint({
    x: xRatio * props.imageWidth,
    y: yRatio * props.imageHeight,
  })
}

function beginPan(event: PointerEvent): void {
  if (!stageEl.value) return
  panState.value = {
    pointerId: event.pointerId,
    startClientX: event.clientX,
    startClientY: event.clientY,
    originOffsetX: offsetX.value,
    originOffsetY: offsetY.value,
  }
  stageEl.value.setPointerCapture(event.pointerId)
}

function updatePan(event: PointerEvent): void {
  const pan = panState.value
  if (!pan || pan.pointerId !== event.pointerId) return
  offsetX.value = pan.originOffsetX + (event.clientX - pan.startClientX)
  offsetY.value = pan.originOffsetY + (event.clientY - pan.startClientY)
}

function endPan(event: PointerEvent): void {
  if (!panState.value || panState.value.pointerId !== event.pointerId) return
  if (stageEl.value?.hasPointerCapture(event.pointerId)) {
    stageEl.value.releasePointerCapture(event.pointerId)
  }
  panState.value = null
}

function beginDraw(event: PointerEvent, point: MaskPoint): void {
  const currentEngine = engine.value
  if (!currentEngine || !stageEl.value) return

  activeDrawPointerId.value = event.pointerId
  stageEl.value.setPointerCapture(event.pointerId)

  if (activeTool.value === TOOL_VALUE_CIRCLE) {
    circlePreview.value = {
      center: point,
      radius: 0,
    }
    transientMask.value = currentEngine.currentMask
    renderMaskCanvas()
    return
  }

  strokePoints.value = [point]
  transientMask.value = currentEngine.currentMask
  applyStrokeToMask(
    transientMask.value,
    props.imageWidth,
    props.imageHeight,
    [point],
    Number(brushSize.value) / 2,
    activeTool.value === TOOL_VALUE_ERASER ? MASK_VALUE_EMPTY : MASK_VALUE_FILLED,
  )
  renderMaskCanvas()
}

function updateDraw(point: MaskPoint): void {
  if (activeDrawPointerId.value === null) return

  if (activeTool.value === TOOL_VALUE_CIRCLE) {
    if (!circlePreview.value) return
    const dx = point.x - circlePreview.value.center.x
    const dy = point.y - circlePreview.value.center.y
    circlePreview.value.radius = Math.max(0.5, Math.hypot(dx, dy))
    renderMaskCanvas()
    return
  }

  const points = strokePoints.value
  const workingMask = transientMask.value
  if (!points || !workingMask) return

  const previousPoint = points[points.length - 1]
  const dx = point.x - previousPoint.x
  const dy = point.y - previousPoint.y
  if ((dx * dx) + (dy * dy) < 0.25) return

  points.push(point)
  applyStrokeToMask(
    workingMask,
    props.imageWidth,
    props.imageHeight,
    [previousPoint, point],
    Number(brushSize.value) / 2,
    activeTool.value === TOOL_VALUE_ERASER ? MASK_VALUE_EMPTY : MASK_VALUE_FILLED,
  )
  renderMaskCanvas()
}

function commitDraw(event: PointerEvent): void {
  if (activeDrawPointerId.value === null || activeDrawPointerId.value !== event.pointerId) return
  const currentEngine = engine.value
  if (!currentEngine) {
    resetInteractionState(false)
    return
  }

  if (activeTool.value === TOOL_VALUE_CIRCLE && circlePreview.value) {
    currentEngine.applyCircle(circlePreview.value.center, circlePreview.value.radius, MASK_VALUE_FILLED)
    resetInteractionState(true)
    syncHistoryFlags()
    renderMaskCanvas()
    return
  }

  if (strokePoints.value && strokePoints.value.length > 0) {
    currentEngine.applyStroke(
      strokePoints.value,
      Number(brushSize.value) / 2,
      activeTool.value === TOOL_VALUE_ERASER ? MASK_VALUE_EMPTY : MASK_VALUE_FILLED,
    )
  }

  resetInteractionState(true)
  syncHistoryFlags()
  renderMaskCanvas()
}

function resetInteractionState(resetPolygon: boolean): void {
  if (activeDrawPointerId.value !== null && stageEl.value?.hasPointerCapture(activeDrawPointerId.value)) {
    stageEl.value.releasePointerCapture(activeDrawPointerId.value)
  }
  activeDrawPointerId.value = null
  strokePoints.value = null
  circlePreview.value = null
  transientMask.value = null
  if (resetPolygon) polygonPoints.value = []
  panState.value = null
}

function onStagePointerDown(event: PointerEvent): void {
  if (!isOpen.value) return

  const point = eventToImagePoint(event)
  previewPointerPoint.value = point

  if (event.button === 1 || event.button === 2) {
    beginPan(event)
    event.preventDefault()
    return
  }

  if (event.button !== 0 || !point) return

  if (activeTool.value === TOOL_VALUE_POLYGON) {
    handlePolygonClick(point, event)
    return
  }

  beginDraw(event, point)
  event.preventDefault()
}

function onStagePointerMove(event: PointerEvent): void {
  const point = eventToImagePoint(event)
  previewPointerPoint.value = point

  if (panState.value) {
    updatePan(event)
    return
  }

  if (!point || activeDrawPointerId.value === null || activeDrawPointerId.value !== event.pointerId) return
  updateDraw(point)
}

function onStagePointerUp(event: PointerEvent): void {
  if (panState.value) {
    endPan(event)
    return
  }
  commitDraw(event)
}

function onStagePointerCancel(event: PointerEvent): void {
  endPan(event)
  if (activeDrawPointerId.value !== null && activeDrawPointerId.value === event.pointerId) {
    resetInteractionState(true)
    renderMaskCanvas()
  }
}

function handlePolygonClick(point: MaskPoint, event: PointerEvent): void {
  const normalizedPoint = normalizePoint(point)
  if (event.detail >= 2) {
    if (polygonPoints.value.length === 0) {
      polygonPoints.value.push(normalizedPoint)
      renderMaskCanvas()
      return
    }
    polygonPoints.value.push(normalizedPoint)
    commitPolygon()
    return
  }
  polygonPoints.value.push(normalizedPoint)
  renderMaskCanvas()
}

function commitPolygon(): void {
  const currentEngine = engine.value
  if (!currentEngine) return
  if (polygonPoints.value.length < 3) return
  currentEngine.applyPolygon(polygonPoints.value, MASK_VALUE_FILLED)
  polygonPoints.value = []
  syncHistoryFlags()
  renderMaskCanvas()
}

function cancelPolygon(): void {
  if (polygonPoints.value.length === 0) return
  polygonPoints.value = []
  renderMaskCanvas()
}

function popPolygonPoint(): void {
  if (polygonPoints.value.length === 0) return
  polygonPoints.value = polygonPoints.value.slice(0, -1)
  renderMaskCanvas()
}

function clearMask(): void {
  if (!engine.value) return
  engine.value.clear(MASK_VALUE_EMPTY)
  polygonPoints.value = []
  syncHistoryFlags()
  renderMaskCanvas()
}

function triggerMaskUpload(): void {
  const input = maskUploadInputEl.value
  if (!input || !engine.value) return
  input.click()
}

async function onMaskUploadInputChange(event: Event): Promise<void> {
  const input = event.target as HTMLInputElement | null
  const file = input?.files?.[0]
  if (input) input.value = ''
  if (!file) return
  if (!engine.value) return
  const requestToken = uploadRequestToken.value + 1
  uploadRequestToken.value = requestToken
  uploadInFlight.value = true
  const sourceFingerprintBeforeUpload = buildSourceFingerprint()
  try {
    const uploadedMask = await loadMaskPlaneFromFile(file)
    if (requestToken !== uploadRequestToken.value) return
    if (!engine.value) return
    if (buildSourceFingerprint() !== sourceFingerprintBeforeUpload) {
      emit('external-reset', 'Mask import canceled: init image source changed during upload.')
      return
    }
    resetInteractionState(true)
    engine.value.replaceMask(uploadedMask)
    syncHistoryFlags()
    renderMaskCanvas()
  } catch (error) {
    if (requestToken !== uploadRequestToken.value) return
    const message = error instanceof Error ? error.message : String(error)
    emit('external-reset', `Mask import failed: ${message}`)
  } finally {
    if (requestToken === uploadRequestToken.value) {
      uploadInFlight.value = false
    }
  }
}

function readFileAsDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error || new Error('Failed to read file.'))
    reader.readAsDataURL(file)
  })
}

async function loadMaskPlaneFromFile(file: File): Promise<Uint8Array> {
  const width = Math.trunc(props.imageWidth)
  const height = Math.trunc(props.imageHeight)
  if (!Number.isFinite(width) || !Number.isFinite(height) || width <= 0 || height <= 0) {
    throw new Error('Mask upload requires positive image dimensions.')
  }
  const src = await readFileAsDataUrl(file)
  const image = await loadImage(src)
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Failed to create canvas context for mask import.')
  }
  context.clearRect(0, 0, width, height)
  context.drawImage(image, 0, 0, width, height)
  const rgba = context.getImageData(0, 0, width, height).data
  return rgbaToMaskPlane(rgba, width, height)
}

function resetDraftFromSource(): void {
  if (!baselineMask.value) return
  engine.value = new InpaintMaskEditorEngine({
    width: props.imageWidth,
    height: props.imageHeight,
    initialMask: baselineMask.value,
  })
  resetInteractionState(true)
  syncHistoryFlags()
  renderMaskCanvas()
}

function undo(): void {
  if (!engine.value) return
  cancelPolygon()
  if (!engine.value.undo()) return
  syncHistoryFlags()
  renderMaskCanvas()
}

function redo(): void {
  if (!engine.value) return
  cancelPolygon()
  if (!engine.value.redo()) return
  syncHistoryFlags()
  renderMaskCanvas()
}

function syncHistoryFlags(): void {
  canUndo.value = Boolean(engine.value?.canUndo)
  canRedo.value = Boolean(engine.value?.canRedo)
}

function closeOverlay(): void {
  emit('update:modelValue', false)
}

function onMainClick(event: MouseEvent): void {
  if (event.target !== event.currentTarget) return
  closeOverlay()
}

function clampZoom(value: number): number {
  const min = Math.min(ZOOM_MIN, computeFitZoom())
  if (!Number.isFinite(value)) return min
  return Math.max(min, Math.min(ZOOM_MAX, value))
}

function computeFitZoom(): number {
  const stage = stageEl.value
  if (!stage) return 1
  const availableWidth = Math.max(1, stage.clientWidth - FIT_PADDING_PX * 2)
  const availableHeight = Math.max(1, stage.clientHeight - FIT_PADDING_PX * 2)
  const fit = Math.min(availableWidth / Math.max(1, props.imageWidth), availableHeight / Math.max(1, props.imageHeight))
  if (!Number.isFinite(fit) || fit <= 0) return 1
  return Math.min(1, fit)
}

function applyFitView(): void {
  zoom.value = computeFitZoom()
  offsetX.value = 0
  offsetY.value = 0
}

function resetView(): void {
  applyFitView()
}

function setZoom(value: number): void {
  zoom.value = clampZoom(value)
}

function zoomIn(): void {
  zoom.value = clampZoom(zoom.value + 0.2)
}

function zoomOut(): void {
  zoom.value = clampZoom(zoom.value - 0.2)
}

function onOverlayWheel(event: WheelEvent): void {
  if (!isOpen.value) return
  const delta = event.deltaY < 0 ? 0.2 : -0.2
  zoom.value = clampZoom(zoom.value + delta)
}

function onBaseImageLoad(): void {
  if (!isOpen.value) return
  applyFitView()
  renderMaskCanvas()
}

function onWindowKeydown(event: KeyboardEvent): void {
  if (!isOpen.value) return

  const keyLower = String(event.key || '').toLowerCase()
  const hasModifier = event.metaKey || event.ctrlKey

  if (keyLower === 'escape') {
    if (polygonPoints.value.length > 0) {
      cancelPolygon()
      event.preventDefault()
      return
    }
    closeOverlay()
    event.preventDefault()
    return
  }

  if (activeTool.value === TOOL_VALUE_POLYGON && keyLower === 'enter') {
    commitPolygon()
    event.preventDefault()
    return
  }

  if (activeTool.value === TOOL_VALUE_POLYGON && keyLower === 'backspace') {
    popPolygonPoint()
    event.preventDefault()
    return
  }

  if (hasModifier && keyLower === 'z') {
    if (event.shiftKey) {
      redo()
    } else {
      undo()
    }
    event.preventDefault()
    return
  }

  if (hasModifier && keyLower === 'y') {
    redo()
    event.preventDefault()
  }
}

function exportCurrentMaskToDataUrl(): string {
  const currentEngine = engine.value
  if (!currentEngine) {
    throw new Error('Mask editor is not initialized.')
  }
  const rgba = maskPlaneToRgba(currentEngine.currentMask, props.imageWidth, props.imageHeight)
  const canvas = document.createElement('canvas')
  canvas.width = props.imageWidth
  canvas.height = props.imageHeight
  const context = canvas.getContext('2d', { willReadFrequently: true })
  if (!context) {
    throw new Error('Failed to create export canvas context.')
  }
  const imageBuffer = new Uint8ClampedArray(rgba.length)
  imageBuffer.set(rgba)
  context.putImageData(new ImageData(imageBuffer, props.imageWidth, props.imageHeight), 0, 0)
  return canvas.toDataURL('image/png')
}

function applyDraftMask(): void {
  if (polygonPoints.value.length >= 3) {
    commitPolygon()
  } else if (polygonPoints.value.length > 0) {
    cancelPolygon()
    emit('external-reset', 'Incomplete polygon was discarded before apply.')
  }
  const dataUrl = exportCurrentMaskToDataUrl()
  emit('apply', dataUrl)
  closeOverlay()
}

onBeforeUnmount(() => {
  window.removeEventListener('keydown', onWindowKeydown)
})
</script>

<!-- styles in styles/components/inpaint-mask-editor.css -->
