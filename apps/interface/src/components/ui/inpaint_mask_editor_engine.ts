/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure mask-editing engine for inpaint canvas tools.
Provides deterministic image-space drawing operations (brush/eraser stroke, circle, polygon),
history with undo/redo and bounded memory policy, and RGBA<->mask helpers used by the UI overlay.

Symbols (top-level; keep in sync; no ghosts):
- `MASK_VALUE_EMPTY` (constant): Canonical unmasked pixel value.
- `MASK_VALUE_FILLED` (constant): Canonical masked pixel value.
- `MaskPoint` (interface): 2D image-space point.
- `MaskHistoryPolicy` (interface): History capacity controls.
- `InpaintMaskEditorEngineOptions` (interface): Constructor options for engine setup.
- `normalizeMaskValue` (function): Normalizes arbitrary values into binary mask pixels.
- `applyStrokeToMask` (function): Mutates a mask buffer by applying a brush/eraser stroke.
- `applyCircleToMask` (function): Mutates a mask buffer by applying a filled circle.
- `applyPolygonToMask` (function): Mutates a mask buffer by applying a filled polygon.
- `rgbaToMaskPlane` (function): Converts RGBA image data into binary mask plane with alpha precedence.
- `maskPlaneToRgba` (function): Converts binary mask plane into opaque grayscale RGBA buffer.
- `InpaintMaskEditorEngine` (class): Stateful editing engine with draw operations and undo/redo.
*/

export const MASK_VALUE_EMPTY = 0
export const MASK_VALUE_FILLED = 255

const DEFAULT_HISTORY_MAX_SNAPSHOTS = 120
const DEFAULT_HISTORY_MAX_BYTES = 192 * 1024 * 1024
const DEFAULT_THRESHOLD = 128

export interface MaskPoint {
  x: number
  y: number
}

export interface MaskHistoryPolicy {
  maxSnapshots: number
  maxBytes: number
}

export interface InpaintMaskEditorEngineOptions {
  width: number
  height: number
  initialMask?: Uint8Array | Uint8ClampedArray | null
  historyPolicy?: Partial<MaskHistoryPolicy>
}

function requirePositiveInt(value: number, fieldName: string): number {
  const integer = Math.trunc(Number(value))
  if (!Number.isFinite(integer) || integer <= 0) {
    throw new Error(`${fieldName} must be a positive integer.`)
  }
  return integer
}

function requireNonNegativeFinite(value: number, fieldName: string): number {
  const numeric = Number(value)
  if (!Number.isFinite(numeric) || numeric < 0) {
    throw new Error(`${fieldName} must be a non-negative finite number.`)
  }
  return numeric
}

function clamp(value: number, min: number, max: number): number {
  if (value < min) return min
  if (value > max) return max
  return value
}

function assertMaskLength(mask: Uint8Array | Uint8ClampedArray, width: number, height: number): void {
  const expected = width * height
  if (mask.length !== expected) {
    throw new Error(`Mask length mismatch: expected ${expected}, got ${mask.length}.`)
  }
}

function cloneMask(mask: Uint8Array | Uint8ClampedArray): Uint8Array {
  return Uint8Array.from(mask)
}

function masksEqual(left: Uint8Array, right: Uint8Array): boolean {
  if (left.length !== right.length) return false
  for (let index = 0; index < left.length; index += 1) {
    if (left[index] !== right[index]) return false
  }
  return true
}

function drawDisk(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  centerX: number,
  centerY: number,
  radius: number,
  value: number,
): void {
  const radiusClamped = Math.max(0.5, Number(radius))
  const radiusSquared = radiusClamped * radiusClamped

  const minX = Math.max(0, Math.floor(centerX - radiusClamped))
  const maxX = Math.min(width - 1, Math.ceil(centerX + radiusClamped))
  const minY = Math.max(0, Math.floor(centerY - radiusClamped))
  const maxY = Math.min(height - 1, Math.ceil(centerY + radiusClamped))

  for (let y = minY; y <= maxY; y += 1) {
    for (let x = minX; x <= maxX; x += 1) {
      const dx = x + 0.5 - centerX
      const dy = y + 0.5 - centerY
      if ((dx * dx) + (dy * dy) > radiusSquared) continue
      mask[(y * width) + x] = value
    }
  }
}

function drawStroke(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  points: readonly MaskPoint[],
  radius: number,
  value: number,
): void {
  if (points.length === 0) return
  if (points.length === 1) {
    const only = points[0]
    drawDisk(mask, width, height, only.x, only.y, radius, value)
    return
  }

  for (let index = 1; index < points.length; index += 1) {
    const start = points[index - 1]
    const end = points[index]
    const deltaX = end.x - start.x
    const deltaY = end.y - start.y
    const steps = Math.max(1, Math.ceil(Math.max(Math.abs(deltaX), Math.abs(deltaY))))
    for (let step = 0; step <= steps; step += 1) {
      const t = step / steps
      const x = start.x + (deltaX * t)
      const y = start.y + (deltaY * t)
      drawDisk(mask, width, height, x, y, radius, value)
    }
  }
}

function fillPolygon(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  points: readonly MaskPoint[],
  value: number,
): void {
  if (points.length < 3) {
    throw new Error('Polygon requires at least 3 points.')
  }

  let minY = Number.POSITIVE_INFINITY
  let maxY = Number.NEGATIVE_INFINITY
  for (const point of points) {
    minY = Math.min(minY, point.y)
    maxY = Math.max(maxY, point.y)
  }

  const yStart = clamp(Math.floor(minY), 0, height - 1)
  const yEnd = clamp(Math.ceil(maxY), 0, height - 1)

  for (let y = yStart; y <= yEnd; y += 1) {
    const scanlineY = y + 0.5
    const intersections: number[] = []

    for (let index = 0; index < points.length; index += 1) {
      const current = points[index]
      const next = points[(index + 1) % points.length]

      if (current.y === next.y) continue

      const edgeMinY = Math.min(current.y, next.y)
      const edgeMaxY = Math.max(current.y, next.y)
      if (scanlineY < edgeMinY || scanlineY >= edgeMaxY) continue

      const ratio = (scanlineY - current.y) / (next.y - current.y)
      const x = current.x + ((next.x - current.x) * ratio)
      intersections.push(x)
    }

    intersections.sort((left, right) => left - right)

    for (let index = 0; index + 1 < intersections.length; index += 2) {
      const startX = clamp(Math.ceil(intersections[index] - 0.5), 0, width - 1)
      const endX = clamp(Math.floor(intersections[index + 1] - 0.5), 0, width - 1)
      for (let x = startX; x <= endX; x += 1) {
        mask[(y * width) + x] = value
      }
    }
  }
}

function resolveHistoryPolicy(policy: Partial<MaskHistoryPolicy> | undefined): MaskHistoryPolicy {
  const maxSnapshots = requirePositiveInt(policy?.maxSnapshots ?? DEFAULT_HISTORY_MAX_SNAPSHOTS, 'historyPolicy.maxSnapshots')
  const maxBytes = requirePositiveInt(policy?.maxBytes ?? DEFAULT_HISTORY_MAX_BYTES, 'historyPolicy.maxBytes')
  return { maxSnapshots, maxBytes }
}

function normalizeMaskPoint(point: MaskPoint): MaskPoint {
  const x = Number(point.x)
  const y = Number(point.y)
  if (!Number.isFinite(x) || !Number.isFinite(y)) {
    throw new Error('Mask point coordinates must be finite numbers.')
  }
  return { x, y }
}

function normalizePointList(points: readonly MaskPoint[]): MaskPoint[] {
  if (!Array.isArray(points) || points.length === 0) {
    throw new Error('Point list must contain at least one point.')
  }
  return points.map((point) => normalizeMaskPoint(point))
}

export function normalizeMaskValue(value: number): number {
  return Number(value) >= 128 ? MASK_VALUE_FILLED : MASK_VALUE_EMPTY
}

function normalizeBrushRadius(value: number): number {
  const numeric = requireNonNegativeFinite(value, 'brushSize')
  return Math.max(0.5, numeric)
}

function normalizeMaskBuffer(
  mask: Uint8Array | Uint8ClampedArray | null | undefined,
  width: number,
  height: number,
): Uint8Array {
  if (!mask) return new Uint8Array(width * height)
  assertMaskLength(mask, width, height)
  const normalized = new Uint8Array(mask.length)
  for (let index = 0; index < mask.length; index += 1) {
    normalized[index] = normalizeMaskValue(mask[index])
  }
  return normalized
}

export function applyStrokeToMask(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  points: readonly MaskPoint[],
  brushSize: number,
  value: number,
): void {
  const widthPx = requirePositiveInt(width, 'width')
  const heightPx = requirePositiveInt(height, 'height')
  assertMaskLength(mask, widthPx, heightPx)
  const normalizedPoints = normalizePointList(points)
  const radius = normalizeBrushRadius(brushSize)
  const fillValue = normalizeMaskValue(value)
  drawStroke(mask, widthPx, heightPx, normalizedPoints, radius, fillValue)
}

export function applyCircleToMask(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  center: MaskPoint,
  radius: number,
  value: number,
): void {
  const widthPx = requirePositiveInt(width, 'width')
  const heightPx = requirePositiveInt(height, 'height')
  assertMaskLength(mask, widthPx, heightPx)
  const normalizedCenter = normalizeMaskPoint(center)
  const normalizedRadius = normalizeBrushRadius(radius)
  const fillValue = normalizeMaskValue(value)
  drawDisk(mask, widthPx, heightPx, normalizedCenter.x, normalizedCenter.y, normalizedRadius, fillValue)
}

export function applyPolygonToMask(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  points: readonly MaskPoint[],
  value: number,
): void {
  const widthPx = requirePositiveInt(width, 'width')
  const heightPx = requirePositiveInt(height, 'height')
  assertMaskLength(mask, widthPx, heightPx)
  const normalizedPoints = normalizePointList(points)
  const fillValue = normalizeMaskValue(value)
  fillPolygon(mask, widthPx, heightPx, normalizedPoints, fillValue)
}

export function rgbaToMaskPlane(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  threshold: number = DEFAULT_THRESHOLD,
): Uint8Array {
  const widthPx = requirePositiveInt(width, 'width')
  const heightPx = requirePositiveInt(height, 'height')
  const expectedLength = widthPx * heightPx * 4
  if (rgba.length !== expectedLength) {
    throw new Error(`RGBA length mismatch: expected ${expectedLength}, got ${rgba.length}.`)
  }

  const thresholdValue = clamp(Math.trunc(Number(threshold)), 0, 255)
  let hasTransparency = false
  for (let index = 3; index < rgba.length; index += 4) {
    if (rgba[index] !== 255) {
      hasTransparency = true
      break
    }
  }

  const mask = new Uint8Array(widthPx * heightPx)
  for (let pixel = 0; pixel < mask.length; pixel += 1) {
    const base = pixel * 4
    const channel = hasTransparency
      ? rgba[base + 3]
      : Math.trunc((rgba[base] + rgba[base + 1] + rgba[base + 2]) / 3)
    mask[pixel] = channel > thresholdValue ? MASK_VALUE_FILLED : MASK_VALUE_EMPTY
  }
  return mask
}

export function maskPlaneToRgba(
  mask: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): Uint8ClampedArray {
  const widthPx = requirePositiveInt(width, 'width')
  const heightPx = requirePositiveInt(height, 'height')
  assertMaskLength(mask, widthPx, heightPx)

  const rgba = new Uint8ClampedArray(widthPx * heightPx * 4)
  for (let index = 0; index < mask.length; index += 1) {
    const value = normalizeMaskValue(mask[index])
    const base = index * 4
    rgba[base] = value
    rgba[base + 1] = value
    rgba[base + 2] = value
    rgba[base + 3] = 255
  }
  return rgba
}

export class InpaintMaskEditorEngine {
  readonly width: number
  readonly height: number
  readonly historyPolicy: MaskHistoryPolicy

  private snapshots: Uint8Array[]
  private historyIndex: number

  constructor(options: InpaintMaskEditorEngineOptions) {
    this.width = requirePositiveInt(options.width, 'width')
    this.height = requirePositiveInt(options.height, 'height')
    this.historyPolicy = resolveHistoryPolicy(options.historyPolicy)

    const initialMask = normalizeMaskBuffer(options.initialMask, this.width, this.height)
    this.snapshots = [initialMask]
    this.historyIndex = 0
    this.enforceHistoryPolicy()
  }

  get canUndo(): boolean {
    return this.historyIndex > 0
  }

  get canRedo(): boolean {
    return this.historyIndex < (this.snapshots.length - 1)
  }

  get currentMask(): Uint8Array {
    return cloneMask(this.snapshots[this.historyIndex])
  }

  get snapshotCount(): number {
    return this.snapshots.length
  }

  get currentSnapshotIndex(): number {
    return this.historyIndex
  }

  clear(value: number = MASK_VALUE_EMPTY): void {
    const next = new Uint8Array(this.width * this.height)
    const normalized = normalizeMaskValue(value)
    if (normalized !== MASK_VALUE_EMPTY) next.fill(normalized)
    this.commit(next)
  }

  replaceMask(mask: Uint8Array | Uint8ClampedArray): void {
    const next = normalizeMaskBuffer(mask, this.width, this.height)
    this.commit(next)
  }

  applyStroke(points: readonly MaskPoint[], brushSize: number, value: number): void {
    const next = cloneMask(this.snapshots[this.historyIndex])
    applyStrokeToMask(next, this.width, this.height, points, brushSize, value)
    this.commit(next)
  }

  applyCircle(center: MaskPoint, radius: number, value: number): void {
    const next = cloneMask(this.snapshots[this.historyIndex])
    applyCircleToMask(next, this.width, this.height, center, radius, value)
    this.commit(next)
  }

  applyPolygon(points: readonly MaskPoint[], value: number): void {
    const next = cloneMask(this.snapshots[this.historyIndex])
    applyPolygonToMask(next, this.width, this.height, points, value)
    this.commit(next)
  }

  undo(): boolean {
    if (!this.canUndo) return false
    this.historyIndex -= 1
    return true
  }

  redo(): boolean {
    if (!this.canRedo) return false
    this.historyIndex += 1
    return true
  }

  private commit(next: Uint8Array): void {
    const current = this.snapshots[this.historyIndex]
    if (masksEqual(current, next)) return

    if (this.historyIndex < (this.snapshots.length - 1)) {
      this.snapshots = this.snapshots.slice(0, this.historyIndex + 1)
    }

    this.snapshots.push(next)
    this.historyIndex = this.snapshots.length - 1
    this.enforceHistoryPolicy()
  }

  private enforceHistoryPolicy(): void {
    const bytesPerSnapshot = this.width * this.height
    const byBytes = Math.max(1, Math.floor(this.historyPolicy.maxBytes / Math.max(1, bytesPerSnapshot)))
    const allowedSnapshots = Math.max(1, Math.min(this.historyPolicy.maxSnapshots, byBytes))

    if (this.snapshots.length <= allowedSnapshots) return

    const removeCount = this.snapshots.length - allowedSnapshots
    this.snapshots = this.snapshots.slice(removeCount)
    this.historyIndex = Math.max(0, this.historyIndex - removeCount)
  }
}
