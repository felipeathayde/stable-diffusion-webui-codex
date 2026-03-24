/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure inpaint preview geometry helpers shared by the img2img card and mask editor.
Mirrors the backend masked-img2img crop math needed for frontend previews: binary mask bbox, blur-support expansion,
masked-padding expansion, aspect-preserving crop expansion, and image-space to contained-preview projection.

Symbols (top-level; keep in sync; no ghosts):
- `InpaintMaskPreviewInput` (interface): Input contract for computing preview geometry from a binary mask plane.
- `InpaintPreviewRect` (interface): Rect contract using image/container pixel coordinates with cached width/height.
- `InpaintMaskPreviewGeometry` (interface): Output contract for mask bounds, blur bounds, blur-support radius, and final crop region.
- `ContainedImageRect` (interface): Projection metadata for an image rendered with `object-fit: contain`.
- `computeInpaintMaskPreviewGeometry` (function): Computes blur-support and masked-padding preview geometry from a binary mask plane.
- `computeContainedImageRect` (function): Computes the rendered image rect inside a containing preview box.
- `projectImageRectToContainer` (function): Projects an image-space rect into the contained preview box.
*/

export interface InpaintMaskPreviewInput {
  imageWidth: number
  imageHeight: number
  processingWidth?: number
  processingHeight?: number
  maskBlur: number
  maskedPadding: number
}

export interface InpaintPreviewRect {
  x1: number
  y1: number
  x2: number
  y2: number
  width: number
  height: number
}

export interface InpaintMaskPreviewGeometry {
  maskBounds: InpaintPreviewRect
  blurBounds: InpaintPreviewRect
  blurSupportRadius: number
  cropRegion: InpaintPreviewRect
}

export interface ContainedImageRect {
  left: number
  top: number
  width: number
  height: number
  scale: number
}

function requirePositiveInt(name: string, value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`Inpaint preview requires ${name} > 0 (got ${String(value)}).`)
  }
  const normalized = Math.trunc(Number(value))
  if (normalized <= 0) {
    throw new Error(`Inpaint preview requires ${name} >= 1 after normalization (got ${String(value)}).`)
  }
  return normalized
}

function normalizeNonNegativeInt(value: number): number {
  if (!Number.isFinite(value) || value <= 0) return 0
  return Math.max(0, Math.trunc(Number(value)))
}

function buildRect(x1: number, y1: number, x2: number, y2: number): InpaintPreviewRect {
  return {
    x1,
    y1,
    x2,
    y2,
    width: Math.max(0, x2 - x1),
    height: Math.max(0, y2 - y1),
  }
}

function expandRect(
  rect: InpaintPreviewRect,
  pad: number,
  imageWidth: number,
  imageHeight: number,
): InpaintPreviewRect {
  const normalizedPad = normalizeNonNegativeInt(pad)
  if (normalizedPad <= 0) return rect
  return buildRect(
    Math.max(0, rect.x1 - normalizedPad),
    Math.max(0, rect.y1 - normalizedPad),
    Math.min(imageWidth, rect.x2 + normalizedPad),
    Math.min(imageHeight, rect.y2 + normalizedPad),
  )
}

function gaussianSupportRadius(sigma: number): number {
  const normalizedSigma = Number(sigma)
  if (!Number.isFinite(normalizedSigma) || normalizedSigma <= 0) return 0
  return Math.max(0, Math.trunc(2.5 * normalizedSigma + 0.5))
}

function computeMaskBounds(
  maskPlane: Uint8Array | Uint8ClampedArray,
  imageWidth: number,
  imageHeight: number,
): InpaintPreviewRect | null {
  let minX = imageWidth
  let minY = imageHeight
  let maxX = -1
  let maxY = -1

  for (let y = 0; y < imageHeight; y += 1) {
    const rowBase = y * imageWidth
    for (let x = 0; x < imageWidth; x += 1) {
      if (maskPlane[rowBase + x] <= 0) continue
      if (x < minX) minX = x
      if (y < minY) minY = y
      if (x > maxX) maxX = x
      if (y > maxY) maxY = y
    }
  }

  if (maxX < 0 || maxY < 0) return null
  return buildRect(minX, minY, maxX + 1, maxY + 1)
}

function expandCropRegion(
  cropRegion: InpaintPreviewRect,
  processingWidth: number,
  processingHeight: number,
  imageWidth: number,
  imageHeight: number,
): InpaintPreviewRect {
  let { x1, y1, x2, y2 } = cropRegion
  const cropWidth = Math.max(1, x2 - x1)
  const cropHeight = Math.max(1, y2 - y1)
  const ratioCrop = cropWidth / cropHeight
  const ratioProcessing = processingWidth / processingHeight

  if (ratioCrop > ratioProcessing) {
    const desiredHeight = cropWidth / ratioProcessing
    const diff = Math.trunc(desiredHeight - cropHeight)
    y1 -= Math.trunc(diff / 2)
    y2 += diff - Math.trunc(diff / 2)
    if (y2 >= imageHeight) {
      const overflow = y2 - imageHeight
      y2 -= overflow
      y1 -= overflow
    }
    if (y1 < 0) {
      y2 -= y1
      y1 = 0
    }
    if (y2 >= imageHeight) y2 = imageHeight
  } else {
    const desiredWidth = cropHeight * ratioProcessing
    const diff = Math.trunc(desiredWidth - cropWidth)
    x1 -= Math.trunc(diff / 2)
    x2 += diff - Math.trunc(diff / 2)
    if (x2 >= imageWidth) {
      const overflow = x2 - imageWidth
      x2 -= overflow
      x1 -= overflow
    }
    if (x1 < 0) {
      x2 -= x1
      x1 = 0
    }
    if (x2 >= imageWidth) x2 = imageWidth
  }

  return buildRect(x1, y1, x2, y2)
}

export function computeInpaintMaskPreviewGeometry(
  maskPlane: Uint8Array | Uint8ClampedArray,
  input: InpaintMaskPreviewInput,
): InpaintMaskPreviewGeometry | null {
  const imageWidth = requirePositiveInt('imageWidth', input.imageWidth)
  const imageHeight = requirePositiveInt('imageHeight', input.imageHeight)
  const expectedMaskLength = imageWidth * imageHeight
  if (maskPlane.length !== expectedMaskLength) {
    throw new Error(`Inpaint preview requires maskPlane length ${expectedMaskLength} (got ${maskPlane.length}).`)
  }

  const maskBounds = computeMaskBounds(maskPlane, imageWidth, imageHeight)
  if (!maskBounds) return null

  const processingWidth = requirePositiveInt('processingWidth', input.processingWidth ?? imageWidth)
  const processingHeight = requirePositiveInt('processingHeight', input.processingHeight ?? imageHeight)
  const blurSupportRadius = gaussianSupportRadius(input.maskBlur)
  const blurBounds = expandRect(maskBounds, blurSupportRadius, imageWidth, imageHeight)
  const paddedBounds = expandRect(blurBounds, input.maskedPadding, imageWidth, imageHeight)
  const cropRegion = expandCropRegion(
    paddedBounds,
    processingWidth,
    processingHeight,
    imageWidth,
    imageHeight,
  )

  return {
    maskBounds,
    blurBounds,
    blurSupportRadius,
    cropRegion,
  }
}

export function computeContainedImageRect(
  containerWidth: number,
  containerHeight: number,
  imageWidth: number,
  imageHeight: number,
): ContainedImageRect {
  const width = requirePositiveInt('containerWidth', containerWidth)
  const height = requirePositiveInt('containerHeight', containerHeight)
  const sourceWidth = requirePositiveInt('imageWidth', imageWidth)
  const sourceHeight = requirePositiveInt('imageHeight', imageHeight)
  const scale = Math.min(width / sourceWidth, height / sourceHeight)
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('Inpaint preview produced invalid contain scale.')
  }
  const renderedWidth = sourceWidth * scale
  const renderedHeight = sourceHeight * scale
  return {
    left: (width - renderedWidth) / 2,
    top: (height - renderedHeight) / 2,
    width: renderedWidth,
    height: renderedHeight,
    scale,
  }
}

export function projectImageRectToContainer(
  rect: InpaintPreviewRect,
  containedImage: ContainedImageRect,
): InpaintPreviewRect {
  const x1 = containedImage.left + (rect.x1 * containedImage.scale)
  const y1 = containedImage.top + (rect.y1 * containedImage.scale)
  const x2 = containedImage.left + (rect.x2 * containedImage.scale)
  const y2 = containedImage.top + (rect.y2 * containedImage.scale)
  return buildRect(x1, y1, x2, y2)
}
