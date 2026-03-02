/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN img2vid frame projection math for no-stretch preview overlays.
Provides deterministic no-stretch crop projection metadata (free image scale + normalized offsets) so the UI and
runtime can agree on which source-image region is used for generation.

Symbols (top-level; keep in sync; no ghosts):
- `WanImg2VidFrameGuideConfig` (interface): Zoom-overlay frame-guide config (target dims + image-scale/crop settings).
- `WanImg2VidFrameProjectionInput` (interface): Projection input contract (source dims + frame dims + image-scale/crop settings).
- `WanImg2VidFrameProjection` (interface): Projection output contract (crop rect + scaled dims + slack/offset metadata).
- `normalizeWanImg2VidImageScale` (function): Validates and normalizes WAN img2vid image-scale values.
- `computeWanImg2VidMinImageScale` (function): Computes the minimum no-stretch image scale required for a target frame.
- `computeWanImg2VidFrameProjection` (function): Computes no-stretch projection metadata for the WAN init-image guide.
*/

export interface WanImg2VidFrameGuideConfig {
  targetWidth: number
  targetHeight: number
  imageScale?: number
  cropOffsetX?: number
  cropOffsetY?: number
}

export interface WanImg2VidFrameProjectionInput {
  sourceWidth: number
  sourceHeight: number
  frameWidth: number
  frameHeight: number
  imageScale?: number
  cropOffsetX?: number
  cropOffsetY?: number
}

export interface WanImg2VidFrameProjection {
  sourceWidth: number
  sourceHeight: number
  frameWidth: number
  frameHeight: number
  minImageScale: number
  imageScale: number
  resizedWidth: number
  resizedHeight: number
  cropOffsetX: number
  cropOffsetY: number
  slackX: number
  slackY: number
  cropX: number
  cropY: number
  cropWidth: number
  cropHeight: number
}

function requirePositiveFinite(name: string, value: number): number {
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`WAN frame projection requires ${name} > 0 (got ${String(value)}).`)
  }
  const normalized = Math.trunc(Number(value))
  if (normalized <= 0) {
    throw new Error(`WAN frame projection requires ${name} >= 1 after normalization (got ${String(value)}).`)
  }
  return normalized
}

function clamp01(value: number): number {
  if (!Number.isFinite(value)) return 0.5
  return Math.max(0, Math.min(1, value))
}

function normalizeOffset(name: string, value: unknown): number {
  if (value === undefined || value === null || value === '') return 0.5
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    throw new Error(`WAN frame projection requires ${name} to be finite in [0,1] (got ${String(value)}).`)
  }
  if (numeric < 0 || numeric > 1) {
    throw new Error(`WAN frame projection requires ${name} in [0,1] (got ${String(value)}).`)
  }
  const clamped = clamp01(numeric)
  if (!Number.isFinite(clamped)) {
    throw new Error(`WAN frame projection requires ${name} to be finite in [0,1].`)
  }
  return clamped
}

export function normalizeWanImg2VidImageScale(
  rawValue: unknown,
  fallback = 1,
): number {
  const fallbackNumeric = Number(fallback)
  if (!Number.isFinite(fallbackNumeric) || fallbackNumeric <= 0) {
    throw new Error(`WAN frame projection requires imageScale fallback > 0 (got ${String(fallback)}).`)
  }
  if (rawValue === undefined || rawValue === null || rawValue === '') return fallbackNumeric
  if (typeof rawValue === 'boolean') {
    throw new Error(`WAN frame projection requires imageScale to be numeric > 0 (got ${String(rawValue)}).`)
  }
  const value = Number(rawValue)
  if (!Number.isFinite(value) || value <= 0) {
    throw new Error(`WAN frame projection requires imageScale > 0 (got ${String(rawValue)}).`)
  }
  return value
}

export function computeWanImg2VidMinImageScale(
  sourceWidth: number,
  sourceHeight: number,
  frameWidth: number,
  frameHeight: number,
): number {
  const widthScale = frameWidth / sourceWidth
  const heightScale = frameHeight / sourceHeight
  const minScale = Math.max(widthScale, heightScale)
  if (!Number.isFinite(minScale) || minScale <= 0) {
    throw new Error('WAN frame projection produced invalid minImageScale.')
  }
  return minScale
}

export function computeWanImg2VidFrameProjection(
  input: WanImg2VidFrameProjectionInput,
): WanImg2VidFrameProjection {
  const sourceWidth = requirePositiveFinite('sourceWidth', input.sourceWidth)
  const sourceHeight = requirePositiveFinite('sourceHeight', input.sourceHeight)
  const frameWidth = requirePositiveFinite('frameWidth', input.frameWidth)
  const frameHeight = requirePositiveFinite('frameHeight', input.frameHeight)
  const minImageScale = computeWanImg2VidMinImageScale(sourceWidth, sourceHeight, frameWidth, frameHeight)
  const imageScale = normalizeWanImg2VidImageScale(input.imageScale, minImageScale)
  const cropOffsetX = normalizeOffset('cropOffsetX', input.cropOffsetX)
  const cropOffsetY = normalizeOffset('cropOffsetY', input.cropOffsetY)

  if (imageScale + 1e-9 < minImageScale) {
    throw new Error(
      `WAN frame projection requires imageScale >= ${minImageScale.toFixed(6)} (got ${imageScale.toFixed(6)}).`,
    )
  }

  const cropWidth = Math.max(1, Math.min(sourceWidth, Math.trunc(frameWidth / imageScale + 0.5)))
  const cropHeight = Math.max(1, Math.min(sourceHeight, Math.trunc(frameHeight / imageScale + 0.5)))
  if (cropWidth <= 0 || cropHeight <= 0 || cropWidth > sourceWidth || cropHeight > sourceHeight) {
    throw new Error(
      `WAN frame projection crop exceeds source bounds (source=${sourceWidth}x${sourceHeight} crop=${cropWidth}x${cropHeight}).`,
    )
  }

  const slackX = Math.max(0, sourceWidth - cropWidth)
  const slackY = Math.max(0, sourceHeight - cropHeight)
  const cropX = slackX > 0 ? Math.min(slackX, Math.max(0, Math.trunc(slackX * cropOffsetX + 0.5))) : 0
  const cropY = slackY > 0 ? Math.min(slackY, Math.max(0, Math.trunc(slackY * cropOffsetY + 0.5))) : 0

  const resizedWidth = Math.max(frameWidth, Math.max(1, Math.trunc(sourceWidth * imageScale + 0.5)))
  const resizedHeight = Math.max(frameHeight, Math.max(1, Math.trunc(sourceHeight * imageScale + 0.5)))

  return {
    sourceWidth,
    sourceHeight,
    frameWidth,
    frameHeight,
    minImageScale,
    imageScale,
    resizedWidth,
    resizedHeight,
    cropOffsetX,
    cropOffsetY,
    slackX,
    slackY,
    cropX,
    cropY,
    cropWidth,
    cropHeight,
  }
}
