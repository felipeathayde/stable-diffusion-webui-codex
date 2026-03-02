/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN img2vid frame projection math for no-stretch preview overlays.
Provides deterministic no-stretch crop projection metadata (resize mode + normalized offsets) so the UI and runtime
can agree on which source-image region is used for generation.

Symbols (top-level; keep in sync; no ghosts):
- `WanImg2VidResizeMode` (type): Supported no-stretch resize modes (`auto|fit_width|fit_height`).
- `WAN_IMG2VID_RESIZE_MODE_LABELS` (constant): Human-readable labels per resize mode.
- `WanImg2VidFrameGuideConfig` (interface): Zoom-overlay frame-guide config (target dims + resize/crop settings).
- `WanImg2VidFrameProjectionInput` (interface): Projection input contract (source dims + frame dims + resize/crop settings).
- `WanImg2VidFrameProjection` (interface): Projection output contract (crop rect + scaled dims + slack/offset metadata).
- `normalizeWanImg2VidResizeMode` (function): Validates and normalizes WAN resize-mode ids.
- `computeWanImg2VidFrameProjection` (function): Computes no-stretch projection metadata for the WAN init-image guide.
*/

export type WanImg2VidResizeMode = 'auto' | 'fit_width' | 'fit_height'
export const WAN_IMG2VID_RESIZE_MODE_LABELS: Record<WanImg2VidResizeMode, string> = {
  auto: 'Auto (no stretch)',
  fit_width: 'Fit Width (no stretch)',
  fit_height: 'Fit Height (no stretch)',
}

export interface WanImg2VidFrameGuideConfig {
  targetWidth: number
  targetHeight: number
  resizeMode?: WanImg2VidResizeMode | string
  cropOffsetX?: number
  cropOffsetY?: number
}

export interface WanImg2VidFrameProjectionInput {
  sourceWidth: number
  sourceHeight: number
  frameWidth: number
  frameHeight: number
  resizeMode?: WanImg2VidResizeMode | string
  cropOffsetX?: number
  cropOffsetY?: number
}

export interface WanImg2VidFrameProjection {
  resizeMode: WanImg2VidResizeMode
  resolvedResizeMode: Exclude<WanImg2VidResizeMode, 'auto'>
  sourceWidth: number
  sourceHeight: number
  frameWidth: number
  frameHeight: number
  scale: number
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

export function normalizeWanImg2VidResizeMode(
  rawValue: unknown,
  fallback: WanImg2VidResizeMode = 'auto',
): WanImg2VidResizeMode {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (!normalized) return fallback
  if (normalized === 'auto' || normalized === 'fit_width' || normalized === 'fit_height') {
    return normalized
  }
  throw new Error(
    `Unsupported WAN resize mode '${String(rawValue)}' (expected 'auto', 'fit_width', or 'fit_height').`,
  )
}

function resolveCropByMode(
  sourceWidth: number,
  sourceHeight: number,
  frameWidth: number,
  frameHeight: number,
  requestedResizeMode: WanImg2VidResizeMode,
): {
  resolvedResizeMode: Exclude<WanImg2VidResizeMode, 'auto'>
  cropWidth: number
  cropHeight: number
} {
  const lhs = sourceWidth * frameHeight
  const rhs = sourceHeight * frameWidth
  const autoResolved: Exclude<WanImg2VidResizeMode, 'auto'> = lhs >= rhs ? 'fit_height' : 'fit_width'
  const resolvedResizeMode = requestedResizeMode === 'auto' ? autoResolved : requestedResizeMode

  if (resolvedResizeMode === 'fit_width') {
    if (lhs > rhs) {
      throw new Error(
        'WAN frame projection invalid combination: fit_width requires frame aspect >= source aspect.',
      )
    }
    const cropHeight = Math.max(
      1,
      Math.min(
        sourceHeight,
        Math.trunc((sourceWidth * frameHeight + Math.trunc(frameWidth / 2)) / frameWidth),
      ),
    )
    return {
      resolvedResizeMode,
      cropWidth: sourceWidth,
      cropHeight,
    }
  }

  if (rhs > lhs) {
    throw new Error(
      'WAN frame projection invalid combination: fit_height requires frame aspect <= source aspect.',
    )
  }
  const cropWidth = Math.max(
    1,
    Math.min(
      sourceWidth,
      Math.trunc((sourceHeight * frameWidth + Math.trunc(frameHeight / 2)) / frameHeight),
    ),
  )
  return {
    resolvedResizeMode,
    cropWidth,
    cropHeight: sourceHeight,
  }
}

export function computeWanImg2VidFrameProjection(
  input: WanImg2VidFrameProjectionInput,
): WanImg2VidFrameProjection {
  const sourceWidth = requirePositiveFinite('sourceWidth', input.sourceWidth)
  const sourceHeight = requirePositiveFinite('sourceHeight', input.sourceHeight)
  const frameWidth = requirePositiveFinite('frameWidth', input.frameWidth)
  const frameHeight = requirePositiveFinite('frameHeight', input.frameHeight)
  const resizeMode = normalizeWanImg2VidResizeMode(input.resizeMode)
  const cropOffsetX = normalizeOffset('cropOffsetX', input.cropOffsetX)
  const cropOffsetY = normalizeOffset('cropOffsetY', input.cropOffsetY)

  const { resolvedResizeMode, cropWidth, cropHeight } = resolveCropByMode(
    sourceWidth,
    sourceHeight,
    frameWidth,
    frameHeight,
    resizeMode,
  )
  if (cropWidth > sourceWidth || cropHeight > sourceHeight) {
    throw new Error(
      `WAN frame projection crop exceeds source bounds (source=${sourceWidth}x${sourceHeight} crop=${cropWidth}x${cropHeight}).`,
    )
  }

  const slackX = Math.max(0, sourceWidth - cropWidth)
  const slackY = Math.max(0, sourceHeight - cropHeight)
  const cropX = slackX > 0 ? Math.min(slackX, Math.max(0, Math.trunc(slackX * cropOffsetX + 0.5))) : 0
  const cropY = slackY > 0 ? Math.min(slackY, Math.max(0, Math.trunc(slackY * cropOffsetY + 0.5))) : 0

  const scale = resolvedResizeMode === 'fit_width'
    ? frameWidth / sourceWidth
    : frameHeight / sourceHeight
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('WAN frame projection produced invalid scale.')
  }

  const resizedWidth = Math.max(1, Math.round(sourceWidth * scale))
  const resizedHeight = Math.max(1, Math.round(sourceHeight * scale))

  return {
    resizeMode,
    resolvedResizeMode,
    sourceWidth,
    sourceHeight,
    frameWidth,
    frameHeight,
    scale,
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
