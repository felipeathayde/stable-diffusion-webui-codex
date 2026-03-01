/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN img2vid frame projection math for no-stretch preview overlays.
Provides a deterministic cover+center-crop projection model so the UI can show which source-image area
is used for generation when target dimensions differ from the init image aspect ratio.

Symbols (top-level; keep in sync; no ghosts):
- `WAN_IMG2VID_FRAME_PROJECTION_POLICY_LABEL` (constant): Human-readable label for WAN no-stretch projection policy.
- `WanImg2VidFrameGuideConfig` (type): Optional zoom-overlay frame-guide config (target dims + policy metadata).
- `WanImg2VidFrameProjectionPolicy` (type): Supported no-stretch projection policy ids.
- `WanImg2VidFrameProjectionInput` (interface): Projection input contract (source dims + frame dims + policy).
- `WanImg2VidFrameProjection` (interface): Projection output contract (crop rect + scaled dims + scale factor).
- `computeWanImg2VidFrameProjection` (function): Computes no-stretch cover+center-crop projection metadata.
*/

export type WanImg2VidFrameProjectionPolicy = 'cover_center_crop'
export const WAN_IMG2VID_FRAME_PROJECTION_POLICY_LABEL = 'Cover + Center Crop (no stretch)'

export interface WanImg2VidFrameGuideConfig {
  targetWidth: number
  targetHeight: number
  policy?: WanImg2VidFrameProjectionPolicy | string
}

export interface WanImg2VidFrameProjectionInput {
  sourceWidth: number
  sourceHeight: number
  frameWidth: number
  frameHeight: number
  policy?: WanImg2VidFrameProjectionPolicy
}

export interface WanImg2VidFrameProjection {
  policy: WanImg2VidFrameProjectionPolicy
  sourceWidth: number
  sourceHeight: number
  frameWidth: number
  frameHeight: number
  scale: number
  resizedWidth: number
  resizedHeight: number
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

export function computeWanImg2VidFrameProjection(
  input: WanImg2VidFrameProjectionInput,
): WanImg2VidFrameProjection {
  const policy: WanImg2VidFrameProjectionPolicy = input.policy || 'cover_center_crop'
  if (policy !== 'cover_center_crop') {
    throw new Error(`Unsupported WAN frame projection policy '${String(policy)}'.`)
  }

  const sourceWidth = requirePositiveFinite('sourceWidth', input.sourceWidth)
  const sourceHeight = requirePositiveFinite('sourceHeight', input.sourceHeight)
  const frameWidth = requirePositiveFinite('frameWidth', input.frameWidth)
  const frameHeight = requirePositiveFinite('frameHeight', input.frameHeight)

  const scale = Math.max(frameWidth / sourceWidth, frameHeight / sourceHeight)
  if (!Number.isFinite(scale) || scale <= 0) {
    throw new Error('WAN frame projection produced invalid scale.')
  }

  const resizedWidth = Math.max(1, Math.round(sourceWidth * scale))
  const resizedHeight = Math.max(1, Math.round(sourceHeight * scale))

  const widthScaledByHeight = sourceWidth * frameHeight
  const heightScaledByWidth = sourceHeight * frameWidth
  let cropWidth = sourceWidth
  let cropHeight = sourceHeight
  if (widthScaledByHeight > heightScaledByWidth) {
    cropWidth = Math.max(1, Math.min(sourceWidth, Math.trunc((sourceHeight * frameWidth + Math.trunc(frameHeight / 2)) / frameHeight)))
  } else if (widthScaledByHeight < heightScaledByWidth) {
    cropHeight = Math.max(1, Math.min(sourceHeight, Math.trunc((sourceWidth * frameHeight + Math.trunc(frameWidth / 2)) / frameWidth)))
  }

  const cropX = Math.trunc((sourceWidth - cropWidth) / 2)
  const cropY = Math.trunc((sourceHeight - cropHeight) / 2)

  return {
    policy,
    sourceWidth,
    sourceHeight,
    frameWidth,
    frameHeight,
    scale,
    resizedWidth,
    resizedHeight,
    cropX,
    cropY,
    cropWidth,
    cropHeight,
  }
}
