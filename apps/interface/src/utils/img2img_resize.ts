/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical img2img resize-mode options used by UI state and payload routing.
Defines the stable option ids for img2img resize behavior selectors and a fail-loud normalizer for
incoming persisted/unknown values.

Symbols (top-level; keep in sync; no ghosts):
- `IMG2IMG_RESIZE_MODE_OPTIONS` (constant): Ordered resize-mode options exposed in img2img controls.
- `Img2ImgResizeMode` (type): Union of allowed resize-mode ids.
- `DEFAULT_IMG2IMG_RESIZE_MODE` (constant): Default resize mode for image-tab params.
- `normalizeImg2ImgResizeMode` (function): Normalizes unknown values to a valid resize-mode id.
*/

export const IMG2IMG_RESIZE_MODE_OPTIONS = [
  { value: 'just_resize', label: 'Just resize' },
  { value: 'crop_and_resize', label: 'Crop and resize' },
  { value: 'resize_and_fill', label: 'Resize and fill' },
  { value: 'just_resize_latent_upscale', label: 'Just resize (latent upscale)' },
  { value: 'upscaler', label: 'Upscaler' },
] as const

export type Img2ImgResizeMode = (typeof IMG2IMG_RESIZE_MODE_OPTIONS)[number]['value']

export const DEFAULT_IMG2IMG_RESIZE_MODE: Img2ImgResizeMode = 'just_resize'

export function normalizeImg2ImgResizeMode(value: unknown): Img2ImgResizeMode {
  const raw = typeof value === 'string' ? value.trim() : ''
  for (const option of IMG2IMG_RESIZE_MODE_OPTIONS) {
    if (option.value === raw) return option.value
  }
  return DEFAULT_IMG2IMG_RESIZE_MODE
}
