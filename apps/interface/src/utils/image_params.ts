/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pure normalization helpers for image-tab parameter controls.
Centralizes normalization used by img2img/inpaint UI updates to keep parent handlers explicit and unit-testable.

Symbols (top-level; keep in sync; no ghosts):
- `MaskEnforcement` (type): Allowed mask enforcement values.
- `normalizeMaskEnforcement` (function): Normalizes a raw select value into a strict mask-enforcement enum.
- `normalizeInpaintingFill` (function): Clamps masked-content fill mode to backend-supported integer range `[0, 3]`.
- `normalizeNonNegativeInt` (function): Truncates and clamps any numeric input to `>= 0`.
*/

export type MaskEnforcement = 'post_blend' | 'per_step_clamp'

export function normalizeMaskEnforcement(value: string): MaskEnforcement {
  return value === 'per_step_clamp' ? 'per_step_clamp' : 'post_blend'
}

export function normalizeInpaintingFill(value: number): number {
  return Math.max(0, Math.min(3, Math.trunc(value)))
}

export function normalizeNonNegativeInt(value: number): number {
  return Math.max(0, Math.trunc(value))
}
