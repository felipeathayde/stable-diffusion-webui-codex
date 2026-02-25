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
- `resolveTextOverride` (function): Uses override text when non-blank; otherwise falls back to base text.
- `isHiresVisibleForMode` (function): Returns whether hires controls should be visible for the active mode/engine.
- `resolveHiresModePolicy` (function): Resolves hires panel visibility and reset behavior for the active mode/engine.
*/

export type MaskEnforcement = 'post_blend' | 'per_step_clamp'

export function normalizeMaskEnforcement(value: string): MaskEnforcement {
  return value === 'post_blend' ? 'post_blend' : 'per_step_clamp'
}

export function normalizeInpaintingFill(value: number): number {
  return Math.max(0, Math.min(3, Math.trunc(value)))
}

export function normalizeNonNegativeInt(value: number): number {
  return Math.max(0, Math.trunc(value))
}

export function resolveTextOverride(baseText: string, overrideText?: string): string {
  const override = String(overrideText ?? '')
  if (override.trim().length > 0) return override
  return String(baseText ?? '')
}

export function isHiresVisibleForMode(useInitImage: boolean, supportsHires: boolean): boolean {
  return !useInitImage && supportsHires
}

export function resolveHiresModePolicy(useInitImage: boolean, supportsHiresForEngine: boolean): { showCard: boolean; resetState: boolean } {
  return {
    showCard: isHiresVisibleForMode(useInitImage, supportsHiresForEngine),
    resetState: !supportsHiresForEngine,
  }
}
