/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared WAN img2vid temporal mode/constraint helpers (frontend).
Provides canonical normalization helpers for mode values and windowed temporal controls so payload/store/view layers enforce the same
`sliding|svi2|svi2_pro` contract (`stride % 4 == 0`, `commit - stride >= 4`) without drift.

Symbols (top-level; keep in sync; no ghosts):
- `WanImg2VidMode` (type): Allowed WAN img2vid temporal mode values (`solo|chunk|sliding|svi2|svi2_pro`).
- `WAN_WINDOW_STRIDE_ALIGNMENT` (const): Required stride alignment for WAN temporal scale.
- `WAN_WINDOW_COMMIT_OVERLAP_MIN` (const): Minimum committed overlap beyond stride.
- `normalizeWanImg2VidMode` (function): Normalizes unknown mode input into canonical `WanImg2VidMode`.
- `isWanWindowedImg2VidMode` (function): Type guard for windowed temporal modes (`sliding|svi2|svi2_pro`).
- `normalizeWanChunkOverlap` (function): Normalizes chunk overlap so `(chunk_frames - overlap_frames) % 4 == 0`.
- `normalizeWanWindowStride` (function): Normalizes stride to window-bounded aligned values compatible with commit-overlap contract.
- `normalizeWanWindowCommit` (function): Normalizes commit into `[stride + overlap_min, window]`.
*/

export type WanImg2VidMode = 'solo' | 'chunk' | 'sliding' | 'svi2' | 'svi2_pro'

export const WAN_WINDOW_STRIDE_ALIGNMENT = 4
export const WAN_WINDOW_COMMIT_OVERLAP_MIN = 4

const WAN_FRAMES_MIN = 9

export function normalizeWanImg2VidMode(value: unknown): WanImg2VidMode {
  const mode = String(value || '').trim().toLowerCase()
  if (mode === 'chunk' || mode === 'sliding' || mode === 'svi2' || mode === 'svi2_pro') return mode
  return 'solo'
}

export function isWanWindowedImg2VidMode(mode: WanImg2VidMode): mode is 'sliding' | 'svi2' | 'svi2_pro' {
  return mode === 'sliding' || mode === 'svi2' || mode === 'svi2_pro'
}

export function normalizeWanChunkOverlap(rawValue: number, chunkFrames: number, fallback: number): number {
  const chunkInt = Math.max(WAN_FRAMES_MIN, Math.trunc(chunkFrames))
  const overlapMax = Math.max(0, chunkInt - 1)
  const candidate = Number.isFinite(rawValue) ? Math.trunc(rawValue) : Math.trunc(fallback)
  const clamped = Math.min(overlapMax, Math.max(0, candidate))
  if ((chunkInt - clamped) % WAN_WINDOW_STRIDE_ALIGNMENT === 0) return clamped

  let best = clamped
  let bestDistance = Number.POSITIVE_INFINITY
  for (let overlap = 0; overlap <= overlapMax; overlap += 1) {
    if ((chunkInt - overlap) % WAN_WINDOW_STRIDE_ALIGNMENT !== 0) continue
    const distance = Math.abs(overlap - clamped)
    if (distance < bestDistance) {
      best = overlap
      bestDistance = distance
    }
  }
  return best
}

export function normalizeWanWindowStride(rawValue: number, windowFrames: number, fallback: number): number {
  const windowInt = Math.max(WAN_FRAMES_MIN, Math.trunc(windowFrames))
  const maxStride = Math.max(WAN_WINDOW_STRIDE_ALIGNMENT, windowInt - WAN_WINDOW_COMMIT_OVERLAP_MIN)
  const maxAlignedStride = maxStride - (maxStride % WAN_WINDOW_STRIDE_ALIGNMENT)

  const candidate = Number.isFinite(rawValue) ? Math.trunc(rawValue) : Math.trunc(fallback)
  const clamped = Math.min(maxStride, Math.max(WAN_WINDOW_STRIDE_ALIGNMENT, candidate))
  const aligned = clamped - (clamped % WAN_WINDOW_STRIDE_ALIGNMENT)

  return Math.max(WAN_WINDOW_STRIDE_ALIGNMENT, Math.min(maxAlignedStride, aligned))
}

export function normalizeWanWindowCommit(rawValue: number, windowFrames: number, windowStride: number, fallback: number): number {
  const windowInt = Math.max(WAN_FRAMES_MIN, Math.trunc(windowFrames))
  const minCommit = Math.max(
    windowStride + WAN_WINDOW_COMMIT_OVERLAP_MIN,
    WAN_WINDOW_STRIDE_ALIGNMENT + WAN_WINDOW_COMMIT_OVERLAP_MIN,
  )
  const candidate = Number.isFinite(rawValue) ? Math.trunc(rawValue) : Math.trunc(fallback)
  return Math.min(windowInt, Math.max(minCommit, candidate))
}
