/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod-validated payload schemas + builders for the generic LTX video lane (`/api/txt2vid` + `/api/img2vid`).
Defines the strict payload contracts used by the dedicated LTX frontend path, normalizing device/settings revision and snapping generic-video
width/height + frame-count fields into the backend-supported domain while failing loud on unsupported sampler/scheduler assumptions.

Symbols (top-level; keep in sync; no ghosts):
- `LTX_ALLOWED_SAMPLERS` (const): Generic-video sampler lanes accepted by the current LTX backend contract.
- `LTX_CANONICAL_SCHEDULER` (const): Canonical scheduler accepted by the current LTX backend contract.
- `LtxTxt2VidPayloadSchema` (const): Zod schema for generic `/api/txt2vid` LTX requests.
- `LtxImg2VidPayloadSchema` (const): Zod schema for generic `/api/img2vid` LTX requests.
- `LtxTxt2VidPayload` (type): Zod-inferred txt2vid payload type.
- `LtxImg2VidPayload` (type): Zod-inferred img2vid payload type.
- `LtxVideoCommonInput` (interface): Shared UI-friendly input shape used by LTX payload builders.
- `LtxTxt2VidInput` (interface): Common txt2vid input.
- `LtxImg2VidInput` (interface): Img2vid input including init image data.
- `normalizeDevice` (function): Validates and normalizes a device token.
- `normalizeSettingsRevision` (function): Normalizes unknown revision input into a non-negative integer.
- `snapLtxDim` (function): Snaps dimensions up to the generic-video 16px grid.
- `normalizeLtxFrameCount` (function): Clamps/snap-normalizes frame counts into the backend `4n+1` domain.
- `normalizeLtxSampler` (function): Enforces the currently supported generic-video sampler lanes.
- `normalizeLtxScheduler` (function): Enforces the canonical generic-video scheduler.
- `buildLtxTxt2VidPayload` (function): Builds a validated generic txt2vid payload for engine `ltx2`.
- `buildLtxImg2VidPayload` (function): Builds a validated generic img2vid payload for engine `ltx2`.
*/

import { z } from 'zod'

const DEVICE_VALUES = ['cuda', 'cpu'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)
const LTX_DIM_STEP = 16
const LTX_DIM_MIN = 16
const LTX_DIM_MAX = 8192
const LTX_FRAMES_MIN = 9
const LTX_FRAMES_MAX = 401

export const LTX_ALLOWED_SAMPLERS = ['uni-pc', 'euler'] as const
export const LTX_CANONICAL_SCHEDULER = 'simple' as const

const LtxSamplerEnum = z.enum(LTX_ALLOWED_SAMPLERS)
const LtxSchedulerEnum = z.literal(LTX_CANONICAL_SCHEDULER)
const Sha256Schema = z
  .string()
  .transform((value) => value.trim().toLowerCase())
  .refine((value) => /^[0-9a-f]{64}$/.test(value), { message: 'Expected sha256 (64 lowercase hex chars)' })
const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })
const NegativePromptSchema = z.string().transform((value) => value.trim())
const LtxDimSchema = z
  .number()
  .int()
  .min(LTX_DIM_MIN)
  .max(LTX_DIM_MAX)
  .refine((value) => value % LTX_DIM_STEP === 0, { message: `Expected dimension aligned to ${LTX_DIM_STEP}px` })
const LtxFrameCountSchema = z
  .number()
  .int()
  .min(LTX_FRAMES_MIN)
  .max(LTX_FRAMES_MAX)
  .refine((value) => (value - 1) % 4 === 0, {
    message: `Expected 4n+1 frame count in [${LTX_FRAMES_MIN}, ${LTX_FRAMES_MAX}]`,
  })

const CommonLtxVideoPayloadSchema = z.object({
  device: DeviceEnum,
  settings_revision: z.number().int().min(0),
  engine: z.literal('ltx2'),
  model: z.string().min(1),
  model_sha: Sha256Schema.optional(),
  tenc_sha: Sha256Schema,
  vae_sha: Sha256Schema.optional(),
  video_save_output: z.literal(true),
  video_save_metadata: z.literal(true),
  video_return_frames: z.boolean(),
}).strict()

export const LtxTxt2VidPayloadSchema = CommonLtxVideoPayloadSchema.extend({
  txt2vid_prompt: PromptSchema,
  txt2vid_neg_prompt: NegativePromptSchema.default(''),
  txt2vid_width: LtxDimSchema,
  txt2vid_height: LtxDimSchema,
  txt2vid_steps: z.number().int().min(1),
  txt2vid_fps: z.number().int().min(1).max(240),
  txt2vid_num_frames: LtxFrameCountSchema,
  txt2vid_sampler: LtxSamplerEnum,
  txt2vid_scheduler: LtxSchedulerEnum,
  txt2vid_seed: z.number().int(),
  txt2vid_cfg_scale: z.number().min(0),
}).strict()

export const LtxImg2VidPayloadSchema = CommonLtxVideoPayloadSchema.extend({
  img2vid_prompt: PromptSchema,
  img2vid_neg_prompt: NegativePromptSchema.default(''),
  img2vid_width: LtxDimSchema,
  img2vid_height: LtxDimSchema,
  img2vid_steps: z.number().int().min(1),
  img2vid_fps: z.number().int().min(1).max(240),
  img2vid_num_frames: LtxFrameCountSchema,
  img2vid_sampler: LtxSamplerEnum,
  img2vid_scheduler: LtxSchedulerEnum,
  img2vid_seed: z.number().int(),
  img2vid_cfg_scale: z.number().min(0),
  img2vid_init_image: z.string().min(1),
}).strict()

export type LtxTxt2VidPayload = z.infer<typeof LtxTxt2VidPayloadSchema>
export type LtxImg2VidPayload = z.infer<typeof LtxImg2VidPayloadSchema>

export interface LtxVideoCommonInput {
  device: string
  settingsRevision: unknown
  model: string
  modelSha?: string | null | undefined
  prompt: string
  negativePrompt?: string
  width: number
  height: number
  fps: number
  frames: number
  steps: number
  cfgScale: number
  sampler: string
  scheduler: string
  seed: number
  textEncoderSha: string
  vaeSha?: string | null | undefined
  videoReturnFrames?: boolean
}

export interface LtxTxt2VidInput extends LtxVideoCommonInput {}

export interface LtxImg2VidInput extends LtxVideoCommonInput {
  initImageData: string
}

export function normalizeDevice(device: string): LtxTxt2VidPayload['device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as LtxTxt2VidPayload['device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

export function normalizeSettingsRevision(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.trunc(value))
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^-?\d+$/.test(trimmed)) return Math.max(0, Math.trunc(Number(trimmed)))
  }
  return 0
}

export function snapLtxDim(rawValue: number): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : LTX_DIM_MIN
  const clamped = Math.min(LTX_DIM_MAX, Math.max(LTX_DIM_MIN, numeric))
  return Math.ceil(clamped / LTX_DIM_STEP) * LTX_DIM_STEP
}

export function normalizeLtxFrameCount(rawValue: number): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : LTX_FRAMES_MIN
  const clamped = Math.min(LTX_FRAMES_MAX, Math.max(LTX_FRAMES_MIN, numeric))
  if ((clamped - 1) % 4 === 0) return clamped

  const down = clamped - (((clamped - 1) % 4 + 4) % 4)
  const up = down + 4
  const downInRange = down >= LTX_FRAMES_MIN
  const upInRange = up <= LTX_FRAMES_MAX
  if (downInRange && upInRange) {
    const downDistance = Math.abs(clamped - down)
    const upDistance = Math.abs(up - clamped)
    return downDistance <= upDistance ? down : up
  }
  if (downInRange) return down
  if (upInRange) return up
  return LTX_FRAMES_MIN
}

export function normalizeLtxSampler(rawValue: string): LtxTxt2VidPayload['txt2vid_sampler'] {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (LTX_ALLOWED_SAMPLERS.includes(normalized as (typeof LTX_ALLOWED_SAMPLERS)[number])) {
    return normalized as LtxTxt2VidPayload['txt2vid_sampler']
  }
  throw new Error(
    `Unsupported LTX sampler '${rawValue}'. Generic video currently accepts only ${LTX_ALLOWED_SAMPLERS.map((value) => `'${value}'`).join(', ')}.`,
  )
}

export function normalizeLtxScheduler(rawValue: string): typeof LTX_CANONICAL_SCHEDULER {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (normalized === LTX_CANONICAL_SCHEDULER) return LTX_CANONICAL_SCHEDULER
  throw new Error(`Unsupported LTX scheduler '${rawValue}'. Generic video currently requires '${LTX_CANONICAL_SCHEDULER}'.`)
}

function normalizeOptionalSha(rawValue: string | null | undefined): string | undefined {
  const normalized = String(rawValue || '').trim().toLowerCase()
  if (!normalized) return undefined
  return normalized
}

function buildCommonFields(input: LtxVideoCommonInput): Omit<LtxTxt2VidPayload, 'txt2vid_prompt' | 'txt2vid_neg_prompt' | 'txt2vid_width' | 'txt2vid_height' | 'txt2vid_steps' | 'txt2vid_fps' | 'txt2vid_num_frames' | 'txt2vid_sampler' | 'txt2vid_scheduler' | 'txt2vid_seed' | 'txt2vid_cfg_scale'> {
  const model = String(input.model || '').trim()
  if (!model) throw new Error('Select a checkpoint to generate.')
  const textEncoderSha = normalizeOptionalSha(input.textEncoderSha)
  if (!textEncoderSha) throw new Error('LTX requests require a text encoder sha.')

  const common: Omit<LtxTxt2VidPayload, 'txt2vid_prompt' | 'txt2vid_neg_prompt' | 'txt2vid_width' | 'txt2vid_height' | 'txt2vid_steps' | 'txt2vid_fps' | 'txt2vid_num_frames' | 'txt2vid_sampler' | 'txt2vid_scheduler' | 'txt2vid_seed' | 'txt2vid_cfg_scale'> = {
    device: normalizeDevice(String(input.device || '')),
    settings_revision: normalizeSettingsRevision(input.settingsRevision),
    engine: 'ltx2',
    model,
    tenc_sha: textEncoderSha,
    video_save_output: true,
    video_save_metadata: true,
    video_return_frames: Boolean(input.videoReturnFrames),
  }
  const modelSha = normalizeOptionalSha(input.modelSha)
  if (modelSha) common.model_sha = modelSha
  const vaeSha = normalizeOptionalSha(input.vaeSha)
  if (vaeSha) common.vae_sha = vaeSha
  return common
}

export function buildLtxTxt2VidPayload(input: LtxTxt2VidInput): LtxTxt2VidPayload {
  const payload: LtxTxt2VidPayload = {
    ...buildCommonFields(input),
    txt2vid_prompt: String(input.prompt || '').trim(),
    txt2vid_neg_prompt: String(input.negativePrompt || '').trim(),
    txt2vid_width: snapLtxDim(Number(input.width)),
    txt2vid_height: snapLtxDim(Number(input.height)),
    txt2vid_steps: Math.max(1, Math.trunc(Number(input.steps))),
    txt2vid_fps: Math.max(1, Math.trunc(Number(input.fps))),
    txt2vid_num_frames: normalizeLtxFrameCount(Number(input.frames)),
    txt2vid_sampler: normalizeLtxSampler(input.sampler),
    txt2vid_scheduler: normalizeLtxScheduler(input.scheduler),
    txt2vid_seed: Number.isFinite(Number(input.seed)) ? Math.trunc(Number(input.seed)) : -1,
    txt2vid_cfg_scale: Number.isFinite(Number(input.cfgScale)) ? Number(input.cfgScale) : 0,
  }
  return LtxTxt2VidPayloadSchema.parse(payload)
}

export function buildLtxImg2VidPayload(input: LtxImg2VidInput): LtxImg2VidPayload {
  const initImageData = String(input.initImageData || '').trim()
  if (!initImageData) throw new Error('Select an initial image for img2vid.')

  const payload: LtxImg2VidPayload = {
    ...buildCommonFields(input),
    img2vid_prompt: String(input.prompt || '').trim(),
    img2vid_neg_prompt: String(input.negativePrompt || '').trim(),
    img2vid_width: snapLtxDim(Number(input.width)),
    img2vid_height: snapLtxDim(Number(input.height)),
    img2vid_steps: Math.max(1, Math.trunc(Number(input.steps))),
    img2vid_fps: Math.max(1, Math.trunc(Number(input.fps))),
    img2vid_num_frames: normalizeLtxFrameCount(Number(input.frames)),
    img2vid_sampler: normalizeLtxSampler(input.sampler),
    img2vid_scheduler: normalizeLtxScheduler(input.scheduler),
    img2vid_seed: Number.isFinite(Number(input.seed)) ? Math.trunc(Number(input.seed)) : -1,
    img2vid_cfg_scale: Number.isFinite(Number(input.cfgScale)) ? Number(input.cfgScale) : 0,
    img2vid_init_image: initImageData,
  }
  return LtxImg2VidPayloadSchema.parse(payload)
}
