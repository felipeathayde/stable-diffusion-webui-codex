/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod-validated payload schemas + builders for WAN video endpoints (txt2vid/img2vid/vid2vid).
Defines the strict API payload schemas and provides helpers that normalize UI inputs (device, stage params, assets, output settings),
handling unset sentinels and producing backend-ready payloads for `/api/*` requests (including `settings_revision`).

	Symbols (top-level; keep in sync; no ghosts):
	- `WanTxt2VidPayloadSchema` (const): Zod schema for WAN `/api/txt2vid` payload.
	- `WanImg2VidPayloadSchema` (const): Zod schema for WAN `/api/img2vid` payload.
	- `WanVid2VidPayloadSchema` (const): Zod schema for WAN `/api/vid2vid` payload.
	- `WanTxt2VidPayload` (type): Zod-inferred payload type for WAN `/api/txt2vid`.
	- `WanImg2VidPayload` (type): Zod-inferred payload type for WAN `/api/img2vid`.
	- `WanVid2VidPayload` (type): Zod-inferred payload type for WAN `/api/vid2vid`.
	- `WanStageInput` (interface): UI-friendly stage params (high/low) that map to WAN stage overrides in payload.
	- `WanVideoOutputInput` (interface): Output options (filename prefix, output folder, format) mapped into payload.
	- `WanInterpolationInput` (interface): Optional interpolation config mapped into payload.
- `WanAssetsInput` (interface): WAN asset selection (metadata/text encoder/VAE) used to fill payload fields.
- `WanVideoCommonInput` (interface): Shared input fields for txt2vid/img2vid (prompt, dims, steps, seed, stage params, assets).
- `WanVid2VidInput` (interface): Vid2vid-specific input (includes init video path + strength/options) extending common input.
- `normalizeDevice` (function): Validates/normalizes device input into the backend enum.
- `snapWanDim` (function): Snaps WAN width/height to a multiple of 16 (rounded up; Diffusers parity).
- `stageToPayload` (function): Converts a `WanStageInput` into the backend stage override object (drops unset fields).
- `isUnsetSentinel` (function): Detects UI sentinel values (e.g., “Automatic”/“Built-in”) that must not be sent as real asset paths.
- `addWanAssets` (function): Injects selected WAN assets into the payload (skips unset/empty values).
- `addWanOutput` (function): Injects output-related fields into the payload.
- `addWanInterpolation` (function): Injects interpolation config into the payload.
- `buildWanTxt2VidPayload` (function): Builds a validated txt2vid payload from UI common input.
- `buildWanImg2VidPayload` (function): Builds a validated img2vid payload from UI input plus init image data.
- `buildWanVid2VidPayload` (function): Builds a validated vid2vid payload from UI vid2vid input.
*/

import { z } from 'zod'

const DEVICE_VALUES = ['cuda', 'cpu', 'mps', 'xpu', 'directml'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)

const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })

const WanFormatEnum = z.enum(['auto', 'diffusers', 'gguf'])

const WAN_DIM_STEP = 16

const Sha256Schema = z
  .string()
  .transform((value) => value.trim().toLowerCase())
  .refine((value) => /^[0-9a-f]{64}$/.test(value), { message: 'Expected sha256 (64 lowercase hex chars)' })

const RepoIdSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.includes('/') && !value.startsWith('/') && !value.endsWith('/'), { message: "Expected repo id like 'Org/Repo'" })

const VideoInterpolationSchema = z
  .object({
    enabled: z.boolean(),
    model: z.string().min(1).optional(),
    times: z.number().int().min(1).optional(),
  })
  .strict()

const WanStageSchema = z
  .object({
    model_sha: Sha256Schema,
    sampler: z.string().min(1).optional(),
    scheduler: z.string().min(1).optional(),
    steps: z.number().int().min(1),
    cfg_scale: z.number(),
    seed: z.number().int().optional(),
    lightning: z.boolean().optional(),
    lora_sha: Sha256Schema.optional(),
    lora_weight: z.number().optional(),
    flow_shift: z.number().optional(),
  })
  .strict()

const CommonWanVideoPayloadSchema = z
  .object({
    codex_device: DeviceEnum,
    settings_revision: z.number().int().min(0),

    video_return_frames: z.boolean().optional(),
    video_filename_prefix: z.string().min(1).optional(),
    video_format: z.string().min(1).optional(),
    video_pix_fmt: z.string().min(1).optional(),
    video_crf: z.number().int().min(0).max(51).optional(),
    video_loop_count: z.number().int().min(0).optional(),
    video_pingpong: z.boolean().optional(),
    video_save_metadata: z.boolean().optional(),
    video_save_output: z.boolean().optional(),
    video_trim_to_audio: z.boolean().optional(),

    video_interpolation: VideoInterpolationSchema.optional(),

    wan_high: WanStageSchema.optional(),
    wan_low: WanStageSchema.optional(),
    wan_format: WanFormatEnum.optional(),
    wan_metadata_repo: RepoIdSchema,
    wan_vae_sha: Sha256Schema,
    wan_tenc_sha: Sha256Schema,
    wan_tokenizer_dir: z.string().min(1).optional(),
  })
  .strict()

export const WanTxt2VidPayloadSchema = CommonWanVideoPayloadSchema.extend({
  txt2vid_prompt: PromptSchema,
  txt2vid_neg_prompt: z.string().optional().default(''),
  txt2vid_width: z.number().int().min(8).max(8192),
  txt2vid_height: z.number().int().min(8).max(8192),
  txt2vid_steps: z.number().int().min(1),
  txt2vid_fps: z.number().int().min(1).max(240),
  txt2vid_num_frames: z.number().int().min(1).max(4096),
  txt2vid_sampler: z.string().min(1).optional(),
  txt2vid_scheduler: z.string().min(1).optional(),
  txt2vid_seed: z.number().int().optional(),
  txt2vid_cfg_scale: z.number().optional(),
}).strict()

export type WanTxt2VidPayload = z.infer<typeof WanTxt2VidPayloadSchema>

export const WanImg2VidPayloadSchema = CommonWanVideoPayloadSchema.extend({
  img2vid_prompt: PromptSchema,
  img2vid_neg_prompt: z.string().optional().default(''),
  img2vid_width: z.number().int().min(8).max(8192),
  img2vid_height: z.number().int().min(8).max(8192),
  img2vid_steps: z.number().int().min(1),
  img2vid_fps: z.number().int().min(1).max(240),
  img2vid_num_frames: z.number().int().min(1).max(4096),
  img2vid_sampler: z.string().min(1).optional(),
  img2vid_scheduler: z.string().min(1).optional(),
  img2vid_seed: z.number().int().optional(),
  img2vid_cfg_scale: z.number().optional(),
  img2vid_init_image: z.string().min(1),
}).strict()

export type WanImg2VidPayload = z.infer<typeof WanImg2VidPayloadSchema>

export const WanVid2VidPayloadSchema = CommonWanVideoPayloadSchema.extend({
  vid2vid_prompt: PromptSchema,
  vid2vid_neg_prompt: z.string().optional().default(''),
  vid2vid_width: z.number().int().min(8).max(8192),
  vid2vid_height: z.number().int().min(8).max(8192),
  vid2vid_steps: z.number().int().min(1),
  vid2vid_fps: z.number().int().min(1).max(240),
  vid2vid_num_frames: z.number().int().min(1).max(4096),
  vid2vid_sampler: z.string().min(1).optional(),
  vid2vid_scheduler: z.string().min(1).optional(),
  vid2vid_seed: z.number().int().optional(),
  vid2vid_cfg_scale: z.number().optional(),
  vid2vid_strength: z.number().min(0).max(1).optional(),
  vid2vid_method: z.enum(['native', 'flow_chunks']).optional(),
  vid2vid_use_source_fps: z.boolean().optional(),
  vid2vid_use_source_frames: z.boolean().optional(),
  vid2vid_start_seconds: z.number().min(0).optional(),
  vid2vid_end_seconds: z.number().min(0).optional(),
  vid2vid_max_frames: z.number().int().min(1).optional(),
  vid2vid_chunk_frames: z.number().int().min(2).max(128).optional(),
  vid2vid_overlap_frames: z.number().int().min(0).max(127).optional(),
  vid2vid_preview_frames: z.number().int().min(1).max(512).optional(),
  vid2vid_flow_enabled: z.boolean().optional(),
  vid2vid_flow_use_large: z.boolean().optional(),
  vid2vid_flow_downscale: z.number().int().min(1).max(8).optional(),
  vid2vid_flow_device: z.string().min(1).optional(),
  // Path-based inputs are supported but restricted server-side; prefer multipart upload.
  vid2vid_video_path: z.string().min(1).optional(),
}).strict()

export type WanVid2VidPayload = z.infer<typeof WanVid2VidPayloadSchema>

export interface WanStageInput {
  modelSha: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  loraSha?: string
  loraWeight?: number
  flowShift?: number
}

export interface WanVideoOutputInput {
  filenamePrefix: string
  format: string
  pixFmt: string
  crf: number
  loopCount: number
  pingpong: boolean
  trimToAudio: boolean
  saveMetadata: boolean
  saveOutput: boolean
  returnFrames?: boolean
}

export interface WanInterpolationInput {
  enabled: boolean
  model: string
  times: number
}

export interface WanAssetsInput {
  metadataRepo: string
  textEncoderSha: string
  vaeSha: string
  tokenizerDir?: string
}

export interface WanVideoCommonInput {
  device: string
  settingsRevision: number
  prompt: string
  negativePrompt: string
  width: number
  height: number
  fps: number
  frames: number
  high: WanStageInput
  low: WanStageInput
  format: 'auto' | 'diffusers' | 'gguf'
  assets: WanAssetsInput
  output: WanVideoOutputInput
  interpolation: WanInterpolationInput
}

export interface WanVid2VidInput extends WanVideoCommonInput {
  strength: number
  method: 'native' | 'flow_chunks'
  useSourceFps: boolean
  useSourceFrames: boolean
  startSeconds?: number
  endSeconds?: number
  maxFrames?: number
  chunkFrames?: number
  overlapFrames?: number
  previewFrames?: number
  flowEnabled: boolean
  flowUseLarge: boolean
  flowDownscale: number
  flowDevice?: string
  videoPath?: string
}

function normalizeDevice(device: string): WanTxt2VidPayload['codex_device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as WanTxt2VidPayload['codex_device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

function normalizeSettingsRevision(value: unknown): number {
  if (typeof value === 'number' && Number.isFinite(value)) return Math.max(0, Math.trunc(value))
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (/^-?\d+$/.test(trimmed)) return Math.max(0, Math.trunc(Number(trimmed)))
  }
  return 0
}

function snapWanDim(value: number): number {
  if (!Number.isFinite(value)) return value
  const v = Math.trunc(value)
  return Math.ceil(v / WAN_DIM_STEP) * WAN_DIM_STEP
}

function stageToPayload(stage: WanStageInput): Record<string, unknown> {
  const modelSha = String(stage.modelSha || '').trim().toLowerCase()
  if (!modelSha) {
    throw new Error('WAN stage requires model_sha (sha256)')
  }
  if (!/^[0-9a-f]{64}$/.test(modelSha)) {
    throw new Error(`WAN stage model_sha must be sha256 (64 lowercase hex), got '${stage.modelSha}'`)
  }
  const payload: Record<string, unknown> = {
    model_sha: modelSha,
    steps: stage.steps,
    cfg_scale: stage.cfgScale,
    seed: stage.seed,
  }
  const sampler = String(stage.sampler || '').trim()
  if (sampler) {
    if (sampler !== sampler.toLowerCase()) {
      throw new Error(`WAN sampler must be canonical lowercase, got '${sampler}'`)
    }
    payload.sampler = sampler
  }
  const scheduler = String(stage.scheduler || '').trim()
  if (scheduler) {
    if (scheduler !== scheduler.toLowerCase()) {
      throw new Error(`WAN scheduler must be canonical lowercase, got '${scheduler}'`)
    }
    payload.scheduler = scheduler
  }
  const loraSha = String(stage.loraSha || '').trim().toLowerCase()
  if (loraSha) {
    if (!/^[0-9a-f]{64}$/.test(loraSha)) {
      throw new Error(`WAN stage lora_sha must be sha256 (64 lowercase hex), got '${stage.loraSha}'`)
    }
    payload.lora_sha = loraSha
    if (typeof stage.loraWeight === 'number') payload.lora_weight = stage.loraWeight
  }
  if (typeof stage.flowShift === 'number') payload.flow_shift = stage.flowShift

  return payload
}

function isUnsetSentinel(raw: string): boolean {
  const v = String(raw || '').trim().toLowerCase()
  if (!v) return true
  return v === 'automatic' || v === 'built in' || v === 'built-in' || v === 'none'
}

function addWanAssets(payload: Record<string, unknown>, assets: WanAssetsInput): void {
  const repo = String(assets.metadataRepo || '').trim()
  if (repo && !isUnsetSentinel(repo)) payload.wan_metadata_repo = repo

  const tokenizerDir = String(assets.tokenizerDir || '').trim()
  if (tokenizerDir && !isUnsetSentinel(tokenizerDir)) payload.wan_tokenizer_dir = tokenizerDir

  const vaeSha = String(assets.vaeSha || '').trim().toLowerCase()
  if (vaeSha) payload.wan_vae_sha = vaeSha

  const tencSha = String(assets.textEncoderSha || '').trim().toLowerCase()
  if (tencSha) payload.wan_tenc_sha = tencSha
}

function addWanOutput(payload: Record<string, unknown>, out: WanVideoOutputInput): void {
  const prefix = String(out.filenamePrefix || '').trim()
  if (prefix) payload.video_filename_prefix = prefix
  const format = String(out.format || '').trim()
  if (format) payload.video_format = format
  const pixFmt = String(out.pixFmt || '').trim()
  if (pixFmt) payload.video_pix_fmt = pixFmt
  if (Number.isFinite(out.crf)) payload.video_crf = out.crf
  if (Number.isFinite(out.loopCount)) payload.video_loop_count = out.loopCount
  payload.video_pingpong = Boolean(out.pingpong)
  payload.video_save_metadata = Boolean(out.saveMetadata)
  payload.video_save_output = Boolean(out.saveOutput)
  payload.video_trim_to_audio = Boolean(out.trimToAudio)
  if (out.returnFrames) payload.video_return_frames = true
}

function addWanInterpolation(payload: Record<string, unknown>, interpolation: WanInterpolationInput): void {
  if (!interpolation.enabled) return
  const model = String(interpolation.model || '').trim()
  payload.video_interpolation = {
    enabled: true,
    model: model || undefined,
    times: Number.isFinite(interpolation.times) ? interpolation.times : undefined,
  }
}

export function buildWanTxt2VidPayload(input: WanVideoCommonInput): WanTxt2VidPayload {
  const totalSteps = input.high.steps + input.low.steps
  const width = snapWanDim(input.width)
  const height = snapWanDim(input.height)
  const payload: Record<string, unknown> = {
    codex_device: normalizeDevice(input.device),
    settings_revision: normalizeSettingsRevision(input.settingsRevision),
    txt2vid_prompt: input.prompt,
    txt2vid_neg_prompt: input.negativePrompt,
    txt2vid_width: width,
    txt2vid_height: height,
    txt2vid_fps: input.fps,
    txt2vid_num_frames: input.frames,
    // Use total steps to keep WAN stage schedules continuous (GGUF runtime) and to avoid inconsistent payloads when
    // high/low stage steps differ.
    txt2vid_steps: totalSteps,
    txt2vid_cfg_scale: input.high.cfgScale,
    txt2vid_seed: input.high.seed,
  }

  const sampler = String(input.high.sampler || '').trim()
  if (sampler) payload.txt2vid_sampler = sampler
  const scheduler = String(input.high.scheduler || '').trim()
  if (scheduler) payload.txt2vid_scheduler = scheduler

  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation)

  payload.wan_high = stageToPayload(input.high)
  payload.wan_low = stageToPayload(input.low)
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)

  return WanTxt2VidPayloadSchema.parse(payload)
}

export function buildWanImg2VidPayload(input: WanVideoCommonInput & { initImageData: string }): WanImg2VidPayload {
  const totalSteps = input.high.steps + input.low.steps
  const width = snapWanDim(input.width)
  const height = snapWanDim(input.height)
  const payload: Record<string, unknown> = {
    codex_device: normalizeDevice(input.device),
    settings_revision: normalizeSettingsRevision(input.settingsRevision),
    img2vid_prompt: input.prompt,
    img2vid_neg_prompt: input.negativePrompt,
    img2vid_width: width,
    img2vid_height: height,
    img2vid_fps: input.fps,
    img2vid_num_frames: input.frames,
    // Use total steps to keep WAN stage schedules continuous (GGUF runtime) and to avoid inconsistent payloads when
    // high/low stage steps differ.
    img2vid_steps: totalSteps,
    img2vid_cfg_scale: input.high.cfgScale,
    img2vid_seed: input.high.seed,
    img2vid_init_image: input.initImageData,
  }

  const sampler = String(input.high.sampler || '').trim()
  if (sampler) payload.img2vid_sampler = sampler
  const scheduler = String(input.high.scheduler || '').trim()
  if (scheduler) payload.img2vid_scheduler = scheduler

  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation)

  payload.wan_high = stageToPayload(input.high)
  payload.wan_low = stageToPayload(input.low)
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)

  return WanImg2VidPayloadSchema.parse(payload)
}

export function buildWanVid2VidPayload(input: WanVid2VidInput): WanVid2VidPayload {
  const totalSteps = input.high.steps + input.low.steps
  const width = snapWanDim(input.width)
  const height = snapWanDim(input.height)
  const payload: Record<string, unknown> = {
    codex_device: normalizeDevice(input.device),
    settings_revision: normalizeSettingsRevision(input.settingsRevision),
    vid2vid_prompt: input.prompt,
    vid2vid_neg_prompt: input.negativePrompt,
    vid2vid_width: width,
    vid2vid_height: height,
    vid2vid_fps: input.fps,
    vid2vid_num_frames: input.frames,
    // Use total steps to keep WAN stage schedules continuous (GGUF runtime) and to avoid inconsistent payloads when
    // high/low stage steps differ.
    vid2vid_steps: totalSteps,
    vid2vid_cfg_scale: input.high.cfgScale,
    vid2vid_seed: input.high.seed,
    vid2vid_strength: input.strength,
    vid2vid_method: input.method,
    vid2vid_use_source_fps: input.useSourceFps,
    vid2vid_use_source_frames: input.useSourceFrames,
    vid2vid_flow_enabled: input.flowEnabled,
    vid2vid_flow_use_large: input.flowUseLarge,
    vid2vid_flow_downscale: input.flowDownscale,
  }

  const sampler = String(input.high.sampler || '').trim()
  if (sampler) payload.vid2vid_sampler = sampler
  const scheduler = String(input.high.scheduler || '').trim()
  if (scheduler) payload.vid2vid_scheduler = scheduler

  if (typeof input.startSeconds === 'number') payload.vid2vid_start_seconds = input.startSeconds
  if (typeof input.endSeconds === 'number') payload.vid2vid_end_seconds = input.endSeconds
  if (typeof input.maxFrames === 'number') payload.vid2vid_max_frames = input.maxFrames
  if (typeof input.chunkFrames === 'number') payload.vid2vid_chunk_frames = input.chunkFrames
  if (typeof input.overlapFrames === 'number') payload.vid2vid_overlap_frames = input.overlapFrames
  if (typeof input.previewFrames === 'number') payload.vid2vid_preview_frames = input.previewFrames
  if (typeof input.flowDevice === 'string' && input.flowDevice.trim()) payload.vid2vid_flow_device = input.flowDevice.trim()

  const vp = String(input.videoPath || '').trim()
  if (vp) payload.vid2vid_video_path = vp

  addWanOutput(payload, input.output)
  addWanInterpolation(payload, input.interpolation)

  payload.wan_high = stageToPayload(input.high)
  payload.wan_low = stageToPayload(input.low)
  if (input.format !== 'auto') payload.wan_format = input.format
  addWanAssets(payload, input.assets)

  return WanVid2VidPayloadSchema.parse(payload)
}
