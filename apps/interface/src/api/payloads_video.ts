import { z } from 'zod'

const DEVICE_VALUES = ['cuda', 'cpu', 'mps', 'xpu', 'directml'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)

const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })

const WanFormatEnum = z.enum(['auto', 'diffusers', 'gguf'])

const VideoInterpolationSchema = z
  .object({
    enabled: z.boolean(),
    model: z.string().min(1).optional(),
    times: z.number().int().min(1).optional(),
  })
  .strict()

const WanStageSchema = z
  .object({
    model_dir: z.string().min(1).optional(),
    sampler: z.string().min(1).optional(),
    scheduler: z.string().min(1).optional(),
    steps: z.number().int().min(1),
    cfg_scale: z.number(),
    seed: z.number().int().optional(),
    lightning: z.boolean().optional(),
    lora_path: z.string().min(1).optional(),
    lora_weight: z.number().optional(),
    flow_shift: z.number().optional(),
  })
  .strict()

const CommonWanVideoPayloadSchema = z
  .object({
    codex_device: DeviceEnum,

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
    wan_metadata_dir: z.string().min(1).optional(),
    wan_vae_path: z.string().min(1).optional(),
    wan_text_encoder_path: z.string().min(1).optional(),
    wan_text_encoder_dir: z.string().min(1).optional(),
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
  modelDir: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  seed: number
  loraPath?: string
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
}

export interface WanInterpolationInput {
  enabled: boolean
  model: string
  times: number
}

export interface WanAssetsInput {
  metadataDir: string
  textEncoder: string
  vaePath: string
  tokenizerDir?: string
}

export interface WanVideoCommonInput {
  device: string
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

function stageToPayload(stage: WanStageInput): Record<string, unknown> {
  const payload: Record<string, unknown> = {
    steps: stage.steps,
    cfg_scale: stage.cfgScale,
    seed: stage.seed,
  }
  const modelDir = String(stage.modelDir || '').trim()
  if (modelDir) payload.model_dir = modelDir
  const sampler = String(stage.sampler || '').trim()
  if (sampler) payload.sampler = sampler
  const scheduler = String(stage.scheduler || '').trim()
  if (scheduler) payload.scheduler = scheduler
  const loraPath = String(stage.loraPath || '').trim()
  if (loraPath) {
    payload.lora_path = loraPath
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
  const metaDir = String(assets.metadataDir || '').trim()
  if (metaDir && !isUnsetSentinel(metaDir)) payload.wan_metadata_dir = metaDir

  const tokenizerDir = String(assets.tokenizerDir || '').trim()
  if (tokenizerDir && !isUnsetSentinel(tokenizerDir)) payload.wan_tokenizer_dir = tokenizerDir

  const vaeRaw = String(assets.vaePath || '').trim()
  if (vaeRaw && !isUnsetSentinel(vaeRaw)) {
    let vae = vaeRaw.replace(/\\+/g, '/')
    if (vae.startsWith('wan22/')) vae = vae.slice('wan22/'.length)
    if (vae.startsWith('vae/')) vae = vae.slice('vae/'.length)
    payload.wan_vae_path = vae
  }

  const teRaw = String(assets.textEncoder || '').trim()
  if (!teRaw || isUnsetSentinel(teRaw)) return
  let normalized = teRaw.replace(/\\+/g, '/')
  // Accept QuickSettings-style labels like "wan22/<abs/path/to/file.safetensors>".
  if (normalized.startsWith('wan22/')) normalized = normalized.slice('wan22/'.length)
  if (normalized.toLowerCase().endsWith('.safetensors')) payload.wan_text_encoder_path = normalized
  else payload.wan_text_encoder_dir = normalized
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
  const payload: Record<string, unknown> = {
    codex_device: normalizeDevice(input.device),
    txt2vid_prompt: input.prompt,
    txt2vid_neg_prompt: input.negativePrompt,
    txt2vid_width: input.width,
    txt2vid_height: input.height,
    txt2vid_fps: input.fps,
    txt2vid_num_frames: input.frames,
    // Use High as the representative defaults for engines that ignore stage overrides.
    txt2vid_steps: input.high.steps,
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
  const payload: Record<string, unknown> = {
    codex_device: normalizeDevice(input.device),
    img2vid_prompt: input.prompt,
    img2vid_neg_prompt: input.negativePrompt,
    img2vid_width: input.width,
    img2vid_height: input.height,
    img2vid_fps: input.fps,
    img2vid_num_frames: input.frames,
    // Use High as the representative defaults for engines that ignore stage overrides.
    img2vid_steps: input.high.steps,
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
  const payload: Record<string, unknown> = {
    codex_device: normalizeDevice(input.device),
    vid2vid_prompt: input.prompt,
    vid2vid_neg_prompt: input.negativePrompt,
    vid2vid_width: input.width,
    vid2vid_height: input.height,
    vid2vid_fps: input.fps,
    vid2vid_num_frames: input.frames,
    // Use High as the representative defaults for engines that ignore stage overrides.
    vid2vid_steps: input.high.steps,
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
