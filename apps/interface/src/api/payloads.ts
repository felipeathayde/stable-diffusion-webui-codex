/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Zod request schemas + payload builders for image generation (txt2img/img2img).
Defines the canonical `Txt2ImgRequestSchema`, UI form-state types, and helpers to build request payloads (including hires/refiner) and to
apply engine-agnostic request normalization/validation.

Symbols (top-level; keep in sync; no ghosts):
- `DISTILLED_CFG_ENGINES` (const): Engine ids treated as distilled-guidance engines (use `distilled_cfg`; CFG/negative prompt omitted).
- `DEVICE_VALUES` (const): Allowed device tokens for requests.
- `DeviceEnum` (const): Zod enum built from `DEVICE_VALUES`.
- `RefinerOptionsSchema` (const): Zod schema for refiner options.
- `UpscalerTileSchema` (const): Zod schema for upscaler tiling config (tile/overlap + OOM fallback + min tile).
- `HiresOptionsSchema` (const): Zod schema for hires options (including nested refiner).
- `PromptSchema` (const): Zod schema for prompt validation/normalization.
- `Txt2ImgRequestSchema` (const): Zod schema for txt2img/img2img request payloads.
- `Txt2ImgRequest` (type): Inferred request type from `Txt2ImgRequestSchema`.
- `UpscalerTileFormState` (interface): UI form state for tile config used by upscaler-driven stages.
- `HiresFormState` (interface): UI form state for hires options.
- `RefinerFormState` (interface): UI form state for refiner options.
- `Txt2ImgFormState` (interface): UI form state for txt2img/img2img payload building.
- `normalizeDevice` (function): Normalizes and validates a device token.
- `buildTxt2ImgPayload` (function): Builds and validates a `Txt2ImgRequest` from UI form state (supports hires tile prefs: fallback + min_tile).
- `formatZodError` (function): Converts Zod errors (or unknown errors) into a readable message.
*/

import { z, ZodError } from 'zod'

// Engines that use distilled guidance (single-branch conditioning) and therefore use distilled_cfg.
const DISTILLED_CFG_ENGINES = ['flux1', 'flux1_kontext', 'flux1_chroma'] as const
const DEVICE_VALUES = ['cuda', 'cpu', 'mps', 'xpu', 'directml'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)

const RefinerOptionsSchema = z
  .object({
    enable: z.literal(true),
    switch_at_step: z.number().int().min(1),
    cfg: z.number(),
    seed: z.number().int(),
    model: z.string().min(1).optional(),
    vae: z.string().min(1).optional(),
  })
  .strict()

const UpscalerTileSchema = z
  .object({
    tile: z.number().int().min(1),
    overlap: z.number().int().min(0),
    fallback_on_oom: z.boolean(),
    min_tile: z.number().int().min(1),
  })
  .strict()
  .superRefine((value, ctx) => {
    if (value.overlap >= value.tile) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "'hires.tile.overlap' must be < tile" })
    }
    if (value.fallback_on_oom && value.min_tile > value.tile) {
      ctx.addIssue({ code: z.ZodIssueCode.custom, message: "'hires.tile.min_tile' must be <= tile when fallback_on_oom is enabled" })
    }
  })

const HiresOptionsSchema = z
  .object({
    enable: z.literal(true),
    denoise: z.number().min(0).max(1),
    scale: z.number().min(1),
    resize_x: z.number().int().min(0),
    resize_y: z.number().int().min(0),
    steps: z.number().int().min(0),
    upscaler: z
      .string()
      .min(1)
      .refine((value) => value.startsWith('latent:') || value.startsWith('spandrel:'), {
        message: "hires.upscaler must be an id like 'latent:*' or 'spandrel:*'",
      }),
    tile: UpscalerTileSchema.optional(),
    checkpoint: z.string().min(1).optional(),
    modules: z.array(z.string().min(1)).optional(),
    sampler: z.string().min(1).optional(),
    scheduler: z.string().min(1).optional(),
    prompt: z.string().optional(),
    negative_prompt: z.string().optional(),
    cfg: z.number().optional(),
    distilled_cfg: z.number().optional(),
    refiner: RefinerOptionsSchema.optional(),
  })
  .strict()

const PromptSchema = z
  .string()
  .transform((value) => value.trim())
  .refine((value) => value.length > 0, { message: 'Prompt must not be empty' })

export const Txt2ImgRequestSchema = z
  .object({
    device: DeviceEnum,
    prompt: PromptSchema,
    negative_prompt: z.string().optional().default(''),
    width: z.number().int().min(8).max(8192),
    height: z.number().int().min(8).max(8192),
    steps: z.number().int().min(1),
    cfg: z.number().optional(),  // Diffusion models (SD, SDXL)
    distilled_cfg: z.number().optional(),  // Distilled-guidance engines (Flux, Chroma)
    sampler: z.string().min(1),
    scheduler: z.string().min(1),
    seed: z.number().int(),
    clip_skip: z.number().int().min(0).max(12).optional(),
    styles: z.array(z.string().min(1)).optional(),
    metadata: z.record(z.any()).optional(),
    engine: z.string().min(1).optional(),
    model: z.string().min(1).optional(),
    smart_offload: z.boolean().optional(),
    smart_fallback: z.boolean().optional(),
    smart_cache: z.boolean().optional(),
    extras: z
      .object({
        hires: HiresOptionsSchema.optional(),
        refiner: RefinerOptionsSchema.optional(),
        text_encoder_override: z
          .object({
            family: z.string().min(1),
            label: z.string().min(1),
            components: z.array(z.string().min(1)).optional(),
          })
          .optional(),
        // Batch params
        batch_size: z.number().int().min(1).optional(),
        batch_count: z.number().int().min(1).optional(),
        // SHA-based model selection
        tenc_sha: z.union([z.string(), z.array(z.string())]).optional(),
        vae_sha: z.string().optional(),
        lora_sha: z.string().optional(),
        model_sha: z.string().optional(),
        // Z-Image variant selection
        zimage_variant: z.enum(['turbo', 'base']).optional(),
      })
      .passthrough()  // Allow additional dynamic keys for engine-specific extras
      .optional(),
  })
  .strict()

export type Txt2ImgRequest = z.infer<typeof Txt2ImgRequestSchema>

export interface UpscalerTileFormState {
  tile: number
  overlap: number
}

export interface HiresFormState {
  enabled: boolean
  denoise: number
  scale: number
  resizeX: number
  resizeY: number
  steps: number
  upscaler: string
  tile: UpscalerTileFormState
  checkpoint?: string
  modules?: string[]
  sampler?: string
  scheduler?: string
  prompt?: string
  negativePrompt?: string
  cfg?: number
  distilledCfg?: number
  refiner?: RefinerFormState
}

export interface RefinerFormState {
  enabled: boolean
  swapAtStep: number
  cfg: number
  seed: number
  model?: string
  vae?: string
}

export interface Txt2ImgFormState {
  prompt: string
  negativePrompt: string
  width: number
  height: number
  steps: number
  guidanceScale: number
  sampler: string
  scheduler: string
  seed: number
  clipSkip: number
  batchSize: number
  batchCount: number
  styles?: string[]
  device: Txt2ImgRequest['device']
  engine?: string
  model?: string
  smartOffload?: boolean
  smartFallback?: boolean
  smartCache?: boolean
  hires?: HiresFormState
  refiner?: RefinerFormState
  extras?: Record<string, unknown>
}

function normalizeDevice(device: string): Txt2ImgRequest['device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as Txt2ImgRequest['device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

export function buildTxt2ImgPayload(
  state: Txt2ImgFormState,
  opts: { hiresFallbackOnOom?: boolean; hiresMinTile?: number } = {},
): Txt2ImgRequest {
  const isDistilledCfgModel = DISTILLED_CFG_ENGINES.includes(state.engine as typeof DISTILLED_CFG_ENGINES[number])
  const hiresFallbackOnOom = opts.hiresFallbackOnOom ?? true
  const hiresMinTilePrefRaw = opts.hiresMinTile
  const hiresMinTilePref = (typeof hiresMinTilePrefRaw === 'number' && Number.isFinite(hiresMinTilePrefRaw))
    ? Math.max(1, Math.trunc(hiresMinTilePrefRaw))
    : 128
  
  const payload: Record<string, unknown> = {
    device: normalizeDevice(state.device),
    prompt: state.prompt.trim(),
    width: state.width,
    height: state.height,
    steps: state.steps,
    sampler: state.sampler,
    scheduler: state.scheduler,
    seed: state.seed,
  }

  if (Number.isFinite(state.clipSkip) && state.clipSkip >= 0) {
    payload.clip_skip = Math.trunc(state.clipSkip)
  }
  
  // Distilled-guidance engines: use distilled_cfg, no CFG/negative prompt (single-branch conditioning)
  // CFG engines: use cfg with negative prompt
  if (isDistilledCfgModel) {
    payload.distilled_cfg = state.guidanceScale
  } else {
    payload.cfg = state.guidanceScale
    if (state.negativePrompt?.trim()) {
      payload.negative_prompt = state.negativePrompt
    }
  }

  const styles = state.styles?.filter((entry) => entry.trim().length > 0) ?? []
  if (styles.length > 0) {
    payload.styles = styles
  }

  if (state.engine) {
    payload.engine = state.engine
  }
  if (state.model) {
    payload.model = state.model
  }
  if (typeof state.smartOffload === 'boolean') {
    payload.smart_offload = state.smartOffload
  }
  if (typeof state.smartFallback === 'boolean') {
    payload.smart_fallback = state.smartFallback
  }
  if (typeof state.smartCache === 'boolean') {
    payload.smart_cache = state.smartCache
  }

  const extras: Record<string, unknown> = {
    batch_size: state.batchSize,
    batch_count: state.batchCount,
  }
  if (state.hires?.enabled) {
    const tile = state.hires.tile ?? { tile: 256, overlap: 16 }
    const tileSize = Math.max(1, Math.trunc(tile.tile))
    const minTile = Math.max(1, Math.min(tileSize, hiresMinTilePref))
    extras.hires = {
      enable: true,
      denoise: state.hires.denoise,
      scale: state.hires.scale,
      resize_x: state.hires.resizeX,
      resize_y: state.hires.resizeY,
      steps: state.hires.steps,
      upscaler: state.hires.upscaler,
      tile: {
        tile: tileSize,
        overlap: Math.trunc(tile.overlap),
        fallback_on_oom: Boolean(hiresFallbackOnOom),
        min_tile: minTile,
      },
      checkpoint: state.hires.checkpoint,
      modules: state.hires.modules && state.hires.modules.length > 0 ? state.hires.modules : undefined,
      sampler: state.hires.sampler,
      scheduler: state.hires.scheduler,
      prompt: state.hires.prompt,
      negative_prompt: state.hires.negativePrompt,
      cfg: state.hires.cfg,
      distilled_cfg: state.hires.distilledCfg,
      refiner: state.hires.refiner?.enabled
        ? {
            enable: true,
            switch_at_step: state.hires.refiner.swapAtStep,
            cfg: state.hires.refiner.cfg,
            seed: state.hires.refiner.seed,
            model: state.hires.refiner.model,
          }
        : undefined,
    }
  }
  if (state.refiner?.enabled) {
    extras.refiner = {
      enable: true,
      switch_at_step: state.refiner.swapAtStep,
      cfg: state.refiner.cfg,
      seed: state.refiner.seed,
      model: state.refiner.model,
    }
  }
  // Merge engine-specific extras from state (e.g., tenc_sha for Z Image)
  if (state.extras) {
    for (const [key, value] of Object.entries(state.extras)) {
      if (value !== undefined) {
        extras[key] = value
      }
    }
  }
  if (Object.keys(extras).length > 0) {
    payload.extras = extras
  }

  return Txt2ImgRequestSchema.parse(payload)
}

export function formatZodError(err: unknown): string {
  if (err instanceof ZodError) {
    return err.errors.map((issue) => issue.message).join('; ')
  }
  return err instanceof Error ? err.message : String(err)
}
