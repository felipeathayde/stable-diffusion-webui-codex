import { z, ZodError } from 'zod'

const DEVICE_VALUES = ['cuda', 'cpu', 'mps', 'xpu', 'directml'] as const
const DeviceEnum = z.enum(DEVICE_VALUES)

const HighresOptionsSchema = z
  .object({
    enable: z.literal(true),
    denoise: z.number().min(0).max(1),
    scale: z.number().min(1),
    resize_x: z.number().int().min(0),
    resize_y: z.number().int().min(0),
    steps: z.number().int().min(0),
    upscaler: z.string().min(1),
    checkpoint: z.string().min(1).optional(),
    modules: z.array(z.string().min(1)).optional(),
    sampler: z.string().min(1).optional(),
    scheduler: z.string().min(1).optional(),
    prompt: z.string().optional(),
    negative_prompt: z.string().optional(),
    cfg: z.number().optional(),
    distilled_cfg: z.number().optional(),
  })
  .strict()

const PromptSchema = z
  .string()
  .transform((value) => value.trim())

export const Txt2ImgRequestSchema = z
  .object({
    __strict_version: z.literal(1),
    codex_device: DeviceEnum,
    prompt: PromptSchema,
    negative_prompt: z.string().optional().default(''),
    width: z.number().int().min(8).max(8192),
    height: z.number().int().min(8).max(8192),
    steps: z.number().int().min(1),
    guidance_scale: z.number(),
    sampler: z.string().min(1),
    scheduler: z.string().min(1),
    seed: z.number().int(),
    batch_size: z.number().int().min(1),
    batch_count: z.number().int().min(1),
    styles: z.array(z.string().min(1)).optional(),
    metadata: z.record(z.any()).optional(),
    engine: z.string().min(1).optional(),
    model: z.string().min(1).optional(),
    extras: z
      .object({
        highres: HighresOptionsSchema.optional(),
      })
      .optional(),
  })
  .strict()

export type Txt2ImgRequest = z.infer<typeof Txt2ImgRequestSchema>

export interface HighresFormState {
  enabled: boolean
  denoise: number
  scale: number
  resizeX: number
  resizeY: number
  steps: number
  upscaler: string
  checkpoint?: string
  modules?: string[]
  sampler?: string
  scheduler?: string
  prompt?: string
  negativePrompt?: string
  cfg?: number
  distilledCfg?: number
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
  batchSize: number
  batchCount: number
  styles?: string[]
  device: Txt2ImgRequest['codex_device']
  engine?: string
  model?: string
  highres?: HighresFormState
}

function normalizeDevice(device: string): Txt2ImgRequest['codex_device'] {
  const normalized = device.trim().toLowerCase()
  if (DEVICE_VALUES.includes(normalized as (typeof DEVICE_VALUES)[number])) {
    return normalized as Txt2ImgRequest['codex_device']
  }
  throw new Error(`Unsupported device '${device}'`)
}

export function buildTxt2ImgPayload(state: Txt2ImgFormState): Txt2ImgRequest {
  const payload: Record<string, unknown> = {
    __strict_version: 1,
    codex_device: normalizeDevice(state.device),
    prompt: state.prompt.trim(),
    negative_prompt: state.negativePrompt ?? '',
    width: state.width,
    height: state.height,
    steps: state.steps,
    guidance_scale: state.guidanceScale,
    sampler: state.sampler,
    scheduler: state.scheduler,
    seed: state.seed,
    batch_size: state.batchSize,
    batch_count: state.batchCount,
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

  const extras: Record<string, unknown> = {}
  if (state.highres?.enabled) {
    extras.highres = {
      enable: true,
      denoise: state.highres.denoise,
      scale: state.highres.scale,
      resize_x: state.highres.resizeX,
      resize_y: state.highres.resizeY,
      steps: state.highres.steps,
      upscaler: state.highres.upscaler,
      checkpoint: state.highres.checkpoint,
      modules: state.highres.modules && state.highres.modules.length > 0 ? state.highres.modules : undefined,
      sampler: state.highres.sampler,
      scheduler: state.highres.scheduler,
      prompt: state.highres.prompt,
      negative_prompt: state.highres.negativePrompt,
      cfg: state.highres.cfg,
      distilled_cfg: state.highres.distilledCfg,
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
