export interface ModelInfo {
  title: string
  name: string
  model_name: string
  hash: string | null
  filename: string
  metadata: Record<string, unknown>
}

export interface SamplerInfo {
  name: string
  label?: string
  aliases: string[]
  supported?: boolean
  options: Record<string, unknown>
}

export interface SchedulerInfo {
  name: string
  label: string
  aliases: string[]
  supported?: boolean
}

export interface ModelsResponse {
  models: ModelInfo[]
  current: string | null
}

export interface SamplersResponse {
  samplers: SamplerInfo[]
}

export interface SchedulersResponse {
  schedulers: SchedulerInfo[]
}

export interface VaesResponse {
  vaes: string[]
  current: string | null
}

export interface TextEncodersResponse {
  text_encoders: string[]
  current: string[]
}

export interface OptionsResponse {
  values: Record<string, unknown>
}

export interface OptionsUpdateResponse {
  updated: string[]
}

export interface Txt2ImgStartResponse {
  task_id: string
}

export interface GeneratedImage {
  format: string
  data: string
}

export type TaskEvent =
  | { type: 'status'; stage: string }
  | {
      type: 'progress'
      stage: string
      percent?: number | null
      step?: number | null
      total_steps?: number | null
      eta_seconds?: number | null
      preview_image?: GeneratedImage
      preview_step?: number | null
    }
  | { type: 'result'; images: GeneratedImage[]; info: unknown; video?: { rel_path?: string | null; mime?: string | null } }
  | { type: 'error'; message: string }
  | { type: 'end' }

export interface TaskResult {
  status: 'running' | 'completed' | 'error'
  error?: string
  result?: {
    images: GeneratedImage[]
    info: unknown
    video?: { rel_path?: string | null; mime?: string | null }
  }
}

export interface MemoryResponse {
  total_vram_mb: number
}

export interface VersionResponse {
  app_version: string
  git_commit: string | null
  python_version: string
  torch_version: string | null
  cuda_version: string | null
}

export interface EngineCapabilities {
  supports_txt2img: boolean
  supports_img2img: boolean
  supports_txt2vid: boolean
  supports_img2vid: boolean
  supports_vid2vid?: boolean
  supports_highres: boolean
  supports_refiner: boolean
  supports_lora: boolean
  supports_controlnet: boolean
  // Optional: restrict UI to only these samplers/schedulers. Null/undefined = allow all.
  samplers?: string[] | null
  schedulers?: string[] | null
}

export interface EngineCapabilitiesResponse {
  engines: Record<string, EngineCapabilities>
  smart_cache?: Record<string, { hits: number; misses: number }>
}

export interface EmbeddingsResponse {
  loaded: Record<string, { step?: number | null; vectors?: number; shape?: number[] | null; sd_checkpoint?: string | null; sd_checkpoint_name?: string | null }>
  skipped: Record<string, { step?: number | null; vectors?: number; shape?: number[] | null; sd_checkpoint?: string | null; sd_checkpoint_name?: string | null }>
}

export interface LoraListResponse {
  loras: { name: string; path: string }[]
}

export interface PathsResponse { paths: Record<string, string[]> }
export interface PathsUpdateResponse { ok: boolean }

// Settings schema (extracted from legacy)
export interface SettingsCategory { id: string; label: string }
export interface SettingsSection { key: string; label: string; category_id?: string | null }
export type SettingsFieldType = 'checkbox' | 'slider' | 'radio' | 'dropdown' | 'number' | 'text' | 'color' | 'html'
export interface SettingsField {
  key: string
  label: string
  type: SettingsFieldType
  section: string
  default?: unknown
  min?: number | null
  max?: number | null
  step?: number | null
  choices?: unknown[] | null
  choices_source?: string | null
}
export interface SettingsSchemaResponse {
  categories: SettingsCategory[]
  sections: SettingsSection[]
  fields: SettingsField[]
  source?: string
  version?: number
}

// UI Blocks (server-driven parameter panels)
export type UiFieldType = 'text' | 'number' | 'checkbox' | 'select' | 'slider' | 'textarea'
export interface UiFieldBind { txt2vid?: string; img2vid?: string }
export interface UiField {
  key: string
  label: string
  type: UiFieldType
  default?: unknown
  help?: string
  min?: number
  max?: number
  step?: number
  options?: (string | number)[]
  bind?: UiFieldBind
  visibleIf?: Record<string, unknown>
}
export interface UiBlockWhen { engines?: string[]; tabs?: string[] }
export interface UiBlockLayout { columns?: number }
export interface UiBlock { id: string; when?: UiBlockWhen; layout?: UiBlockLayout; fields: UiField[] }
export interface UiBlocksResponse { version: number; blocks: UiBlock[]; semantic_engine?: string }

// UI Presets (Model UI)
export interface UiPreset { id: string; title: string; tabs?: string[]; model_select: { type: 'exact' | 'pattern'; value: string }; options?: Record<string, unknown> }
export interface UiPresetsResponse { version: number; presets: UiPreset[] }
export interface UiPresetApplyResponse { applied: boolean; model: string }

// Tabs/workflows persistence
export interface ApiTabMeta { createdAt: string; updatedAt: string }
export interface ApiTab { id: string; type: 'sd15' | 'sdxl' | 'flux' | 'zimage' | 'wan'; title: string; order: number; enabled: boolean; params: Record<string, unknown>; meta: ApiTabMeta }
export interface TabsResponse { version: number; tabs: ApiTab[] }
export interface WorkflowsResponse { version: number; workflows: Array<{ id: string; name: string; source_tab_id: string; type: string; created_at: string; engine_semantics: string; params_snapshot: Record<string, unknown> }>} 

// Model inventory (for populating selects)
export interface InventoryResponse {
  vaes: Array<{ name: string; path: string; sha256?: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }>
  text_encoders: Array<{ name: string; path: string; sha256?: string }>
  loras: Array<{ name: string; path: string }>
  wan22: { gguf: Array<{ name: string; path: string; stage: 'high' | 'low' | 'unknown' }> }
  metadata: Array<{ name: string; path: string }>
}
