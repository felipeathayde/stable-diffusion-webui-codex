/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend API DTOs and response/payload types.
Defines TypeScript interfaces/types for backend responses (models/options/samplers/tasks/events/inventory) and UI-driven schemas (settings schema, UI blocks/presets, tabs/workflows).

Symbols (top-level; keep in sync; no ghosts):
- `ModelInfo` (interface): Model list entry returned by `/api/models`.
- `SamplerInfo` (interface): Sampler metadata entry returned by `/api/samplers`.
- `SchedulerInfo` (interface): Scheduler metadata entry returned by `/api/schedulers`.
- `ModelsResponse` (interface): `/api/models` response shape.
- `SamplersResponse` (interface): `/api/samplers` response shape.
- `SchedulersResponse` (interface): `/api/schedulers` response shape.
- `VaesResponse` (interface): `/api/vaes` response shape.
- `TextEncodersResponse` (interface): `/api/text-encoders` response shape.
- `OptionsResponse` (interface): `/api/options` response shape.
- `OptionsUpdateResponse` (interface): `/api/options` update response shape.
- `Txt2ImgStartResponse` (interface): Start-task response shape (`task_id`).
- `GeneratedImage` (interface): Base64-encoded image payload used in task results and previews.
- `TaskEvent` (type): Task SSE event union emitted by `/api/tasks/:id/stream`.
- `TaskResult` (interface): Polled task result shape returned by `/api/tasks/:id`.
- `MemoryResponse` (interface): `/api/memory` response shape.
- `VersionResponse` (interface): `/api/version` response shape.
- `EngineCapabilities` (interface): Per-engine capability flags used to gate UI features.
- `EngineCapabilitiesResponse` (interface): `/api/engines/capabilities` response shape.
- `EmbeddingsResponse` (interface): `/api/embeddings` response shape.
- `LoraListResponse` (interface): `/api/loras` response shape.
- `PathsResponse` (interface): `/api/paths` response shape.
- `PathsUpdateResponse` (interface): `/api/paths` update response shape.
- `SettingsCategory` (interface): Settings category entry in settings schema responses.
- `SettingsSection` (interface): Settings section entry in settings schema responses.
- `SettingsFieldType` (type): Allowed field types in settings schema definitions.
- `SettingsField` (interface): Settings field entry in settings schema responses.
- `SettingsSchemaResponse` (interface): `/api/settings/schema` response shape.
- `UiFieldType` (type): Allowed field types for server-driven UI blocks.
- `UiFieldBind` (interface): Optional binds mapping UI fields to payload keys.
- `UiField` (interface): UI block field definition.
- `UiBlockWhen` (interface): Conditional activation for a UI block.
- `UiBlockLayout` (interface): Layout metadata for a UI block.
- `UiBlock` (interface): Server-driven UI block definition.
- `UiBlocksResponse` (interface): `/api/ui/blocks` response shape.
- `UiPreset` (interface): UI preset definition used by the frontend.
- `UiPresetsResponse` (interface): `/api/ui/presets` response shape.
- `UiPresetApplyResponse` (interface): `/api/ui/presets/apply` response shape.
- `ApiTabMeta` (interface): Per-tab metadata timestamps.
- `ApiTab` (interface): Persisted model tab definition.
- `TabsResponse` (interface): `/api/ui/tabs` response shape.
- `WorkflowsResponse` (interface): `/api/ui/workflows` response shape.
- `InventoryResponse` (interface): `/api/models/inventory` response shape.
*/

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
