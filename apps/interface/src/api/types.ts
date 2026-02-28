/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Frontend API DTOs and response/payload types.
Defines TypeScript interfaces/types for backend responses (models/options/samplers/tasks/events/inventory) and UI-driven schemas (settings schema, UI blocks/presets, tabs/workflows), including options revision/apply metadata fields used by strict generation contracts.

Symbols (top-level; keep in sync; no ghosts):
- `ModelInfo` (interface): Model list entry returned by `/api/models`.
- `SamplerInfo` (interface): Sampler metadata entry returned by `/api/samplers`.
- `SchedulerInfo` (interface): Scheduler metadata entry returned by `/api/schedulers`.
- `ModelsResponse` (interface): `/api/models` response shape.
- `SamplersResponse` (interface): `/api/samplers` response shape.
- `SchedulersResponse` (interface): `/api/schedulers` response shape.
- `OptionsResponse` (interface): `/api/options` response shape.
- `OptionsUpdateResponse` (interface): `/api/options` update response shape.
- `TaskStartResponse` (interface): Start-task response shape (`task_id`) used by multiple endpoints.
- `Txt2ImgStartResponse` (interface): Start-task response shape (`task_id`).
- `UpscalerKind` (type): Allowed upscaler kind values (`latent`/`spandrel`).
- `UpscalerDefinition` (interface): Upscaler entry returned by `/api/upscalers`.
- `UpscalersResponse` (interface): `/api/upscalers` response shape.
- `UpscalersHfManifestV1` (interface): Canonical schema for `upscalers/manifest.json` (HF curated metadata).
- `UpscalersHfManifestV1Weight` (interface): One HF manifest weight entry.
- `RemoteUpscalerWeight` (type): Remote HF weight entry (either raw listing or curated + metadata).
- `RemoteUpscalersResponse` (interface): `/api/upscalers/remote` response shape (manifest + raw weights fallback).
- `GeneratedImage` (interface): Base64-encoded image payload used in task results and previews.
- `TaskEvent` (type): Task SSE event union emitted by `/api/tasks/:id/events` (supports replay via `id:` / `after` and emits `gap` on truncation).
- `TaskResult` (interface): Polled task result shape returned by `/api/tasks/:id`.
- `MemoryResponse` (interface): `/api/memory` response shape.
- `ObliterateVramProcessInfo` (interface): One external GPU process row returned by `/api/obliterate-vram`.
- `ObliterateVramFailure` (interface): One failed external process termination row from `/api/obliterate-vram`.
- `ObliterateVramSkippedProcess` (interface): One skipped external process row from `/api/obliterate-vram`.
- `ObliterateVramExternalKillMode` (type): External process termination mode for `/api/obliterate-vram`.
- `ObliterateVramRequest` (interface): Request payload for `/api/obliterate-vram`.
- `ObliterateVramResponse` (interface): `/api/obliterate-vram` response shape.
- `VersionResponse` (interface): `/api/version` response shape.
- `EngineCapabilities` (interface): Per-engine capability flags used to gate UI features.
- `GuidanceAdvancedCapabilities` (interface): Per-engine support map for advanced CFG/APG controls.
- `FamilyCapabilities` (interface): Per-family capability flags from backend (`families`) used to gate prompt/clip controls.
- `EngineDependencyCheckRow` (interface): One dependency-check row returned by backend readiness contract.
- `EngineDependencyStatus` (interface): Aggregated dependency status (`ready + checks`) for one semantic engine.
- `EngineCapabilitiesResponse` (interface): `/api/engines/capabilities` response shape.
- `PromptTokenCountRequest` (interface): Request payload for `/api/models/prompt-token-count`.
- `PromptTokenCountResponse` (interface): Response payload for `/api/models/prompt-token-count`.
- `EngineAssetContract` (interface): Per-engine asset requirements contract exposed by the backend (VAE/text encoders).
- `EngineAssetContractVariants` (interface): Base vs core-only contract variants for one engine id.
- `EmbeddingsResponse` (interface): `/api/embeddings` response shape.
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
- `ApiTab` (interface): Persisted model tab definition (`sd15|sdxl|flux1|zimage|chroma|wan|anima`).
- `TabsResponse` (interface): `/api/ui/tabs` response shape.
- `WorkflowsResponse` (interface): `/api/ui/workflows` response shape.
- `InventoryResponse` (interface): `/api/models/inventory` response shape.
- `PngInfoAnalyzeResponse` (interface): `/api/tools/pnginfo/analyze` response shape.
*/

export interface ModelInfo {
  title: string
  name: string
  model_name: string
  hash: string | null
  filename: string
  metadata: Record<string, unknown>
  core_only: boolean
  core_only_reason?: string | null
  family_hint?: string | null
}

export interface SamplerInfo {
  name: string
  label?: string
  supported?: boolean
  default_scheduler: string
  allowed_schedulers: string[]
}

export interface SchedulerInfo {
  name: string
  label?: string
  supported?: boolean
}

export interface ModelsResponse {
  models: ModelInfo[]
  current: string | null
}

export interface FileMetadataResponse {
  path: string
  kind: 'gguf' | 'safetensors'
  flat: Record<string, unknown>
  nested: Record<string, unknown>
  summary: Record<string, unknown>
}

export interface PngInfoAnalyzeResponse {
  width: number
  height: number
  metadata: Record<string, string>
}

export interface CheckpointMetadataResponse {
  hash: string | null
  'file.name': string
  'file.path': string
  'file.size.bytes': number
  'file.size.megabytes': number
  'file.size.gigabytes': number
  metadata: { raw: Record<string, unknown>; nested: Record<string, unknown> }
}

export interface SamplersResponse {
  samplers: SamplerInfo[]
}

export interface SchedulersResponse {
  schedulers: SchedulerInfo[]
}

export interface OptionsResponse {
  values: Record<string, unknown>
  revision?: number | null
}

export interface OptionsUpdateResponse {
  updated: string[]
  revision?: number | null
  applied_now?: string[] | null
  restart_required?: string[] | null
}

export interface TaskStartResponse {
  task_id: string
}

export interface Txt2ImgStartResponse extends TaskStartResponse {}

export type UpscalerKind = 'latent' | 'spandrel'

export interface UpscalerDefinition {
  id: string
  label: string
  kind: UpscalerKind
  meta: Record<string, unknown>
}

export interface UpscalersResponse {
  upscalers: UpscalerDefinition[]
}

export interface UpscalersHfManifestV1Weight {
  id: string
  hf_path: string
  label: string
  arch: string
  scale: number
  license_name: string
  license_url: string
  license_spdx: string | null
  sha256: string
  tags: string[]
  notes: string | null
}

export interface UpscalersHfManifestV1 {
  schema_version: 1
  weights: UpscalersHfManifestV1Weight[]
}

export interface RemoteUpscalerWeightMeta {
  id: string
  arch: string
  scale: number
  license_name: string
  license_url: string
  license_spdx: string | null
  sha256: string
  tags: string[]
  notes: string | null
}

export type RemoteUpscalerWeight =
  | {
      hf_path: string
      label: string
      curated: false
      meta: null
    }
  | {
      hf_path: string
      label: string
      curated: true
      meta: RemoteUpscalerWeightMeta
    }

export interface RemoteUpscalersResponse {
  repo_id: string
  revision: string | null
  manifest_path: string
  manifest_found: boolean
  manifest_error: string | null
  manifest_errors: string[]
  manifest: UpscalersHfManifestV1 | null
  weights: RemoteUpscalerWeight[]
  safeweights_enabled: boolean
  allowed_weight_suffixes: string[]
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
  | { type: 'gap'; oldest_event_id: number; newest_event_id: number }
  | { type: 'result'; images: GeneratedImage[]; info: unknown; video?: { rel_path?: string | null; mime?: string | null } }
  | { type: 'error'; message: string }
  | { type: 'end' }

export interface TaskResult {
  status: 'running' | 'completed' | 'error'
  task_id?: string
  stage?: string
  progress?: {
    stage?: string
    percent?: number | null
    step?: number | null
    total_steps?: number | null
    eta_seconds?: number | null
  } | null
  preview_image?: GeneratedImage
  preview_step?: number | null
  last_event_id?: number
  buffer_oldest_event_id?: number
  buffer_newest_event_id?: number
  started_at_ms?: number | null
  error?: string
  result?: {
    images: GeneratedImage[]
    info: unknown
    video?: { rel_path?: string | null; mime?: string | null }
  }
}

export interface MemoryResponse {
  total_vram_mb: number
  attention?: {
    backend?: string
    sdpa_policy?: string
    force_upcast?: boolean
    enable_flash?: boolean
    enable_mem_efficient?: boolean
    pytorch_sdp_enabled?: boolean
  }
}

export interface ObliterateVramProcessInfo {
  pid: number
  process_name: string
  used_gpu_memory_mb: number | null
  gpu_uuid: string
}

export interface ObliterateVramFailure {
  pid: number
  error: string
}

export interface ObliterateVramSkippedProcess {
  pid: number
  reason: string
}

export type ObliterateVramExternalKillMode = 'disabled' | 'all'

export interface ObliterateVramRequest {
  external_kill_mode?: ObliterateVramExternalKillMode
}

export interface ObliterateVramResponse {
  ok: boolean
  message: string
  internal: {
    runtime_unload_models: boolean
    runtime_soft_empty_cache: boolean
    gguf_cache_cleared: boolean
    gc_collect_ran: boolean
    torch_cuda_cache_cleared: boolean
  }
  internal_failures: string[]
  external: {
    kill_mode: ObliterateVramExternalKillMode
    nvidia_smi_available: boolean
    detected_processes: ObliterateVramProcessInfo[]
    terminated_pids: number[]
    skipped: ObliterateVramSkippedProcess[]
    failures: ObliterateVramFailure[]
  }
  warnings: string[]
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
  supports_hires: boolean
  supports_refiner: boolean
  supports_lora: boolean
  supports_controlnet: boolean
  // Optional: restrict UI to only these samplers/schedulers. Null/undefined = allow all.
  samplers?: string[] | null
  schedulers?: string[] | null
  default_sampler?: string | null
  default_scheduler?: string | null
  guidance_advanced?: GuidanceAdvancedCapabilities | null
}

export interface GuidanceAdvancedCapabilities {
  apg_enabled: boolean
  apg_start_step: boolean
  apg_eta: boolean
  apg_momentum: boolean
  apg_norm_threshold: boolean
  apg_rescale: boolean
  guidance_rescale: boolean
  cfg_trunc_ratio: boolean
  renorm_cfg: boolean
}

export interface FamilyCapabilities {
  supports_negative_prompt: boolean
  shows_clip_skip: boolean
}

export interface EngineDependencyCheckRow {
  id: string
  label: string
  ok: boolean
  message: string
}

export interface EngineDependencyStatus {
  ready: boolean
  checks: EngineDependencyCheckRow[]
}

export interface EngineCapabilitiesResponse {
  engines: Record<string, EngineCapabilities>
  families?: Record<string, FamilyCapabilities>
  smart_cache?: Record<string, { hits: number; misses: number }>
  asset_contracts?: Record<string, EngineAssetContractVariants>
  engine_id_to_semantic_engine: Record<string, string>
  dependency_checks: Record<string, EngineDependencyStatus>
}

export interface PromptTokenCountRequest {
  engine: string
  prompt: string
}

export interface PromptTokenCountResponse {
  engine: string
  prompt_len: number
  count: number
}

export interface EngineAssetContract {
  requires_vae: boolean
  tenc_count: number
  tenc_slots?: string[]
  tenc_slot_labels?: string[]
  tenc_kind: string
  tenc_kind_label?: string
  sha_only: boolean
  notes: string
}

export interface EngineAssetContractVariants {
  base: EngineAssetContract
  core_only: EngineAssetContract
}

export interface EmbeddingsResponse {
  loaded: Record<string, { step?: number | null; vectors?: number; shape?: number[] | null; sd_checkpoint?: string | null; sd_checkpoint_name?: string | null }>
  skipped: Record<string, { step?: number | null; vectors?: number; shape?: number[] | null; sd_checkpoint?: string | null; sd_checkpoint_name?: string | null }>
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
export interface UiPresetApplyResponse { applied: boolean; model: string; checkpoint: string; updated: string[] }

// Tabs/workflows persistence
export interface ApiTabMeta { createdAt: string; updatedAt: string }
export interface ApiTab { id: string; type: 'sd15' | 'sdxl' | 'flux1' | 'zimage' | 'chroma' | 'wan' | 'anima'; title: string; order: number; enabled: boolean; params: Record<string, unknown>; meta: ApiTabMeta }
export interface TabsResponse { version: number; tabs: ApiTab[] }
export interface WorkflowsResponse { version: number; workflows: Array<{ id: string; name: string; source_tab_id: string; type: string; created_at: string; engine_semantics: string; params_snapshot: Record<string, unknown> }>}

// Model inventory (for populating selects)
export interface InventoryResponse {
  vaes: Array<{ name: string; path: string; sha256?: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }>
  text_encoders: Array<{ name: string; path: string; sha256?: string }>
  loras: Array<{ name: string; path: string; sha256?: string }>
  wan22: { gguf: Array<{ name: string; path: string; sha256?: string; stage: 'high' | 'low' | 'unknown' }> }
  metadata: Array<{ name: string; path: string }>
}
