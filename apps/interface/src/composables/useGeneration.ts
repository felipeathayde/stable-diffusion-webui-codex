/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Unified generation composable for image tabs (SD/Flux/Chroma/ZImage; txt2img/img2img/inpaint).
Owns per-tab generation state (progress/live preview/gallery/history), builds request payloads using Model Tabs + QuickSettings,
starts `/api/txt2img` and `/api/img2img` (txt2img can include hires settings; img2img stays hires-free),
includes `settings_revision` in payloads, handles
stale-revision conflicts (`409` + `current_revision`), and consumes task SSE events to update UI state.
Persists a minimal per-tab resume marker to `localStorage` and auto-reattaches to in-flight tasks after reload (SSE replay via `after` / `lastEventId`).

Symbols (top-level; keep in sync; no ghosts):
- `ImageRunHistoryItem` (interface): Persisted per-tab run history entry (task id, status, summary, params snapshot, error message).
- `GenerationState` (interface): Per-tab reactive runtime state (status/progress/preview/gallery/history selection).
- `defaultState` (function): Creates a fresh `GenerationState` with empty progress/gallery/history.
- `getTabState` (function): Returns (and initializes) the `GenerationState` for a given tab id from internal maps.
- `resolveEngineForRequest` (function): Canonical tab-type/mode -> backend engine mapping used for capability checks and request dispatch.
- `BuildImg2ImgPayloadArgs` (interface): Input contract for deterministic img2img payload assembly (no hires keys).
- `buildImg2ImgPayload` (function): Builds img2img start payload at the source and enforces the no-`img2img_hires_*` invariant fail-loud.
- `buildGuidancePayload` (function): Builds `extras.guidance` payload from tab state + per-engine advanced-guidance support matrix.
- `extractLoraNamesFromPrompt` (function): Extracts LoRA token names from prompt text (`<lora:name:weight>`).
- `isGenerationRunningForTab` (function): Returns whether the cached generation state for a tab id is currently `running`.
- `useGeneration` (function): Main composable API; wires payload building, task start, SSE handling, and history updates, enforcing GGUF-required
  `vae_sha`/`tenc_sha` (core-only checkpoints) and enforcing engine-level external asset requirements via backend `asset_contracts`.
*/

import { computed, reactive, ref } from 'vue'
import { useModelTabsStore, type BaseTab, type GuidanceAdvancedParams, type ImageBaseParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import { useUpscalersStore } from '../stores/upscalers'
import { getEngineConfig, type EngineType } from '../stores/engine_config'
import { buildTxt2ImgPayload, type Txt2ImgRequest } from '../api/payloads'
import { fetchTaskResult, startImg2Img, startTxt2Img, subscribeTask } from '../api/client'
import type { GeneratedImage, GuidanceAdvancedCapabilities, TaskEvent } from '../api/types'
import { resolveImageRequestEngineId } from '../utils/engine_taxonomy'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'

export interface ImageRunHistoryItem {
  taskId: string
  mode: 'txt2img' | 'img2img'
  createdAtMs: number
  status: 'completed' | 'error' | 'cancelled'
  summary: string
  promptPreview: string
  paramsSnapshot: Record<string, unknown>
  errorMessage?: string
}

export interface GenerationState {
  status: 'idle' | 'running' | 'done' | 'error'
  progress: {
    stage: string
    percent: number | null
    etaSeconds: number | null
    step: number | null
    totalSteps: number | null
  }
  previewImage: GeneratedImage | null
  previewStep: number | null
  gallery: GeneratedImage[]
  info: unknown | null
  errorMessage: string
  taskId: string
  lastSeed: number | null
  startedAtMs: number | null
  finishedAtMs: number | null
  history: ImageRunHistoryItem[]
  selectedTaskId: string
  historyLoadingTaskId: string
  currentRun: ImageRunHistoryItem | null
}

const MAX_HISTORY = 8
const RESUME_STORAGE_PREFIX = 'codex.resume.image'
const resumeAttempts = new Set<string>()
const LORA_TAG_RE = /<\s*lora\s*:\s*([^:>]+)\s*(?::[^>]*)?>/gi

type ResumeState = {
  taskId: string
  lastEventId: number
  createdAtMs: number
  paramsSnapshot: Record<string, unknown>
}

function resumeKey(tabId: string): string {
  return `${RESUME_STORAGE_PREFIX}.${tabId}`
}

function loadResumeState(key: string): ResumeState | null {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const obj = JSON.parse(raw) as any
    if (!obj || typeof obj !== 'object') return null
    if (typeof obj.taskId !== 'string' || !obj.taskId.trim()) return null
    const lastEventId = typeof obj.lastEventId === 'number' && Number.isFinite(obj.lastEventId) ? Math.trunc(obj.lastEventId) : 0
    const createdAtMs = typeof obj.createdAtMs === 'number' && Number.isFinite(obj.createdAtMs) ? Math.trunc(obj.createdAtMs) : 0
    const paramsSnapshot = obj.paramsSnapshot && typeof obj.paramsSnapshot === 'object' ? (obj.paramsSnapshot as Record<string, unknown>) : {}
    return { taskId: obj.taskId, lastEventId: Math.max(0, lastEventId), createdAtMs, paramsSnapshot }
  } catch {
    return null
  }
}

function saveResumeState(key: string, state: ResumeState): void {
  try {
    localStorage.setItem(key, JSON.stringify(state))
  } catch {
    // ignore localStorage failures (private mode/quota)
  }
}

function clearResumeState(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore
  }
}

function updateResumeEventId(key: string, eventId: number): void {
  const v = Math.trunc(Number(eventId))
  if (!Number.isFinite(v) || v <= 0) return
  const cur = loadResumeState(key)
  if (!cur) return
  if (v <= cur.lastEventId) return
  saveResumeState(key, { ...cur, lastEventId: v })
}

function defaultState(): GenerationState {
  return {
    status: 'idle',
    progress: { stage: 'none', percent: null, etaSeconds: null, step: null, totalSteps: null },
    previewImage: null,
    previewStep: null,
    gallery: [],
    info: null,
    errorMessage: '',
    taskId: '',
    lastSeed: null,
    startedAtMs: null,
    finishedAtMs: null,
    history: [],
    selectedTaskId: '',
    historyLoadingTaskId: '',
    currentRun: null,
  }
}

export function resolveEngineForRequest(tabType: string, useInitImage: boolean): string {
  return resolveImageRequestEngineId(tabType, useInitImage)
}

export interface BuildImg2ImgPayloadArgs {
  params: ImageBaseParams
  supportsNegativePrompt: boolean
  isDistilledCfgModel: boolean
  batchCount: number
  batchSize: number
  device: string
  settingsRevision: number
  engineId: string
  modelOverride: string
  extras: Record<string, unknown>
}

export function buildImg2ImgPayload(args: BuildImg2ImgPayloadArgs): Record<string, unknown> {
  const params = args.params
  const payload: Record<string, unknown> = {
    img2img_init_image: params.initImageData,
    img2img_mask: params.useMask ? params.maskImageData : '',
    img2img_prompt: params.prompt,
    img2img_neg_prompt: args.supportsNegativePrompt ? params.negativePrompt : '',
    img2img_styles: [],
    img2img_batch_count: args.batchCount,
    img2img_batch_size: args.batchSize,
    img2img_steps: params.steps,
    img2img_cfg_scale: args.isDistilledCfgModel ? 1.0 : params.cfgScale,
    img2img_distilled_cfg_scale: args.isDistilledCfgModel ? params.cfgScale : undefined,
    img2img_denoising_strength: params.denoiseStrength,
    img2img_width: params.width,
    img2img_height: params.height,
    img2img_sampling: params.sampler,
    img2img_scheduler: params.scheduler,
    img2img_seed: params.seed,
    img2img_clip_skip: params.clipSkip,
    device: args.device,
    settings_revision: args.settingsRevision,
    engine: args.engineId,
    model: args.modelOverride,
    img2img_extras: { ...args.extras },
  }
  if (params.useMask) {
    payload.img2img_mask_enforcement = params.maskEnforcement
    payload.img2img_inpainting_fill = Math.max(0, Math.min(3, Math.trunc(Number(params.inpaintingFill))))
    payload.img2img_inpaint_full_res = Boolean(params.inpaintFullRes)
    payload.img2img_inpaint_full_res_padding = Math.max(0, Math.trunc(Number(params.inpaintFullResPadding)))
    payload.img2img_inpainting_mask_invert = params.maskInvert ? 1 : 0
    payload.img2img_mask_blur = Math.max(0, Math.trunc(Number(params.maskBlur)))
    payload.img2img_mask_round = Boolean(params.maskRound)
  }
  for (const key of Object.keys(payload)) {
    if (key.startsWith('img2img_hires_')) {
      throw new Error(`Invalid img2img payload key '${key}'. Fix payload builder source.`)
    }
  }
  return payload
}

export function buildGuidancePayload(
  guidanceAdvanced: GuidanceAdvancedParams,
  support: GuidanceAdvancedCapabilities | null | undefined,
): Record<string, unknown> | null {
  if (!guidanceAdvanced.enabled) return null
  if (!support) return null

  const toFinite = (value: unknown, fallback: number): number => {
    const numeric = Number(value)
    return Number.isFinite(numeric) ? numeric : fallback
  }
  const clamp = (value: unknown, fallback: number, min?: number, max?: number): number => {
    const numeric = toFinite(value, fallback)
    if (min !== undefined && numeric < min) return min
    if (max !== undefined && numeric > max) return max
    return numeric
  }
  const clampInt = (value: unknown, fallback: number, min?: number): number => {
    const numeric = Math.trunc(toFinite(value, fallback))
    if (min !== undefined && numeric < min) return min
    return numeric
  }

  const payload: Record<string, unknown> = {}

  if (support.apg_enabled) payload.apg_enabled = true
  if (support.apg_start_step) payload.apg_start_step = clampInt(guidanceAdvanced.apgStartStep, 0, 0)
  if (support.apg_eta) payload.apg_eta = toFinite(guidanceAdvanced.apgEta, 0)
  if (support.apg_momentum) payload.apg_momentum = clamp(guidanceAdvanced.apgMomentum, 0, 0, 0.999999)
  if (support.apg_norm_threshold) payload.apg_norm_threshold = clamp(guidanceAdvanced.apgNormThreshold, 15, 0)
  if (support.apg_rescale) payload.apg_rescale = clamp(guidanceAdvanced.apgRescale, 0, 0, 1)
  if (support.guidance_rescale) payload.guidance_rescale = clamp(guidanceAdvanced.guidanceRescale, 0, 0, 1)
  if (support.cfg_trunc_ratio) {
    payload.cfg_trunc_ratio = clamp(guidanceAdvanced.cfgTruncRatio, 0.8, 0, 1)
  }
  if (support.renorm_cfg) payload.renorm_cfg = clamp(guidanceAdvanced.renormCfg, 0, 0)

  return Object.keys(payload).length > 0 ? payload : null
}

function extractLoraNamesFromPrompt(prompt: string): string[] {
  const names: string[] = []
  const seen = new Set<string>()
  const text = String(prompt || '')
  let match: RegExpExecArray | null = null
  LORA_TAG_RE.lastIndex = 0
  while ((match = LORA_TAG_RE.exec(text)) !== null) {
    const name = String(match[1] || '').trim()
    if (!name) continue
    const key = name.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    names.push(name)
  }
  return names
}

export function isGenerationRunningForTab(tabId: string): boolean {
  return getTabState(tabId).status === 'running'
}

// Per-tab generation state (keyed by tab ID)
const tabStates = new Map<string, GenerationState>()
const unsubscribers = new Map<string, () => void>()

function getTabState(tabId: string): GenerationState {
  if (!tabStates.has(tabId)) {
    tabStates.set(tabId, reactive(defaultState()) as GenerationState)
  }
  return tabStates.get(tabId)!
}

export function useGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()
  const backendCaps = useEngineCapabilitiesStore()
  const upscalersStore = useUpscalersStore()
  
  // Reactive state for this tab
  const state = ref(getTabState(tabId))
  const resumeNotice = ref('')
  
  // Tab info
  const tab = computed(() => modelTabs.tabs.find(t => t.id === tabId) as BaseTab | undefined)
  const params = computed(() => tab.value?.params as ImageBaseParams | undefined)
  const engineType = computed(() => tab.value?.type as EngineType | undefined)
  const engineConfig = computed(() => engineType.value ? getEngineConfig(engineType.value) : null)

  function buildRunSummary(p: ImageBaseParams): string {
    const sampler = p.sampler
    const scheduler = p.scheduler
    const seedLabel = p.seed === -1 ? 'seed random' : `seed ${p.seed}`
    const clipSkipLabel = Number.isFinite(p.clipSkip) && p.clipSkip > 0 && p.clipSkip !== 1 ? ` · clip-skip ${p.clipSkip}` : ''
    return `${p.width}×${p.height} px · ${p.steps} steps · cfg ${p.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel}${clipSkipLabel} · batch ${p.batchCount}×${p.batchSize}`
  }

  function buildParamsSnapshot(p: ImageBaseParams): Record<string, unknown> {
    return {
      checkpoint: p.checkpoint,
      textEncoders: p.textEncoders,

      prompt: p.prompt,
      negativePrompt: p.negativePrompt,

      width: p.width,
      height: p.height,
      sampler: p.sampler,
      scheduler: p.scheduler,
      steps: p.steps,
      cfgScale: p.cfgScale,
      seed: p.seed,
      clipSkip: p.clipSkip,

      batchSize: p.batchSize,
      batchCount: p.batchCount,
      img2imgResizeMode: p.img2imgResizeMode,
      img2imgUpscaler: p.img2imgUpscaler,
      guidanceAdvanced: p.guidanceAdvanced,

      hires: p.hires,
      refiner: p.refiner,

      useInitImage: p.useInitImage,
      initImageName: p.initImageName,
      denoiseStrength: p.denoiseStrength,
    }
  }

  function pushHistory(item: ImageRunHistoryItem): void {
    state.value.history.unshift(item)
    if (state.value.history.length > MAX_HISTORY) state.value.history.length = MAX_HISTORY
  }
  
  function stopStream(): void {
    const unsub = unsubscribers.get(tabId)
    if (unsub) {
      unsub()
      unsubscribers.delete(tabId)
    }
  }
  
  function resetProgress(): void {
    state.value.progress = { stage: 'none', percent: null, etaSeconds: null, step: null, totalSteps: null }
  }
  
  async function generate(): Promise<void> {
    if (!tab.value || !params.value || !engineType.value) {
      state.value.status = 'error'
      state.value.errorMessage = 'Tab not found'
      return
    }

    await backendCaps.init()
    
    stopStream()
    state.value.status = 'running'
    state.value.errorMessage = ''
    state.value.gallery = []
    state.value.info = null
    state.value.previewImage = null
    state.value.previewStep = null
    resetProgress()
    state.value.progress.stage = 'starting'
    state.value.startedAtMs = performance.now()
    state.value.finishedAtMs = null

    const p = params.value
    const config = engineConfig.value!
    const checkpoint = String((p as any).checkpoint || '').trim()
    const modelIsCoreOnly = quicksettings.isModelCoreOnly(checkpoint)
    const resolvedModelSha = quicksettings.resolveModelSha(checkpoint)
    const modelOverride = resolvedModelSha || checkpoint
    if (!modelOverride) {
      state.value.status = 'error'
      state.value.errorMessage = 'Select a checkpoint to generate.'
      return
    }

    const createdAtMs = Date.now()
    const promptPreview = String(p.prompt || '').trim().slice(0, 120)
    const summary = buildRunSummary(p)
    const paramsSnapshot = buildParamsSnapshot(p)

    const batchSize = Math.max(1, Math.trunc(Number(p.batchSize)))
    const batchCount = Math.max(1, Math.trunc(Number(p.batchCount)))
    const settingsRevision = quicksettings.getSettingsRevision()

    const tabType = String(engineType.value)
    const engineOverrideForRequest = resolveEngineForRequest(tabType, Boolean(p.useInitImage))

    const engineSurface = backendCaps.get(engineOverrideForRequest)
    if (!engineSurface) {
      const message = `Engine capabilities missing for '${engineOverrideForRequest}'.`
      console.error(`[useGeneration] ${message}`)
      state.value.status = 'error'
      state.value.errorMessage = message
      return
    }
    const familyCaps = backendCaps.getFamilyForEngine(engineOverrideForRequest)
    if (!familyCaps) {
      const message = `Family capabilities missing for '${engineOverrideForRequest}'.`
      console.error(`[useGeneration] ${message}`)
      state.value.status = 'error'
      state.value.errorMessage = message
      return
    }
    const supportsNegative = familyCaps.supports_negative_prompt
    const guidanceSupport = engineSurface.guidance_advanced ?? null

    const assetContract = backendCaps.getAssetContract(engineOverrideForRequest, { checkpointCoreOnly: modelIsCoreOnly })

    if (!p.useInitImage && !engineSurface.supports_txt2img) {
      const message = `This engine does not support txt2img (${engineOverrideForRequest}).`
      console.error(`[useGeneration] ${message}`)
      state.value.status = 'error'
      state.value.errorMessage = message
      return
    }

    if (p.useInitImage) {
      if (!engineSurface.supports_img2img) {
        const message = `This engine does not support img2img (${engineOverrideForRequest}).`
        console.error(`[useGeneration] ${message}`)
        state.value.status = 'error'
        state.value.errorMessage = message
        return
      }
      if (!p.initImageData) {
        state.value.status = 'error'
        state.value.errorMessage = 'Select an initial image for img2img.'
        return
      }
      if (p.useMask) {
        if (engineOverrideForRequest === 'flux1_kontext') {
          state.value.status = 'error'
          state.value.errorMessage = 'Masking is not supported for Flux.1 img2img (Kontext) yet.'
          return
        }
        if (!p.maskImageData) {
          state.value.status = 'error'
          state.value.errorMessage = 'Select a mask image for inpaint.'
          return
        }
      }
    }

    const textEncoders = Array.isArray((p as any).textEncoders)
      ? (p as any).textEncoders.map((it: unknown) => String(it || '').trim()).filter((it: string) => it.length > 0)
      : []
    const loraNames = [
      ...extractLoraNamesFromPrompt(p.prompt),
      ...(supportsNegative ? extractLoraNamesFromPrompt(p.negativePrompt) : []),
    ]

    // Build extras based on engine capabilities (e.g. tenc_sha)
    const extras: Record<string, unknown> = {}
    const guidancePayload = buildGuidancePayload(p.guidanceAdvanced, guidanceSupport)
    if (guidancePayload) {
      extras.guidance = guidancePayload
    }

    const needsTencSha = (assetContract?.tenc_count ?? 0) > 0
    if (needsTencSha) {
      const shas: string[] = []
      for (const label of textEncoders) {
        const sha = quicksettings.resolveTextEncoderSha(label)
        if (!sha) {
          state.value.status = 'error'
          state.value.errorMessage = `Text encoder SHA not found for '${label}'.`
          return
        }
        shas.push(sha)
      }
      if (needsTencSha && shas.length === 0) {
        state.value.status = 'error'
        state.value.errorMessage = 'Select a text encoder so the request can include tenc_sha.'
        return
      }
      const requiredCount = Math.trunc(Number(assetContract?.tenc_count ?? 0))
      if (requiredCount > 0 && shas.length !== requiredCount) {
        const label = String(assetContract?.tenc_kind_label || assetContract?.tenc_kind || '').trim()
        state.value.status = 'error'
        state.value.errorMessage = label
          ? `This engine requires exactly ${requiredCount} text encoder(s) (${label}).`
          : `This engine requires exactly ${requiredCount} text encoder(s).`
        return
      }
      if (shas.length > 0) {
        extras.tenc_sha = shas.length === 1 ? shas[0] : shas
      }
    }

    let selectedVae = ''
    try {
      selectedVae = quicksettings.requireVaeSelection()
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
      return
    }
    const resolvedVaeSha = quicksettings.resolveVaeSha(selectedVae)
    const needsVaeSha = Boolean(assetContract?.requires_vae)
    if (needsVaeSha) {
      if (!resolvedVaeSha) {
        state.value.status = 'error'
        state.value.errorMessage = 'Select a VAE so the request can include vae_sha.'
        return
      }
      extras.vae_sha = resolvedVaeSha
    } else if (resolvedVaeSha) {
      // Optional override: if user picked an explicit VAE, include its sha.
      extras.vae_sha = resolvedVaeSha
    }
    
    // Z-Image variant selection lives in request extras so the backend can pick flow_shift (shift=3.0 turbo / shift=6.0 base).
    const zimageTurbo = engineOverrideForRequest === 'zimage'
      ? Boolean((p as any)?.zimageTurbo ?? true)
      : false
    if (engineOverrideForRequest === 'zimage') {
      extras.zimage_variant = zimageTurbo ? 'turbo' : 'base'
    }

    if (loraNames.length > 0) {
      const loraShas: string[] = []
      const seenShas = new Set<string>()
      for (const loraName of loraNames) {
        const sha = quicksettings.resolveLoraSha(loraName)
        if (!sha) {
          state.value.status = 'error'
          state.value.errorMessage = `LoRA SHA not found for '${loraName}'. Refresh inventory and retry.`
          return
        }
        if (seenShas.has(sha)) continue
        seenShas.add(sha)
        loraShas.push(sha)
      }
      if (loraShas.length > 0) {
        extras.lora_sha = loraShas.length === 1 ? loraShas[0] : loraShas
      }
    }

    const device = (quicksettings.currentDevice || 'cpu') as any

    try {
      let taskId = ''
      if (p.useInitImage) {
        const payload = buildImg2ImgPayload({
          params: p,
          supportsNegativePrompt: supportsNegative,
          isDistilledCfgModel: Boolean(config.capabilities.usesDistilledCfg) && !config.capabilities.usesCfg,
          batchCount,
          batchSize,
          device,
          settingsRevision,
          engineId: engineOverrideForRequest,
          modelOverride,
          extras,
        })
        const { task_id } = await startImg2Img(payload)
        taskId = task_id
      } else {
        let payload: Txt2ImgRequest
        try {
          payload = buildTxt2ImgPayload({
            prompt: p.prompt,
            negativePrompt: supportsNegative ? p.negativePrompt : '',
            width: p.width,
            height: p.height,
            steps: p.steps,
            guidanceScale: p.cfgScale,
            sampler: p.sampler,
            scheduler: p.scheduler,
            seed: p.seed,
            clipSkip: p.clipSkip,
            batchSize,
            batchCount,
            styles: [],
            device,
            settingsRevision,
            engine: engineOverrideForRequest,
            model: modelOverride,
            hires: p.hires,
            refiner: p.refiner,
            extras,
          }, { hiresFallbackOnOom: Boolean(upscalersStore.fallbackOnOom), hiresMinTile: Number(upscalersStore.minTile) })
        } catch (error) {
          state.value.status = 'error'
          state.value.errorMessage = error instanceof Error ? error.message : String(error)
          return
        }
        const { task_id } = await startTxt2Img(payload)
        taskId = task_id
      }

      state.value.taskId = taskId
      state.value.currentRun = {
        taskId,
        mode: p.useInitImage ? 'img2img' : 'txt2img',
        createdAtMs,
        status: 'completed',
        summary,
        promptPreview,
        paramsSnapshot,
      }

      state.value.progress.stage = 'submitted'

      const key = resumeKey(tabId)
      saveResumeState(key, { taskId, lastEventId: 0, createdAtMs, paramsSnapshot })
      const unsub = subscribeTask(state.value.taskId, handleTaskEvent, undefined, {
        onMeta: ({ eventId }) => {
          if (typeof eventId === 'number') updateResumeEventId(key, eventId)
        },
      })
      unsubscribers.set(tabId, unsub)
    } catch (error) {
      const conflictRevision = resolveSettingsRevisionConflict(error)
      if (conflictRevision !== null) {
        try {
          await quicksettings.refreshSettingsRevision(conflictRevision)
        } catch {
          // Ignore refresh failures; fallback revision is already applied.
        }
        state.value.status = 'error'
        state.value.errorMessage = formatSettingsRevisionConflictMessage(quicksettings.getSettingsRevision())
        return
      }
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
    }
  }
  
  function handleTaskEvent(event: TaskEvent): void {
    switch (event.type) {
      case 'status':
        state.value.progress.stage = event.stage
        break
      case 'progress':
        state.value.progress = {
          stage: event.stage,
          percent: event.percent ?? null,
          etaSeconds: event.eta_seconds ?? null,
          step: event.step ?? null,
          totalSteps: event.total_steps ?? null,
        }
        if (event.preview_image) {
          state.value.previewImage = event.preview_image
          state.value.previewStep = event.preview_step ?? null
        }
        break
      case 'result':
        state.value.gallery = event.images || []
        state.value.info = event.info ?? null
        state.value.previewImage = null
        state.value.previewStep = null
        if (state.value.currentRun?.taskId) {
          state.value.currentRun.status = 'completed'
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        try {
          const infoObj = (typeof event.info === 'string' ? JSON.parse(event.info) : event.info) as any
          const rawSeed = infoObj?.seed ?? infoObj?.all_seeds?.[0]
          const resolvedSeed = typeof rawSeed === 'number' ? rawSeed : Number(rawSeed)
          if (Number.isFinite(resolvedSeed)) {
            state.value.lastSeed = resolvedSeed
          }
        } catch {
          // ignore seed parsing; keep lastSeed as-is
        }
        state.value.finishedAtMs = performance.now()
        state.value.status = 'done'
        break
      case 'gap':
        // History truncated while disconnected; refresh snapshot and keep streaming.
        if (state.value.taskId) void refreshTaskSnapshot(state.value.taskId)
        break
      case 'error':
        state.value.status = 'error'
        state.value.errorMessage = event.message
        state.value.finishedAtMs = performance.now()
        state.value.previewImage = null
        state.value.previewStep = null
        if (state.value.currentRun?.taskId) {
          state.value.currentRun.status = 'error'
          state.value.currentRun.errorMessage = event.message
          pushHistory(state.value.currentRun)
          state.value.selectedTaskId = state.value.currentRun.taskId
          state.value.currentRun = null
        }
        clearResumeState(resumeKey(tabId))
        stopStream()
        break
      case 'end':
        clearResumeState(resumeKey(tabId))
        if (state.value.status !== 'error') {
          state.value.status = 'done'
        }
        if (state.value.finishedAtMs === null) {
          state.value.finishedAtMs = performance.now()
        }
        state.value.previewImage = null
        state.value.previewStep = null
        stopStream()
        break
    }
  }

  async function refreshTaskSnapshot(taskId: string): Promise<void> {
    try {
      const res = await fetchTaskResult(taskId)
      if (res.status !== 'running') return
      if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
      const p = res.progress
      if (p && typeof p === 'object') {
        state.value.progress = {
          stage: String(p.stage ?? state.value.progress.stage),
          percent: p.percent ?? null,
          etaSeconds: p.eta_seconds ?? null,
          step: p.step ?? null,
          totalSteps: p.total_steps ?? null,
        }
      }
      if (res.preview_image) state.value.previewImage = res.preview_image
      if (res.preview_step !== undefined) state.value.previewStep = res.preview_step ?? null
    } catch {
      // ignore snapshot refresh failures
    }
  }

  async function tryAutoResume(): Promise<void> {
    if (resumeAttempts.has(tabId)) return
    resumeAttempts.add(tabId)

    const key = resumeKey(tabId)
    const saved = loadResumeState(key)
    if (!saved) return

    let res
    try {
      res = await fetchTaskResult(saved.taskId)
    } catch {
      clearResumeState(key)
      return
    }

    if (res.status === 'running') {
      state.value.status = 'running'
      state.value.taskId = saved.taskId
      state.value.errorMessage = ''
      state.value.finishedAtMs = null
      if (typeof res.stage === 'string' && res.stage.trim()) state.value.progress.stage = res.stage
      const p = res.progress
      if (p && typeof p === 'object') {
        state.value.progress = {
          stage: String(p.stage ?? state.value.progress.stage),
          percent: p.percent ?? null,
          etaSeconds: p.eta_seconds ?? null,
          step: p.step ?? null,
          totalSteps: p.total_steps ?? null,
        }
      }
      if (res.preview_image) state.value.previewImage = res.preview_image
      if (res.preview_step !== undefined) state.value.previewStep = res.preview_step ?? null
      const unsub = subscribeTask(saved.taskId, handleTaskEvent, undefined, {
        after: saved.lastEventId,
        onMeta: ({ eventId }) => {
          if (typeof eventId === 'number') updateResumeEventId(key, eventId)
        },
      })
      unsubscribers.set(tabId, unsub)
      resumeNotice.value = 'Reconnected (resumed task).'
      return
    }

    // Task is terminal; hydrate UI and clear resume marker.
    clearResumeState(key)
    if (res.status === 'completed' && res.result) {
      state.value.gallery = res.result.images || []
      state.value.info = res.result.info ?? null
      state.value.status = 'done'
      state.value.previewImage = null
      state.value.previewStep = null
      state.value.finishedAtMs = performance.now()
      return
    }
    if (res.status === 'error') {
      state.value.status = 'error'
      state.value.errorMessage = String(res.error || 'Task failed.')
      state.value.finishedAtMs = performance.now()
      state.value.previewImage = null
      state.value.previewStep = null
    }
  }

  // Attempt to resume an in-flight task after a browser reload/crash.
  void tryAutoResume()

  async function loadHistory(taskId: string): Promise<void> {
    if (!taskId || state.value.status === 'running') return
    stopStream()
    state.value.historyLoadingTaskId = taskId
    try {
      const result = await fetchTaskResult(taskId)
      if (result.status === 'error') {
        state.value.status = 'error'
        state.value.errorMessage = result.error || 'Task failed.'
        return
      }
      if (result.status === 'completed' && result.result) {
        state.value.gallery = result.result.images || []
        state.value.info = result.result.info ?? null
        state.value.status = 'done'
        state.value.selectedTaskId = taskId
        return
      }
      state.value.status = 'error'
      state.value.errorMessage = 'Task is still running.'
    } catch (err) {
      state.value.status = 'error'
      state.value.errorMessage = err instanceof Error ? err.message : String(err)
    } finally {
      state.value.historyLoadingTaskId = ''
    }
  }

  function clearHistory(): void {
    state.value.history = []
    state.value.selectedTaskId = ''
  }
  
  // Expose reactive state and methods
  return {
    // State
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    previewImage: computed(() => state.value.previewImage),
    previewStep: computed(() => state.value.previewStep),
    gallery: computed(() => state.value.gallery),
    info: computed(() => state.value.info),
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    lastSeed: computed(() => state.value.lastSeed),
    history: computed(() => state.value.history),
    selectedTaskId: computed(() => state.value.selectedTaskId),
    historyLoadingTaskId: computed(() => state.value.historyLoadingTaskId),
    gentimeMs: computed(() => {
      if (state.value.startedAtMs === null || state.value.finishedAtMs === null) return null
      return Math.max(0, state.value.finishedAtMs - state.value.startedAtMs)
    }),
    isRunning: computed(() => state.value.status === 'running'),
    
    // Tab info
    tab,
    params,
    engineType,
    engineConfig,
    
    // Actions
    generate,
    stopStream,
    loadHistory,
    clearHistory,
    resumeNotice,
  }
}
