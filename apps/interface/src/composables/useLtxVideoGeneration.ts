/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dedicated LTX video generation composable for the generic backend video contract.
Owns per-tab runtime state for `ltx2` txt2vid/img2vid runs, validates LTX-specific frontend assumptions against the backend capability/asset
contracts, builds generic payloads, starts `/api/txt2vid` or `/api/img2vid` tasks, and consumes task SSE events to surface progress/results.
Unlike the WAN composable, this lane has no queue/history/stage orchestration; it stays fail-loud on unsupported generic-video assumptions.

Symbols (top-level; keep in sync; no ghosts):
- `Status` (type): Runtime status union (`idle|running|error|done`).
- `LtxProgressState` (interface): Current task progress snapshot.
- `LtxGenerationState` (interface): Per-tab LTX runtime state.
- `ResumeState` (type): Persisted auto-resume marker for an in-flight LTX task.
- `freshState` (function): Creates empty per-tab state.
- `getTabState` (function): Returns/initializes per-tab state storage.
- `useLtxVideoGeneration` (function): Main LTX video composable.
*/

import { computed, ref } from 'vue'

import { cancelTask, fetchTaskResult, startImg2Vid, startTxt2Vid, subscribeTask } from '../api/client'
import { formatZodError } from '../api/payloads'
import {
  buildLtxImg2VidPayload,
  buildLtxTxt2VidPayload,
  normalizeDevice,
  normalizeLtxSampler,
  normalizeLtxScheduler,
  type LtxImg2VidPayload,
  type LtxTxt2VidPayload,
  type LtxVideoCommonInput,
} from '../api/payloads_ltx_video'
import type { GeneratedImage, TaskEvent } from '../api/types'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import { useModelTabsStore, type LtxGenerationMode, type LtxTabParams, type TabByType } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { formatSettingsRevisionConflictMessage, resolveSettingsRevisionConflict } from './settings_revision_conflict'

type Status = 'idle' | 'running' | 'error' | 'done'

type ResumeState = {
  taskId: string
  lastEventId: number
  mode: LtxGenerationMode
}

type PreparedRun =
  | { mode: 'txt2vid'; payload: LtxTxt2VidPayload }
  | { mode: 'img2vid'; payload: LtxImg2VidPayload }

export interface LtxProgressState {
  stage: string
  percent: number | null
  etaSeconds: number | null
  step: number | null
  totalSteps: number | null
  message: string | null
  totalPercent: number | null
  totalPhase: string | null
  totalPhaseStep: number | null
  totalPhaseTotalSteps: number | null
}

export interface LtxGenerationState {
  status: Status
  progress: LtxProgressState
  frames: GeneratedImage[]
  info: unknown | null
  video: { rel_path?: string | null; mime?: string | null } | null
  errorMessage: string
  taskId: string
  cancelRequested: boolean
}

const DEFAULT_PROGRESS: LtxProgressState = {
  stage: 'idle',
  percent: null,
  etaSeconds: null,
  step: null,
  totalSteps: null,
  message: null,
  totalPercent: null,
  totalPhase: null,
  totalPhaseStep: null,
  totalPhaseTotalSteps: null,
}
const tabStates = new Map<string, LtxGenerationState>()
const unsubscribers = new Map<string, () => void>()
const resumeAttempts = new Set<string>()
const resumeToastShown = new Set<string>()

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function dependencyCheckMessage(
  dependency: { checks?: Array<{ id?: string; ok?: boolean; message?: string | null }> } | null | undefined,
  id: string,
): string {
  const row = Array.isArray(dependency?.checks)
    ? dependency.checks.find((entry) => String(entry?.id || '').trim() === id)
    : undefined
  if (!row) return ''
  if (row.ok) return ''
  return String(row.message || '').trim() || `Dependency check '${id}' is not ready.`
}

function freshState(): LtxGenerationState {
  return {
    status: 'idle',
    progress: { ...DEFAULT_PROGRESS },
    frames: [],
    info: null,
    video: null,
    errorMessage: '',
    taskId: '',
    cancelRequested: false,
  }
}

function getTabState(tabId: string): LtxGenerationState {
  if (!tabStates.has(tabId)) tabStates.set(tabId, freshState())
  return tabStates.get(tabId)!
}

function resumeKey(tabId: string): string {
  return `codex.resume.ltx2.${tabId}`
}

function parseResumeMode(value: unknown): LtxGenerationMode | null {
  const normalized = String(value || '').trim().toLowerCase()
  if (normalized === 'txt2vid' || normalized === 'img2vid') return normalized
  return null
}

function loadResumeState(key: string): ResumeState | null {
  try {
    const raw = localStorage.getItem(key)
    if (!raw) return null
    const parsed: unknown = JSON.parse(raw)
    if (!isRecordObject(parsed)) return null
    const taskId = typeof parsed.taskId === 'string' ? parsed.taskId.trim() : ''
    if (!taskId) return null
    const lastEventId = typeof parsed.lastEventId === 'number' && Number.isFinite(parsed.lastEventId)
      ? Math.max(0, Math.trunc(parsed.lastEventId))
      : 0
    const mode = parseResumeMode(parsed.mode)
    if (!mode) return null
    return { taskId, lastEventId, mode }
  } catch {
    return null
  }
}

function saveResumeState(key: string, state: ResumeState): void {
  try {
    localStorage.setItem(key, JSON.stringify(state))
  } catch {
    // ignore localStorage failures
  }
}

function clearResumeState(key: string): void {
  try {
    localStorage.removeItem(key)
  } catch {
    // ignore localStorage failures
  }
}

function updateResumeEventId(key: string, eventId: number): void {
  const value = Math.trunc(Number(eventId))
  if (!Number.isFinite(value) || value <= 0) return
  const current = loadResumeState(key)
  if (!current || value <= current.lastEventId) return
  saveResumeState(key, { ...current, lastEventId: value })
}

function outputUrl(relPath: string): string {
  const clean = String(relPath || '').replace(/\\+/g, '/').replace(/^\/+/, '')
  const encoded = clean.split('/').map((part) => encodeURIComponent(part)).join('/')
  return `/api/output/${encoded}`
}

export function useLtxVideoGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()
  const engineCaps = useEngineCapabilitiesStore()

  const state = ref<LtxGenerationState>(getTabState(tabId))
  const resumeNotice = ref('')

  const tab = computed<TabByType<'ltx2'> | null>(() => {
    const candidate = modelTabs.tabs.find((entry) => entry.id === tabId) || null
    if (!candidate || candidate.type !== 'ltx2') return null
    return candidate as unknown as TabByType<'ltx2'>
  })
  const params = computed<LtxTabParams | null>(() => tab.value?.params || null)
  const mode = computed<LtxGenerationMode>(() => {
    const explicit = String(params.value?.mode || '').trim().toLowerCase()
    if (explicit === 'img2vid' || explicit === 'txt2vid') return explicit
    return params.value?.useInitImage ? 'img2vid' : 'txt2vid'
  })
  const engineSurface = computed(() => engineCaps.get('ltx2'))
  const dependencyStatus = computed(() => engineCaps.getDependencyStatus('ltx2'))
  const dependencyError = computed(() => engineCaps.firstDependencyError('ltx2'))
  const checkpoint = computed(() => String(params.value?.checkpoint || '').trim())
  const checkpointCoreOnly = computed(() => Boolean(checkpoint.value) && quicksettings.isModelCoreOnly(checkpoint.value))
  const assetContract = computed(() => engineCaps.getAssetContract('ltx2', { checkpointCoreOnly: checkpointCoreOnly.value }))

  void engineCaps.init().catch(() => {
    // Leave the UI fail-loud via blockedReason/generate() instead of throwing from setup.
  })

  function stopStream(): void {
    const unsub = unsubscribers.get(tabId)
    if (!unsub) return
    unsub()
    unsubscribers.delete(tabId)
  }

  function resetProgress(): void {
    state.value.progress = { ...DEFAULT_PROGRESS }
  }

  function clearResultState(): void {
    state.value.frames = []
    state.value.info = null
    state.value.video = null
  }

  function setError(message: string): void {
    state.value.status = 'error'
    state.value.errorMessage = message
  }

  function setErrorMessage(message: string): void {
    state.value.errorMessage = message
  }

  function isVaeSentinel(value: string): boolean {
    const normalized = String(value || '').trim().toLowerCase()
    return normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in' || normalized === 'none'
  }

  function blockedReasonFor(currentParams: LtxTabParams | null): string {
    if (!currentParams) return 'LTX tab params are not available.'

    const dependency = dependencyStatus.value
    if (!dependency) return "Dependency checks for 'ltx2' are not available."
    const checkpointInventoryMessage = dependencyCheckMessage(dependency, 'checkpoint_inventory')
    if (checkpointInventoryMessage) return checkpointInventoryMessage
    const vendoredMetadataMessage = dependencyCheckMessage(dependency, 'vendored_metadata')
    if (vendoredMetadataMessage) return vendoredMetadataMessage
    if (checkpointCoreOnly.value) {
      const connectorsMessage = dependencyCheckMessage(dependency, 'connectors_inventory')
      if (connectorsMessage) return connectorsMessage
      const audioBundleMessage = dependencyCheckMessage(dependency, 'audio_bundle_inventory')
      if (audioBundleMessage) return audioBundleMessage
    }

    const surface = engineSurface.value
    if (!surface) return "Capabilities for 'ltx2' are not loaded."
    if (mode.value === 'txt2vid' && !surface.supports_txt2vid) return 'LTX does not currently expose txt2vid on the generic backend contract.'
    if (mode.value === 'img2vid' && !surface.supports_img2vid) return 'LTX does not currently expose img2vid on the generic backend contract.'

    const prompt = String(currentParams.prompt || '').trim()
    if (!prompt) return 'Prompt must not be empty.'

    const checkpointLabel = String(currentParams.checkpoint || '').trim()
    if (!checkpointLabel) return 'Select an LTX checkpoint in QuickSettings.'

    try {
      normalizeDevice(quicksettings.currentDevice || 'cpu')
    } catch (error) {
      return error instanceof Error ? error.message : String(error)
    }

    try {
      normalizeLtxSampler(currentParams.sampler)
    } catch (error) {
      return error instanceof Error ? error.message : String(error)
    }
    try {
      normalizeLtxScheduler(currentParams.scheduler)
    } catch (error) {
      return error instanceof Error ? error.message : String(error)
    }

    const contract = assetContract.value
    if (!contract) return "Asset contract for 'ltx2' is not available."

    const requiredTencCount = Math.max(0, Math.trunc(Number(contract.tenc_count ?? 0)))
    if (requiredTencCount !== 1) {
      return `Unsupported LTX asset contract: expected exactly 1 text encoder, got ${requiredTencCount}.`
    }

    const textEncoderLabel = String(currentParams.textEncoder || '').trim()
    if (!textEncoderLabel) return 'Select the LTX text encoder in QuickSettings.'
    if (!quicksettings.resolveTextEncoderSha(textEncoderLabel)) {
      return `Text encoder SHA not found for '${textEncoderLabel}'. Refresh inventory and re-select it.`
    }

    const vaeLabel = String(currentParams.vae || '').trim()
    const explicitExternalVae = Boolean(vaeLabel) && !isVaeSentinel(vaeLabel)
    const resolvedVaeSha = explicitExternalVae ? quicksettings.resolveVaeSha(vaeLabel) : undefined
    if (contract.requires_vae) {
      if (!explicitExternalVae) return 'This LTX checkpoint requires an external VAE selected in QuickSettings.'
      if (!resolvedVaeSha) return `VAE SHA not found for '${vaeLabel}'. Refresh inventory and re-select it.`
    } else if (explicitExternalVae && !resolvedVaeSha) {
      return `VAE SHA not found for '${vaeLabel}'. Refresh inventory and re-select it.`
    }

    if (mode.value === 'img2vid' && !String(currentParams.initImageData || '').trim()) {
      return 'Image mode requires an initial image; select a file or switch to Text mode.'
    }

    return ''
  }

  const blockedReason = computed(() => blockedReasonFor(params.value))
  const canGenerate = computed(() => blockedReason.value.length === 0)

  function buildCommonInput(currentParams: LtxTabParams): LtxVideoCommonInput {
    const checkpointLabel = String(currentParams.checkpoint || '').trim()
    if (!checkpointLabel) throw new Error('Select an LTX checkpoint in QuickSettings.')
    const modelSha = quicksettings.resolveModelSha(checkpointLabel)

    const contract = assetContract.value
    if (!contract) throw new Error("Asset contract for 'ltx2' is not available.")

    const requiredTencCount = Math.max(0, Math.trunc(Number(contract.tenc_count ?? 0)))
    if (requiredTencCount !== 1) {
      throw new Error(`Unsupported LTX asset contract: expected exactly 1 text encoder, got ${requiredTencCount}.`)
    }

    const textEncoderLabel = String(currentParams.textEncoder || '').trim()
    const textEncoderSha = quicksettings.resolveTextEncoderSha(textEncoderLabel)
    if (!textEncoderSha) {
      throw new Error(`Text encoder SHA not found for '${textEncoderLabel || '<empty>'}'.`)
    }

    const vaeLabel = String(currentParams.vae || '').trim()
    const explicitExternalVae = Boolean(vaeLabel) && !isVaeSentinel(vaeLabel)
    const resolvedVaeSha = explicitExternalVae ? quicksettings.resolveVaeSha(vaeLabel) : undefined
    if (contract.requires_vae && !resolvedVaeSha) {
      throw new Error('This LTX checkpoint requires an external VAE selected in QuickSettings.')
    }
    if (explicitExternalVae && !resolvedVaeSha) {
      throw new Error(`VAE SHA not found for '${vaeLabel}'.`)
    }

    return {
      device: normalizeDevice(quicksettings.currentDevice || 'cpu'),
      settingsRevision: quicksettings.getSettingsRevision(),
      model: checkpointLabel,
      modelSha,
      prompt: currentParams.prompt,
      negativePrompt: currentParams.negativePrompt,
      width: currentParams.width,
      height: currentParams.height,
      fps: currentParams.fps,
      frames: currentParams.frames,
      steps: currentParams.steps,
      cfgScale: currentParams.cfgScale,
      sampler: currentParams.sampler,
      scheduler: currentParams.scheduler,
      seed: currentParams.seed,
      textEncoderSha,
      vaeSha: resolvedVaeSha,
      videoReturnFrames: Boolean(currentParams.videoReturnFrames),
    }
  }

  function prepareRun(currentParams: LtxTabParams): PreparedRun {
    const common = buildCommonInput(currentParams)
    if (mode.value === 'img2vid') {
      return {
        mode: 'img2vid',
        payload: buildLtxImg2VidPayload({
          ...common,
          initImageData: String(currentParams.initImageData || '').trim(),
        }),
      }
    }
    return {
      mode: 'txt2vid',
      payload: buildLtxTxt2VidPayload(common),
    }
  }

  function onTaskEvent(event: TaskEvent): void {
    const key = resumeKey(tabId)
    switch (event.type) {
      case 'status':
        state.value.progress.stage = event.stage
        break
      case 'progress':
        const eventProgress: LtxProgressState = {
          stage: event.stage,
          percent: event.percent ?? null,
          etaSeconds: event.eta_seconds ?? null,
          step: event.step ?? null,
          totalSteps: event.total_steps ?? null,
          message: event.message ?? null,
          totalPercent: null,
          totalPhase: null,
          totalPhaseStep: null,
          totalPhaseTotalSteps: null,
        }
        state.value.progress = eventProgress
        break
      case 'gap':
        if (state.value.taskId) void refreshTaskSnapshot(state.value.taskId)
        break
      case 'result':
        state.value.frames = Array.isArray(event.images) ? event.images : []
        state.value.info = event.info ?? null
        state.value.video = event.video ?? null
        state.value.status = 'done'
        break
      case 'error':
        clearResumeState(key)
        setError(String(event.message || 'Task failed.'))
        break
      case 'end':
        clearResumeState(key)
        if (state.value.status !== 'error') state.value.status = 'done'
        stopStream()
        break
    }
  }

  async function startPreparedRun(run: PreparedRun): Promise<void> {
    stopStream()
    resumeNotice.value = ''
    state.value.status = 'running'
    state.value.errorMessage = ''
    clearResultState()
    resetProgress()
    state.value.progress.stage = 'starting'
    state.value.taskId = ''
    state.value.cancelRequested = false

    try {
      const response = await (run.mode === 'img2vid' ? startImg2Vid(run.payload) : startTxt2Vid(run.payload))
      const taskId = String(response.task_id || '').trim()
      if (!taskId) throw new Error('Backend returned an empty task id for LTX video generation.')
      state.value.taskId = taskId

      const key = resumeKey(tabId)
      saveResumeState(key, { taskId, lastEventId: 0, mode: run.mode })
      const unsub = subscribeTask(taskId, onTaskEvent, undefined, {
        onMeta: ({ eventId }) => {
          if (typeof eventId === 'number') updateResumeEventId(key, eventId)
        },
      })
      unsubscribers.set(tabId, unsub)
    } catch (error) {
      clearResumeState(resumeKey(tabId))
      const conflictRevision = resolveSettingsRevisionConflict(error)
      if (conflictRevision !== null) {
        try {
          await quicksettings.refreshSettingsRevision(conflictRevision)
        } catch {
          // Ignore refresh failures; fallback revision is already applied.
        }
        setError(formatSettingsRevisionConflictMessage(quicksettings.getSettingsRevision()))
        return
      }
      setError(formatZodError(error))
    }
  }

  async function generate(): Promise<void> {
    if (!tab.value) {
      setError(`useLtxVideoGeneration: tab '${tabId}' not found or not available.`)
      return
    }
    if (tab.value.type !== 'ltx2') {
      setError(`useLtxVideoGeneration: unsupported tab type '${String(tab.value.type)}'.`)
      return
    }

    try {
      await engineCaps.init()
    } catch {
      // The fail-loud message comes from blockedReason below.
    }

    const reason = blockedReasonFor(tab.value.params)
    if (reason) {
      setError(reason)
      return
    }

    try {
      await startPreparedRun(prepareRun(tab.value.params))
    } catch (error) {
      setError(formatZodError(error))
    }
  }

  async function refreshTaskSnapshot(taskId: string): Promise<void> {
    try {
      const result = await fetchTaskResult(taskId)
      if (result.status === 'running') {
        if (typeof result.stage === 'string' && result.stage.trim()) state.value.progress.stage = result.stage
        const progress = result.progress
        if (progress && typeof progress === 'object') {
          const snapshotProgress: LtxProgressState = {
            stage: String(progress.stage ?? state.value.progress.stage),
            percent: progress.percent ?? null,
            etaSeconds: progress.eta_seconds ?? null,
            step: progress.step ?? null,
            totalSteps: progress.total_steps ?? null,
            message: progress.message ?? null,
            totalPercent: null,
            totalPhase: null,
            totalPhaseStep: null,
            totalPhaseTotalSteps: null,
          }
          state.value.progress = snapshotProgress
        }
        return
      }

      if (result.status === 'completed' && result.result) {
        state.value.frames = Array.isArray(result.result.images) ? result.result.images : []
        state.value.info = result.result.info ?? null
        state.value.video = result.result.video ?? null
        state.value.status = 'done'
        return
      }

      if (result.status === 'error') {
        setError(String(result.error || 'Task failed.'))
      }
    } catch {
      // Ignore snapshot refresh failures.
    }
  }

  async function tryAutoResume(): Promise<void> {
    if (resumeAttempts.has(tabId)) return
    resumeAttempts.add(tabId)

    const key = resumeKey(tabId)
    const saved = loadResumeState(key)
    if (!saved) return

    let result
    try {
      result = await fetchTaskResult(saved.taskId)
    } catch {
      clearResumeState(key)
      return
    }

    if (result.status === 'running') {
      stopStream()
      state.value.status = 'running'
      state.value.taskId = saved.taskId
      state.value.errorMessage = ''
      state.value.cancelRequested = false
      clearResultState()
      resetProgress()

      if (typeof result.stage === 'string' && result.stage.trim()) state.value.progress.stage = result.stage
      const progress = result.progress
      if (progress && typeof progress === 'object') {
        const resumedProgress: LtxProgressState = {
          stage: String(progress.stage ?? state.value.progress.stage),
          percent: progress.percent ?? null,
          etaSeconds: progress.eta_seconds ?? null,
          step: progress.step ?? null,
          totalSteps: progress.total_steps ?? null,
          message: progress.message ?? null,
          totalPercent: null,
          totalPhase: null,
          totalPhaseStep: null,
          totalPhaseTotalSteps: null,
        }
        state.value.progress = resumedProgress
      }

      const unsub = subscribeTask(saved.taskId, onTaskEvent, undefined, {
        after: saved.lastEventId,
        onMeta: ({ eventId }) => {
          if (typeof eventId === 'number') updateResumeEventId(key, eventId)
        },
      })
      unsubscribers.set(tabId, unsub)

      if (!resumeToastShown.has(saved.taskId)) {
        resumeNotice.value = 'Reconnected (resumed task).'
        resumeToastShown.add(saved.taskId)
      }
      return
    }

    clearResumeState(key)

    if (result.status === 'completed' && result.result) {
      state.value.frames = Array.isArray(result.result.images) ? result.result.images : []
      state.value.info = result.result.info ?? null
      state.value.video = result.result.video ?? null
      state.value.status = 'done'
      state.value.taskId = saved.taskId
      return
    }

    if (result.status === 'error') {
      state.value.taskId = saved.taskId
      setError(String(result.error || 'Task failed.'))
    }
  }

  void tryAutoResume()

  async function cancel(modeValue: 'immediate' | 'after_current' = 'immediate'): Promise<void> {
    const taskId = state.value.taskId
    if (!taskId || state.value.status !== 'running') return
    state.value.cancelRequested = true
    try {
      await cancelTask(taskId, modeValue)
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : String(error))
    }
  }

  const videoExport = computed(() => state.value.video)
  const videoUrl = computed(() => {
    const relPath = state.value.video?.rel_path
    if (!relPath) return ''
    return outputUrl(relPath)
  })

  return {
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    frames: computed(() => state.value.frames),
    info: computed(() => state.value.info),
    videoExport,
    videoUrl,
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    isRunning: computed(() => state.value.status === 'running'),
    cancelRequested: computed(() => state.value.cancelRequested),
    tab,
    params,
    mode,
    checkpoint,
    checkpointCoreOnly,
    engineSurface,
    dependencyStatus,
    dependencyError,
    assetContract,
    blockedReason,
    canGenerate,
    generate,
    stopStream,
    cancel,
    resumeNotice,
  }
}
