<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Renderless LTX runtime helper for the canonical video tab view.
Mounts the existing LTX composable/watcher runtime under a view-local seam and exposes reactive slot props to `VideoModelTab.vue`,
while keeping the route-owned video view as the only live body/layout owner.

Symbols (top-level; keep in sync; no ghosts):
- `VideoModelTabLtxRuntime` (component): Renderless LTX runtime helper for `VideoModelTab.vue`.
- `ExecutionProfileOption` (type): Workspace-facing execution-profile selector row.
- `normalizeExecutionProfileName` (function): Normalizes raw execution-profile names for selector/display checks.
- `executionProfileLabel` (function): Formats a user-facing label for a known or stale execution profile.
- `ensureExecutionProfileVisible` (function): Preserves stale persisted execution-profile values in the local selector option list.
- `slotProps` (const): Reactive slot-prop bundle exposed to `VideoModelTab.vue`.
-->

<template>
  <slot v-bind="slotProps" />
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'

import type { GeneratedImage } from '../../api/types'
import {
  LTX_DIM_ALIGNMENT,
  LTX_DIM_MAX,
  LTX_DIM_MIN,
  LTX_FRAME_ALIGNMENT,
  LTX_FRAMES_MAX,
  LTX_FRAMES_MIN,
  LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT,
  resolveLtxDimAlignmentForExecutionProfile,
} from '../../api/payloads_ltx_video'
import { useLtxVideoGeneration } from '../../composables/useLtxVideoGeneration'
import { useResultsCard } from '../../composables/useResultsCard'
import { useModelTabsStore, type LtxTabParams } from '../../stores/model_tabs'

const props = defineProps<{ tabId: string }>()

const store = useModelTabsStore()

const {
  status,
  progress,
  frames,
  info,
  videoUrl,
  errorMessage,
  isRunning,
  tab,
  params,
  mode,
  checkpointCoreOnly,
  ltxExecutionSurface,
  checkpointExecutionMetadata,
  assetContract,
  blockedReason,
  generate,
  cancel,
  resumeNotice,
} = useLtxVideoGeneration(props.tabId)

const { notice: copyNotice, copyJson, formatJson, toast } = useResultsCard()

type ExecutionProfileOption = {
  value: string
  label: string
  supported: boolean
}

function normalizeExecutionProfileName(rawValue: string): string {
  return String(rawValue || '').trim().toLowerCase()
}

function executionProfileLabel(value: string): string {
  const normalized = normalizeExecutionProfileName(value)
  if (normalized === 'one_stage') return 'One-stage'
  if (normalized === 'two_stage') return 'Two-stage'
  if (normalized === 'distilled') return 'Distilled'
  return value || 'Unknown'
}

function ensureExecutionProfileVisible(options: ExecutionProfileOption[], currentValue: string): ExecutionProfileOption[] {
  const current = String(currentValue || '').trim()
  if (!current) return options
  if (options.some((entry) => entry.value === current)) return options
  const normalizedCurrent = normalizeExecutionProfileName(current)
  const canonicalMatch = options.find((entry) => normalizeExecutionProfileName(entry.value) === normalizedCurrent)
  return [{
    value: current,
    label: canonicalMatch
      ? `${executionProfileLabel(current)} (stored raw value; reselect the canonical profile)`
      : `${executionProfileLabel(current)} (unsupported; reselect a supported profile)`,
    supported: false,
  }, ...options]
}

const executionProfileOptions = computed<ExecutionProfileOption[]>(() => {
  const checkpointExecution = checkpointExecutionMetadata.value
  const checkpointAllowed = checkpointExecution?.allowedExecutionProfiles ?? []
  const surfaceAllowed = ltxExecutionSurface.value?.allowed_execution_profiles ?? []
  const allowed = checkpointExecution ? checkpointAllowed : surfaceAllowed
  const base = allowed.map((value: string) => ({
    value,
    label: executionProfileLabel(value),
    supported: true,
  }))
  return ensureExecutionProfileVisible(base, String(params.value?.executionProfile || ''))
})

const selectedExecutionProfile = computed(() => String(params.value?.executionProfile || '').trim())
const dimensionAlignment = computed(() => {
  if (selectedExecutionProfile.value !== 'two_stage') return LTX_DIM_ALIGNMENT
  return resolveLtxDimAlignmentForExecutionProfile(selectedExecutionProfile.value)
})
const executionProfileCaption = computed(() => {
  const profile = selectedExecutionProfile.value
  if (!profile) {
    return 'Execution profiles choose the truthful LTX lane. Sampler and scheduler stay derived from the selected profile; they are not raw sampler controls.'
  }
  if (profile === 'one_stage') {
    return 'One-stage runs a single-stage lane to final output resolution. Sampler and scheduler stay derived from this profile.'
  }
  if (profile === 'two_stage') {
    return 'Two-stage runs stage 1 at half final resolution, then a fixed x2 latent upscale plus a fixed stage-2 distilled refine. Steps and CFG control only stage 1; this is not a raw sampler lane.'
  }
  if (profile === 'distilled') {
    return 'Distilled runs the fixed distilled one-stage lane. Sampler and scheduler stay derived from this profile.'
  }
  return `Stored profile '${profile}' is not a supported LTX execution-profile id on this frontend slice. Re-select a supported checkpoint/profile pair.`
})
const dimensionRuleCaption = computed(() => {
  if (selectedExecutionProfile.value === 'two_stage') {
    return `Width and height stay the final output dimensions. two_stage runs stage 1 at half resolution, so both final dimensions must be divisible by ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}.`
  }
  return `Width and height stay the final output dimensions. one_stage and distilled require multiples of ${LTX_DIM_ALIGNMENT}.`
})
const dimensionWarning = computed(() => {
  const current = params.value
  if (!current || selectedExecutionProfile.value !== 'two_stage') return ''
  if (
    current.width % LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT === 0
    && current.height % LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT === 0
  ) {
    return ''
  }
  return `two_stage requires final width and height divisible by ${LTX_TWO_STAGE_FINAL_DIM_ALIGNMENT}. Current size ${current.width}×${current.height} is blocking.`
})
const executionProfileWarning = computed(() => {
  const currentProfile = selectedExecutionProfile.value
  const currentOption = executionProfileOptions.value.find((entry) => entry.value === currentProfile)
  if (currentProfile && currentOption && !currentOption.supported) {
    const normalized = normalizeExecutionProfileName(currentProfile)
    const canonicalMatch = executionProfileOptions.value.find(
      (entry) => entry.supported && normalizeExecutionProfileName(entry.value) === normalized,
    )
    if (canonicalMatch) {
      return `Stored raw profile '${currentProfile}' is blocking because the canonical supported value is '${canonicalMatch.value}'. Re-select the supported profile instead of relying on silent remapping.`
    }
    return `Stored profile '${currentProfile}' is unsupported for the selected checkpoint. Re-select a supported profile.`
  }
  const message = String(blockedReason.value || '')
  if (
    message.includes('execution profile')
    || message.includes('checkpoint metadata')
    || message.includes('not executable')
  ) {
    return message
  }
  return ''
})

const checkpointDisplay = computed(() => String(params.value?.checkpoint || '').trim() || 'Not selected')
const vaeDisplay = computed(() => String(params.value?.vae || '').trim() || (checkpointCoreOnly.value ? 'Not selected' : 'Built-in / omitted'))
const textEncoderDisplay = computed(() => String(params.value?.textEncoder || '').trim() || 'Not selected')
const hideNegativePrompt = computed(() => Number(params.value?.cfgScale) === 1)
const promptModeLabel = computed(() => (mode.value === 'img2vid' ? 'IMG2VID' : 'TXT2VID'))
const runGenerateDisabled = computed(() => isRunning.value || Boolean(blockedReason.value))
const runGenerateTitle = computed(() => (isRunning.value ? '' : blockedReason.value))
const runSummary = computed(() => {
  const current = params.value
  if (!current) return ''
  const profile = String(current.executionProfile || '').trim()
  const profileLabel = profile ? executionProfileLabel(profile) : 'Profile unresolved'
  return `${current.width}×${current.height} · ${current.frames}f @ ${current.fps}fps · ${profileLabel} · steps ${current.steps} · cfg ${current.cfgScale}`
})
const successMessage = computed(() => {
  const parts: string[] = []
  if (videoUrl.value) parts.push('Video ready')
  if (frames.value.length > 0) parts.push(`${frames.value.length} frame${frames.value.length === 1 ? '' : 's'} returned`)
  return parts.join(' · ') || 'Task finished.'
})

watch(
  () => {
    const metadata = checkpointExecutionMetadata.value
    return {
      checkpoint: String(params.value?.checkpoint || '').trim(),
      checkpointKind: String(metadata?.checkpointKind || '').trim(),
      defaultProfile: String(metadata?.defaultExecutionProfile || '').trim(),
      defaultStepsKey: typeof metadata?.defaultSteps === 'number' ? String(metadata.defaultSteps) : '',
      defaultGuidanceKey: typeof metadata?.defaultGuidanceScale === 'number' ? String(metadata.defaultGuidanceScale) : '',
      allowedProfilesKey: (metadata?.allowedExecutionProfiles ?? []).join('|'),
    }
  },
  (nextState, previousState) => {
    const current = params.value
    const isInitialRun = previousState === undefined

    const metadata = checkpointExecutionMetadata.value
    if (!current || !metadata) return
    if (metadata.checkpointKind === 'unknown') return
    const defaultProfile = String(metadata.defaultExecutionProfile || '').trim()
    if (!defaultProfile) return
    const metadataArrived = !String(previousState?.checkpointKind || '').trim() && Boolean(nextState.checkpointKind)
    const defaultsReady = nextState.defaultProfile !== '' && nextState.defaultStepsKey !== '' && nextState.defaultGuidanceKey !== ''
    const previousDefaultsReady =
      String(previousState?.defaultProfile || '').trim() !== ''
      && String(previousState?.defaultStepsKey || '').trim() !== ''
      && String(previousState?.defaultGuidanceKey || '').trim() !== ''
    const defaultsCompleted = defaultsReady && !previousDefaultsReady
    const previousCheckpoint = String(previousState?.checkpoint || '').trim()
    const checkpointChanged = previousCheckpoint !== nextState.checkpoint
    const currentProfile = String(current.executionProfile || '').trim()
    if (currentProfile) return
    const shouldApplyDefaults = isInitialRun || metadataArrived || checkpointChanged || defaultsCompleted
    if (!shouldApplyDefaults) return

    const patch: Partial<LtxTabParams> = {}
    if (current.executionProfile !== defaultProfile) patch.executionProfile = defaultProfile
    if (typeof metadata.defaultSteps === 'number' && current.steps !== metadata.defaultSteps) patch.steps = metadata.defaultSteps
    if (
      typeof metadata.defaultGuidanceScale === 'number'
      && Number(current.cfgScale) !== Number(metadata.defaultGuidanceScale)
    ) {
      patch.cfgScale = metadata.defaultGuidanceScale
    }
    if (Object.keys(patch).length > 0) updateParamsPatch(patch)
  },
  { immediate: true },
)

function normalizePositiveInt(rawValue: number, fallback: number, minimum = 1, maximum?: number): number {
  const numeric = Number.isFinite(rawValue) ? Math.trunc(rawValue) : Math.max(minimum, Math.trunc(fallback))
  const lowerClamped = Math.max(minimum, numeric)
  if (maximum === undefined) return lowerClamped
  return Math.min(maximum, lowerClamped)
}

function normalizeFiniteNumber(rawValue: number, fallback: number, minimum?: number, maximum?: number): number {
  const numeric = Number.isFinite(rawValue) ? Number(rawValue) : Number(fallback)
  let next = Number.isFinite(numeric) ? numeric : 0
  if (minimum !== undefined) next = Math.max(minimum, next)
  if (maximum !== undefined) next = Math.min(maximum, next)
  return next
}

function updateParamsPatch(patch: Partial<LtxTabParams>): void {
  store.updateParams(props.tabId, patch as Partial<Record<string, unknown>>).catch((error) => {
    toast(error instanceof Error ? error.message : String(error))
  })
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result || ''))
    reader.onerror = () => reject(reader.error ?? new Error('Failed to read file.'))
    reader.readAsDataURL(file)
  })
}

async function onInitImageFile(file: File): Promise<void> {
  try {
    const dataUrl = await readFileAsDataURL(file)
    updateParamsPatch({
      mode: 'img2vid',
      useInitImage: true,
      initImageData: dataUrl,
      initImageName: file.name,
    })
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function onInitImageRejected(payload: { reason: string; files: File[] }): void {
  const suffix = payload.files.length > 0 ? ` (${payload.files.map((file) => file.name).join(', ')})` : ''
  toast(`${payload.reason}${suffix}`)
}

function clearInit(): void {
  updateParamsPatch({ initImageData: '', initImageName: '' })
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

const slotProps = computed(() => ({
  tab: tab.value,
  params: params.value,
  mode: mode.value,
  status: status.value,
  progress: progress.value,
  frames: frames.value,
  info: info.value,
  videoUrl: videoUrl.value,
  errorMessage: errorMessage.value,
  isRunning: isRunning.value,
  checkpointCoreOnly: checkpointCoreOnly.value,
  checkpointExecutionMetadata: checkpointExecutionMetadata.value,
  assetContract: assetContract.value,
  copyNotice: copyNotice.value,
  resumeNotice: resumeNotice.value,
  dimensionAlignment: dimensionAlignment.value,
  executionProfileOptions: executionProfileOptions.value,
  executionProfileCaption: executionProfileCaption.value,
  dimensionRuleCaption: dimensionRuleCaption.value,
  dimensionWarning: dimensionWarning.value,
  executionProfileWarning: executionProfileWarning.value,
  checkpointDisplay: checkpointDisplay.value,
  vaeDisplay: vaeDisplay.value,
  textEncoderDisplay: textEncoderDisplay.value,
  hideNegativePrompt: hideNegativePrompt.value,
  promptModeLabel: promptModeLabel.value,
  runGenerateDisabled: runGenerateDisabled.value,
  runGenerateTitle: runGenerateTitle.value,
  runSummary: runSummary.value,
  successMessage: successMessage.value,
  generate,
  cancel,
  updateParamsPatch,
  onInitImageFile,
  onInitImageRejected,
  clearInit,
  normalizePositiveInt,
  normalizeFiniteNumber,
  copyJson,
  formatJson,
  toDataUrl,
  ltxDimMin: LTX_DIM_MIN,
  ltxDimMax: LTX_DIM_MAX,
  ltxFramesMin: LTX_FRAMES_MIN,
  ltxFramesMax: LTX_FRAMES_MAX,
  ltxFrameAlignment: LTX_FRAME_ALIGNMENT,
  LTX_DIM_MIN,
  LTX_DIM_MAX,
  LTX_FRAMES_MIN,
  LTX_FRAMES_MAX,
  LTX_FRAME_ALIGNMENT,
}))

</script>
