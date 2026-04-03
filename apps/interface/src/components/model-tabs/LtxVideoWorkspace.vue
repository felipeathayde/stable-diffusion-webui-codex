<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Family-owned LTX video workspace backed by the generic video endpoints.
Uses the shared video presentation structure while keeping LTX-specific controls truthful to the strict LTX generic backend contract
(profile-aware `32px` / `64px` geometry, `8n+1` frames, checkpoint-aware execution profiles, no silent snapping).
Stale persisted unsupported execution profiles stay visible as blocking state until the user repairs them. Checkpoint/VAE/text-encoder selection
stays in QuickSettings; the workspace owns prompt/init-image, generation parameters, run/cancel, progress, exported video, and optional
returned frames.

Symbols (top-level; keep in sync; no ghosts):
- `LtxVideoWorkspace` (component): LTX video generation workspace.
- `ExecutionProfileOption` (type): Workspace-facing execution-profile selector row.
- `readFileAsDataURL` (function): Reads the selected init image into a data URL.
- `normalizePositiveInt` (function): Clamps/sanitizes positive integer field updates.
- `normalizeExecutionProfileName` (function): Normalizes raw execution-profile names for selector/display checks.
- `executionProfileLabel` (function): Formats a user-facing label for a known or stale execution profile.
- `ensureExecutionProfileVisible` (function): Preserves stale persisted execution-profile values in the local selector option list.
- `updateParamsPatch` (function): Persists top-level LTX param patches.
- `onInitImageFile` (function): Loads an init image into tab params.
- `clearInit` (function): Clears init-image fields without changing mode.
-->

<template>
  <section v-if="tab && params" class="panels video-panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Prompt</div>
        <div class="panel-body">
          <div class="gen-card">
            <div class="row-split mb-2">
              <span class="label-muted">Prompt</span>
              <span class="caption">{{ mode === 'img2vid' ? 'IMG2VID' : 'TXT2VID' }}</span>
            </div>
            <PromptFields
              :prompt="params.prompt"
              :negative="params.negativePrompt"
              token-engine="ltx2"
              @update:prompt="(value) => updateParamsPatch({ prompt: value })"
              @update:negative="(value) => updateParamsPatch({ negativePrompt: value })"
            />
            <p class="caption mt-2">Mode, checkpoint, VAE, and text encoder are selected in QuickSettings.</p>
          </div>

          <div v-if="mode === 'img2vid'" class="gen-card mt-3">
            <div class="row-split mb-2">
              <span class="label-muted">Init Image</span>
            </div>
            <Img2ImgInpaintParamsCard
              embedded
              :disabled="isRunning"
              sectionTitle="Img2Vid Parameters"
              sectionSubtitle="Initial image"
              initImageLabel="Image"
              :initImageData="params.initImageData"
              :initImageName="params.initImageName"
              :imageWidth="params.width"
              :imageHeight="params.height"
              :useMask="false"
              maskImageData=""
              maskImageName=""
              maskEnforcement="per_step_clamp"
              :inpaintingFill="1"
              :inpaintFullResPadding="0"
              :maskBlur="0"
              @set:initImage="onInitImageFile"
              @clear:initImage="clearInit"
              @reject:initImage="onInitImageRejected"
            />
            <p class="caption mt-2">
              Generic LTX img2vid currently does not expose denoise strength on `/api/img2vid`.
            </p>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <div class="gen-card mb-3">
            <div class="row-split mb-2">
              <span class="label-muted">Video</span>
            </div>
            <div class="grid gap-3 md:grid-cols-2 mb-3">
	              <SliderField
	                label="Width (px)"
	                :modelValue="params.width"
	                :min="LTX_DIM_MIN"
	                :max="LTX_DIM_MAX"
	                :step="dimensionAlignment"
	                :inputStep="1"
	                :nudgeStep="dimensionAlignment"
	                inputClass="cdx-input-w-md"
	                :disabled="isRunning"
	                @update:modelValue="(value) => updateParamsPatch({ width: normalizePositiveInt(value, params?.width ?? LTX_DIM_MIN, LTX_DIM_MIN, LTX_DIM_MAX) })"
	              />
	              <SliderField
	                label="Height (px)"
	                :modelValue="params.height"
	                :min="LTX_DIM_MIN"
	                :max="LTX_DIM_MAX"
	                :step="dimensionAlignment"
	                :inputStep="1"
	                :nudgeStep="dimensionAlignment"
	                inputClass="cdx-input-w-md"
	                :disabled="isRunning"
	                @update:modelValue="(value) => updateParamsPatch({ height: normalizePositiveInt(value, params?.height ?? LTX_DIM_MIN, LTX_DIM_MIN, LTX_DIM_MAX) })"
	              />
	            </div>
            <VideoSettingsCard
              embedded
              :frames="params.frames"
              :fps="params.fps"
              :minFrames="LTX_FRAMES_MIN"
              :maxFrames="LTX_FRAMES_MAX"
              :frameStep="LTX_FRAME_ALIGNMENT"
              :frameNudgeStep="LTX_FRAME_ALIGNMENT"
              frameRuleLabel="8n+1"
              :minFps="1"
              :maxFps="60"
	              @update:frames="(value) => updateParamsPatch({ frames: normalizePositiveInt(value, params?.frames ?? LTX_FRAMES_MIN, LTX_FRAMES_MIN, LTX_FRAMES_MAX) })"
	              @update:fps="(value) => updateParamsPatch({ fps: normalizePositiveInt(value, params?.fps ?? 24, 1, 240) })"
	            />
	            <p class="caption mt-2">{{ dimensionRuleCaption }}</p>
	            <p v-if="dimensionWarning" class="panel-status mt-2">{{ dimensionWarning }}</p>
	          </div>

          <div class="gen-card mb-3">
            <div class="row-split mb-2">
              <span class="label-muted">Sampling</span>
            </div>
            <div class="grid gap-3 md:grid-cols-2 xl:grid-cols-3 mb-3">
              <SliderField
                label="Steps"
                :modelValue="params.steps"
                :min="1"
                :max="100"
                :step="1"
                :inputStep="1"
                :nudgeStep="1"
                inputClass="cdx-input-w-sm"
                :disabled="isRunning"
                @update:modelValue="(value) => updateParamsPatch({ steps: normalizePositiveInt(value, params?.steps ?? 3, 1) })"
              />
              <SliderField
                label="CFG"
                :modelValue="params.cfgScale"
                :min="0"
                :max="20"
                :step="0.1"
                :inputStep="0.1"
                :nudgeStep="0.5"
                inputClass="cdx-input-w-sm"
                :disabled="isRunning"
                @update:modelValue="(value) => updateParamsPatch({ cfgScale: normalizeFiniteNumber(value, params?.cfgScale ?? 1, 0) })"
              />
              <div class="form-field">
                <label class="label-muted">Seed</label>
                <NumberStepperInput
                  :modelValue="params.seed"
                  :min="-1"
                  :step="1"
                  :nudgeStep="1"
                  inputClass="cdx-input-w-sm"
                  :disabled="isRunning"
                  updateOnInput
                  @update:modelValue="(value) => updateParamsPatch({ seed: Math.trunc(value) })"
                />
              </div>
            </div>
            <div class="grid gap-3 md:grid-cols-2">
              <div class="form-field">
                <label class="label-muted">Execution Profile</label>
                <select
                  class="select-md"
                  :value="params.executionProfile"
                  :disabled="isRunning"
                  @change="(event) => updateParamsPatch({ executionProfile: (event.target as HTMLSelectElement).value })"
	                >
	                  <option value="">Select profile</option>
                  <option
                    v-for="option in executionProfileOptions"
                    :key="option.value"
                    :value="option.value"
                    :disabled="!option.supported"
                  >
	                    {{ option.label }}
	                  </option>
	                </select>
	                <p class="caption mt-2">{{ executionProfileCaption }}</p>
	              </div>
	            </div>
	            <p v-if="executionProfileWarning" class="panel-status mt-2">{{ executionProfileWarning }}</p>
          </div>

          <div class="gen-card">
            <div class="row-split mb-2">
              <span class="label-muted">Output / Assets</span>
            </div>
            <div class="row-split mb-2">
              <span class="label-muted">Return frames</span>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', params.videoReturnFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="isRunning"
                :aria-pressed="params.videoReturnFrames"
                @click="updateParamsPatch({ videoReturnFrames: !params.videoReturnFrames })"
              >
                {{ params.videoReturnFrames ? 'Enabled' : 'Disabled' }}
              </button>
            </div>

            <div class="space-y-2 text-sm">
              <div class="row-split"><span class="label-muted">Checkpoint</span><code>{{ checkpointDisplay }}</code></div>
              <div class="row-split"><span class="label-muted">VAE</span><code>{{ vaeDisplay }}</code></div>
              <div class="row-split"><span class="label-muted">Text Encoder</span><code>{{ textEncoderDisplay }}</code></div>
            </div>

            <p v-if="checkpointCoreOnly && assetContract?.notes" class="caption mt-2">
              {{ assetContract.notes }}
            </p>
            <details v-else-if="assetContract?.notes" class="accordion mt-2">
              <summary>Contract notes</summary>
              <div class="accordion-body">
                <p class="text-xs break-words">{{ assetContract.notes }}</p>
              </div>
            </details>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-stack panel-stack--sticky">
      <RunCard
        :isRunning="isRunning"
        :generateDisabled="runGenerateDisabled"
        :generateTitle="runGenerateTitle"
        :showBatchControls="false"
        @generate="generate()"
        @cancel="cancel()"
      >
        <div v-if="resumeNotice || copyNotice" class="caption">
          {{ resumeNotice || copyNotice }}
        </div>
        <RunSummaryChips class="video-results-summary" :text="runSummary" />
        <RunProgressStatus
          v-if="isRunning"
          :stage="progress.stage"
          :percent="progress.percent"
          :step="progress.step"
          :total-steps="progress.totalSteps"
          :eta-seconds="progress.etaSeconds"
          :show-progress-bar="true"
        />
        <RunProgressStatus
          v-else-if="errorMessage"
          variant="error"
          title="Run failed"
          :message="errorMessage"
          :show-progress-bar="false"
        />
        <RunProgressStatus
          v-else-if="status === 'done' && (videoUrl || frames.length)"
          variant="success"
          title="Run complete"
          :message="successMessage"
          :show-progress-bar="false"
        />
      </RunCard>

      <ResultsCard class="video-results-panel" :showGenerate="false" headerRightClass="results-header-actions">
        <template #header-right>
          <button v-if="info" class="btn btn-sm btn-outline" type="button" @click="copyJson(info, 'Copied generation info.')">
            Copy info
          </button>
        </template>

        <div class="gen-card mb-3">
          <div class="row-split">
            <span class="label-muted">Exported Video</span>
            <a v-if="videoUrl" class="btn btn-sm btn-outline" :href="videoUrl" target="_blank" rel="noreferrer">Open</a>
          </div>
          <video v-if="videoUrl" class="rounded mt-2" :src="videoUrl" controls />
          <div v-else class="caption mt-2">No exported video yet.</div>
        </div>

        <div class="gen-card">
          <div class="row-split mb-2">
            <span class="label-muted">Returned Frames</span>
            <span class="caption">{{ params.videoReturnFrames ? 'Requested' : 'Disabled' }}</span>
          </div>
          <ResultViewer mode="video" :frames="frames" :toDataUrl="toDataUrl" emptyText="No frames yet.">
            <template #empty>
              <div class="results-empty-state">
                <div class="results-empty-title">
                  <template v-if="isRunning">Generating…</template>
                  <template v-else-if="videoUrl">Frames not returned</template>
                  <template v-else>No frames yet</template>
                </div>
                <div v-if="videoUrl" class="caption">
                  <template v-if="params.videoReturnFrames">The backend completed without returned frames.</template>
                  <template v-else>Enable “Return frames” to include frames in the result payload.</template>
                </div>
                <div v-else-if="!isRunning" class="caption">Generate to see returned frames here.</div>
              </div>
            </template>
          </ResultViewer>
        </div>

        <div v-if="info" class="gen-card mt-3">
          <div class="row-split">
            <span class="label-muted">Generation Info</span>
          </div>
          <pre class="text-xs break-words mt-2">{{ formatJson(info) }}</pre>
        </div>
      </ResultsCard>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
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
import Img2ImgInpaintParamsCard from '../../components/Img2ImgInpaintParamsCard.vue'
import PromptFields from '../../components/prompt/PromptFields.vue'
import ResultViewer from '../../components/ResultViewer.vue'
import ResultsCard from '../../components/results/ResultsCard.vue'
import RunCard from '../../components/results/RunCard.vue'
import RunProgressStatus from '../../components/results/RunProgressStatus.vue'
import RunSummaryChips from '../../components/results/RunSummaryChips.vue'
import VideoSettingsCard from '../../components/VideoSettingsCard.vue'
import NumberStepperInput from '../../components/ui/NumberStepperInput.vue'
import SliderField from '../../components/ui/SliderField.vue'
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
</script>
