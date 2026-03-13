<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dedicated LTX video tab UI backed by the generic video endpoints.
Renders a minimal truthful `ltx2` txt2vid/img2vid workspace, reusing shared prompt/init-image/progress/results components while keeping
checkpoint/VAE/text-encoder selection in QuickSettings instead of duplicating local selectors. Exposes only the generic backend contract:
mode, prompts, dimensions, fps, frames, steps, cfg, sampler, scheduler, seed, optional init image, run/cancel, progress, exported video,
and optional returned frames.

Symbols (top-level; keep in sync; no ghosts):
- `LTXTab` (component): LTX video generation workspace for `/models/:tabId`.
- `fallbackSamplers` / `fallbackSchedulers` (const): Local selector fallbacks when sampler catalogs are unavailable.
- `readFileAsDataURL` (function): Reads the selected init image into a data URL.
- `normalizePositiveInt` (function): Clamps/sanitizes positive integer field updates.
- `updateParamsPatch` (function): Persists top-level LTX param patches.
- `setMode` (function): Keeps `mode` and `useInitImage` in sync.
- `onInitImageFile` (function): Loads an init image into tab params.
- `clearInit` (function): Clears init-image fields without changing mode.
-->

<template>
  <section v-if="tab && params" class="panels ltx-panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Prompt</div>
        <div class="panel-body">
          <div class="gen-card mb-3">
            <div class="row-split">
              <span class="label-muted">Mode</span>
              <div class="qs-row">
                <button
                  :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', mode === 'txt2vid' ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                  type="button"
                  :disabled="isRunning"
                  :aria-pressed="mode === 'txt2vid'"
                  @click="setMode('txt2vid')"
                >
                  TXT2VID
                </button>
                <button
                  :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', mode === 'img2vid' ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                  type="button"
                  :disabled="isRunning"
                  :aria-pressed="mode === 'img2vid'"
                  @click="setMode('img2vid')"
                >
                  IMG2VID
                </button>
              </div>
            </div>
            <p class="caption mt-2">Checkpoint, VAE, and text encoder are selected in QuickSettings.</p>
          </div>

          <div class="gen-card mb-3">
            <PromptFields
              :prompt="params.prompt"
              :negative="params.negativePrompt"
              token-engine="ltx2"
              @update:prompt="(value) => updateParamsPatch({ prompt: value })"
              @update:negative="(value) => updateParamsPatch({ negativePrompt: value })"
            />
          </div>

          <div v-if="mode === 'img2vid'" class="gen-card">
            <InitialImageCard
              label="Initial Image"
              :src="params.initImageData"
              :hasImage="Boolean(params.initImageData)"
              :disabled="isRunning"
              placeholder="Drop an image or click to browse."
              dropzone
              zoomable
              @set="onInitImageFile"
              @clear="clearInit"
              @rejected="onInitImageRejected"
            >
              <template #footer>
                <p class="caption mt-2">
                  Generic LTX img2vid currently does not expose denoise strength on `/api/img2vid`; the stored tab value is not sent.
                </p>
              </template>
            </InitialImageCard>
          </div>
        </div>
      </div>

      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <div class="gen-card mb-3">
            <div class="grid gap-3 md:grid-cols-2">
              <SliderField
                label="Width (px)"
                :modelValue="params.width"
                :min="16"
                :max="2048"
                :step="16"
                :inputStep="16"
                :nudgeStep="16"
                inputClass="cdx-input-w-md"
                :disabled="isRunning"
                @update:modelValue="(value) => updateParamsPatch({ width: snapLtxDim(value) })"
              />
              <SliderField
                label="Height (px)"
                :modelValue="params.height"
                :min="16"
                :max="2048"
                :step="16"
                :inputStep="16"
                :nudgeStep="16"
                inputClass="cdx-input-w-md"
                :disabled="isRunning"
                @update:modelValue="(value) => updateParamsPatch({ height: snapLtxDim(value) })"
              />
            </div>
          </div>

          <div class="gen-card mb-3">
            <VideoSettingsCard
              embedded
              :frames="params.frames"
              :fps="params.fps"
              :minFrames="9"
              :maxFrames="401"
              :minFps="1"
              :maxFps="60"
              @update:frames="(value) => updateParamsPatch({ frames: normalizeLtxFrameCount(value) })"
              @update:fps="(value) => updateParamsPatch({ fps: normalizePositiveInt(value, params?.fps ?? 24, 1, 240) })"
            />
          </div>

          <div class="gen-card mb-3">
            <div class="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
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
          </div>

          <div class="gen-card mb-3">
            <div class="grid gap-3 md:grid-cols-2">
              <SamplerSelector
                :samplers="filteredSamplers"
                :modelValue="params.sampler"
                :recommendedNames="recommendedSamplers"
                :disabled="isRunning"
                @update:modelValue="(value) => updateParamsPatch({ sampler: value })"
              />
              <SchedulerSelector
                :schedulers="filteredSchedulers"
                :modelValue="params.scheduler"
                :recommendedNames="recommendedSchedulers"
                :disabled="isRunning"
                @update:modelValue="(value) => updateParamsPatch({ scheduler: value })"
              />
            </div>
            <p v-if="samplingWarning" class="panel-status mt-2">{{ samplingWarning }}</p>
          </div>

          <div class="gen-card">
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

            <p v-if="checkpointCoreOnly" class="caption mt-2">
              Core-only LTX checkpoint detected: external VAE, text encoder, embeddings connectors, and the audio bundle are required.
            </p>
            <details v-if="assetContract?.notes" class="accordion mt-2">
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
        <div class="caption">{{ runSummary }}</div>
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

      <ResultsCard :showGenerate="false" headerRightClass="wan-header-actions">
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
          <video v-if="videoUrl" class="w-full rounded mt-2" :src="videoUrl" controls />
          <div v-else class="caption mt-2">No exported video yet.</div>
        </div>

        <div class="gen-card">
          <div class="row-split mb-2">
            <span class="label-muted">Returned Frames</span>
            <span class="caption">{{ params.videoReturnFrames ? 'Requested' : 'Disabled' }}</span>
          </div>
          <ResultViewer mode="video" :frames="frames" :toDataUrl="toDataUrl" emptyText="No frames yet.">
            <template #empty>
              <div class="wan-results-empty">
                <div class="wan-empty-title">
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
import { computed, onMounted, ref, watch } from 'vue'

import { fetchSamplers, fetchSchedulers } from '../api/client'
import type { GeneratedImage, SamplerInfo, SchedulerInfo } from '../api/types'
import {
  LTX_ALLOWED_SAMPLERS,
  LTX_CANONICAL_SCHEDULER,
  normalizeLtxFrameCount,
  snapLtxDim,
} from '../api/payloads_ltx_video'
import InitialImageCard from '../components/InitialImageCard.vue'
import PromptFields from '../components/prompt/PromptFields.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunProgressStatus from '../components/results/RunProgressStatus.vue'
import SamplerSelector from '../components/SamplerSelector.vue'
import SchedulerSelector from '../components/SchedulerSelector.vue'
import VideoSettingsCard from '../components/VideoSettingsCard.vue'
import NumberStepperInput from '../components/ui/NumberStepperInput.vue'
import SliderField from '../components/ui/SliderField.vue'
import { useLtxVideoGeneration } from '../composables/useLtxVideoGeneration'
import { useResultsCard } from '../composables/useResultsCard'
import { useModelTabsStore, type LtxGenerationMode, type LtxTabParams } from '../stores/model_tabs'

const props = defineProps<{ tabId: string }>()

const store = useModelTabsStore()
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])
const samplingCatalogError = ref('')

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
  engineSurface,
  assetContract,
  blockedReason,
  generate,
  cancel,
  resumeNotice,
} = useLtxVideoGeneration(props.tabId)

const { notice: copyNotice, copyJson, formatJson, toast } = useResultsCard()

const fallbackSamplers: SamplerInfo[] = LTX_ALLOWED_SAMPLERS.map((name) => ({
  name,
  label: name,
  supported: true,
  default_scheduler: LTX_CANONICAL_SCHEDULER,
  allowed_schedulers: [LTX_CANONICAL_SCHEDULER],
}))
const fallbackSchedulers: SchedulerInfo[] = [{ name: LTX_CANONICAL_SCHEDULER, label: LTX_CANONICAL_SCHEDULER, supported: true }]

function normalizeSamplerName(rawValue: string): string {
  return String(rawValue || '').trim().toLowerCase()
}

function normalizeSchedulerName(rawValue: string): string {
  return String(rawValue || '').trim().toLowerCase()
}

function ensureSamplerVisible(options: SamplerInfo[], currentValue: string): SamplerInfo[] {
  const current = normalizeSamplerName(currentValue)
  if (!current) return options
  if (options.some((entry) => normalizeSamplerName(entry.name) === current)) return options
  if (!LTX_ALLOWED_SAMPLERS.includes(current as (typeof LTX_ALLOWED_SAMPLERS)[number])) return options
  return [{
    name: current,
    label: current,
    supported: true,
    default_scheduler: LTX_CANONICAL_SCHEDULER,
    allowed_schedulers: [LTX_CANONICAL_SCHEDULER],
  }, ...options]
}

function ensureSchedulerVisible(options: SchedulerInfo[], currentValue: string): SchedulerInfo[] {
  const current = normalizeSchedulerName(currentValue)
  if (!current) return options
  if (options.some((entry) => normalizeSchedulerName(entry.name) === current)) return options
  if (current !== LTX_CANONICAL_SCHEDULER) return options
  return [{ name: current, label: current, supported: true }, ...options]
}

const filteredSamplers = computed(() => {
  const allowed = new Set<string>(LTX_ALLOWED_SAMPLERS)
  const filtered = samplers.value.filter((entry) => allowed.has(String(entry.name || '').trim()))
  const base = filtered.length > 0 ? filtered : fallbackSamplers
  return ensureSamplerVisible(base, String(params.value?.sampler || ''))
})
const filteredSchedulers = computed(() => {
  const filtered = schedulers.value.filter((entry) => String(entry.name || '').trim() === LTX_CANONICAL_SCHEDULER)
  const base = filtered.length > 0 ? filtered : fallbackSchedulers
  return ensureSchedulerVisible(base, String(params.value?.scheduler || ''))
})
const recommendedSamplers = computed(() => {
  const allowed = new Set<string>(LTX_ALLOWED_SAMPLERS)
  const raw = Array.isArray(engineSurface.value?.recommended_samplers) ? engineSurface.value?.recommended_samplers : []
  return raw.map((entry) => String(entry || '').trim()).filter((entry) => allowed.has(entry))
})
const recommendedSchedulers = computed(() => {
  const raw = Array.isArray(engineSurface.value?.recommended_schedulers) ? engineSurface.value?.recommended_schedulers : []
  return raw.map((entry) => String(entry || '').trim()).filter((entry) => entry === LTX_CANONICAL_SCHEDULER)
})
const samplingWarning = computed(() => {
  const message = String(blockedReason.value || '')
  if (message.includes('sampler') || message.includes('scheduler')) return message
  return samplingCatalogError.value
})

const checkpointDisplay = computed(() => String(params.value?.checkpoint || '').trim() || 'Not selected')
const vaeDisplay = computed(() => String(params.value?.vae || '').trim() || (checkpointCoreOnly.value ? 'Not selected' : 'Built-in / omitted'))
const textEncoderDisplay = computed(() => String(params.value?.textEncoder || '').trim() || 'Not selected')
const runGenerateDisabled = computed(() => isRunning.value || Boolean(blockedReason.value))
const runGenerateTitle = computed(() => (isRunning.value ? '' : blockedReason.value))
const runSummary = computed(() => {
  const current = params.value
  if (!current) return ''
  return `${current.width}×${current.height} · ${current.frames}f @ ${current.fps}fps · steps ${current.steps} · cfg ${current.cfgScale}`
})
const successMessage = computed(() => {
  const parts: string[] = []
  if (videoUrl.value) parts.push('Video ready')
  if (frames.value.length > 0) parts.push(`${frames.value.length} frame${frames.value.length === 1 ? '' : 's'} returned`)
  return parts.join(' · ') || 'Task finished.'
})

onMounted(async () => {
  try {
    const [samplerResponse, schedulerResponse] = await Promise.all([fetchSamplers(), fetchSchedulers()])
    samplers.value = samplerResponse.samplers
    schedulers.value = schedulerResponse.schedulers
    samplingCatalogError.value = ''
  } catch (error) {
    samplers.value = []
    schedulers.value = []
    samplingCatalogError.value = error instanceof Error ? error.message : String(error)
  }
})

function syncSamplingSelections(): void {
  const current = params.value
  if (!current) return

  const patch: Partial<LtxTabParams> = {}
  const normalizedSampler = normalizeSamplerName(current.sampler)
  if (
    normalizedSampler
    && normalizedSampler !== current.sampler
    && LTX_ALLOWED_SAMPLERS.includes(normalizedSampler as (typeof LTX_ALLOWED_SAMPLERS)[number])
  ) {
    patch.sampler = normalizedSampler
  }

  const normalizedScheduler = normalizeSchedulerName(current.scheduler)
  if (normalizedScheduler === LTX_CANONICAL_SCHEDULER && normalizedScheduler !== current.scheduler) {
    patch.scheduler = normalizedScheduler
  }

  if (Object.keys(patch).length > 0) updateParamsPatch(patch)
}

watch(
  () => [String(params.value?.sampler || ''), String(params.value?.scheduler || '')] as const,
  () => {
    syncSamplingSelections()
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

function setMode(nextMode: LtxGenerationMode): void {
  if (!params.value) return
  if (mode.value === nextMode && Boolean(params.value.useInitImage) === (nextMode === 'img2vid')) return
  updateParamsPatch({ mode: nextMode, useInitImage: nextMode === 'img2vid' })
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
