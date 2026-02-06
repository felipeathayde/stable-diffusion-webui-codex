<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Image model tab view (txt2img/img2img/inpaint) UI for SD/Flux/ZImage-family engines.
Owns prompt + parameter controls, init-image + mask handling for img2img/inpaint, per-tab history, and integrates with the generation composable to
submit `/api/txt2img`/`/api/img2img` tasks and render progress/results (Z-Image Turbo/Base UI is variant-dependent: CFG label + negative prompt gating).
Hires settings list upscalers from `/api/upscalers` and share tile controls + explicit OOM fallback preference with `/upscale`.
Also shares the global `min_tile` preference (tiled OOM fallback lower bound) with `/upscale`.
Surfaces a one-shot toast when the generation composable auto-reattaches to an in-flight task after a reload/crash.
Generate CTA and run preflight are capability-driven (`/api/engines/capabilities`) and fail loud when the current mode is unsupported.

Symbols (top-level; keep in sync; no ghosts):
- `ImageModelTab` (component): Main image model tab view; handles prompt/params/profile persistence, init-image UX, history reuse, and actions.
- `sendToWorkflows` (function): Sends the current params snapshot to the workflows subsystem (async).
- `copyCurrentParams` (function): Copies current params snapshot to clipboard (async).
- `copyHistoryParams` (function): Copies a history entry’s params snapshot to clipboard (async).
- `applyHistory` (function): Applies a history entry back into current state (prompt/params/assets).
- `formatHistoryTitle` (function): Builds a human-friendly history title from a run entry.
- `profileStorageKeyFor` (function): Computes the localStorage key for saving/loading per-engine profiles.
- `loadProfile` (function): Loads a saved profile into current params (with validation/defaulting).
- `saveProfile` (function): Saves current params as a profile in localStorage.
- `setParams` (function): Applies partial updates to the current tab params state.
- `setHires` (function): Applies partial updates to the hires config object.
- `setHiresRefiner` (function): Applies partial updates to the hires-refiner config object.
- `setRefiner` (function): Applies partial updates to the refiner config object.
- `clampFloat` (function): Clamps a float to `[min, max]` (input sanitation).
- `setFallbackOnOom` (function): Updates the global “fallback on OOM” preference used by upscaler tiling (hires-fix + `/upscale`).
- `setMinTile` (function): Updates the global `min_tile` preference used as the tiled OOM fallback lower bound (hires-fix + `/upscale`).
- `snapInitImageDim` (function): Snaps init-image derived dimensions to model constraints (e.g., multiples of 8).
- `toggleInitImage` (function): Toggles init-image usage (img2img).
- `onInitFileSet` (function): Reads an init image file into a data URL and stores name/data, then syncs dims (async).
- `clearInit` (function): Clears init image fields.
- `toggleMask` (function): Toggles mask usage (inpaint) for masked img2img.
- `onMaskFileSet` (function): Reads a mask file into a data URL and stores it after validating dimensions (async).
- `clearMask` (function): Clears mask fields.
- `toDataUrl` (function): Converts a generated image payload to a data URL for preview.
- `randomizeSeed` (function): Randomizes the seed field for the current tab params.
- `reuseSeed` (function): Reuses the last seed from history/current run as the next seed.
- `download` (function): Downloads a generated image artifact to disk.
- `sendToImg2Img` (function): Sends a generated image back into img2img init-image fields (async).
- `readFileAsDataURL` (function): Reads a File into a data URL (used for init-image handling).
- `readImageDimensions` (function): Reads width/height from an image source URL (used for init-image dimension sync).
- `syncInitImageDims` (function): Synchronizes init-image derived dimensions into width/height params (async).
- `maybeApplyKontextDefaults` (function): Applies Kontext-specific default params when relevant to the current engine/tab.
- `syncPreviewHeight` (function): Keeps the preview panel height aligned with layout changes (uses DOM measurements).
-->

<template>
  <section v-if="tab" class="panels">
    <!-- Left column: Prompt + Parameters -->
    <div class="panel-stack" ref="leftStack">
      <PromptCard
        v-model:prompt="promptText"
        v-model:negative="negativeText"
        :defaultShowNegative="defaultShowNegative"
        :supportsNegative="supportsNegative"
        :allowNegativeToggle="supportsNegative"
        :enableAssets="enableAssets"
        :enableStyles="enableStyles"
        :toolbarLabel="toolbarLabel"
        :fieldsId="`image-modeltab-prompt-${tabId}`"
      >
        <div v-if="isRunning" class="panel-progress">
          <p><strong>Stage:</strong> {{ progress.stage }}</p>
          <p v-if="progressPercent !== null">Progress: {{ progressPercent.toFixed(1) }}%</p>
          <p v-if="progress.totalSteps && progress.step !== null">
            Step {{ progress.step }} / {{ progress.totalSteps }}
          </p>
          <p v-if="progress.etaSeconds !== null" class="caption">ETA ~ {{ progress.etaSeconds.toFixed(0) }}s</p>
        </div>
        <div v-if="errorMessage" class="panel-error">
          {{ errorMessage }}
        </div>

        <div v-if="supportsImg2Img" class="panel-section">
          <button
            :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', params.useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
            type="button"
            :aria-pressed="params.useInitImage"
            :disabled="isRunning"
            @click="toggleInitImage"
          >
            Use Initial Image (img2img)
          </button>

          <div v-if="params.useInitImage">
            <InitialImageCard
              label="Initial Image"
              :src="params.initImageData"
              :has-image="Boolean(params.initImageData)"
              :disabled="isRunning"
              @set="onInitFileSet"
              @clear="clearInit"
            >
              <template #footer>
                <p v-if="params.initImageName" class="caption">{{ params.initImageName }}</p>
              </template>
            </InitialImageCard>

            <SliderField
              label="Denoise"
              :modelValue="params.denoiseStrength"
              :min="0"
              :max="1"
              :step="0.01"
              :inputStep="0.05"
              inputClass="cdx-input-w-xs"
              :disabled="isRunning"
              @update:modelValue="(v) => setParams({ denoiseStrength: clampFloat(v, 0, 1) })"
            />

            <button
              :class="[
                'btn',
                'qs-toggle-btn',
                'qs-toggle-btn--sm',
                params.useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off',
              ]"
              type="button"
              :aria-pressed="params.useMask"
              :disabled="isRunning || props.type === 'flux1'"
              @click="toggleMask"
            >
              Use Mask (inpaint)
            </button>
            <p v-if="props.type === 'flux1'" class="caption">
              Masking is not supported for Flux.1 img2img (Kontext) yet.
            </p>

            <div v-if="params.useMask">
              <InitialImageCard
                label="Mask"
                accept="image/*"
                :src="params.maskImageData"
                :has-image="Boolean(params.maskImageData)"
                :disabled="isRunning"
                placeholder="Select a mask image (RGBA/alpha supported)."
                @set="onMaskFileSet"
                @clear="clearMask"
              >
                <template #footer>
                  <p v-if="params.maskImageName" class="caption">{{ params.maskImageName }}</p>
                </template>
              </InitialImageCard>

              <div class="panel-section">
                <label class="label-muted">Enforcement</label>
                <select
                  class="select-md"
                  :disabled="isRunning"
                  :value="params.maskEnforcement"
                  @change="setParams({ maskEnforcement: ($event.target as HTMLSelectElement).value as any })"
                >
                  <option value="post_blend">Forge-style (post-sample blend)</option>
                  <option value="per_step_clamp">Clamp per step</option>
                </select>
              </div>

              <div class="panel-section">
                <label class="label-muted">Masked content</label>
                <select
                  class="select-md"
                  :disabled="isRunning"
                  :value="params.inpaintingFill"
                  @change="setParams({ inpaintingFill: Math.max(0, Math.min(3, Math.trunc(Number(($event.target as HTMLSelectElement).value)))) })"
                >
                  <option :value="1">Original</option>
                  <option :value="0">Fill</option>
                  <option :value="2">Latent noise</option>
                  <option :value="3">Latent nothing</option>
                </select>
              </div>

              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', params.inpaintFullRes ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="params.inpaintFullRes"
                :disabled="isRunning"
                @click="setParams({ inpaintFullRes: !params.inpaintFullRes })"
              >
                Inpaint area: Only masked (full-res)
              </button>

              <SliderField
                v-if="params.inpaintFullRes"
                label="Only masked padding"
                :modelValue="params.inpaintFullResPadding"
                :min="0"
                :max="256"
                :step="1"
                :inputStep="1"
                inputClass="cdx-input-w-xs"
                :disabled="isRunning"
                @update:modelValue="(v) => setParams({ inpaintFullResPadding: Math.max(0, Math.trunc(v)) })"
              />

              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', params.maskInvert ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="params.maskInvert"
                :disabled="isRunning"
                @click="setParams({ maskInvert: !params.maskInvert })"
              >
                Invert mask
              </button>

              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', params.maskRound ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :aria-pressed="params.maskRound"
                :disabled="isRunning"
                @click="setParams({ maskRound: !params.maskRound })"
              >
                Round mask
              </button>

              <SliderField
                label="Mask blur"
                :modelValue="params.maskBlur"
                :min="0"
                :max="64"
                :step="1"
                :inputStep="1"
                inputClass="cdx-input-w-xs"
                :disabled="isRunning"
                @update:modelValue="(v) => setParams({ maskBlur: Math.max(0, Math.trunc(v)) })"
              />
            </div>
          </div>
        </div>
      </PromptCard>

      <div class="panel">
        <div class="panel-header">
          Generation Parameters
          <div class="toolbar">
            <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="loadProfile">Load profile</button>
            <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="saveProfile">Save profile</button>
          </div>
        </div>
        <div class="panel-body">
          <BasicParametersCard
            :samplers="filteredSamplers"
            :schedulers="filteredSchedulers"
            :sampler="params.sampler"
            :scheduler="params.scheduler"
            :steps="params.steps"
            :width="params.width"
            :height="params.height"
            :cfg-scale="params.cfgScale"
            :seed="params.seed"
            :clip-skip="params.clipSkip"
            section-title="Basic Parameters"
            :resolutionPresets="resolutionPresets"
            :show-cfg="true"
            :cfg-label="cfgLabel"
            :show-clip-skip="showClipSkip"
            :min-clip-skip="minClipSkip"
            :max-clip-skip="12"
            :show-init-image-dims="params.useInitImage && Boolean(params.initImageData)"
            :disabled="isRunning"
            @update:sampler="onSamplerChange"
            @update:scheduler="(v: string) => setParams({ scheduler: v })"
            @update:steps="(v: number) => setParams({ steps: Math.max(1, Math.trunc(v)) })"
            @update:width="(v: number) => setParams({ width: Math.max(64, Math.trunc(v)) })"
            @update:height="(v: number) => setParams({ height: Math.max(64, Math.trunc(v)) })"
            @update:cfgScale="(v: number) => setParams({ cfgScale: v })"
            @update:seed="(v: number) => setParams({ seed: Math.trunc(v) })"
            @update:clipSkip="(v: number) => setParams({ clipSkip: Math.max(minClipSkip, Math.trunc(v)) })"
            @random-seed="randomizeSeed"
            @reuse-seed="reuseSeed"
            @sync-init-image-dims="syncInitImageDims"
          />

          <HiresSettingsCard
            v-if="showHires"
            :disabled="isRunning"
            :enabled="params.hires.enabled"
            :denoise="params.hires.denoise"
            :scale="params.hires.scale"
            :steps="params.hires.steps"
            :upscaler="params.hires.upscaler"
            :tile="params.hires.tile"
            :minTile="minTile"
            :fallbackOnOom="fallbackOnOom"
            :upscalers="upscalers"
            :upscalersLoading="upscalersLoading"
            :upscalersError="upscalersError"
            :base-width="params.width"
            :base-height="params.height"
            :refinerEnabled="showHiresRefiner ? params.hires.refiner?.enabled : undefined"
            :refinerSteps="showHiresRefiner ? params.hires.refiner?.steps : undefined"
            :refinerCfg="showHiresRefiner ? params.hires.refiner?.cfg : undefined"
            :refinerSeed="showHiresRefiner ? params.hires.refiner?.seed : undefined"
            :refinerModel="showHiresRefiner ? params.hires.refiner?.model : undefined"
            :refinerVae="showHiresRefiner ? params.hires.refiner?.vae : undefined"
            @update:enabled="(v: boolean) => setHires({ enabled: v })"
            @update:denoise="(v: number) => setHires({ denoise: clampFloat(v, 0, 1) })"
            @update:scale="(v: number) => setHires({ scale: v })"
            @update:steps="(v: number) => setHires({ steps: Math.max(0, Math.trunc(v)) })"
            @update:upscaler="(v: string) => setHires({ upscaler: v })"
            @update:tile="(v: { tile: number; overlap: number }) => setHires({ tile: v })"
            @update:minTile="setMinTile"
            @update:fallbackOnOom="setFallbackOnOom"
            @update:refinerEnabled="(v: boolean) => setHiresRefiner({ enabled: v })"
            @update:refinerSteps="(v: number) => setHiresRefiner({ steps: Math.max(0, Math.trunc(v)) })"
            @update:refinerCfg="(v: number) => setHiresRefiner({ cfg: v })"
            @update:refinerSeed="(v: number) => setHiresRefiner({ seed: Math.trunc(v) })"
            @update:refinerModel="(v: string) => setHiresRefiner({ model: v })"
            @update:refinerVae="(v: string) => setHiresRefiner({ vae: v })"
          />

          <RefinerSettingsCard
            v-if="showGlobalRefiner"
            :enabled="params.refiner.enabled"
            :steps="params.refiner.steps"
            :cfg="params.refiner.cfg"
            :seed="params.refiner.seed"
            :model="params.refiner.model"
            :vae="params.refiner.vae"
            @update:enabled="(v: boolean) => setRefiner({ enabled: v })"
            @update:steps="(v: number) => setRefiner({ steps: Math.max(0, Math.trunc(v)) })"
            @update:cfg="(v: number) => setRefiner({ cfg: v })"
            @update:seed="(v: number) => setRefiner({ seed: Math.trunc(v) })"
            @update:model="(v: string) => setRefiner({ model: v })"
            @update:vae="(v: string) => setRefiner({ vae: v })"
          />
        </div>
      </div>
    </div>

    <!-- Right column: Run + Results -->
    <div class="panel-stack">
      <RunCard
        :generateDisabled="isRunning || !canGenerateForCurrentMode"
        :generateTitle="generateDisabledReason"
        :isRunning="isRunning"
        :showBatchControls="true"
        :batchCount="params.batchCount"
        :batchSize="params.batchSize"
        :disabled="isRunning"
        @generate="generate"
        @update:batchCount="(v: number) => setParams({ batchCount: Math.max(1, Math.trunc(v)) })"
        @update:batchSize="(v: number) => setParams({ batchSize: Math.max(1, Math.trunc(v)) })"
      >
        <div v-if="copyNotice" class="caption">{{ copyNotice }}</div>
        <RunSummaryChips :text="runSummary" />
      </RunCard>

      <ResultsCard :showGenerate="false" headerClass="three-cols" headerRightClass="results-actions">
        <template #header-right>
          <div class="gentime-display" v-if="gentimeSeconds !== null">
            <span class="caption">Time: {{ gentimeSeconds.toFixed(2) }}s</span>
          </div>
          <button class="btn btn-sm btn-secondary" type="button" :disabled="workflowBusy" @click="sendToWorkflows">
            {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
          </button>
          <button class="btn btn-sm btn-outline" type="button" @click="copyCurrentParams">Copy params</button>
        </template>

        <div class="gen-card mb-3">
          <WanSubHeader title="History">
            <button class="btn btn-sm btn-ghost" type="button" title="Clear history" :disabled="!history.length || isRunning" @click="clearHistory">Clear</button>
          </WanSubHeader>
          <div v-if="history.length" class="cdx-history-list">
            <div v-for="item in history" :key="item.taskId" :class="['cdx-history-item', { 'is-selected': item.taskId === selectedTaskId }]">
              <div class="cdx-history-meta">
                <div class="cdx-history-title">{{ formatHistoryTitle(item) }}</div>
                <div class="cdx-history-sub">{{ item.summary }}</div>
                <div v-if="item.promptPreview" class="cdx-history-sub">{{ item.promptPreview }}</div>
                <div v-if="item.status !== 'completed'" class="caption">Status: {{ item.status }}</div>
                <div v-if="item.errorMessage" class="caption">Error: {{ item.errorMessage }}</div>
              </div>
              <div class="cdx-history-actions">
                <button class="btn btn-sm btn-secondary" type="button" :disabled="isRunning || historyLoadingTaskId === item.taskId" @click="loadHistory(item.taskId)">
                  {{ historyLoadingTaskId === item.taskId ? 'Loading…' : 'Load' }}
                </button>
                <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="applyHistory(item)">Apply</button>
                <button class="btn btn-sm btn-outline" type="button" :disabled="isRunning" @click="copyHistoryParams(item)">Copy</button>
              </div>
            </div>
          </div>
          <div v-else class="caption">No runs yet.</div>
        </div>

        <ResultViewer
          mode="image"
          :images="images"
          :previewImage="previewImage"
          :previewCaption="previewCaption"
          :isRunning="isRunning"
          :width="params.width"
          :height="params.height"
          :emptyText="resultsEmptyText"
          :style="previewStyle"
        >
          <template #image-actions="{ image, index }">
            <button
              v-if="supportsImg2Img"
              class="gallery-action"
              type="button"
              title="Send to Img2Img"
              @click="sendToImg2Img(image)"
            >
              Send to Img2Img
            </button>
            <button class="gallery-action" type="button" title="Download Image" @click="download(image, index)">
              Download
            </button>
          </template>
        </ResultViewer>
      </ResultsCard>

      <div class="panel" v-if="info">
        <div class="panel-header">Generation Info</div>
        <div class="panel-body">
          <pre class="text-xs break-words">{{ formatJson(info) }}</pre>
        </div>
      </div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { fetchSamplers, fetchSchedulers } from '../api/client'
import type { GeneratedImage, SamplerInfo, SchedulerInfo } from '../api/types'
import { formatJson, useResultsCard } from '../composables/useResultsCard'
import { resolveEngineForRequest, useGeneration, type ImageRunHistoryItem } from '../composables/useGeneration'
import { useModelTabsStore, type ImageBaseParams } from '../stores/model_tabs'
import { getEngineConfig, getEngineDefaults, type EngineType } from '../stores/engine_config'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import { useUpscalersStore } from '../stores/upscalers'
import { useWorkflowsStore } from '../stores/workflows'
import BasicParametersCard from '../components/BasicParametersCard.vue'
import HiresSettingsCard from '../components/HiresSettingsCard.vue'
import InitialImageCard from '../components/InitialImageCard.vue'
import PromptCard from '../components/prompt/PromptCard.vue'
import RefinerSettingsCard from '../components/RefinerSettingsCard.vue'
import WanSubHeader from '../components/wan/WanSubHeader.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import SliderField from '../components/ui/SliderField.vue'

const props = defineProps<{ tabId: string; type: EngineType }>()
const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const workflows = useWorkflowsStore()
const upscalersStore = useUpscalersStore()
const { upscalers, loading: upscalersLoading, error: upscalersError, fallbackOnOom, minTile } = storeToRefs(upscalersStore)

// Use unified generation composable
const {
  generate,
  stopStream,
  gallery,
  progress,
  previewImage,
  previewStep,
  errorMessage,
  isRunning,
  lastSeed,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  tab,
  info,
  gentimeMs,
  loadHistory,
  clearHistory,
  resumeNotice,
} = useGeneration(props.tabId)

const leftStack = ref<HTMLElement | null>(null)
const previewStyle = ref<Record<string, string>>({})
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

onMounted(async () => {
  if (!store.tabs.length) store.load()
  void engineCaps.init()
  void upscalersStore.load()
  const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers
  syncPreviewHeight()
  window.addEventListener('resize', syncPreviewHeight)
})

onBeforeUnmount(() => {
  stopStream()
  window.removeEventListener('resize', syncPreviewHeight)
})

const workflowBusy = ref(false)
const { notice: copyNotice, toast, copyJson } = useResultsCard()

watch(
  resumeNotice,
  (msg) => {
    const text = String(msg || '').trim()
    if (!text) return
    toast(text)
    resumeNotice.value = ''
  },
  { immediate: true },
)

const params = computed<ImageBaseParams>(() => (tab.value?.params as any) as ImageBaseParams)
const engineConfig = computed(() => getEngineConfig(props.type))
const resolvedEngineForMode = computed(() => resolveEngineForRequest(props.type, Boolean(params.value.useInitImage)))
const engineSurface = computed(() => engineCaps.get(resolvedEngineForMode.value))

const zimageTurbo = computed(() => props.type === 'zimage' ? Boolean((params.value as any)?.zimageTurbo ?? true) : false)
const supportsNegative = computed(() => engineConfig.value.capabilities.usesNegativePrompt)
const supportsTxt2Img = computed(() => {
  const surf = engineSurface.value
  if (!surf) return false
  return Boolean(surf.supports_txt2img)
})
const supportsImg2Img = computed(() => {
  const surf = engineSurface.value
  if (!surf) return false
  return Boolean(surf.supports_img2img)
})
const canGenerateForCurrentMode = computed(() => (params.value.useInitImage ? supportsImg2Img.value : supportsTxt2Img.value))
const generateDisabledReason = computed(() => {
  if (isRunning.value) return ''
  if (!engineSurface.value) return `Capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (params.value.useInitImage && !supportsImg2Img.value) return `${engineConfig.value.label} does not support img2img.`
  if (!params.value.useInitImage && !supportsTxt2Img.value) return `${engineConfig.value.label} does not support txt2img.`
  return ''
})

const enableAssets = computed(() => true)
const enableStyles = computed(() => true)
const toolbarLabel = computed(() => {
  if (props.type !== 'zimage') return ''
  return zimageTurbo.value ? 'Z Image Turbo' : 'Z Image Base'
})

const cfgLabel = computed(() => (engineConfig.value.capabilities.usesDistilledCfg ? 'Distilled CFG' : 'CFG'))
const showClipSkip = computed(() => props.type === 'sd15' || props.type === 'sdxl' || props.type === 'flux1')
const minClipSkip = computed(() => 0)
const defaultShowNegative = computed(() => props.type === 'sdxl' && supportsNegative.value)

const showHires = computed(() => {
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_hires
})

const showHiresRefiner = computed(() => !Boolean((params.value as any)?.useInitImage))

const showGlobalRefiner = computed(() => {
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_refiner
})

const filteredSamplers = computed(() => {
  const allowed = engineSurface.value?.samplers as string[] | null | undefined
  if (!allowed || allowed.length === 0) return samplers.value
  return samplers.value.filter(s => allowed.includes(s.name))
})

const activeSamplerSpec = computed(() => samplers.value.find(s => s.name === params.value.sampler) ?? null)

const filteredSchedulers = computed(() => {
  let list = schedulers.value
  const allowed = engineSurface.value?.schedulers as string[] | null | undefined
  if (allowed && allowed.length > 0) list = list.filter(s => allowed.includes(s.name))
  const allowedBySampler = activeSamplerSpec.value?.allowed_schedulers
  if (Array.isArray(allowedBySampler) && allowedBySampler.length > 0) {
    const set = new Set(allowedBySampler)
    list = list.filter(s => set.has(s.name))
  }
  return list
})

function onSamplerChange(value: string): void {
  const spec = samplers.value.find(s => s.name === value)
  const scheduler = params.value.scheduler
  if (spec && Array.isArray(spec.allowed_schedulers) && spec.allowed_schedulers.length > 0) {
    if (!spec.allowed_schedulers.includes(scheduler)) {
      setParams({ sampler: value, scheduler: spec.default_scheduler })
      return
    }
  }
  setParams({ sampler: value })
}

watch([() => params.value.sampler, () => params.value.scheduler, samplers], () => {
  const spec = samplers.value.find(s => s.name === params.value.sampler)
  if (!spec || !Array.isArray(spec.allowed_schedulers) || spec.allowed_schedulers.length === 0) return
  if (spec.allowed_schedulers.includes(params.value.scheduler)) return
  setParams({ scheduler: spec.default_scheduler })
}, { immediate: true })

const promptText = computed({
  get: () => params.value.prompt,
  set: (value: string) => setParams({ prompt: value }),
})

const negativeText = computed({
  get: () => params.value.negativePrompt,
  set: (value: string) => {
    if (!supportsNegative.value) return
    setParams({ negativePrompt: value })
  },
})

watch([supportsImg2Img, () => engineCaps.loaded], ([supported, capsLoaded]) => {
  if (!capsLoaded || supported) return
  if (!params.value.useInitImage) return
  setParams({
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  })
}, { immediate: true })

watch(showHires, (show) => {
  if (show) return
  if (!params.value.hires.enabled && !params.value.hires.refiner?.enabled) return
  setHires({
    enabled: false,
    refiner: { ...(params.value.hires.refiner || {}), enabled: false },
  } as any)
})

watch(showGlobalRefiner, (show) => {
  if (show) return
  if (!params.value.refiner.enabled) return
  setRefiner({ enabled: false })
})

watch([supportsImg2Img, showHires, showGlobalRefiner, () => params.value.useInitImage], () => {
  void nextTick(syncPreviewHeight)
})

const images = computed(() => gallery.value)

const gentimeSeconds = computed(() => {
  if (gentimeMs.value == null) return null
  return gentimeMs.value / 1000
})

const progressPercent = computed(() => {
  if (progress.value.percent !== null) return progress.value.percent
  if (!progress.value.totalSteps || progress.value.step === null) return null
  return (progress.value.step / progress.value.totalSteps) * 100
})

const resultsEmptyText = computed(() => {
  if (!isRunning.value) return 'No images yet. Generate to see results here.'
  const stage = String(progress.value.stage || 'starting')
  if (stage === 'starting' || stage === 'submitted' || stage === 'queued') return 'Starting inference…'
  if (progressPercent.value !== null) return `Generating… (${progressPercent.value.toFixed(1)}%)`
  return `Generating… (${stage})`
})

const previewCaption = computed(() => {
  const step = previewStep.value
  if (step !== null && progress.value.totalSteps) return `Live preview · step ${step}/${progress.value.totalSteps}`
  if (step !== null) return `Live preview · step ${step}`
  return 'Live preview'
})

const resolutionPresets = computed((): [number, number][] => {
  if (props.type === 'sd15') return [[512, 512], [512, 768], [768, 512]]
  return [[1024, 1024], [1152, 896], [1216, 832], [1344, 768]]
})

const runSummary = computed(() => {
  const sampler = params.value.sampler || engineSurface.value?.default_sampler || ''
  const scheduler = params.value.scheduler || engineSurface.value?.default_scheduler || ''
  const seedLabel = params.value.seed === -1 ? 'seed random' : `seed ${params.value.seed}`
  return `${params.value.width}×${params.value.height} px · ${params.value.steps} steps · ${cfgLabel.value} ${params.value.cfgScale} · ${sampler} / ${scheduler} · ${seedLabel} · batch ${params.value.batchCount}×${params.value.batchSize}`
})

async function sendToWorkflows(): Promise<void> {
  if (!tab.value) return
  workflowBusy.value = true
  try {
    await workflows.createSnapshot({
      name: `${tab.value.title} — ${new Date().toLocaleString()}`,
      source_tab_id: tab.value.id,
      type: tab.value.type,
      engine_semantics: tab.value.type === 'wan' ? 'wan22' : tab.value.type,
      params_snapshot: tab.value.params as Record<string, unknown>,
    })
    toast('Snapshot saved to Workflows.')
  } catch (e) {
    toast(e instanceof Error ? e.message : String(e))
  } finally {
    workflowBusy.value = false
  }
}

async function copyCurrentParams(): Promise<void> {
  if (!tab.value) return
  await copyJson(tab.value.params, 'Copied params.')
}

async function copyHistoryParams(item: ImageRunHistoryItem): Promise<void> {
  await copyJson(item.paramsSnapshot, 'Copied history params.')
}

function applyHistory(item: ImageRunHistoryItem): void {
  const snap = item.paramsSnapshot as Partial<ImageBaseParams>
  setParams({
    ...(snap as any),
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  })
  toast('Applied history params.')
}

function formatHistoryTitle(item: { mode: string; createdAtMs: number; taskId: string }): string {
  const dt = new Date(item.createdAtMs || Date.now())
  const hh = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const label = item.mode === 'img2img' ? 'Img2Img' : 'Txt2Img'
  return `${label} · ${hh}`
}

function profileStorageKeyFor(type: EngineType): string {
  if (type === 'flux1') return 'codex.flux1.profile.v1'
  if (type === 'sdxl') return 'codex.sdxl.profile.v1'
  if (type === 'zimage') return 'codex.zimage.profile'
  if (type === 'sd15') return 'codex.sd15.profile.v1'
  return `codex.${type}.profile.v1`
}

function loadProfile(): void {
  const key = profileStorageKeyFor(props.type)
  try {
    const raw = localStorage.getItem(key)
    if (!raw) {
      toast('No saved profile found.')
      return
    }

    const snapshot = JSON.parse(raw) as Record<string, unknown>
    const next: Partial<ImageBaseParams> = {}

    const numberOrNull = (value: unknown): number | null => {
      const n = Number(value)
      return Number.isFinite(n) ? n : null
    }

    if (typeof snapshot.prompt === 'string') next.prompt = snapshot.prompt
    if (supportsNegative.value && typeof snapshot.negativePrompt === 'string') next.negativePrompt = snapshot.negativePrompt
    const steps = numberOrNull(snapshot.steps); if (steps !== null) next.steps = Math.max(1, Math.trunc(steps))
    const cfgScale = numberOrNull(snapshot.cfgScale); if (cfgScale !== null) next.cfgScale = cfgScale
    const width = numberOrNull(snapshot.width); if (width !== null) next.width = Math.max(64, Math.trunc(width))
    const height = numberOrNull(snapshot.height); if (height !== null) next.height = Math.max(64, Math.trunc(height))
    const seed = numberOrNull(snapshot.seed); if (seed !== null) next.seed = Math.trunc(seed)
    const clipSkip = numberOrNull(snapshot.clipSkip); if (clipSkip !== null) next.clipSkip = Math.max(minClipSkip.value, Math.trunc(clipSkip))
    const batchSize = numberOrNull(snapshot.batchSize); if (batchSize !== null) next.batchSize = Math.max(1, Math.trunc(batchSize))
    const batchCount = numberOrNull(snapshot.batchCount); if (batchCount !== null) next.batchCount = Math.max(1, Math.trunc(batchCount))

    const selectedModel = typeof snapshot.selectedModel === 'string' ? snapshot.selectedModel : ''
    const selectedSampler = typeof snapshot.selectedSampler === 'string' ? snapshot.selectedSampler : ''
    const selectedScheduler = typeof snapshot.selectedScheduler === 'string' ? snapshot.selectedScheduler : ''

    if (selectedModel) next.checkpoint = selectedModel
    if (selectedSampler) next.sampler = selectedSampler
    if (selectedScheduler) next.scheduler = selectedScheduler

    setParams(next)
    toast('Loaded saved profile.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function saveProfile(): void {
  const key = profileStorageKeyFor(props.type)
  try {
    const snapshot = {
      prompt: params.value.prompt,
      negativePrompt: supportsNegative.value ? params.value.negativePrompt : '',
      steps: params.value.steps,
      cfgScale: params.value.cfgScale,
      width: params.value.width,
      height: params.value.height,
      seed: params.value.seed,
      clipSkip: params.value.clipSkip,
      batchSize: params.value.batchSize,
      batchCount: params.value.batchCount,
      selectedModel: params.value.checkpoint,
      selectedSampler: params.value.sampler,
      selectedScheduler: params.value.scheduler,
    }
    localStorage.setItem(key, JSON.stringify(snapshot))
    toast('Profile saved.')
  } catch (error) {
    toast(error instanceof Error ? error.message : String(error))
  }
}

function setParams(patch: Partial<ImageBaseParams>): void {
  if (!tab.value) return
  store.updateParams(props.tabId, patch as any)
}

function setHires(patch: Record<string, unknown>): void {
  setParams({ hires: { ...(params.value.hires as any), ...patch } as any })
}

function setHiresRefiner(patch: Record<string, unknown>): void {
  const nextRefiner = { ...((params.value.hires as any).refiner || {}), ...patch }
  setHires({ refiner: nextRefiner })
}

function setRefiner(patch: Record<string, unknown>): void {
  setParams({ refiner: { ...(params.value.refiner as any), ...patch } as any })
}

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function setFallbackOnOom(value: boolean): void {
  fallbackOnOom.value = Boolean(value)
}

function setMinTile(value: number): void {
  const v = Math.max(1, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  minTile.value = v
}

const _KONTEXT_DEFAULT_STEPS = 28
const _KONTEXT_DEFAULT_DISTILLED_CFG = 2.5
const _INIT_IMAGE_DIM_MIN = 64
const _INIT_IMAGE_DIM_MAX = 8192
const _INIT_IMAGE_DIM_STEP = 8

function snapInitImageDim(value: number): number {
  const clamped = Math.max(_INIT_IMAGE_DIM_MIN, Math.min(_INIT_IMAGE_DIM_MAX, Math.trunc(value)))
  const snapped = Math.round(clamped / _INIT_IMAGE_DIM_STEP) * _INIT_IMAGE_DIM_STEP
  return Math.max(_INIT_IMAGE_DIM_MIN, Math.min(_INIT_IMAGE_DIM_MAX, snapped))
}

function toggleInitImage(): void {
  const checked = !Boolean(params.value.useInitImage)
  const wasEnabled = Boolean(params.value.useInitImage)
  setParams({ useInitImage: checked })
  if (!checked) {
    setParams({
      initImageData: '',
      initImageName: '',
      useMask: false,
      maskImageData: '',
      maskImageName: '',
    })
    return
  }
  if (!wasEnabled) maybeApplyKontextDefaults()
}

async function onInitFileSet(file: File): Promise<void> {
  const wasEnabled = Boolean(params.value.useInitImage)
  const dataUrl = await readFileAsDataURL(file)
  const patch: Partial<ImageBaseParams> = {
    initImageData: dataUrl,
    initImageName: file.name,
    useInitImage: true,
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  }
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    patch.width = snapInitImageDim(width)
    patch.height = snapInitImageDim(height)
  } catch {
    // ignore: keep current dims
  }
  setParams(patch)
  if (!wasEnabled) maybeApplyKontextDefaults()
}

function clearInit(): void {
  setParams({
    initImageData: '',
    initImageName: '',
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  })
}

function toggleMask(): void {
  if (props.type === 'flux1') return
  const next = !Boolean(params.value.useMask)
  const patch: Partial<ImageBaseParams> = { useMask: next }
  if (!next) {
    patch.maskImageData = ''
    patch.maskImageName = ''
  }
  setParams(patch)
}

async function onMaskFileSet(file: File): Promise<void> {
  if (!params.value.initImageData) {
    toast('Select an initial image before setting a mask.')
    return
  }
  const dataUrl = await readFileAsDataURL(file)
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    if (width !== params.value.width || height !== params.value.height) {
      toast(`Mask size must match init image size: expected ${params.value.width}×${params.value.height}, got ${width}×${height}.`)
      return
    }
  } catch {
    toast('Failed to load mask image.')
    return
  }
  setParams({ useMask: true, maskImageData: dataUrl, maskImageName: file.name })
}

function clearMask(): void {
  setParams({ maskImageData: '', maskImageName: '' })
}

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function randomizeSeed(): void {
  setParams({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) setParams({ seed: lastSeed.value })
}

function download(image: GeneratedImage, index: number): void {
  const link = document.createElement('a')
  link.href = toDataUrl(image)
  link.download = `${props.type}_${index + 1}.png`
  link.click()
}

async function sendToImg2Img(image: GeneratedImage): Promise<void> {
  if (!supportsImg2Img.value) return
  const wasEnabled = Boolean(params.value.useInitImage)
  const dataUrl = toDataUrl(image)
  const patch: Partial<ImageBaseParams> = {
    useInitImage: true,
    initImageData: dataUrl,
    initImageName: `from_${props.type}.png`,
    useMask: false,
    maskImageData: '',
    maskImageName: '',
  }
  try {
    const { width, height } = await readImageDimensions(dataUrl)
    patch.width = snapInitImageDim(width)
    patch.height = snapInitImageDim(height)
  } catch {
    // ignore
  }
  setParams(patch)
  if (!wasEnabled) maybeApplyKontextDefaults()
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader(); reader.onload = () => resolve(String(reader.result)); reader.onerror = () => reject(reader.error); reader.readAsDataURL(file)
  })
}

function readImageDimensions(src: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve({ width: img.naturalWidth || img.width, height: img.naturalHeight || img.height })
    img.onerror = () => reject(new Error('Failed to load image'))
    img.src = src
  })
}

async function syncInitImageDims(): Promise<void> {
  const src = String(params.value.initImageData || '')
  if (!src) return
  try {
    const { width, height } = await readImageDimensions(src)
    setParams({ width: snapInitImageDim(width), height: snapInitImageDim(height) })
  } catch {
    // ignore
  }
}

function maybeApplyKontextDefaults(): void {
  if (props.type !== 'flux1') return
  const defaults = getEngineDefaults('flux1')
  const defaultCfg = defaults.distilledCfg ?? defaults.cfg
  // Only apply when user hasn't customized away from the Flux defaults.
  if (params.value.steps === defaults.steps) setParams({ steps: _KONTEXT_DEFAULT_STEPS })
  if (params.value.cfgScale === defaultCfg) setParams({ cfgScale: _KONTEXT_DEFAULT_DISTILLED_CFG })
}

function syncPreviewHeight(): void {
  const el = leftStack.value
  if (!el) return
  const h = el.getBoundingClientRect().height
  previewStyle.value = { minHeight: `${Math.max(300, Math.floor(h))}px` }
}

defineExpose({ generate })
</script>
