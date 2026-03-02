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
When `useInitImage=true`, generation parameters render through `Img2ImgBasicParametersCard` (hires-like layout only; no img2img hires payload).
CFG Advanced/APG controls are capability-gated (`engineSurface.guidance_advanced`) and persist through tab params/profile snapshots.
Hires settings list upscalers from `/api/upscalers` and share tile controls with `/upscale`.
Also shares the global `min_tile` preference (tiled lower bound) with `/upscale`.
Surfaces a one-shot toast when the generation composable auto-reattaches to an in-flight task after a reload/crash.
Generate CTA and run preflight are capability-driven (`/api/engines/capabilities`) and fail loud when the current mode is unsupported.
Run status in the RUN card is centralized via `RunProgressStatus` variants (progress/error/info/success/warning), so errors are visible even when Prompt is off-screen.

Symbols (top-level; keep in sync; no ghosts):
- `ImageModelTab` (component): Main image model tab view; handles prompt/params/profile persistence, init-image UX, history reuse, and actions.
- `sendToWorkflows` (function): Sends the current params snapshot to the workflows subsystem (async).
- `copyCurrentParams` (function): Copies current params snapshot to clipboard (async).
- `onCancelRun` (function): Cancels the active run (XYZ sweep immediate stop or current image task cancel).
- `copyHistoryParams` (function): Copies a history entry’s params snapshot to clipboard (async).
- `applyHistory` (function): Applies a history entry back into current state (prompt/params/assets).
- `formatHistoryTitle` (function): Builds a human-friendly history title from a run entry.
- `profileStorageKeyFor` (function): Computes the localStorage key for saving/loading per-engine profiles.
- `loadProfile` (function): Loads a saved profile into current params (with validation/defaulting).
- `saveProfile` (function): Saves current params as a profile in localStorage.
- `setParams` (function): Applies partial updates to the current tab params state.
- `normalizeGuidanceAdvancedPatch` (function): Sanitizes/normalizes advanced-guidance payload fragments (profile + UI patch merges).
- `setGuidanceAdvanced` (function): Applies partial advanced-guidance updates into `params.guidanceAdvanced`.
- `setHires` (function): Applies partial updates to the hires config object.
- `setHiresRefiner` (function): Applies partial updates to the hires-refiner config object.
- `setRefiner` (function): Applies partial updates to the refiner config object.
- `clampFloat` (function): Clamps a float to `[min, max]` (input sanitation).
- `setMinTile` (function): Updates the global `min_tile` preference used as the tiled OOM fallback lower bound (hires-fix + `/upscale`).
- `snapInitImageDim` (function): Snaps init-image derived dimensions to model constraints (e.g., multiples of 8).
- `onInitFileSet` (function): Reads an init image file into a data URL and stores name/data, then syncs dims (async).
- `onInitImageRejected` (function): Surfaces dropzone reject reasons for init-image input.
- `clearInit` (function): Clears init image fields.
- `clearMask` (function): Clears mask fields.
- `onMaskEditorApply` (function): Validates and stores an edited mask exported from the inpaint mask editor overlay.
- `onMaskEditorResetNotice` (function): Surfaces inpaint mask editor source-reset notices as toasts.
- `toDataUrl` (function): Converts a generated image payload to a data URL for preview.
- `randomizeSeed` (function): Randomizes the seed field for the current tab params.
- `reuseSeed` (function): Reuses the last seed from history/current run as the next seed.
- `download` (function): Downloads a generated image artifact to disk.
- `sendToImg2Img` (function): Sends a generated image back into img2img init-image fields (async).
- `readFileAsDataURL` (function): Reads a File into a data URL (used for init-image handling).
- `readImageDimensions` (function): Reads width/height from an image source URL (used for init-image dimension sync).
- `syncInitImageDims` (function): Synchronizes init-image derived dimensions into width/height params (async).
- `maskEditorImageWidth`/`maskEditorImageHeight` (const): Derived init-image dimensions used by the inpaint mask editor canvas (keeps backend mask-dimension contract).
- `maybeApplyKontextDefaults` (function): Applies Kontext-specific default params when relevant to the current engine/tab.
- `syncPreviewHeight` (function): Keeps the preview panel height aligned with layout changes (uses DOM measurements).
- `onGenerate` (function): Run handler for the Run card; dispatches standard generation or XYZ sweep depending on XYZ enable state.
- `runGenerateDisabled`/`runGenerateTitle` (const): Run CTA state/title derived from capabilities + active mode + XYZ running/enabled state.
- `missingInpaintMask` (const): Derived guard flag used to disable generation when INPAINT is enabled without an applied mask.
- `xyzSamplerChoices`/`xyzSchedulerChoices` (const): Sampler/scheduler names passed to embedded XYZ autofill (scheduler list is sampler-compatible).
-->

<template>
  <section v-if="tab" class="panels">
    <!-- Left column: Prompt + Parameters -->
    <div class="panel-stack" ref="leftStack">
      <PromptCard
        v-model:prompt="promptText"
        v-model:negative="negativeText"
        :supportsNegative="supportsNegative"
        :token-engine="resolvedEngineForMode"
        :enableAssets="enableAssets"
        :enableStyles="enableStyles"
        :toolbarLabel="toolbarLabel"
        :fieldsId="`image-modeltab-prompt-${tabId}`"
      >
        <div v-if="supportsImg2Img && params.useInitImage" class="panel-section">
          <Img2ImgInpaintParamsCard
            :disabled="isRunning"
            :initImageData="params.initImageData"
            :initImageName="params.initImageName"
            :imageWidth="maskEditorImageWidth"
            :imageHeight="maskEditorImageHeight"
            :useMask="params.useMask"
            :maskImageData="params.maskImageData"
            :maskImageName="params.maskImageName"
            :maskEnforcement="params.maskEnforcement"
            :inpaintingFill="params.inpaintingFill"
            :inpaintFullResPadding="params.inpaintFullResPadding"
            :maskInvert="params.maskInvert"
            :maskRound="params.maskRound"
            :maskBlur="params.maskBlur"
            :maskRegionSplit="params.maskRegionSplit"
            @set:initImage="onInitFileSet"
            @clear:initImage="clearInit"
            @reject:initImage="onInitImageRejected"
            @clear:maskImage="clearMask"
            @apply:maskImageData="onMaskEditorApply"
            @notice:maskEditorReset="onMaskEditorResetNotice"
            @update:maskEnforcement="(v) => setParams({ maskEnforcement: normalizeMaskEnforcement(v) })"
            @update:inpaintingFill="(v) => setParams({ inpaintingFill: normalizeInpaintingFill(v) })"
            @update:inpaintFullResPadding="(v) => setParams({ inpaintFullResPadding: normalizeNonNegativeInt(v) })"
            @toggle:maskInvert="toggleMaskInvert"
            @toggle:maskRound="setParams({ maskRound: !params.maskRound })"
            @update:maskBlur="(v) => setParams({ maskBlur: normalizeNonNegativeInt(v) })"
            @toggle:maskRegionSplit="setParams({ maskRegionSplit: !params.maskRegionSplit })"
          />
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
          <Img2ImgBasicParametersCard
            v-if="params.useInitImage"
            :samplers="filteredSamplers"
            :schedulers="filteredSchedulers"
            :upscalers="upscalers"
            :upscalersLoading="upscalersLoading"
            :upscalersError="upscalersError"
            :sampler="params.sampler"
            :scheduler="params.scheduler"
            :steps="params.steps"
            :width="params.width"
            :height="params.height"
            :cfg-scale="params.cfgScale"
            :cfg-label="cfgLabel"
            :denoise-strength="params.denoiseStrength"
            :seed="params.seed"
            :clip-skip="params.clipSkip"
            :show-clip-skip="showClipSkip"
            :min-clip-skip="minClipSkip"
            :max-clip-skip="12"
            :guidance-advanced="params.guidanceAdvanced"
            :guidance-support="guidanceAdvancedSupport"
            :upscaler="params.img2imgUpscaler"
            :resize-mode="params.img2imgResizeMode"
            :show-init-image-dims="Boolean(params.initImageData)"
            :disabled="isRunning"
            @update:sampler="onSamplerChange"
            @update:scheduler="(v) => setParams({ scheduler: v })"
            @update:steps="(v) => setParams({ steps: Math.max(1, Math.trunc(v)) })"
            @update:width="(v) => setParams({ width: Math.max(64, Math.trunc(v)) })"
            @update:height="(v) => setParams({ height: Math.max(64, Math.trunc(v)) })"
            @update:cfgScale="(v) => setParams({ cfgScale: v })"
            @update:denoiseStrength="(v) => setParams({ denoiseStrength: clampFloat(v, 0, 1) })"
            @update:seed="(v) => setParams({ seed: Math.trunc(v) })"
            @update:clipSkip="(v) => setParams({ clipSkip: Math.max(minClipSkip, Math.trunc(v)) })"
            @update:guidanceAdvanced="setGuidanceAdvanced"
            @update:upscaler="(v) => setParams({ img2imgUpscaler: String(v || '').trim() })"
            @update:resizeMode="(v) => setParams({ img2imgResizeMode: normalizeImg2ImgResizeMode(v) })"
            @random-seed="randomizeSeed"
            @reuse-seed="reuseSeed"
            @sync-init-image-dims="syncInitImageDims"
          />

          <BasicParametersCard
            v-else
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
            :show-denoise="false"
            :denoise-strength="params.denoiseStrength"
            :cfg-label="cfgLabel"
            :show-clip-skip="showClipSkip"
            :min-clip-skip="minClipSkip"
            :max-clip-skip="12"
            :guidance-advanced="params.guidanceAdvanced"
            :guidance-support="guidanceAdvancedSupport"
            :show-init-image-dims="false"
            :disabled="isRunning"
            @update:sampler="onSamplerChange"
            @update:scheduler="(v) => setParams({ scheduler: v })"
            @update:steps="(v) => setParams({ steps: Math.max(1, Math.trunc(v)) })"
            @update:width="(v) => setParams({ width: Math.max(64, Math.trunc(v)) })"
            @update:height="(v) => setParams({ height: Math.max(64, Math.trunc(v)) })"
            @update:cfgScale="(v) => setParams({ cfgScale: v })"
            @update:seed="(v) => setParams({ seed: Math.trunc(v) })"
            @update:clipSkip="(v) => setParams({ clipSkip: Math.max(minClipSkip, Math.trunc(v)) })"
            @update:guidanceAdvanced="setGuidanceAdvanced"
            @random-seed="randomizeSeed"
            @reuse-seed="reuseSeed"
          />

          <HiresSettingsCard
            v-if="showHires"
            :disabled="isRunning"
            :enabled="params.hires.enabled"
            :samplers="filteredSamplers"
            :schedulers="filteredHiresSchedulers"
            :sampler="hiresSampler"
            :scheduler="hiresScheduler"
            :denoise="params.hires.denoise"
            :scale="params.hires.scale"
            :steps="params.hires.steps"
            :cfg-label="cfgLabel"
            :cfg="hiresCfgValue"
            :resize-x="params.hires.resizeX"
            :resize-y="params.hires.resizeY"
            :checkpoint="params.hires.checkpoint"
            :model-choices="swapModelChoices"
            :prompt="params.hires.prompt ?? ''"
            :negative-prompt="params.hires.negativePrompt ?? ''"
            :supports-negative="supportsNegative"
            :upscaler="params.hires.upscaler"
            :tile="params.hires.tile"
            :minTile="minTile"
            :upscalers="upscalers"
            :upscalersLoading="upscalersLoading"
            :upscalersError="upscalersError"
            :base-width="params.width"
            :base-height="params.height"
            :refinerEnabled="showHiresRefiner ? params.hires.refiner?.enabled : undefined"
            :refinerSwapAtStep="showHiresRefiner ? params.hires.refiner?.swapAtStep : undefined"
            :refinerCfg="showHiresRefiner ? params.hires.refiner?.cfg : undefined"
            :refinerSeed="showHiresRefiner ? params.hires.refiner?.seed : undefined"
            :refinerModel="showHiresRefiner ? params.hires.refiner?.model : undefined"
            :refinerModelChoices="showHiresRefiner ? swapModelChoices : undefined"
            @update:enabled="(v) => setHires({ enabled: v })"
            @update:denoise="(v) => setHires({ denoise: clampFloat(v, 0, 1) })"
            @update:scale="(v) => setHires({ scale: clampFloat(v, 1, 4) })"
            @update:steps="(v) => setHires({ steps: Math.max(0, Math.trunc(v)) })"
            @update:cfg="onHiresCfgChange"
            @update:resizeX="(v) => setHires({ resizeX: Math.max(0, Math.trunc(v)) })"
            @update:resizeY="(v) => setHires({ resizeY: Math.max(0, Math.trunc(v)) })"
            @update:checkpoint="(v) => setHires({ checkpoint: String(v || '').trim() || undefined })"
            @update:prompt="(v) => setHires({ prompt: String(v || '') })"
            @update:negativePrompt="(v) => setHires({ negativePrompt: String(v || '') })"
            @update:sampler="onHiresSamplerChange"
            @update:scheduler="onHiresSchedulerChange"
            @update:upscaler="(v) => setHires({ upscaler: v })"
            @update:tile="(v) => setHires({ tile: v })"
            @update:minTile="setMinTile"
            @update:refinerEnabled="(v) => setHiresRefiner({ enabled: v })"
            @update:refinerSwapAtStep="(v) => setHiresRefiner({ swapAtStep: Math.max(1, Math.trunc(v)) })"
            @update:refinerCfg="(v) => setHiresRefiner({ cfg: v })"
            @update:refinerSeed="(v) => setHiresRefiner({ seed: Math.trunc(v) })"
            @update:refinerModel="(v) => setHiresRefiner({ model: v })"
          />

          <RefinerSettingsCard
            v-if="showGlobalRefiner"
            :enabled="params.refiner.enabled"
            :swapAtStep="params.refiner.swapAtStep"
            :cfg="params.refiner.cfg"
            :seed="params.refiner.seed"
            :model="params.refiner.model"
            :modelChoices="swapModelChoices"
            @update:enabled="(v) => setRefiner({ enabled: v })"
            @update:swapAtStep="(v) => setRefiner({ swapAtStep: Math.max(1, Math.trunc(v)) })"
            @update:cfg="(v) => setRefiner({ cfg: v })"
            @update:seed="(v) => setRefiner({ seed: Math.trunc(v) })"
            @update:model="(v) => setRefiner({ model: v })"
          />

          <XyzSweepCard
            :samplers="xyzSamplerChoices"
            :schedulers="xyzSchedulerChoices"
          />
        </div>
      </div>
    </div>

    <!-- Right column: Run + Results -->
    <div class="panel-stack panel-stack--sticky">
      <RunCard
        :generateLabel="generateLabel"
        :generateDisabled="runGenerateDisabled"
        :generateTitle="runGenerateTitle"
        :isRunning="isRunBusy"
        :showBatchControls="true"
        :batchCount="params.batchCount"
        :batchSize="params.batchSize"
        :disabled="isRunBusy"
        @generate="onGenerate"
        @cancel="onCancelRun"
        @update:batchCount="(v) => setParams({ batchCount: Math.max(1, Math.trunc(v)) })"
        @update:batchSize="(v) => setParams({ batchSize: Math.max(1, Math.trunc(v)) })"
      >
        <RunProgressStatus
          v-if="isRunning"
          :stage="progress.stage"
          :percent="progressPercent"
          :step="progress.step"
          :total-steps="progress.totalSteps"
          :eta-seconds="progress.etaSeconds"
        />
        <RunProgressStatus
          v-else-if="xyzRunning"
          stage="xyz sweep"
          title="XYZ sweep"
          :message="xyzStore.progress.current || ''"
          :percent="xyzProgressPercent"
          :step="xyzStore.progress.total ? xyzStore.progress.completed : null"
          :total-steps="xyzStore.progress.total || null"
          :show-progress-bar="Boolean(xyzStore.progress.total)"
        />
        <RunProgressStatus
          v-else-if="errorMessage"
          variant="error"
          title="Run failed"
          :message="errorMessage"
          :show-progress-bar="false"
        />
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
            <button
              v-for="item in history"
              :key="item.taskId"
              type="button"
              :class="['cdx-history-item', { 'is-selected': item.taskId === selectedTaskId }]"
              :aria-label="`Open history details for ${formatHistoryTitle(item)}`"
              @click="openHistoryDetails(item)"
            >
              <img
                v-if="item.thumbnail"
                class="cdx-history-thumb"
                :src="toDataUrl(item.thumbnail)"
                :alt="formatHistoryTitle(item)"
                loading="lazy"
              >
              <div v-else class="cdx-history-thumb cdx-history-thumb--empty">
                <span>No preview</span>
              </div>
            </button>
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
          <template #empty>
            <div class="wan-results-empty">
              <div class="wan-empty-title">
                <template v-if="isRunning">{{ resultsEmptyText }}</template>
                <template v-else>No images yet</template>
              </div>
              <div v-if="!isRunning" class="caption">Generate to see results here.</div>
            </div>
          </template>
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

    <Modal v-model="historyDetailsOpen" :title="historyDetailsTitle">
      <div v-if="historyDetailsItem" class="cdx-history-modal">
        <div class="cdx-history-modal__top">
          <img
            v-if="historyDetailsImageUrl"
            class="cdx-history-modal__preview"
            :src="historyDetailsImageUrl"
            :alt="historyDetailsTitle"
          >
          <div v-else class="cdx-history-modal__preview cdx-history-modal__preview--empty">No preview</div>
          <div class="cdx-history-modal__meta">
            <div class="cdx-history-modal__meta-row"><span>Mode</span><strong>{{ historyDetailsModeLabel }}</strong></div>
            <div class="cdx-history-modal__meta-row"><span>Created</span><strong>{{ historyDetailsCreatedAtLabel }}</strong></div>
            <div class="cdx-history-modal__meta-row"><span>Status</span><strong>{{ historyDetailsItem.status }}</strong></div>
            <div class="cdx-history-modal__meta-row"><span>Task</span><code>{{ historyDetailsItem.taskId }}</code></div>
          </div>
        </div>

        <div class="cdx-history-modal__section">
          <p class="label-muted">Summary</p>
          <p class="cdx-history-modal__summary">{{ historyDetailsItem.summary }}</p>
        </div>

        <div v-if="historyDetailsPrompt" class="cdx-history-modal__section">
          <p class="label-muted">Prompt</p>
          <pre class="text-xs break-words">{{ historyDetailsPrompt }}</pre>
        </div>
        <div v-if="historyDetailsNegativePrompt" class="cdx-history-modal__section">
          <p class="label-muted">Negative Prompt</p>
          <pre class="text-xs break-words">{{ historyDetailsNegativePrompt }}</pre>
        </div>
        <div v-if="historyDetailsItem.errorMessage" class="cdx-history-modal__section">
          <p class="label-muted">Error</p>
          <pre class="text-xs break-words">{{ historyDetailsItem.errorMessage }}</pre>
        </div>
        <details class="accordion">
          <summary>Params snapshot</summary>
          <div class="accordion-body">
            <pre class="text-xs break-words">{{ formatJson(historyDetailsItem.paramsSnapshot) }}</pre>
          </div>
        </details>
      </div>
      <template #footer>
        <button
          class="btn btn-sm btn-secondary"
          type="button"
          :disabled="!historyDetailsItem || isRunning || historyLoadingTaskId === historyDetailsItem.taskId"
          @click="onLoadHistoryDetails"
        >
          {{ historyDetailsItem && historyLoadingTaskId === historyDetailsItem.taskId ? 'Loading…' : 'Load' }}
        </button>
        <button class="btn btn-sm btn-outline" type="button" :disabled="!historyDetailsItem || isRunning" @click="onApplyHistoryDetails">Apply</button>
        <button class="btn btn-sm btn-outline" type="button" :disabled="!historyDetailsItem || isRunning" @click="onCopyHistoryDetails">Copy</button>
        <button class="btn btn-sm btn-outline" type="button" @click="historyDetailsOpen = false">Close</button>
      </template>
    </Modal>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { storeToRefs } from 'pinia'
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { fetchPaths, fetchSamplers, fetchSchedulers } from '../api/client'
import type { GeneratedImage, GuidanceAdvancedCapabilities, SamplerInfo, SchedulerInfo } from '../api/types'
import { formatJson, useResultsCard } from '../composables/useResultsCard'
import { resolveEngineForRequest, useGeneration, type ImageRunHistoryItem } from '../composables/useGeneration'
import {
  defaultImageParamsForType,
  useModelTabsStore,
  type GuidanceAdvancedParams,
  type ImageBaseParams,
  type ImageTabType,
  type TabByType,
} from '../stores/model_tabs'
import { getEngineConfig, getEngineDefaults } from '../stores/engine_config'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useBootstrapStore } from '../stores/bootstrap'
import { useUpscalersStore } from '../stores/upscalers'
import { useWorkflowsStore } from '../stores/workflows'
import { useXyzStore } from '../stores/xyz'
import { normalizeTabFamily } from '../utils/engine_taxonomy'
import { filterModelTitlesForFamily } from '../utils/model_family_filters'
import { normalizeImg2ImgResizeMode } from '../utils/img2img_resize'
import { normalizeInpaintingFill, normalizeMaskEnforcement, normalizeNonNegativeInt, resolveHiresModePolicy } from '../utils/image_params'
import BasicParametersCard from '../components/BasicParametersCard.vue'
import HiresSettingsCard from '../components/HiresSettingsCard.vue'
import Img2ImgBasicParametersCard from '../components/Img2ImgBasicParametersCard.vue'
import Img2ImgInpaintParamsCard from '../components/Img2ImgInpaintParamsCard.vue'
import PromptCard from '../components/prompt/PromptCard.vue'
import RefinerSettingsCard from '../components/RefinerSettingsCard.vue'
import WanSubHeader from '../components/wan/WanSubHeader.vue'
import ResultViewer from '../components/ResultViewer.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunProgressStatus from '../components/results/RunProgressStatus.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import XyzSweepCard from '../components/XyzSweepCard.vue'
import Modal from '../components/ui/Modal.vue'

const props = defineProps<{ tabId: string; type: ImageTabType }>()
const store = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const quicksettingsStore = useQuicksettingsStore()
const bootstrap = useBootstrapStore()
const workflows = useWorkflowsStore()
const upscalersStore = useUpscalersStore()
const xyzStore = useXyzStore()
const { upscalers, loading: upscalersLoading, error: upscalersError, minTile } = storeToRefs(upscalersStore)

// Use unified generation composable
const {
  generate: generateBase,
  cancel: cancelBase,
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
const modelPaths = ref<Record<string, string[]>>({})
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])
const historyDetailsOpen = ref(false)
const historyDetailsItem = ref<ImageRunHistoryItem | null>(null)

onMounted(() => {
  bootstrap
    .runRequired('Failed to initialize image tab controls', async () => {
      await upscalersStore.load({ refresh: true })
      await quicksettingsStore.init()
      const [samp, sched, pathRes] = await Promise.all([fetchSamplers(), fetchSchedulers(), fetchPaths()])
      samplers.value = samp.samplers
      schedulers.value = sched.schedulers
      modelPaths.value = (pathRes.paths || {}) as Record<string, string[]>
    })
    .catch(() => {
      // Fatal state is already set by bootstrap store.
    })
  syncPreviewHeight()
  window.addEventListener('resize', syncPreviewHeight)
})

onBeforeUnmount(() => {
  stopStream()
  window.removeEventListener('resize', syncPreviewHeight)
})

const workflowBusy = ref(false)
const { notice: copyNotice, toast, copyJson } = useResultsCard()
type ImageTab = TabByType<ImageTabType>

const historyDetailsTitle = computed(() => (historyDetailsItem.value ? formatHistoryTitle(historyDetailsItem.value) : 'History details'))
const historyDetailsCreatedAtLabel = computed(() => {
  const timestamp = historyDetailsItem.value?.createdAtMs
  if (!timestamp) return '—'
  return new Date(timestamp).toLocaleString()
})
const historyDetailsModeLabel = computed(() => {
  const mode = historyDetailsItem.value?.mode
  return mode === 'img2img' ? 'Img2Img' : 'Txt2Img'
})
const historyDetailsImageUrl = computed(() => {
  const thumbnail = historyDetailsItem.value?.thumbnail
  return thumbnail ? toDataUrl(thumbnail) : ''
})
const historyDetailsPrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  const prompt = readHistorySnapshotText(item, 'prompt')
  if (prompt) return prompt
  return item.promptPreview || ''
})
const historyDetailsNegativePrompt = computed(() => {
  const item = historyDetailsItem.value
  if (!item) return ''
  return readHistorySnapshotText(item, 'negativePrompt')
})

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

const imageTab = computed<ImageTab | null>(() => {
  const candidate = tab.value
  if (!candidate || candidate.type === 'wan') return null
  return candidate as unknown as ImageTab
})
const fallbackParams = computed<ImageBaseParams>(() => defaultImageParamsForType(props.type))
const params = computed<ImageBaseParams>(() => imageTab.value?.params ?? fallbackParams.value)

const initImageNaturalWidth = ref(0)
const initImageNaturalHeight = ref(0)
const initImageDimsToken = ref(0)

const maskEditorImageWidth = computed(() => {
  if (!String(params.value.initImageData || '').trim()) return params.value.width
  const w = Math.trunc(Number(initImageNaturalWidth.value))
  return Number.isFinite(w) && w > 0 ? w : params.value.width
})
const maskEditorImageHeight = computed(() => {
  if (!String(params.value.initImageData || '').trim()) return params.value.height
  const h = Math.trunc(Number(initImageNaturalHeight.value))
  return Number.isFinite(h) && h > 0 ? h : params.value.height
})

watch(
  () => String(params.value.initImageData || '').trim(),
  (src) => {
    const token = (initImageDimsToken.value += 1)
    if (!src) {
      initImageNaturalWidth.value = 0
      initImageNaturalHeight.value = 0
      return
    }

    readImageDimensions(src)
      .then(({ width, height }) => {
        if (initImageDimsToken.value !== token) return
        const initW = Math.max(0, Math.trunc(width))
        const initH = Math.max(0, Math.trunc(height))
        initImageNaturalWidth.value = initW
        initImageNaturalHeight.value = initH

        const maskSrc = String(params.value.maskImageData || '').trim()
        if (!maskSrc) return
        readImageDimensions(maskSrc)
          .then(({ width: maskW, height: maskH }) => {
            if (initImageDimsToken.value !== token) return
            if (Math.trunc(maskW) === initW && Math.trunc(maskH) === initH) return
            toast(
              `Mask cleared: init image size is ${initW}×${initH}, but mask is ${maskW}×${maskH}. Re-open the editor to reapply.`,
            )
            setParams({ maskImageData: '', maskImageName: '' })
          })
          .catch(() => {
            if (initImageDimsToken.value !== token) return
            toast('Mask cleared: failed to load the stored mask image.')
            setParams({ maskImageData: '', maskImageName: '' })
          })
      })
      .catch(() => {
        if (initImageDimsToken.value !== token) return
        initImageNaturalWidth.value = 0
        initImageNaturalHeight.value = 0
      })
  },
  { immediate: true },
)

const engineConfig = computed(() => getEngineConfig(props.type))
const resolvedEngineForMode = computed(() => resolveEngineForRequest(props.type, Boolean(params.value.useInitImage)))
const engineSurface = computed(() => engineCaps.get(resolvedEngineForMode.value))
const guidanceAdvancedSupport = computed<GuidanceAdvancedCapabilities | null>(() => {
  const guidance = engineSurface.value?.guidance_advanced
  return guidance ?? null
})
const familyCapabilities = computed(() => engineCaps.getFamilyForEngine(resolvedEngineForMode.value))
const dependencyStatus = computed(() => engineCaps.getDependencyStatus(resolvedEngineForMode.value))
const dependencyError = computed(() => engineCaps.firstDependencyError(resolvedEngineForMode.value))
const dependencyReady = computed(() => Boolean(dependencyStatus.value?.ready))

const zimageTurbo = computed(() => props.type === 'zimage' ? Boolean(params.value.zimageTurbo ?? true) : false)
const supportsNegative = computed(() => Boolean(familyCapabilities.value?.supports_negative_prompt))
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
const canGenerateForCurrentMode = computed(() =>
  dependencyReady.value
  && Boolean(familyCapabilities.value)
  && (params.value.useInitImage ? supportsImg2Img.value : supportsTxt2Img.value),
)
const generateDisabledReason = computed(() => {
  if (isRunning.value) return ''
  if (!dependencyStatus.value) return `Dependency checks for '${resolvedEngineForMode.value}' are not available.`
  if (!dependencyStatus.value.ready) return dependencyError.value || `Dependencies for '${resolvedEngineForMode.value}' are not ready.`
  if (!engineSurface.value) return `Capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (!familyCapabilities.value) return `Family capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (params.value.useInitImage && !supportsImg2Img.value) return `${engineConfig.value.label} does not support img2img.`
  if (!params.value.useInitImage && !supportsTxt2Img.value) return `${engineConfig.value.label} does not support txt2img.`
  return ''
})
const xyzSamplerChoices = computed(() => filteredSamplers.value.map((entry) => entry.name))
const xyzSchedulerChoices = computed(() => filteredSchedulers.value.map((entry) => entry.name))
const xyzRunning = computed(() => xyzStore.status === 'running')
const isRunBusy = computed(() => isRunning.value || xyzRunning.value)
const generateLabel = computed(() => (xyzStore.enabled ? 'Generate xyz' : 'Generate'))
const xyzProgressPercent = computed(() => {
  if (!xyzStore.progress.total) return null
  return (xyzStore.progress.completed / xyzStore.progress.total) * 100
})
const missingInpaintMask = computed(() =>
  Boolean(params.value.useInitImage)
  && Boolean(params.value.useMask)
  && !String(params.value.maskImageData || '').trim(),
)
const runGenerateDisabled = computed(() => {
  if (isRunBusy.value) return true
  if (xyzStore.enabled) {
    return !(dependencyReady.value && Boolean(familyCapabilities.value) && supportsTxt2Img.value)
  }
  if (missingInpaintMask.value) return true
  return !canGenerateForCurrentMode.value
})
const runGenerateTitle = computed(() => {
  if (xyzRunning.value) return 'XYZ sweep is running.'
  if (!xyzStore.enabled) {
    if (missingInpaintMask.value) return 'INPAINT is enabled but no mask is applied. Open the mask editor and apply a mask.'
    return generateDisabledReason.value
  }
  if (!dependencyStatus.value) return `Dependency checks for '${resolvedEngineForMode.value}' are not available.`
  if (!dependencyStatus.value.ready) return dependencyError.value || `Dependencies for '${resolvedEngineForMode.value}' are not ready.`
  if (!engineSurface.value) return `Capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (!familyCapabilities.value) return `Family capabilities for '${resolvedEngineForMode.value}' are not loaded.`
  if (!supportsTxt2Img.value) return `${engineConfig.value.label} does not support txt2img.`
  return ''
})

const enableAssets = computed(() => true)
const enableStyles = computed(() => true)
const toolbarLabel = computed(() => {
  if (props.type !== 'zimage') return ''
  return zimageTurbo.value ? 'Z Image Turbo' : 'Z Image Base'
})

const cfgLabel = computed(() => (engineConfig.value.capabilities.usesDistilledCfg ? 'Distilled CFG' : 'CFG'))
const showClipSkip = computed(() => Boolean(familyCapabilities.value?.shows_clip_skip))
const minClipSkip = computed(() => 0)
const swapModelChoices = computed(() => {
  const family = normalizeTabFamily(props.type)
  if (!family || family === 'wan') return []
  return filterModelTitlesForFamily(quicksettingsStore.models, family, modelPaths.value)
})

const supportsHiresForEngine = computed(() => {
  if (props.type === 'zimage') return false
  const surf = engineSurface.value
  if (!surf) return true
  return surf.supports_hires
})
const hiresModePolicy = computed(() => resolveHiresModePolicy(Boolean(params.value.useInitImage), supportsHiresForEngine.value))
const showHires = computed(() => hiresModePolicy.value.showCard)

const showHiresRefiner = computed(() => !Boolean(params.value.useInitImage))

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

const hiresSampler = computed(() => {
  const override = String(params.value.hires.sampler || '').trim()
  if (override) return override
  return params.value.sampler
})

const hiresScheduler = computed(() => {
  const override = String(params.value.hires.scheduler || '').trim()
  if (override) return override
  return params.value.scheduler
})

const hiresCfgValue = computed(() => {
  if (engineConfig.value.capabilities.usesDistilledCfg) {
    const value = Number(params.value.hires.distilledCfg)
    if (Number.isFinite(value)) return value
    return params.value.cfgScale
  }
  const value = Number(params.value.hires.cfg)
  if (Number.isFinite(value)) return value
  return params.value.cfgScale
})

const filteredHiresSchedulers = computed(() => {
  let list = schedulers.value
  const allowed = engineSurface.value?.schedulers as string[] | null | undefined
  if (allowed && allowed.length > 0) list = list.filter((entry) => allowed.includes(entry.name))
  const spec = samplers.value.find((entry) => entry.name === hiresSampler.value)
  const allowedBySampler = spec?.allowed_schedulers
  if (Array.isArray(allowedBySampler) && allowedBySampler.length > 0) {
    const allowedSet = new Set(allowedBySampler)
    list = list.filter((entry) => allowedSet.has(entry.name))
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

function onHiresSamplerChange(value: string): void {
  const spec = samplers.value.find((entry) => entry.name === value)
  const currentScheduler = hiresScheduler.value
  if (spec && Array.isArray(spec.allowed_schedulers) && spec.allowed_schedulers.length > 0) {
    if (!spec.allowed_schedulers.includes(currentScheduler)) {
      setHires({ sampler: value, scheduler: spec.default_scheduler })
      return
    }
  }
  setHires({ sampler: value })
}

function onHiresSchedulerChange(value: string): void {
  setHires({ scheduler: value })
}

function onHiresCfgChange(value: number): void {
  const normalized = clampFloat(value, 0, 30)
  if (engineConfig.value.capabilities.usesDistilledCfg) {
    setHires({ distilledCfg: normalized, cfg: undefined })
    return
  }
  setHires({ cfg: normalized, distilledCfg: undefined })
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

watch(() => hiresModePolicy.value.resetState, (shouldReset) => {
  if (!shouldReset) return
  if (!params.value.hires.enabled && !params.value.hires.refiner?.enabled) return
  setHires({ enabled: false })
  setHiresRefiner({ enabled: false })
})

watch(showGlobalRefiner, (show) => {
  if (show) return
  if (!params.value.refiner.enabled) return
  setRefiner({ enabled: false })
})

watch([supportsImg2Img, showHires, showGlobalRefiner, () => params.value.useInitImage], () => {
  void nextTick(syncPreviewHeight)
})

watch(
  () => params.value.useInitImage,
  (enabled, wasEnabled) => {
    if (!enabled || wasEnabled) return
    maybeApplyKontextDefaults()
  },
)

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

async function onGenerate(): Promise<void> {
  if (xyzStore.enabled) {
    await xyzStore.run()
    return
  }
  await generateBase()
}

async function onCancelRun(): Promise<void> {
  try {
    if (xyzRunning.value) {
      await xyzStore.stop('immediate')
      return
    }
    await cancelBase()
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

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

function openHistoryDetails(item: ImageRunHistoryItem): void {
  historyDetailsItem.value = item
  historyDetailsOpen.value = true
}

async function onLoadHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await loadHistory(item.taskId)
}

function onApplyHistoryDetails(): void {
  const item = historyDetailsItem.value
  if (!item) return
  applyHistory(item)
}

async function onCopyHistoryDetails(): Promise<void> {
  const item = historyDetailsItem.value
  if (!item) return
  await copyHistoryParams(item)
}

function applyHistory(item: ImageRunHistoryItem): void {
  const snap = item.paramsSnapshot as Partial<ImageBaseParams>
  setParams({
    ...snap,
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

function readHistorySnapshotText(item: ImageRunHistoryItem, key: string): string {
  const snapshot = item.paramsSnapshot
  if (!snapshot || typeof snapshot !== 'object') return ''
  const value = (snapshot as Record<string, unknown>)[key]
  if (typeof value !== 'string') return ''
  return value.trim()
}

function profileStorageKeyFor(type: ImageTabType): string {
  if (type === 'flux1') return 'codex.flux1.profile.v1'
  if (type === 'sdxl') return 'codex.sdxl.profile.v1'
  if (type === 'zimage') return 'codex.zimage.profile'
  if (type === 'sd15') return 'codex.sd15.profile.v1'
  return `codex.${type}.profile.v1`
}

function normalizeGuidanceAdvancedPatch(raw: unknown, base: GuidanceAdvancedParams): GuidanceAdvancedParams {
  const source = raw && typeof raw === 'object' && !Array.isArray(raw)
    ? (raw as Record<string, unknown>)
    : {}
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
  return {
    enabled: typeof source.enabled === 'boolean' ? source.enabled : base.enabled,
    apgEnabled: typeof source.apgEnabled === 'boolean' ? source.apgEnabled : base.apgEnabled,
    apgStartStep: clampInt(source.apgStartStep, base.apgStartStep, 0),
    apgEta: clamp(source.apgEta, base.apgEta),
    apgMomentum: clamp(source.apgMomentum, base.apgMomentum, 0, 0.99),
    apgNormThreshold: clamp(source.apgNormThreshold, base.apgNormThreshold, 0),
    apgRescale: clamp(source.apgRescale, base.apgRescale, 0, 1),
    guidanceRescale: clamp(source.guidanceRescale, base.guidanceRescale, 0, 1),
    cfgTruncEnabled: typeof source.cfgTruncEnabled === 'boolean' ? source.cfgTruncEnabled : base.cfgTruncEnabled,
    cfgTruncRatio: clamp(source.cfgTruncRatio, base.cfgTruncRatio, 0, 1),
    renormCfg: clamp(source.renormCfg, base.renormCfg, 0),
  }
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
    if (snapshot.guidanceAdvanced && typeof snapshot.guidanceAdvanced === 'object') {
      next.guidanceAdvanced = normalizeGuidanceAdvancedPatch(snapshot.guidanceAdvanced, params.value.guidanceAdvanced)
    }

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
      guidanceAdvanced: params.value.guidanceAdvanced,
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
  store.updateParams(props.tabId, patch as Partial<Record<string, unknown>>).catch((error) => {
    toast(error instanceof Error ? error.message : String(error))
  })
}

function toggleMaskInvert(): void {
  const next = !params.value.maskInvert
  setParams({
    maskInvert: next,
    maskRegionSplit: next ? false : params.value.maskRegionSplit,
  })
}

function setGuidanceAdvanced(patch: Partial<GuidanceAdvancedParams>): void {
  const next = normalizeGuidanceAdvancedPatch(
    patch,
    params.value.guidanceAdvanced,
  )
  setParams({ guidanceAdvanced: next })
}

function setHires(patch: Partial<ImageBaseParams['hires']>): void {
  setParams({ hires: { ...params.value.hires, ...patch } })
}

function setHiresRefiner(patch: Partial<NonNullable<ImageBaseParams['hires']['refiner']>>): void {
  const nextRefiner = {
    enabled: false,
    swapAtStep: 1,
    cfg: 3.5,
    seed: -1,
    model: undefined,
    ...(params.value.hires.refiner || {}),
    ...patch,
  }
  const swapAtStep = Number(nextRefiner.swapAtStep)
  nextRefiner.swapAtStep = Number.isFinite(swapAtStep) && swapAtStep >= 1 ? Math.trunc(swapAtStep) : 1
  setHires({ refiner: nextRefiner })
}

function setRefiner(patch: Partial<ImageBaseParams['refiner']>): void {
  const nextRefiner = { ...params.value.refiner, ...patch }
  const swapAtStep = Number(nextRefiner.swapAtStep)
  nextRefiner.swapAtStep = Number.isFinite(swapAtStep) && swapAtStep >= 1 ? Math.trunc(swapAtStep) : 1
  setParams({ refiner: nextRefiner })
}

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
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

async function onInitFileSet(file: File): Promise<void> {
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
}

function onInitImageRejected(payload: { reason: string; files: File[] }): void {
  const fileName = payload.files[0]?.name || 'file'
  toast(`Init image rejected (${fileName}): ${payload.reason}`)
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

function clearMask(): void {
  setParams({ maskImageData: '', maskImageName: '' })
}

async function onMaskEditorApply(maskDataUrl: string): Promise<void> {
  if (!params.value.initImageData) {
    toast('Select an initial image before editing a mask.')
    return
  }

  let initDims: { width: number; height: number }
  try {
    initDims = await readImageDimensions(params.value.initImageData)
  } catch {
    toast('Failed to load init image for mask validation.')
    return
  }

  try {
    const { width, height } = await readImageDimensions(maskDataUrl)
    if (width !== initDims.width || height !== initDims.height) {
      toast(`Mask size must match init image size: expected ${initDims.width}×${initDims.height}, got ${width}×${height}.`)
      return
    }
  } catch {
    toast('Failed to load edited mask image.')
    return
  }

  setParams({ useMask: true, maskImageData: maskDataUrl, maskImageName: 'edited-mask.png' })
}

function onMaskEditorResetNotice(message: string): void {
  const text = String(message || '').trim()
  if (!text) return
  toast(text)
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

defineExpose({ generate: onGenerate })
</script>
