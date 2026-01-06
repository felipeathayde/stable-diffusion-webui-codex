<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN video generation tab (txt2vid/img2vid/vid2vid) UI.
Owns prompt + init media inputs, stage params, assets selection, guided-generation overlay, and history; submits tasks via `/api/*` and
renders progress/results via task events.

Symbols (top-level; keep in sync; no ghosts):
- `WANTab` (component): WAN video tab view; handles input modes, generation start/queue, history apply/reuse, and guided-generation UX.
- `WanAssetsParams` (interface): Minimal WAN assets triple (metadata dir + text encoder + VAE) used for validation and payload building.
- `GuidedStep` (type): Guided-generation step definition (message + CSS selector to highlight/focus).
- `AspectMode` (type): Aspect ratio mode presets for width/height controls.
- `normalizePath` (function): Normalizes paths for stable comparisons (used by root filtering and UI label handling).
- `fileInRoots` (function): Checks whether a file path is under any configured root (used to constrain selectable WAN assets).
- `defaultStage` (function): Returns default WAN stage params (high/low) for new tabs/resets.
- `defaultVideo` (function): Returns default video params (prompt/dims/init media fields) for new tabs/resets.
- `defaultAssets` (function): Returns default (empty) assets selection.
- `setVideo` (function): Applies partial updates to the video params in state (triggers dependent sync where needed).
- `setHigh` (function): Applies partial updates to the high stage (and can drive low-stage sync when enabled).
- `setLow` (function): Applies partial updates to the low stage.
- `syncLowFromHighIfNeeded` (function): Keeps low stage params aligned with high stage when the “low follows high” toggle is enabled.
- `onLowFollowsHighChange` (function): Toggles low-follow behavior and applies an immediate sync.
- `toggleLowNoise` (function): Toggles low-stage noise-related behavior/flags.
- `toInt` (function): Parses an integer from an `<input>` event with fallback.
- `onInitImageFile` (function): Reads an init image file into a data URL and stores name/data for img2vid (async).
- `clearInit` (function): Clears init image fields.
- `onGenerateClick` (function): Starts a generation run for the current input mode (builds payload, submits, and wires streaming) (async).
- `onInitVideoFile` (function): Handles vid2vid init-video selection and preview state.
- `clearInitVideo` (function): Clears init video selection/preview state.
- `clampNumber` (function): Clamps a numeric value to `[min, max]`.
- `computeGuidedTooltipPosition` (function): Computes tooltip position for guided-generation overlay based on current highlight rect.
- `isFocusable` (function): Type guard for focusable DOM elements.
- `findFocusTarget` (function): Resolves the element to focus for a guided step (selector + fallbacks).
- `clearGuidedHighlight` (function): Clears guided highlight/tooltip state.
- `updateGuidedRect` (function): Recomputes the guided highlight rectangle from DOM measurements.
- `scheduleGuidedRectUpdate` (function): Schedules highlight-rect recomputation (debounced via timers/rAF).
- `scheduleGuidedSettleUpdate` (function): Schedules a “settle” recompute after layout/scroll changes.
- `stopGuided` (function): Stops the guided-generation flow and removes transient UI state/listeners.
- `focusGuided` (function): Scrolls/focuses the UI control for a guided step.
- `startGuided` (function): Starts guided-generation flow (initial step + listeners + rect scheduling).
- `onGuidedGenEvent` (function): Handles guided-generation events emitted by other UI surfaces.
- `onWanModeChangeEvent` (function): Handles WAN mode change events (syncs tab input mode/state).
- `setInputMode` (function): Sets the tab input mode and resets/validates init-media state for that mode.
- `buildCurrentSnapshot` (function): Builds a JSON-serializable snapshot of current params (used for history/clipboard/workflows).
- `copyCurrentParams` (function): Copies current params snapshot to clipboard (async).
- `copyInfo` (function): Copies current run info/metadata to clipboard (async).
- `copyHistoryParams` (function): Copies a history entry’s params snapshot to clipboard (async).
- `queueNext` (function): Queues a next run based on current params/history (async).
- `applyHistory` (function): Applies a history entry back into current state (prompt/params/assets).
- `reuseLast` (function): Convenience helper to reuse the most recent history entry.
- `isRecord` (function): Type guard for `Record<string, unknown>`.
- `formatDiffValue` (function): Formats values for the “params diff” UI.
- `diffObjects` (function): Recursively diffs two objects into `{path, before, after}` entries (used for history diff).
- `snapDim` (function): Snaps a dimension to model constraints (e.g., multiple-of-8).
- `ratioForMode` (function): Returns the target aspect ratio for a given `AspectMode` preset.
- `onAspectModeChange` (function): Applies aspect-mode changes and updates width/height accordingly.
- `applyWidth` (function): Applies width updates (snapping + aspect-mode handling).
- `applyHeight` (function): Applies height updates (snapping + aspect-mode handling).
- `sendToWorkflows` (function): Sends the current snapshot into the workflows subsystem (async).
- `toDataUrl` (function): Converts a generated image payload to a data URL for preview.
- `formatHistoryTitle` (function): Builds a human-friendly history title from a run entry.
- `readFileAsDataURL` (function): Reads a File into a data URL (used by init-image handling).
-->

<template>
  <section v-if="tab" class="panels wan-panels">
    <div class="panel-stack">
      <PromptCard v-model:prompt="videoPrompt" v-model:negative="videoNegative" fieldsId="wan-guided-prompt">
        <div v-if="mode !== 'txt2vid'" class="gen-card">
          <div class="row-split">
            <span class="label-muted">Input</span>
            <span class="caption">Mode is set in QuickSettings.</span>
          </div>
          <div v-if="mode === 'img2vid'" id="wan-guided-init-image" class="mt-2">
            <InitialImageCard
              label="Image"
              :disabled="isRunning"
              :src="video.initImageData"
              :hasImage="Boolean(video.initImageData)"
              @set="onInitImageFile"
              @clear="clearInit"
            >
              <template #footer>
                <div v-if="video.initImageName" class="caption mt-1">{{ video.initImageName }}</div>
              </template>
            </InitialImageCard>
          </div>
          <div v-else id="wan-guided-init-video" class="mt-2">
            <InitialVideoCard
              label="Video"
              :disabled="isRunning"
              :src="initVideoPreviewUrl"
              :hasVideo="Boolean(initVideoPreviewUrl)"
              @set="onInitVideoFile"
              @clear="clearInitVideo"
            >
              <template #footer>
                <div class="cdx-form-grid mt-2">
                  <div>
                    <label class="label-muted">Video path (optional)</label>
                    <input class="ui-input" type="text" :disabled="isRunning" :value="video.initVideoPath" placeholder="relative/path/to/video.mp4" @change="setVideo({ initVideoPath: ($event.target as HTMLInputElement).value })" />
                    <p class="caption mt-1">Paths are restricted server-side; upload is recommended.</p>
                  </div>
                  <div>
                    <label class="label-muted">Selected file</label>
                    <div class="caption break-words">{{ video.initVideoName || 'None' }}</div>
                  </div>
                </div>
              </template>
            </InitialVideoCard>
          </div>
        </div>

        <div v-if="errorMessage" class="panel-error">{{ errorMessage }}</div>
      </PromptCard>

      <div class="panel">
        <div class="panel-header">Generation Parameters</div>
        <div class="panel-body">
          <div class="gen-card">
            <WanSubHeader title="Video" />
            <div class="gc-row">
              <SliderField
                class="gc-col gc-col--wide"
                label="Width (px)"
                :modelValue="video.width"
                :min="64"
                :max="2048"
                :step="64"
                :inputStep="8"
                :nudgeStep="8"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                @update:modelValue="applyWidth"
              >
                <template #right>
                  <NumberStepperInput
                    :modelValue="video.width"
                    :min="64"
                    :max="2048"
                    :step="8"
                    :nudgeStep="8"
                    inputClass="cdx-input-w-md"
                    :disabled="isRunning"
                    @update:modelValue="applyWidth"
                  />
                  <select
                    class="ui-input ui-input-sm select-md cdx-input-w-sm"
                    :disabled="isRunning"
                    :value="aspectMode"
                    aria-label="Aspect ratio"
                    title="Aspect ratio"
                    @change="onAspectModeChange"
                  >
                    <option value="free">Free</option>
                    <option value="current">Lock</option>
                    <option value="16:9">16:9</option>
                    <option value="1:1">1:1</option>
                    <option value="9:16">9:16</option>
                    <option value="4:3">4:3</option>
                    <option value="3:4">3:4</option>
                  </select>
                </template>
                <template #below>
                  <span v-if="aspectMode !== 'free'" class="caption">Keeps ratio while editing width/height.</span>
                </template>
              </SliderField>
              <SliderField
                class="gc-col gc-col--wide"
                label="Height (px)"
                :modelValue="video.height"
                :min="64"
                :max="2048"
                :step="64"
                :inputStep="8"
                :nudgeStep="8"
                :disabled="isRunning"
                inputClass="cdx-input-w-md"
                @update:modelValue="applyHeight"
              />
            </div>
            <VideoSettingsCard
              embedded
              :frames="video.frames"
              :fps="video.fps"
              @update:frames="(v:number)=>setVideo({ frames: v })"
              @update:fps="(v:number)=>setVideo({ fps: v })"
            />
          </div>

          <div class="gen-card">
            <WanSubHeader title="Video Output" />
            <WanVideoOutputPanel embedded :video="video" :disabled="isRunning" @update:video="setVideo" />
          </div>

          <div v-if="mode === 'vid2vid'" class="gen-card">
            <WanSubHeader title="Video2Video" />
            <div class="gc-row">
              <div class="gc-col">
                <label class="label-muted">Strength</label>
                <input class="ui-input" type="number" min="0" max="1" step="0.05" :disabled="isRunning" :value="video.vid2vidStrength" @change="setVideo({ vid2vidStrength: Number(($event.target as HTMLInputElement).value) })" />
                <p class="caption mt-1">Higher = more change. Lower = closer to source video.</p>
              </div>
              <div class="gc-col">
                <label class="label-muted">Method</label>
                <select class="select-md" :disabled="isRunning" :value="video.vid2vidMethod" @change="setVideo({ vid2vidMethod: (($event.target as HTMLSelectElement).value === 'native' ? 'native' : 'flow_chunks') })">
                  <option value="flow_chunks">Flow chunks (GGUF-friendly)</option>
                  <option value="native">Native (Diffusers video input)</option>
                </select>
              </div>
              <div class="gc-col">
                <label class="label-muted">Chunk Frames</label>
                <input class="ui-input" type="number" min="2" max="128" :disabled="isRunning" :value="video.vid2vidChunkFrames" @change="setVideo({ vid2vidChunkFrames: toInt($event, video.vid2vidChunkFrames) })" />
              </div>
              <div class="gc-col">
                <label class="label-muted">Overlap</label>
                <input class="ui-input" type="number" min="0" max="127" :disabled="isRunning" :value="video.vid2vidOverlapFrames" @change="setVideo({ vid2vidOverlapFrames: toInt($event, video.vid2vidOverlapFrames) })" />
              </div>
            </div>
            <div class="cdx-form-row">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.vid2vidUseSourceFps ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="isRunning"
                :aria-pressed="video.vid2vidUseSourceFps"
                @click="setVideo({ vid2vidUseSourceFps: !video.vid2vidUseSourceFps })"
              >
                Use source FPS
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.vid2vidUseSourceFrames ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="isRunning"
                :aria-pressed="video.vid2vidUseSourceFrames"
                @click="setVideo({ vid2vidUseSourceFrames: !video.vid2vidUseSourceFrames })"
              >
                Use source length
              </button>
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', video.vid2vidFlowEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="isRunning"
                :aria-pressed="video.vid2vidFlowEnabled"
                @click="setVideo({ vid2vidFlowEnabled: !video.vid2vidFlowEnabled })"
              >
                Optical flow
              </button>
            </div>
            <div v-if="video.vid2vidFlowEnabled" class="gc-row">
              <div class="gc-col">
                <label class="label-muted">Flow downscale</label>
                <input class="ui-input" type="number" min="1" max="8" :disabled="isRunning" :value="video.vid2vidFlowDownscale" @change="setVideo({ vid2vidFlowDownscale: toInt($event, video.vid2vidFlowDownscale) })" />
                <p class="caption mt-1">Higher = faster/rougher. 2 is a good default.</p>
              </div>
              <div class="gc-col">
                <label class="label-muted">Model</label>
                <select class="select-md" :disabled="isRunning" :value="video.vid2vidFlowUseLarge ? 'large' : 'small'" @change="setVideo({ vid2vidFlowUseLarge: (($event.target as HTMLSelectElement).value === 'large') })">
                  <option value="small">RAFT small</option>
                  <option value="large">RAFT large</option>
                </select>
              </div>
              <div class="gc-col">
                <label class="label-muted">Preview frames</label>
                <input class="ui-input" type="number" min="1" max="512" :disabled="isRunning" :value="video.vid2vidPreviewFrames" @change="setVideo({ vid2vidPreviewFrames: toInt($event, video.vid2vidPreviewFrames) })" />
                <p class="caption mt-1">UI preview only; full video is exported to disk.</p>
              </div>
            </div>
          </div>

          <div id="wan-guided-high-stage" class="gen-card">
            <WanSubHeader title="High Noise" />
            <WanStagePanel
              title="High Noise"
              embedded
              :stage="high"
              :samplers="samplers"
              :schedulers="schedulers"
              :lightx2v="lightx2v"
              :lora-choices="wanLoraChoices"
              :disabled="isRunning"
              @update:stage="setHigh"
            />
          </div>

          <div class="gen-card">
            <WanSubHeader title="Low Noise">
              <button
                :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', lowFollowsHigh ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                type="button"
                :disabled="isRunning"
                :aria-pressed="lowFollowsHigh"
                @click="onLowFollowsHighChange(!lowFollowsHigh)"
              >
                Use High settings
              </button>
              <button class="btn-icon" type="button" :aria-expanded="lowNoiseOpen ? 'true' : 'false'" :title="lowNoiseOpen ? 'Collapse' : 'Expand'" aria-label="Toggle Low Noise" @click="toggleLowNoise">
                <span aria-hidden="true">{{ lowNoiseOpen ? '▾' : '▸' }}</span>
              </button>
            </WanSubHeader>
            <div v-if="lowFollowsHigh" class="caption">Low stage mirrors High (sampler/scheduler/steps/CFG/seed/LoRA).</div>
            <div v-if="lowNoiseOpen" class="mt-2" id="wan-guided-low-stage">
              <WanStagePanel
                title="Low Noise"
                embedded
                :stage="low"
                :samplers="samplers"
                :schedulers="schedulers"
                :lightx2v="lightx2v"
                :lora-choices="wanLoraChoices"
                :disabled="isRunning || lowFollowsHigh"
                @update:stage="setLow"
              />
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right column: Results -->
    <div class="panel-stack">
      <RunCard
        :isRunning="isRunning"
        :generateDisabled="isRunning"
        :generateTitle="!canGenerate ? 'Guided gen: click to see what is missing.' : ''"
        generateId="wan-guided-generate"
        :showBatchControls="false"
        @generate="onGenerateClick"
      >
        <template #header-right>
          <div class="wan-header-actions">
            <button
              v-if="isRunning"
              class="btn btn-sm btn-secondary"
              type="button"
              :disabled="queue.length >= queueMax || !canGenerate"
              :title="queue.length >= queueMax ? `Queue full (max ${queueMax}).` : (!canGenerate ? blockedReason : '')"
              @click="queueNext"
            >
              Queue ({{ queue.length }}/{{ queueMax }})
            </button>
            <button v-else-if="history.length" class="btn btn-sm btn-secondary" type="button" :disabled="isRunning" @click="reuseLast">
              Reuse last
            </button>
            <button v-if="isRunning" class="btn btn-sm btn-secondary" type="button" :disabled="cancelRequested" @click="cancel()">
              {{ cancelRequested ? 'Cancelling…' : 'Cancel' }}
            </button>
          </div>
        </template>

        <div v-if="copyNotice" class="caption">{{ copyNotice }}</div>
        <RunSummaryChips class="wan-results-summary" :text="runSummary" />
      </RunCard>

      <ResultsCard
        class="wan-results-panel"
        headerClass="three-cols"
        headerRightClass="wan-header-actions"
        :showGenerate="false"
      >
        <template #header-right>
          <button class="btn btn-sm btn-outline" type="button" :disabled="workflowBusy" @click="sendToWorkflows">
            {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
          </button>
          <button class="btn btn-sm btn-outline" type="button" @click="copyCurrentParams">Copy params</button>
        </template>

        <div v-if="isRunning" class="panel-progress">
          <p><strong>Stage:</strong> {{ progress.stage }}</p>
          <p v-if="progress.percent !== null">Progress: {{ progress.percent.toFixed(1) }}%</p>
          <progress v-if="progress.percent !== null" class="wan-progress" :value="progress.percent" max="100"></progress>
          <p v-if="progress.step !== null && progress.totalSteps !== null">
            Step {{ progress.step }} / {{ progress.totalSteps }}
          </p>
          <p v-if="progress.etaSeconds !== null" class="caption">ETA ~ {{ progress.etaSeconds.toFixed(0) }}s</p>
          <div v-if="queue.length" class="wan-callout-actions mt-2">
            <span class="caption">Queued: {{ queue.length }} / {{ queueMax }}</span>
            <button class="btn btn-sm btn-ghost" type="button" @click="clearQueue">Clear queue</button>
          </div>
        </div>

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

          <details v-if="diffText" class="accordion">
            <summary>Diff vs previous run</summary>
            <div class="accordion-body">
              <pre class="text-xs break-words">{{ diffText }}</pre>
            </div>
          </details>
        </div>

        <div v-if="videoUrl" class="gen-card mb-3">
          <div class="row-split">
            <span class="label-muted">Exported Video</span>
            <a class="btn btn-sm btn-outline" :href="videoUrl" target="_blank" rel="noreferrer">Open</a>
          </div>
          <video class="w-full rounded" :src="videoUrl" controls />
          <p class="caption mt-1">Tip: if playback fails, install ffmpeg and ensure CODEX_ROOT/output is writable.</p>
        </div>
        <ResultViewer mode="video" :frames="framesResult" :toDataUrl="toDataUrl" emptyText="No results yet.">
          <template #empty>
            <div class="wan-results-empty">
              <div class="wan-empty-title">{{ isRunning ? 'Generating…' : 'No results yet' }}</div>
              <div v-if="isRunning" class="caption">
                Stage: {{ progress.stage }}
                <template v-if="progress.percent !== null"> · {{ progress.percent.toFixed(1) }}%</template>
              </div>
              <div v-else class="caption">Need help? Press Generate to see what is missing.</div>
            </div>
          </template>
        </ResultViewer>

        <div v-if="info" class="gen-card mt-3">
          <div class="row-split">
            <span class="label-muted">Generation Info</span>
            <div class="wan-header-actions">
              <button class="btn btn-sm btn-outline" type="button" @click="copyInfo">Copy info</button>
            </div>
          </div>
          <pre class="text-xs break-words">{{ formatJson(info) }}</pre>
        </div>
      </ResultsCard>
    </div>

    <Teleport to="body">
      <div
        v-if="guidedActive && guidedRect"
        ref="guidedTooltipEl"
        class="codex-guided-tooltip"
        :data-placement="guidedTooltipPlacement"
        :style="guidedTooltipStyle"
      >
        <div class="codex-guided-tooltip-title">Guided gen</div>
        <div class="codex-guided-tooltip-body">{{ guidedMessage }}</div>
        <div class="codex-guided-tooltip-actions">
          <button class="btn btn-sm btn-secondary" type="button" @click="stopGuided">Close</button>
        </div>
      </div>
    </Teleport>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab not found.</div></div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, computed, ref, watch, nextTick } from 'vue'
import { useModelTabsStore, type WanStageParams, type WanVideoParams } from '../stores/model_tabs'
import type { SamplerInfo, SchedulerInfo, GeneratedImage } from '../api/types'
import { fetchSamplers, fetchSchedulers, fetchLoras, fetchPaths } from '../api/client'
import ResultViewer from '../components/ResultViewer.vue'
import InitialImageCard from '../components/InitialImageCard.vue'
import InitialVideoCard from '../components/InitialVideoCard.vue'
import VideoSettingsCard from '../components/VideoSettingsCard.vue'
import ResultsCard from '../components/results/ResultsCard.vue'
import RunCard from '../components/results/RunCard.vue'
import RunSummaryChips from '../components/results/RunSummaryChips.vue'
import SliderField from '../components/ui/SliderField.vue'
import PromptCard from '../components/prompt/PromptCard.vue'
import WanStagePanel from '../components/wan/WanStagePanel.vue'
import WanSubHeader from '../components/wan/WanSubHeader.vue'
import WanVideoOutputPanel from '../components/wan/WanVideoOutputPanel.vue'
import { useVideoGeneration, type VideoRunHistoryItem } from '../composables/useVideoGeneration'
import { useResultsCard } from '../composables/useResultsCard'
import { useWorkflowsStore } from '../stores/workflows'
import NumberStepperInput from '../components/ui/NumberStepperInput.vue'

const props = defineProps<{ tabId: string }>()
const store = useModelTabsStore()
const workflows = useWorkflowsStore()

// Load option lists
const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])
const wanLoras = ref<Array<{ name: string; path: string }>>([])

onMounted(async () => {
  if (!store.tabs.length) store.load()
  const [samp, sched, pathsRes, lorasRes] = await Promise.all([
    fetchSamplers(),
    fetchSchedulers(),
    fetchPaths().catch(() => ({ paths: {} as Record<string, string[]> })),
    fetchLoras().catch(() => ({ loras: [] as Array<{ name: string; path: string }> })),
  ])
  samplers.value = samp.samplers
  schedulers.value = sched.schedulers

  const roots = Array.isArray((pathsRes as any)?.paths?.wan22_loras) ? ((pathsRes as any).paths.wan22_loras as string[]) : []
  wanLoras.value = (lorasRes.loras || []).filter((l) => fileInRoots(l.path, roots))
})

const tab = computed(() => store.tabs.find(t => t.id === props.tabId) || null)
const lightx2v = computed<boolean>(() => Boolean((tab.value?.params as any)?.lightx2v))
const wanLoraChoices = computed(() => wanLoras.value)

function normalizePath(path: string): string {
  return String(path || '').replace(/\\+/g, '/').replace(/\/+$/, '')
}

function fileInRoots(file: string, roots: string[]): boolean {
  const fNorm = normalizePath(file)
  if (!fNorm) return false
  for (const root of roots || []) {
    const rNorm = normalizePath(root)
    if (!rNorm) continue
    if (fNorm === rNorm || fNorm.startsWith(rNorm + '/')) return true
    const rel = rNorm.startsWith('/') ? rNorm.slice(1) : rNorm
    if (fNorm.includes('/' + rel + '/') || fNorm.endsWith('/' + rel)) return true
  }
  return false
}

function defaultStage(): WanStageParams {
  return { modelDir: '', sampler: '', scheduler: '', steps: 30, cfgScale: 7, seed: -1, loraPath: '', loraWeight: 1.0 }
}
function defaultVideo(): WanVideoParams {
  return {
    prompt: '',
    negativePrompt: '',
    width: 768,
    height: 432,
    fps: 24,
    frames: 16,
    useInitImage: false,
    initImageData: '',
    initImageName: '',
    useInitVideo: false,
    initVideoPath: '',
    initVideoName: '',
    vid2vidStrength: 0.8,
    vid2vidMethod: 'flow_chunks',
    vid2vidUseSourceFps: true,
    vid2vidUseSourceFrames: true,
    vid2vidChunkFrames: 16,
    vid2vidOverlapFrames: 4,
    vid2vidPreviewFrames: 48,
    vid2vidFlowEnabled: true,
    vid2vidFlowUseLarge: false,
    vid2vidFlowDownscale: 2,
    filenamePrefix: 'wan22',
    format: 'video/h264-mp4',
    pixFmt: 'yuv420p',
    crf: 15,
    loopCount: 0,
    pingpong: false,
    trimToAudio: false,
    saveMetadata: true,
    saveOutput: true,
    rifeEnabled: true,
    rifeModel: 'rife47.pth',
    rifeTimes: 2,
  }
}

const video = computed<WanVideoParams>(() => ((tab.value?.params as any)?.video as WanVideoParams) || defaultVideo())
const high = computed<WanStageParams>(() => ((tab.value?.params as any)?.high as WanStageParams) || defaultStage())
const low = computed<WanStageParams>(() => ((tab.value?.params as any)?.low as WanStageParams) || defaultStage())

interface WanAssetsParams { metadata: string; textEncoder: string; vae: string }
function defaultAssets(): WanAssetsParams { return { metadata: '', textEncoder: '', vae: '' } }

const assets = computed<WanAssetsParams>(() => ((tab.value?.params as any)?.assets as WanAssetsParams) || defaultAssets())

function setVideo(patch: Partial<WanVideoParams>): void {
  if (!tab.value) return
  const current = (tab.value.params as any).video as WanVideoParams
  store.updateParams(props.tabId, { video: { ...current, ...patch } })
}
function setHigh(patch: Partial<WanStageParams>): void {
  if (!tab.value) return
  const current = (tab.value.params as any).high as WanStageParams
  store.updateParams(props.tabId, { high: { ...current, ...patch } })
}
function setLow(patch: Partial<WanStageParams>): void {
  if (!tab.value) return
  const current = (tab.value.params as any).low as WanStageParams
  store.updateParams(props.tabId, { low: { ...current, ...patch } })
}

const lowFollowsHigh = computed<boolean>(() => Boolean((tab.value?.params as any)?.lowFollowsHigh))
const lowNoiseOpen = ref(true)

function syncLowFromHighIfNeeded(): void {
  const patch: Partial<WanStageParams> = {
    sampler: high.value.sampler,
    scheduler: high.value.scheduler,
    steps: high.value.steps,
    cfgScale: high.value.cfgScale,
    seed: high.value.seed,
    loraPath: high.value.loraPath,
    loraWeight: high.value.loraWeight,
  }
  const needsUpdate = Object.entries(patch).some(([k, v]) => (low.value as any)[k] !== v)
  if (!needsUpdate) return
  setLow(patch)
}

function onLowFollowsHighChange(enabled: boolean): void {
  if (!tab.value) return
  if (!enabled) {
    store.updateParams(props.tabId, { lowFollowsHigh: false } as any)
    return
  }

  const nextLow: Partial<WanStageParams> = {
    sampler: high.value.sampler,
    scheduler: high.value.scheduler,
    steps: high.value.steps,
    cfgScale: high.value.cfgScale,
    seed: high.value.seed,
    loraPath: high.value.loraPath,
    loraWeight: high.value.loraWeight,
  }
  store.updateParams(props.tabId, { lowFollowsHigh: true, low: { ...(low.value as any), ...nextLow } } as any)
}

function toggleLowNoise(): void {
  lowNoiseOpen.value = !lowNoiseOpen.value
}

watch(
  () => ([
    lowFollowsHigh.value,
    high.value.sampler,
    high.value.scheduler,
    high.value.steps,
    high.value.cfgScale,
    high.value.seed,
    high.value.loraPath,
    high.value.loraWeight,
  ] as const),
  ([enabled]) => {
    if (!enabled) return
    syncLowFromHighIfNeeded()
  },
)

watch(
  () => ([
    lowFollowsHigh.value,
    low.value.sampler,
    low.value.scheduler,
    low.value.steps,
    low.value.cfgScale,
    low.value.seed,
    low.value.loraPath,
    low.value.loraWeight,
  ] as const),
  ([enabled]) => {
    if (!enabled) return
    syncLowFromHighIfNeeded()
  },
)

const videoPrompt = computed({
  get: () => video.value.prompt,
  set: (value: string) => setVideo({ prompt: value }),
})

const videoNegative = computed({
  get: () => video.value.negativePrompt,
  set: (value: string) => setVideo({ negativePrompt: value }),
})

function toInt(e: Event, fallback: number): number { const v = Number((e.target as HTMLInputElement).value); return Number.isFinite(v) ? Math.trunc(v) : fallback }

async function onInitImageFile(file: File): Promise<void> {
  const dataUrl = await readFileAsDataURL(file)
  setVideo({ initImageData: dataUrl, initImageName: file.name, useInitImage: true })
}

function clearInit(): void { setVideo({ initImageData: '', initImageName: '' }) }

// Generation wiring (composable)
const {
  generate,
  isRunning,
  canGenerate,
  blockedReason,
  cancel,
  cancelRequested,
  progress,
  frames: framesResult,
  info,
  videoUrl,
  errorMessage,
  mode,
  history,
  selectedTaskId,
  historyLoadingTaskId,
  loadHistory,
  clearHistory,
  queue,
  queueMax,
  enqueue,
  clearQueue,
  setInitVideoFile,
  clearInitVideoFile,
} = useVideoGeneration(props.tabId)

async function onGenerateClick(): Promise<void> {
  if (isRunning.value) return
  if (!canGenerate.value) {
    startGuided()
    return
  }
  stopGuided()
  await generate()
}

const initVideoPreviewUrl = ref('')

function onInitVideoFile(file: File): void {
  try {
    if (initVideoPreviewUrl.value) URL.revokeObjectURL(initVideoPreviewUrl.value)
  } catch { /* ignore */ }
  initVideoPreviewUrl.value = URL.createObjectURL(file)
  setInitVideoFile(file)
  setVideo({ useInitVideo: true, initVideoName: file.name, initVideoPath: '' })
}

function clearInitVideo(): void {
  clearInitVideoFile()
  try {
    if (initVideoPreviewUrl.value) URL.revokeObjectURL(initVideoPreviewUrl.value)
  } catch { /* ignore */ }
  initVideoPreviewUrl.value = ''
  setVideo({ initVideoName: '', initVideoPath: '' })
}

onBeforeUnmount(() => {
  try {
    if (initVideoPreviewUrl.value) URL.revokeObjectURL(initVideoPreviewUrl.value)
  } catch { /* ignore */ }
})

const { notice: copyNotice, toast, copyJson, formatJson } = useResultsCard()

type GuidedStep = { id: string; message: string; selector: string; focusSelector?: string }
const guidedActive = ref(false)
const guidedMessage = ref('')
const guidedRect = ref<DOMRect | null>(null)
const guidedCurrentId = ref('')
let guidedHighlightedEl: HTMLElement | null = null
let guidedRaf: number | null = null
let guidedSettleTimer: number | null = null

const guidedTooltipEl = ref<HTMLElement | null>(null)
const guidedTooltipPos = ref<{ left: number; top: number; placement: 'top' | 'bottom' } | null>(null)

function clampNumber(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function computeGuidedTooltipPosition(): void {
  const rect = guidedRect.value
  const el = guidedTooltipEl.value
  if (!rect || !el) {
    guidedTooltipPos.value = null
    return
  }

  const tooltipW = el.offsetWidth || 0
  const tooltipH = el.offsetHeight || 0
  if (tooltipW <= 0 || tooltipH <= 0) {
    guidedTooltipPos.value = null
    return
  }

  const margin = 12
  const spaceAbove = rect.top
  const spaceBelow = window.innerHeight - rect.bottom
  const placement: 'top' | 'bottom' = (spaceBelow >= tooltipH + margin || spaceBelow >= spaceAbove) ? 'bottom' : 'top'

  const centerX = rect.left + rect.width / 2
  const rawLeft = centerX - tooltipW / 2
  const left = clampNumber(rawLeft, margin, window.innerWidth - margin - tooltipW)

  const rawTop = placement === 'bottom' ? (rect.bottom + 10) : (rect.top - 10 - tooltipH)
  const top = clampNumber(rawTop, margin, window.innerHeight - margin - tooltipH)

  guidedTooltipPos.value = { left, top, placement }
}

const guidedTooltipPlacement = computed<'top' | 'bottom'>(() => guidedTooltipPos.value?.placement || 'bottom')
const guidedTooltipStyle = computed<Record<string, string>>(() => {
  const pos = guidedTooltipPos.value
  if (!pos) return { left: '0px', top: '0px', opacity: '0' }
  return { left: `${Math.round(pos.left)}px`, top: `${Math.round(pos.top)}px`, opacity: '1' }
})

function isFocusable(el: Element | null): el is HTMLElement {
  if (!(el instanceof HTMLElement)) return false
  const tag = el.tagName.toLowerCase()
  if (tag === 'input' || tag === 'select' || tag === 'textarea' || tag === 'button') return true
  if (el.getAttribute('contenteditable') === 'true') return true
  return typeof el.focus === 'function'
}

function findFocusTarget(root: HTMLElement, selector?: string): HTMLElement | null {
  if (selector) {
    const el = document.querySelector(selector)
    return isFocusable(el) ? el : null
  }
  if (isFocusable(root)) return root
  const inside = root.querySelector('input,select,textarea,button,[contenteditable=\"true\"]')
  return isFocusable(inside) ? inside : null
}

function clearGuidedHighlight(): void {
  if (guidedHighlightedEl) guidedHighlightedEl.classList.remove('codex-guided-attention')
  guidedHighlightedEl = null
}

function updateGuidedRect(): void {
  if (!guidedHighlightedEl) {
    guidedRect.value = null
    return
  }
  guidedRect.value = guidedHighlightedEl.getBoundingClientRect()
}

function scheduleGuidedRectUpdate(): void {
  if (guidedRaf !== null) return
  guidedRaf = window.requestAnimationFrame(() => {
    guidedRaf = null
    updateGuidedRect()
    computeGuidedTooltipPosition()
  })
}

function scheduleGuidedSettleUpdate(): void {
  if (guidedSettleTimer !== null) window.clearTimeout(guidedSettleTimer)
  guidedSettleTimer = window.setTimeout(() => {
    guidedSettleTimer = null
    updateGuidedRect()
    computeGuidedTooltipPosition()
  }, 250)
}

function stopGuided(): void {
  guidedActive.value = false
  guidedMessage.value = ''
  guidedRect.value = null
  guidedTooltipPos.value = null
  guidedCurrentId.value = ''
  clearGuidedHighlight()
  if (guidedSettleTimer !== null) window.clearTimeout(guidedSettleTimer)
  guidedSettleTimer = null
}

function focusGuided(step: GuidedStep): void {
  const target = document.querySelector(step.selector) as HTMLElement | null
  if (!target) return

  const focusEl = findFocusTarget(target, step.focusSelector) || target
  clearGuidedHighlight()
  guidedHighlightedEl = focusEl
  guidedHighlightedEl.classList.add('codex-guided-attention')

  guidedMessage.value = step.message
  guidedCurrentId.value = step.id
  guidedHighlightedEl.scrollIntoView({ behavior: 'smooth', block: 'center' })
  try {
    guidedHighlightedEl.focus({ preventScroll: true } as any)
  } catch {
    try { guidedHighlightedEl.focus() } catch { /* ignore */ }
  }
  updateGuidedRect()
  scheduleGuidedRectUpdate()
  scheduleGuidedSettleUpdate()
}

function startGuided(): void {
  guidedActive.value = true
}

const guidedSteps = computed<GuidedStep[]>(() => {
  const steps: GuidedStep[] = []

  const prompt = String(video.value.prompt || '').trim()
  if (!prompt) {
    steps.push({
      id: 'prompt',
      message: 'Write a prompt to generate.',
      selector: '#wan-guided-prompt',
      focusSelector: '#wan-guided-prompt [contenteditable=\"true\"]',
    })
    return steps
  }

  if (!high.value.modelDir && !low.value.modelDir) {
    steps.push({
      id: 'wan_models',
      message: 'Select WAN High/Low models in QuickSettings (header).',
      selector: '#qs-wan-high',
    })
    return steps
  }

  if (mode.value === 'img2vid' && !video.value.initImageData) {
    steps.push({
      id: 'init_image',
      message: 'Image mode needs an input image. Upload one (or switch to Text mode).',
      selector: '#wan-guided-init-image',
    })
    return steps
  }

  if (mode.value === 'vid2vid') {
    const path = String(video.value.initVideoPath || '').trim()
    const hasFile = Boolean(initVideoPreviewUrl.value) || Boolean(video.value.initVideoName)
    if (!hasFile && !path) {
      steps.push({
        id: 'init_video',
        message: 'Video mode needs an input video. Upload a file (or provide a path).',
        selector: '#wan-guided-init-video',
      })
      return steps
    }
  }

  return steps
})

watch(guidedActive, (active) => {
  if (active) {
    window.addEventListener('scroll', scheduleGuidedRectUpdate, true)
    window.addEventListener('resize', scheduleGuidedRectUpdate)
    scheduleGuidedRectUpdate()
  } else {
    window.removeEventListener('scroll', scheduleGuidedRectUpdate, true)
    window.removeEventListener('resize', scheduleGuidedRectUpdate)
    if (guidedRaf !== null) window.cancelAnimationFrame(guidedRaf)
    guidedRaf = null
  }
})

watch(isRunning, (running) => {
  if (running) stopGuided()
})

watch([guidedActive, guidedSteps], async ([active, steps]) => {
  if (!active) return
  await nextTick()

  if (!steps.length) {
    focusGuided({
      id: 'ready',
      message: 'Ready. Click Generate.',
      selector: '#wan-guided-generate',
      focusSelector: '#wan-guided-generate',
    })
    return
  }

  const step = steps[0]!
  if (step.id === guidedCurrentId.value && guidedRect.value) return
  focusGuided(step)
}, { deep: true })

function onGuidedGenEvent(event: Event): void {
  const e = event as CustomEvent<{ tabId?: string }>
  if (e.detail?.tabId && e.detail.tabId !== props.tabId) return
  startGuided()
}

function onWanModeChangeEvent(event: Event): void {
  const e = event as CustomEvent<{ tabId?: string; mode?: string }>
  if (e.detail?.tabId && e.detail.tabId !== props.tabId) return
  const raw = String(e.detail?.mode || '').trim().toLowerCase()
  const next: 'txt2vid' | 'img2vid' | 'vid2vid' = raw === 'vid2vid' ? 'vid2vid' : (raw === 'img2vid' ? 'img2vid' : 'txt2vid')
  setInputMode(next)
}

onMounted(() => {
  window.addEventListener('codex-wan-guided-gen', onGuidedGenEvent as EventListener)
  window.addEventListener('codex-wan-mode-change', onWanModeChangeEvent as EventListener)
})

onBeforeUnmount(() => {
  window.removeEventListener('codex-wan-guided-gen', onGuidedGenEvent as EventListener)
  window.removeEventListener('codex-wan-mode-change', onWanModeChangeEvent as EventListener)
  stopGuided()
})

function setInputMode(next: 'txt2vid' | 'img2vid' | 'vid2vid'): void {
  if (isRunning.value) return
  if (next === 'txt2vid') {
    clearInitVideo()
    setVideo({ useInitVideo: false, initVideoName: '', initVideoPath: '', useInitImage: false, initImageData: '', initImageName: '' })
    return
  }
  if (next === 'img2vid') {
    clearInitVideo()
    setVideo({ useInitVideo: false, initVideoName: '', initVideoPath: '', useInitImage: true })
    return
  }
  // vid2vid
  setVideo({ useInitVideo: true, useInitImage: false, initImageData: '', initImageName: '' })
}

const durationLabel = computed(() => {
  const fps = Number(video.value.fps) || 0
  const frames = Number(video.value.frames) || 0
  if (fps <= 0) return '0.00'
  return (frames / fps).toFixed(2)
})

const runSummary = computed(() => {
  const v = video.value
  const highStage = high.value
  const lowStage = low.value
  const base = `${mode.value} · ${v.width}×${v.height} px · ${v.frames} frames @ ${v.fps} fps (~ ${durationLabel.value}s) · High ${highStage.steps} steps · CFG ${highStage.cfgScale} · Low ${lowStage.steps} steps · CFG ${lowStage.cfgScale}`
  return lightx2v.value ? `${base} · lightx2v` : base
})

function buildCurrentSnapshot(): Record<string, unknown> {
  return {
    mode: video.value.useInitVideo ? 'vid2vid' : (video.value.useInitImage ? 'img2vid' : 'txt2vid'),
    initImageName: video.value.initImageName || '',
    initVideoName: video.value.initVideoName || '',
    initVideoPath: video.value.initVideoPath || '',
    vid2vid: {
      strength: video.value.vid2vidStrength,
      method: video.value.vid2vidMethod,
      useSourceFps: video.value.vid2vidUseSourceFps,
      useSourceFrames: video.value.vid2vidUseSourceFrames,
      chunkFrames: video.value.vid2vidChunkFrames,
      overlapFrames: video.value.vid2vidOverlapFrames,
      previewFrames: video.value.vid2vidPreviewFrames,
      flowEnabled: video.value.vid2vidFlowEnabled,
      flowUseLarge: video.value.vid2vidFlowUseLarge,
      flowDownscale: video.value.vid2vidFlowDownscale,
    },
    prompt: String(video.value.prompt || ''),
    negativePrompt: String(video.value.negativePrompt || ''),
    width: video.value.width,
    height: video.value.height,
    frames: video.value.frames,
    fps: video.value.fps,
    lightx2v: lightx2v.value,
    assets: {
      metadata: String(assets.value.metadata || ''),
      textEncoder: String(assets.value.textEncoder || ''),
      vae: String(assets.value.vae || ''),
    },
    high: {
      modelDir: high.value.modelDir,
      sampler: high.value.sampler,
      scheduler: high.value.scheduler,
      steps: high.value.steps,
      cfgScale: high.value.cfgScale,
      seed: high.value.seed,
      loraPath: lightx2v.value ? high.value.loraPath : '',
      loraWeight: high.value.loraWeight,
    },
    low: {
      modelDir: low.value.modelDir,
      sampler: low.value.sampler,
      scheduler: low.value.scheduler,
      steps: low.value.steps,
      cfgScale: low.value.cfgScale,
      seed: low.value.seed,
      loraPath: lightx2v.value ? low.value.loraPath : '',
      loraWeight: low.value.loraWeight,
    },
    output: {
      filenamePrefix: video.value.filenamePrefix,
      format: video.value.format,
      pixFmt: video.value.pixFmt,
      crf: video.value.crf,
      loopCount: video.value.loopCount,
      pingpong: video.value.pingpong,
      trimToAudio: video.value.trimToAudio,
      saveMetadata: video.value.saveMetadata,
      saveOutput: video.value.saveOutput,
    },
    interpolation: {
      enabled: video.value.rifeEnabled,
      model: video.value.rifeModel,
      times: video.value.rifeTimes,
    },
  }
}

async function copyCurrentParams(): Promise<void> {
  await copyJson(buildCurrentSnapshot(), 'Copied current params JSON.')
}

async function copyInfo(): Promise<void> {
  await copyJson(info.value, 'Copied info JSON.')
}

async function copyHistoryParams(item: VideoRunHistoryItem): Promise<void> {
  await copyJson(item.paramsSnapshot, 'Copied history params JSON.')
}

async function queueNext(): Promise<void> {
  try {
    await enqueue()
    toast('Queued next run.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  }
}

function applyHistory(item: VideoRunHistoryItem): void {
  const snap = (item.paramsSnapshot || {}) as any

  const rawMode = String(snap.mode || '').toLowerCase()
  const nextMode: 'txt2vid' | 'img2vid' | 'vid2vid' = rawMode === 'vid2vid' ? 'vid2vid' : (rawMode === 'img2vid' ? 'img2vid' : 'txt2vid')
  setInputMode(nextMode)

  const output = snap.output || {}
  const interpolation = snap.interpolation || {}
  const v2v = snap.vid2vid || {}

  setVideo({
    prompt: String(snap.prompt || ''),
    negativePrompt: String(snap.negativePrompt || ''),
    width: Number(snap.width) || video.value.width,
    height: Number(snap.height) || video.value.height,
    frames: Number(snap.frames) || video.value.frames,
    fps: Number(snap.fps) || video.value.fps,
    initVideoName: String(snap.initVideoName || video.value.initVideoName),
    initVideoPath: String(snap.initVideoPath || video.value.initVideoPath),
    vid2vidStrength: Number.isFinite(v2v.strength) ? Number(v2v.strength) : video.value.vid2vidStrength,
    vid2vidMethod: (String(v2v.method || '').toLowerCase() === 'native' ? 'native' : 'flow_chunks'),
    vid2vidUseSourceFps: typeof v2v.useSourceFps === 'boolean' ? Boolean(v2v.useSourceFps) : video.value.vid2vidUseSourceFps,
    vid2vidUseSourceFrames: typeof v2v.useSourceFrames === 'boolean' ? Boolean(v2v.useSourceFrames) : video.value.vid2vidUseSourceFrames,
    vid2vidChunkFrames: Number.isFinite(v2v.chunkFrames) ? Number(v2v.chunkFrames) : video.value.vid2vidChunkFrames,
    vid2vidOverlapFrames: Number.isFinite(v2v.overlapFrames) ? Number(v2v.overlapFrames) : video.value.vid2vidOverlapFrames,
    vid2vidPreviewFrames: Number.isFinite(v2v.previewFrames) ? Number(v2v.previewFrames) : video.value.vid2vidPreviewFrames,
    vid2vidFlowEnabled: typeof v2v.flowEnabled === 'boolean' ? Boolean(v2v.flowEnabled) : video.value.vid2vidFlowEnabled,
    vid2vidFlowUseLarge: typeof v2v.flowUseLarge === 'boolean' ? Boolean(v2v.flowUseLarge) : video.value.vid2vidFlowUseLarge,
    vid2vidFlowDownscale: Number.isFinite(v2v.flowDownscale) ? Number(v2v.flowDownscale) : video.value.vid2vidFlowDownscale,
    filenamePrefix: String(output.filenamePrefix || video.value.filenamePrefix),
    format: String(output.format || video.value.format),
    pixFmt: String(output.pixFmt || video.value.pixFmt),
    crf: Number.isFinite(output.crf) ? Number(output.crf) : video.value.crf,
    loopCount: Number.isFinite(output.loopCount) ? Number(output.loopCount) : video.value.loopCount,
    pingpong: Boolean(output.pingpong),
    trimToAudio: Boolean(output.trimToAudio),
    saveMetadata: Boolean(output.saveMetadata),
    saveOutput: Boolean(output.saveOutput),
    rifeEnabled: Boolean(interpolation.enabled),
    rifeModel: String(interpolation.model || ''),
    rifeTimes: Number.isFinite(interpolation.times) ? Number(interpolation.times) : video.value.rifeTimes,
  })

  const hi = snap.high || {}
  const lo = snap.low || {}
  const snapLightx2v =
    typeof snap.lightx2v === 'boolean'
      ? Boolean(snap.lightx2v)
      : Boolean((hi as any).loraEnabled || (lo as any).loraEnabled || (hi as any).loraPath || (lo as any).loraPath)
  store.updateParams(props.tabId, { lightx2v: snapLightx2v } as any)

  const snapAssets = snap.assets || {}
  if (snapAssets && typeof snapAssets === 'object') {
    store.updateParams(props.tabId, { assets: { ...assets.value, ...snapAssets } })
  }

  setHigh({
    modelDir: String(hi.modelDir || ''),
    sampler: String(hi.sampler || ''),
    scheduler: String(hi.scheduler || ''),
    steps: Number(hi.steps) || high.value.steps,
    cfgScale: Number(hi.cfgScale) || high.value.cfgScale,
    seed: Number.isFinite(hi.seed) ? Number(hi.seed) : high.value.seed,
    loraPath: snapLightx2v ? String(hi.loraPath || '') : '',
    loraWeight: Number.isFinite(hi.loraWeight) ? Number(hi.loraWeight) : high.value.loraWeight,
  })

  setLow({
    modelDir: String(lo.modelDir || ''),
    sampler: String(lo.sampler || ''),
    scheduler: String(lo.scheduler || ''),
    steps: Number(lo.steps) || low.value.steps,
    cfgScale: Number(lo.cfgScale) || low.value.cfgScale,
    seed: Number.isFinite(lo.seed) ? Number(lo.seed) : low.value.seed,
    loraPath: snapLightx2v ? String(lo.loraPath || '') : '',
    loraWeight: Number.isFinite(lo.loraWeight) ? Number(lo.loraWeight) : low.value.loraWeight,
  })

  toast('Applied params from history.')
}

function reuseLast(): void {
  if (!history.value.length) return
  applyHistory(history.value[0] as VideoRunHistoryItem)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function formatDiffValue(value: unknown): string {
  if (typeof value === 'string') {
    const v = value.length > 160 ? value.slice(0, 160) + '…' : value
    return JSON.stringify(v)
  }
  if (typeof value === 'number' || typeof value === 'boolean' || value === null || value === undefined) {
    return String(value)
  }
  try {
    const raw = JSON.stringify(value)
    if (raw.length > 180) return raw.slice(0, 180) + '…'
    return raw
  } catch {
    return String(value)
  }
}

function diffObjects(before: unknown, after: unknown, prefix = '', out: Array<{ path: string; before: unknown; after: unknown }> = []): Array<{ path: string; before: unknown; after: unknown }> {
  if (out.length > 80) return out
  if (before === after) return out

  const aObj = isRecord(before)
  const bObj = isRecord(after)
  if (aObj && bObj) {
    const keys = new Set([...Object.keys(before), ...Object.keys(after)])
    for (const k of keys) {
      const nextPrefix = prefix ? `${prefix}.${k}` : k
      diffObjects((before as any)[k], (after as any)[k], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  if (Array.isArray(before) && Array.isArray(after)) {
    const max = Math.max(before.length, after.length)
    for (let i = 0; i < max; i++) {
      const nextPrefix = `${prefix}[${i}]`
      diffObjects(before[i], after[i], nextPrefix, out)
      if (out.length > 80) break
    }
    return out
  }

  out.push({ path: prefix || '(root)', before, after })
  return out
}

const selectedHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const id = String(selectedTaskId.value || '')
  if (!id) return null
  return (history.value as VideoRunHistoryItem[]).find((h) => h.taskId === id) || null
})

const previousHistoryItem = computed<VideoRunHistoryItem | null>(() => {
  const selected = selectedHistoryItem.value
  if (!selected) return null
  const idx = (history.value as VideoRunHistoryItem[]).findIndex((h) => h.taskId === selected.taskId)
  if (idx < 0) return null
  return (history.value as VideoRunHistoryItem[])[idx + 1] || null
})

const diffText = computed(() => {
  const selected = selectedHistoryItem.value
  const prev = previousHistoryItem.value
  if (!selected || !prev) return ''

  const rows = diffObjects(prev.paramsSnapshot, selected.paramsSnapshot)
  if (!rows.length) return ''

  return rows
    .map((r) => `${r.path}: ${formatDiffValue(r.before)} → ${formatDiffValue(r.after)}`)
    .join('\n')
})

type AspectMode = 'free' | 'current' | '16:9' | '1:1' | '9:16' | '4:3' | '3:4'
const aspectMode = ref<AspectMode>('free')
const aspectRatio = ref<number | null>(null)

function snapDim(value: number): number {
  const step = 8
  const min = 64
  const max = 2048
  const v = Number.isFinite(value) ? value : min
  return Math.min(max, Math.max(min, Math.round(v / step) * step))
}

function ratioForMode(mode: AspectMode): number | null {
  if (mode === 'current') {
    const w = Number(video.value.width) || 0
    const h = Number(video.value.height) || 0
    return h > 0 ? w / h : null
  }
  if (mode === '16:9') return 16 / 9
  if (mode === '1:1') return 1
  if (mode === '9:16') return 9 / 16
  if (mode === '4:3') return 4 / 3
  if (mode === '3:4') return 3 / 4
  return null
}

function onAspectModeChange(e: Event): void {
  const mode = String((e.target as HTMLSelectElement).value || 'free') as AspectMode
  aspectMode.value = mode
  if (mode === 'free') {
    aspectRatio.value = null
    return
  }
  const ratio = ratioForMode(mode)
  aspectRatio.value = ratio
  if (!ratio) return

  // For fixed presets, snap the current size into the chosen ratio (preserve width).
  if (mode !== 'current') {
    const w = snapDim(Number(video.value.width) || 64)
    const h = snapDim(w / ratio)
    setVideo({ width: w, height: h })
  }
}

function applyWidth(value: number): void {
  const nextW = snapDim(value)
  const r = aspectRatio.value
  if (r && r > 0) {
    const nextH = snapDim(nextW / r)
    setVideo({ width: nextW, height: nextH })
    return
  }
  setVideo({ width: nextW })
}

function applyHeight(value: number): void {
  const nextH = snapDim(value)
  const r = aspectRatio.value
  if (r && r > 0) {
    const nextW = snapDim(nextH * r)
    setVideo({ width: nextW, height: nextH })
    return
  }
  setVideo({ height: nextH })
}

const workflowBusy = ref(false)

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

function toDataUrl(image: GeneratedImage): string { return `data:image/${image.format};base64,${image.data}` }

function formatHistoryTitle(item: VideoRunHistoryItem): string {
  const dt = new Date(item.createdAtMs || Date.now())
  const hh = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
  const label = item.mode === 'vid2vid'
    ? 'Vid2Vid'
    : (item.mode === 'img2vid' ? 'Img2Vid' : 'Txt2Vid')
  return `${label} · ${hh}`
}

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

defineExpose({ generate })
</script>
