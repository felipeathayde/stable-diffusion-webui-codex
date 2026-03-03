<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared QuickSettings top bar for Model Tabs (SD/Flux/Chroma/ZImage/WAN).
Loads `/api/options`, `/api/models`, `/api/models/inventory`, and `/api/paths`, then filters/presents per-family selectors (models/TE/VAE)
and commits overrides (device + runtime flags + tab-scoped Z-Image variant) used by generation payload builders.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsBar` (component): Main QuickSettings SFC; includes “advanced” UI, per-family subcomponents, and selector filtering logic.
- `cancelAdvancedAnimation` (function): Cancels in-flight advanced-row animations (used by toggling/resize logic).
- `easeOutCubic` (function): Easing helper used for advanced-row animations.
- `syncAdvancedHeight` (function): Measures/synchronizes advanced-row height for smooth expand/collapse transitions.
- `toggleAdvancedRow` (function): Toggles the advanced row (uses animation helpers and persisted UI state).
- `currentTab` (function): Determines the current tab kind (`txt2img`/`img2img`/`txt2vid`/`img2vid`) from routing/state.
- `tabFamilyFromStorage` (function): Loads persisted per-tab family from local storage (used to keep UI consistent on reload).
- `normalizePath` (function): Normalizes paths for stable comparisons (slash/case handling).
- `MetadataKind` (type): Discriminant for inline metadata popups (checkpoint/TE/VAE/WAN stage).
- `isRecordObject` (function): Type guard for plain object payloads used by metadata parsers.
- `parseMetadataKind` (function): Narrows a raw metadata kind string to supported `MetadataKind` values.
- `parseMetadataPayload` (function): Parses/validates dynamic metadata event payload into `{ kind, value }`.
- `extractSizeBytes` (function): Reads validated file size bytes from `/models/file-metadata` summary payload.
- `onShowMetadata` (function): Resolves selection metadata and opens a modal.
- `fileInPaths` (function): Checks whether a file path belongs to the configured roots for a key from `/api/paths` (drives selector filtering).
- `isVaeForFamily` (function): Filters VAE entries to those relevant for the current family.
- `withBuiltInVaeChoice` (function): Prepends canonical `built-in` to filtered VAE choices and removes legacy aliases/duplicates.
- `canonicalizeVaeChoiceForActiveFamily` (function): Normalizes `currentVae` to an active-family option (direct match, sentinel alias, or SHA-equivalent fallback).
- `isQuicksettingsReady` (ref): Becomes true only after component-local inventory/paths initialization completes; gates mount-time VAE canonicalization.
- `normalizeTextEncoderLabels` (function): Normalizes raw TE values into a stable label list (used for Flux/WAN multi-TE cases).
- `WanAssetsParams` (type): Minimal WAN assets triple used for payload building (metadata dir + TE + VAE).
- `currentWanAssets` (function): Builds `WanAssetsParams` from current UI selections (used by WAN payload generation).
- `textEncoderLabel` (function): Converts raw TE selector values into a canonical label (handles WAN-style prefixes).
- `onPrimaryTextEncoderChange` (function): Applies primary text-encoder selection changes (and triggers dependent updates).
- `onSecondaryTextEncoderChange` (function): Applies secondary text-encoder selection changes (Flux/Kontext dual-encoder workflows).
- `onSmartOffloadChange` (function): Updates Smart Offload toggle (impacts per-request memory behavior).
- `onSmartFallbackChange` (function): Updates Smart Fallback toggle (best-effort OOM fallback behavior).
- `onSmartCacheChange` (function): Updates Smart Cache toggle (conditioning caching behavior).
- `onCoreStreamingChange` (function): Updates core streaming toggle (runtime streaming behavior).
- `isObliteratingVram` (ref): Tracks in-flight `/api/obliterate-vram` requests to prevent repeated fire.
- `onObliterateVram` (function): Triggers safe VRAM cleanup and surfaces fail-loud status in quicksettings toasts/logs.
- `resolveWanFlowShiftForMode` (function): Resolves automatic WAN stage `flowShift` policy for the selected WAN mode + LightX2V toggle.
- `patchWanStageFlowShift` (function): Applies/removes managed WAN stage `flowShift` values without clobbering unrelated manual overrides.
- `finiteStageFlowShift` (function): Normalizes a stage `flowShift` into a finite number or `undefined` for stable policy comparisons.
- `ensureWanFlowShiftPolicy` (function): Enforces managed WAN `flowShift` policy on the active tab (including initial load) without update loops.
- `onWanModeChange` (function): Updates WAN mode selection and derived controls.
- `onWanBrowseModels` (function): Opens the shared add-path modal for WAN model roots (`wan22_ckpt`) from the WAN quicksettings `+` action.
- `onWanGuidedGen` (function): Opens WAN guided generation flow (UI navigation/CTA).
- `onUseInitImageChange` (function): Toggles active image-tab mode between txt2img and img2img from quick settings.
- `canShowModeToggles` (computed): Enables IMG2IMG/INPAINT quicksettings controls when the active image tab supports img2img.
- `useMask` (computed): Reflects active image-tab inpaint toggle state (`tab.params.useMask`).
- `supportsInpaint` (computed): Flags whether inpaint toggle is supported for the active image family.
- `isActiveImageTabRunning` (computed): Tracks whether the active image tab currently has an in-flight generation task.
- `inpaintToggleDisabled` (computed): Disables INPAINT when unsupported, when IMG2IMG is off, or when no init image is loaded.
- `inpaintToggleTitle` (computed): Tooltip reason for INPAINT enabled/disabled state.
- `onUseMaskChange` (function): Toggles inpaint mode (`useMask`) from quick settings with explicit IMG2IMG/Flux guards.
- `zimageTurbo` (computed): Returns the current Z-Image Turbo toggle state for the active tab.
- `zimageTurboLocked` (ref): When true, the Z-Image Turbo toggle is fixed by trusted checkpoint metadata.
- `_trustedZImageVariantFromCheckpointMeta` (function): Extracts `codex.zimage.variant` when metadata is trusted (Codex provenance).
- `onZImageTurboChange` (function): Applies Turbo toggle updates to the active Z-Image tab (with default migration).
- `enginePrefixForFamily` (function): Maps a `TabFamily` to the engine prefix used in options/labels.
- `openAddPathModal` (function): Opens the reusable add-path modal for checkpoint/VAE/text-encoder library keys.
- `onAddPathModalAdded` (function): Refreshes quicksettings lists after add-path operations mutate library paths.
- `onAddPathModalError` (function): Surfaces add-path scan/add failures through quicksettings toasts.
- `applyInventorySnapshot` (function): Applies one inventory payload to local quicksettings selector sources.
- `parseInventoryTaskResult` (function): Parses inventory payloads from task `result` SSE events.
- `runAsyncInventoryRefreshTask` (function): Starts `/api/models/inventory/refresh/async` and resolves when SSE emits terminal inventory data.
- `openPathInputModal` (function): Opens the in-app path input modal and registers async apply behavior.
- `confirmPathInputModal` (function): Validates/applies modal-entered path values.
- `closePathInputModal` (function): Closes and clears the in-app path input modal state.
- `openOverrides` (function): Opens the overrides UI surface (advanced controls entrypoint).
-->

<template>
  <section :class="['quicksettings', { 'quicksettings-loading': isLoadingQuicksettings }]">
    <div class="quicksettings-row quicksettings-row--main">
      <div class="quicksettings-group qs-group-advanced-toggle">
        <div class="qs-row">
          <button
            class="btn qs-btn-outline qs-advanced-handle"
            type="button"
            :aria-expanded="advancedOpen ? 'true' : 'false'"
            :aria-label="advancedOpen ? 'Collapse options' : 'Expand options'"
            :title="advancedOpen ? 'Collapse options' : 'Expand options'"
            @click="toggleAdvancedRow"
          >
            <svg v-if="!advancedOpen" class="qs-advanced-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
              <path
                d="M6 10L12 16L18 10"
                fill="none"
                stroke="currentColor"
                stroke-width="2.5"
                stroke-linecap="round"
                stroke-linejoin="round"
                transform="rotate(-90 12 12)"
              />
            </svg>
            <svg v-else class="qs-advanced-icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">
              <path d="M6 10L12 16L18 10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
            </svg>
          </button>
        </div>
      </div>
      <!-- WAN-specific quicksettings -->
      <template v-if="activeFamily === 'wan'">
        <QuickSettingsWan
          :mode="wanModelMode"
          :lightx2v="wanLightx2v"
          :high-model="wanHighModel"
          :high-choices="wanHighDirChoices"
          :low-model="wanLowModel"
          :low-choices="wanLowDirChoices"
          :text-encoder="wanTextEncoder"
          :text-encoder-choices="wanTextEncoderChoices"
          :vae="wanVae"
          :vae-choices="wanVaeChoices"
          @update:mode="onWanModeChange"
          @update:lightx2v="onWanLightx2vChange"
          @update:highModel="onWanHighModelChange"
          @update:lowModel="onWanLowModelChange"
          @update:textEncoder="onWanTextEncoderChange"
          @update:vae="onWanVaeChange"
          @browseModels="onWanBrowseModels"
          @browseTe="onWanBrowseTe"
          @browseVae="onWanBrowseVae"
          @refresh="refreshAll"
          @showMetadata="onShowMetadata"
        />
      </template>

      <!-- FLUX.1-specific quicksettings -->
      <template v-else-if="activeFamily === 'flux1'">
        <QuickSettingsFlux
          :checkpoint="effectiveCheckpoint"
          :checkpoints="filteredModelTitles"
          :vae="store.currentVae"
          :vae-choices="filteredVaeChoices"
          :text-encoder-primary="flux1TextEncoderPrimary"
          :text-encoder-secondary="flux1TextEncoderSecondary"
          :text-encoder-choices="filteredTextEncoderChoices"
          @update:checkpoint="onModelChange"
          @update:vae="onVaeChange"
          @update:textEncoderPrimary="onPrimaryTextEncoderChange"
          @update:textEncoderSecondary="onSecondaryTextEncoderChange"
          @addCheckpointPath="onAddCheckpointPath"
          @addVaePath="onAddVaePath"
          @addTencPath="onAddTencPath"
          @showMetadata="onShowMetadata"
        />
        <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
          <label class="label-muted">Mode</label>
          <div class="qs-row">
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useInitImage"
              @click="onUseInitImageChange(!useInitImage)"
            >
              IMG2IMG
            </button>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useMask"
              :disabled="inpaintToggleDisabled"
              :title="inpaintToggleTitle"
              @click="onUseMaskChange(!useMask)"
            >
              INPAINT
            </button>
          </div>
        </div>
        <div class="quicksettings-group qs-group-models">
          <label class="label-muted">Models</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
          </div>
        </div>
      </template>

      <!-- Z Image-specific quicksettings -->
      <template v-else-if="activeFamily === 'zimage'">
        <QuickSettingsZImage
          :checkpoint="effectiveCheckpoint"
          :checkpoints="filteredModelTitles"
          :turbo="zimageTurbo"
          :turbo-locked="zimageTurboLocked"
          :vae="store.currentVae"
          :vae-choices="filteredVaeChoices"
          :text-encoder="primaryTextEncoder"
          :text-encoder-choices="filteredTextEncoderChoices"
          @update:checkpoint="onModelChange"
          @update:turbo="onZImageTurboChange"
          @update:vae="onVaeChange"
          @update:textEncoder="onPrimaryTextEncoderChange"
          @addCheckpointPath="onAddCheckpointPath"
          @addVaePath="onAddVaePath"
          @addTencPath="onAddTencPath"
          @showMetadata="onShowMetadata"
        />
        <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle qs-group-mode-toggle--end">
          <label class="label-muted">Mode</label>
          <div class="qs-row">
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useInitImage"
              @click="onUseInitImageChange(!useInitImage)"
            >
              IMG2IMG
            </button>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useMask"
              :disabled="inpaintToggleDisabled"
              :title="inpaintToggleTitle"
              @click="onUseMaskChange(!useMask)"
            >
              INPAINT
            </button>
          </div>
        </div>
        <div class="quicksettings-group qs-group-models">
          <label class="label-muted">Models</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
          </div>
        </div>
      </template>

      <!-- Chroma-specific quicksettings -->
      <template v-else-if="activeFamily === 'chroma'">
        <QuickSettingsChroma
          :checkpoint="effectiveCheckpoint"
          :checkpoints="filteredModelTitles"
          :vae="store.currentVae"
          :vae-choices="filteredVaeChoices"
          :text-encoder="primaryTextEncoder"
          :text-encoder-choices="filteredTextEncoderChoices"
          :show-text-encoder="store.isModelCoreOnly(effectiveCheckpoint)"
          @update:checkpoint="onModelChange"
          @update:vae="onVaeChange"
          @update:textEncoder="onPrimaryTextEncoderChange"
          @addCheckpointPath="onAddCheckpointPath"
          @addVaePath="onAddVaePath"
          @addTencPath="onAddTencPath"
          @showMetadata="onShowMetadata"
        />
        <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
          <label class="label-muted">Mode</label>
          <div class="qs-row">
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useInitImage"
              @click="onUseInitImageChange(!useInitImage)"
            >
              IMG2IMG
            </button>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useMask"
              :disabled="inpaintToggleDisabled"
              :title="inpaintToggleTitle"
              @click="onUseMaskChange(!useMask)"
            >
              INPAINT
            </button>
          </div>
        </div>
        <div class="quicksettings-group qs-group-models">
          <label class="label-muted">Models</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
          </div>
        </div>
      </template>

      <!-- Default (SD15/SDXL) quicksettings -->
      <template v-else>
        <QuickSettingsBase
          :checkpoint="effectiveCheckpoint"
          :checkpoints="filteredModelTitles"
          :vae="store.currentVae"
          :vae-choices="filteredVaeChoices"
          :text-encoder="primaryTextEncoder"
          :text-encoder-choices="filteredTextEncoderChoices"
          text-encoder-automatic-label="Built-in"
          :show-text-encoder="activeFamily !== 'sd15' && activeFamily !== 'sdxl'"
          @update:checkpoint="onModelChange"
          @update:vae="onVaeChange"
          @update:textEncoder="onPrimaryTextEncoderChange"
          @addCheckpointPath="onAddCheckpointPath"
          @addVaePath="onAddVaePath"
          @showMetadata="onShowMetadata"
        />
        <div v-if="canShowModeToggles" class="quicksettings-group qs-group-mode-toggle">
          <label class="label-muted">Mode</label>
          <div class="qs-row">
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useInitImage ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useInitImage"
              @click="onUseInitImageChange(!useInitImage)"
            >
              IMG2IMG
            </button>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useMask ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="useMask"
              :disabled="inpaintToggleDisabled"
              :title="inpaintToggleTitle"
              @click="onUseMaskChange(!useMask)"
            >
              INPAINT
            </button>
          </div>
        </div>
        <div class="quicksettings-group qs-group-models">
          <label class="label-muted">Models</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
          </div>
        </div>
      </template>
    </div>

    <div v-if="qsNotice" class="caption">{{ qsNotice }}</div>

    <div ref="advancedRowEl" class="quicksettings-advanced-collapse" :data-state="advancedOpen ? 'open' : 'closed'">
      <div ref="advancedRowInnerEl" class="quicksettings-row quicksettings-row--advanced-inner">
        <QuickSettingsPerf
          :smart-offload="store.smartOffload"
          :smart-fallback="store.smartFallback"
          :smart-cache="store.smartCache"
          :core-streaming="store.coreStreaming"
          :obliterate-busy="isObliteratingVram"
          @update:smartOffload="onSmartOffloadChange"
          @update:smartFallback="onSmartFallbackChange"
          @update:smartCache="onSmartCacheChange"
          @update:coreStreaming="onCoreStreamingChange"
          @obliterateVram="onObliterateVram"
        />

        <div class="quicksettings-group qs-group-overrides">
          <label class="label-muted">Overrides</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-overrides-btn" type="button" @click="openOverrides">
              Set overrides
            </button>
          </div>
        </div>
      </div>
    </div>

    <QuickSettingsOverridesModal v-model="showOverridesModal" />
    <QuickSettingsAddPathModal
      v-model="showAddPathModal"
      :title="addPathModalTitle"
      :label="addPathModalLabel"
      :target-key="addPathModalTargetKey"
      :target-kind="addPathModalTargetKind"
      :placeholder="addPathModalPlaceholder"
      @added="onAddPathModalAdded"
      @error="onAddPathModalError"
    />
    <AssetMetadataModal v-model="showMetadataModal" :title="metadataModalTitle" :subtitle="metadataModalSubtitle" :payload="metadataModalPayload" />
    <Modal v-model="showPathInputModal" :title="pathInputModalTitle">
      <div class="quicksettings-path-modal">
        <label class="label-muted" for="quicksettings-path-input">{{ pathInputModalLabel }}</label>
        <input
          id="quicksettings-path-input"
          ref="pathInputEl"
          class="ui-input"
          type="text"
          :placeholder="pathInputModalPlaceholder"
          v-model="pathInputModalValue"
          @keydown.enter.prevent="confirmPathInputModal"
        />
      </div>
      <template #footer>
        <button class="btn btn-md btn-outline" type="button" @click="closePathInputModal">Cancel</button>
        <button class="btn btn-md btn-secondary" type="button" @click="confirmPathInputModal">Apply</button>
      </template>
    </Modal>
  </section>
</template>


<script setup lang="ts">
import { onBeforeUnmount, onMounted, computed, nextTick, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useUiBlocksStore } from '../stores/ui_blocks'
import { MODEL_TABS_STORAGE_KEY, useModelTabsStore, type ImageBaseParams, type TabByType, type WanAssetsParams, type WanStageParams } from '../stores/model_tabs'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import { useBootstrapStore } from '../stores/bootstrap'
import {
  cacheModelInventorySnapshot,
  fetchCheckpointMetadata,
  fetchFileMetadata,
  fetchModelInventory,
  fetchPaths,
  fetchObliterateVram,
  refreshModelInventory,
  startModelInventoryRefreshTask,
  subscribeTask,
} from '../api/client'
import type { InventoryResponse, ModelInfo, TaskEvent } from '../api/types'
import { isGenerationRunningForTab } from '../composables/useGeneration'
import { useResultsCard } from '../composables/useResultsCard'
import { normalizeTabFamily, tabFamilyFromSemanticEngine, type TabFamily } from '../utils/engine_taxonomy'
import { filterModelTitlesForFamily, enginePrefixForFamily } from '../utils/model_family_filters'
import QuickSettingsBase from './quicksettings/QuickSettingsBase.vue'
import QuickSettingsPerf from './quicksettings/QuickSettingsPerf.vue'
import QuickSettingsWan from './quicksettings/QuickSettingsWan.vue'
import QuickSettingsFlux from './quicksettings/QuickSettingsFlux.vue'
import QuickSettingsChroma from './quicksettings/QuickSettingsChroma.vue'
import QuickSettingsZImage from './quicksettings/QuickSettingsZImage.vue'
import QuickSettingsOverridesModal from './modals/QuickSettingsOverridesModal.vue'
import QuickSettingsAddPathModal from './modals/QuickSettingsAddPathModal.vue'
import AssetMetadataModal from './modals/AssetMetadataModal.vue'
import Modal from './ui/Modal.vue'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const uiBlocks = useUiBlocksStore()
const tabsStore = useModelTabsStore()
const engineCaps = useEngineCapabilitiesStore()
const bootstrap = useBootstrapStore()
const pathsConfig = ref<Record<string, string[]>>({})
type InventoryVae = { name: string; path: string; sha256?: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }
type InventoryWanGguf = { name: string; path: string; sha256?: string; stage: string }
type InventoryTextEncoder = { name: string; path: string; sha256?: string }
type ImageTab = TabByType<'sd15' | 'sdxl' | 'flux1' | 'zimage' | 'chroma' | 'anima'>
type WanTab = TabByType<'wan'>
type AddPathTargetKind = 'checkpoint' | 'vae' | 'text_encoder'
const inventoryVaes = ref<InventoryVae[]>([])
const inventoryWan = ref<InventoryWanGguf[]>([])
const inventoryTextEncoders = ref<InventoryTextEncoder[]>([])
const showOverridesModal = ref(false)
const showMetadataModal = ref(false)
const showAddPathModal = ref(false)
const showPathInputModal = ref(false)
const metadataModalTitle = ref('Metadata')
const metadataModalSubtitle = ref('')
const metadataModalPayload = ref<unknown>(null)
const addPathModalTitle = ref('Add Model Path')
const addPathModalLabel = ref('Path')
const addPathModalTargetKey = ref('')
const addPathModalTargetKind = ref<AddPathTargetKind>('checkpoint')
const addPathModalPlaceholder = ref('')
const pathInputModalTitle = ref('Update Path')
const pathInputModalLabel = ref('Path')
const pathInputModalPlaceholder = ref('')
const pathInputModalValue = ref('')
const pathInputEl = ref<HTMLInputElement | null>(null)
let pathInputApply: ((value: string) => Promise<void>) | null = null
const { notice: qsNotice, toast: qsToast } = useResultsCard({ noticeDurationMs: 4000 })
const isLoadingQuicksettings = ref(false)
const isQuicksettingsReady = ref(false)
const isObliteratingVram = ref(false)
const QUICKSETTINGS_ADVANCED_OPEN_STORAGE_KEY = 'codex.quicksettings.advanced_open'
const advancedOpen = ref(true)
const advancedRowEl = ref<HTMLElement | null>(null)
const advancedRowInnerEl = ref<HTMLElement | null>(null)
const advancedAnimating = ref(false)
let advancedRafId: number | null = null

try {
  const stored = localStorage.getItem(QUICKSETTINGS_ADVANCED_OPEN_STORAGE_KEY)
  if (stored === '0') advancedOpen.value = false
  if (stored === '1') advancedOpen.value = true
} catch {
  // ignore localStorage failures
}

watch(advancedOpen, (isOpen) => {
  try {
    localStorage.setItem(QUICKSETTINGS_ADVANCED_OPEN_STORAGE_KEY, isOpen ? '1' : '0')
  } catch {
    // ignore localStorage failures
  }
})

function cancelAdvancedAnimation(): void {
  if (advancedRafId !== null) cancelAnimationFrame(advancedRafId)
  advancedRafId = null
}

function easeOutCubic(t: number): number {
  const clamped = Math.min(1, Math.max(0, t))
  return 1 - Math.pow(1 - clamped, 3)
}

function syncAdvancedHeight(): void {
  const el = advancedRowEl.value
  const inner = advancedRowInnerEl.value
  if (!el || !inner) return
  if (advancedAnimating.value) return
  if (!advancedOpen.value) {
    el.style.height = '0px'
    el.style.opacity = '0'
    return
  }
  const nextHeight = inner.getBoundingClientRect().height
  if (nextHeight > 0) {
    el.style.height = `${nextHeight}px`
    el.style.opacity = ''
  }
}

function toggleAdvancedRow(): void {
  const el = advancedRowEl.value
  if (!el) {
    advancedOpen.value = !advancedOpen.value
    return
  }
  if (advancedAnimating.value) return

  cancelAdvancedAnimation()

  const startHeight = el.getBoundingClientRect().height

  const next = !advancedOpen.value
  advancedOpen.value = next
  advancedAnimating.value = true

  el.style.pointerEvents = 'none'

  const inner = advancedRowInnerEl.value
  const targetHeight = next ? (inner?.getBoundingClientRect().height ?? el.scrollHeight) : 0
  const durationMs = next ? 280 : 260
  const fromOpacity = next ? 0 : 1
  const toOpacity = next ? 1 : 0

  el.style.height = `${startHeight}px`
  el.style.opacity = `${fromOpacity}`

  const startMs = performance.now()
  const tick = (nowMs: number) => {
    const t = (nowMs - startMs) / durationMs
    const eased = easeOutCubic(t)
    const currentHeight = startHeight + (targetHeight - startHeight) * eased
    const currentOpacity = fromOpacity + (toOpacity - fromOpacity) * eased
    el.style.height = `${currentHeight}px`
    el.style.opacity = `${currentOpacity}`

    if (t < 1) {
      advancedRafId = requestAnimationFrame(tick)
      return
    }

    advancedRafId = null
    el.style.height = `${targetHeight}px`
    el.style.opacity = next ? '' : '0'
    el.style.pointerEvents = ''
    advancedAnimating.value = false
  }

  advancedRafId = requestAnimationFrame(tick)
}

function currentTab(): 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' {
  const p = route.path
  if (p.startsWith('/img2img')) return 'img2img'
  if (p.startsWith('/txt2vid')) return 'txt2vid'
  if (p.startsWith('/img2vid')) return 'img2vid'
  return 'txt2img'
}

const routeTabId = computed(() => String(route.params.tabId || ''))
const activeModelTab = computed(() => {
  if (!route.path.startsWith('/models/')) return null
  const id = routeTabId.value
  if (!id) return null
  const fromList = tabsStore.tabs.find(t => t.id === id) || null
  if (fromList) return fromList
  const active = tabsStore.activeTab
  if (active && active.id === id) return active
  return null
})

function asImageTab(value: unknown): ImageTab | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as { type?: unknown }
  const type = normalizeTabFamily(candidate.type)
  if (!type || type === 'wan') return null
  return value as ImageTab
}

function asWanTab(value: unknown): WanTab | null {
  if (!value || typeof value !== 'object') return null
  const candidate = value as { type?: unknown }
  return normalizeTabFamily(candidate.type) === 'wan' ? (value as WanTab) : null
}

function tabFamilyFromStorage(tabId: string): TabFamily | null {
  if (!tabId) return null
  try {
    const raw = localStorage.getItem(MODEL_TABS_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { tabs?: unknown[] }
    const list = Array.isArray(parsed.tabs) ? parsed.tabs : []
    const match = list.find((entry) => {
      if (!entry || typeof entry !== 'object') return false
      const id = String((entry as { id?: unknown }).id || '')
      return id === tabId
    }) as { type?: unknown } | undefined
    return normalizeTabFamily(match?.type)
  } catch {
    return null
  }
}

let routeActiveSyncToken = 0
watch(routeTabId, async (tabId) => {
  const token = ++routeActiveSyncToken
  if (!tabId) return
  try {
    if (!tabsStore.tabs.length) {
      await tabsStore.load()
    }
    if (token !== routeActiveSyncToken) return
    tabsStore.setActive(tabId)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}, { immediate: true })

const activeFamily = computed<TabFamily>(() => {
  if (route.path.startsWith('/models/') && routeTabId.value) {
    const type = normalizeTabFamily(activeModelTab.value?.type) || tabFamilyFromStorage(routeTabId.value)
    if (type) return type
  }

  // Fallback when no model tab is active (settings/tools pages etc.).
  if (!engineCaps.loaded) return 'sd15'
  const semantic = engineCaps.semanticEngineForId(uiBlocks.semanticEngine || 'sd15')
  const family = tabFamilyFromSemanticEngine(semantic)
  if (family) return family

  return 'sd15'
})
const semanticEngine = computed<string>(() => {
  // Prefer semantic engine from UI blocks when available (video tabs etc.).
  if (uiBlocks.semanticEngine) return uiBlocks.semanticEngine
  return 'sd15'
})

async function loadInventory(options?: { forceRefresh?: boolean }): Promise<void> {
  const inv = options?.forceRefresh ? await refreshModelInventory() : await fetchModelInventory()
  applyInventorySnapshot(inv)
}

function applyInventorySnapshot(inv: InventoryResponse): void {
  inventoryVaes.value = inv.vaes
  inventoryWan.value = (inv.wan22?.gguf ?? []).map((g) => ({
    name: String(g.name),
    path: String(g.path),
    sha256: typeof g?.sha256 === 'string' ? String(g.sha256) : undefined,
    stage: String(g.stage || 'unknown'),
  }))
  // Text encoder files are available via inventory for future use (e.g., Flux overrides).
  inventoryTextEncoders.value = inv.text_encoders ?? []
}

async function loadPaths(): Promise<void> {
  const res = await fetchPaths()
  pathsConfig.value = (res.paths || {}) as Record<string, string[]>
}

function normalizePath(path: string): string {
  return path.replace(/\\+/g, '/').replace(/\/+$/, '')
}

type MetadataKind =
  | 'checkpoint'
  | 'vae'
  | 'text_encoder'
  | 'text_encoder_primary'
  | 'text_encoder_secondary'
  | 'wan_high_model'
  | 'wan_low_model'
  | 'wan_text_encoder'
  | 'wan_vae'

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function parseMetadataKind(value: unknown): MetadataKind | null {
  const kind = String(value || '').trim()
  if (
    kind === 'checkpoint'
    || kind === 'vae'
    || kind === 'text_encoder'
    || kind === 'text_encoder_primary'
    || kind === 'text_encoder_secondary'
    || kind === 'wan_high_model'
    || kind === 'wan_low_model'
    || kind === 'wan_text_encoder'
    || kind === 'wan_vae'
  ) {
    return kind
  }
  return null
}

function parseMetadataPayload(payload: unknown): { kind: string; value: string } | null {
  if (!isRecordObject(payload)) return null
  const kind = String(payload.kind ?? '').trim()
  if (!kind) return null
  const value = String(payload.value ?? '').trim()
  if (!value) return null
  return { kind, value }
}

function extractSizeBytes(summary: Record<string, unknown>): number | null {
  const summaryFile = summary.file
  if (!isRecordObject(summaryFile)) return null
  const size = summaryFile.size_bytes
  if (typeof size !== 'number' || !Number.isFinite(size) || size < 0) return null
  return size
}

function isSha256(value: string): boolean {
  const lower = value.toLowerCase().trim()
  return lower.length === 64 && /^[0-9a-f]+$/.test(lower)
}

function stripFamilyPrefix(label: string): string {
  const norm = label.replace(/\\+/g, '/').trim()
  const idx = norm.indexOf('/')
  if (idx <= 0) return norm
  const prefix = norm.slice(0, idx)
  const rest = norm.slice(idx + 1)
  if (!rest) return norm
  if (['sd15', 'sdxl', 'flux1', 'chroma', 'wan22', 'zimage'].includes(prefix)) return rest
  return norm
}

function findModelByTitle(title: string): ModelInfo | undefined {
  const raw = String(title || '').trim()
  if (!raw) return undefined
  for (const m of store.models) {
    if (!m) continue
    if (m.title === raw) return m
  }
  return undefined
}

function findVaeRecord(label: string): InventoryVae | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined
  const norm = normalizePath(raw)
  const tail = norm.split('/').pop() || raw
  return inventoryVaes.value.find((v) => {
    if (!v) return false
    if (v.name === raw) return true
    const vPath = normalizePath(String(v.path || ''))
    return vPath === norm || (tail ? vPath.endsWith('/' + tail) : false)
  })
}

function findTextEncoderRecord(label: string): InventoryTextEncoder | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined
  const unprefixed = stripFamilyPrefix(raw)
  const candidates = [raw, unprefixed]
  for (const cand of candidates) {
    const norm = normalizePath(cand)
    const tail = norm.split('/').pop() || cand
    const found = inventoryTextEncoders.value.find((te) => {
      if (!te) return false
      if (te.name === cand) return true
      const tePath = normalizePath(String(te.path || ''))
      return tePath === norm || (tail ? tePath.endsWith('/' + tail) : false)
    })
    if (found) return found
  }
  return undefined
}

function findWanGgufRecord(label: string, stage: 'high' | 'low'): InventoryWanGguf | undefined {
  const raw = String(label || '').trim()
  if (!raw) return undefined
  const norm = normalizePath(raw)
  const tail = norm.split('/').pop() || raw
  return inventoryWan.value.find((w) => {
    if (!w) return false
    if (String(w.stage || '') !== stage) return false
    if (w.name === raw) return true
    const wPath = normalizePath(String(w.path || ''))
    return wPath === norm || (tail ? wPath.endsWith('/' + tail) : false)
  })
}

function onShowMetadata(payload: unknown): void {
  const parsed = parseMetadataPayload(payload)
  if (!parsed) return
  const { value } = parsed
  const kind = parseMetadataKind(parsed.kind)

  let title = 'Metadata'
  let subtitle = ''
  let out: Record<string, unknown> = {}
  let filePathForMetadata: string | null = null

	  if (kind === 'checkpoint') {
	    title = 'Checkpoint metadata'
	    subtitle = value
	    out = { selection: value, metadata: { status: 'loading' } }
	  } else if (kind === 'vae' || kind === 'wan_vae') {
	    const rec = findVaeRecord(value)
	    const sha = store.resolveVaeSha(value) || (rec?.sha256 ? String(rec.sha256) : undefined)
	    title = kind === 'wan_vae' ? 'WAN VAE metadata' : 'VAE metadata'
	    subtitle = rec?.name ? String(rec.name) : value
    filePathForMetadata = rec?.path ? String(rec.path) : null
    out = {
      selection: value,
      sha256: sha,
      inventory: rec
        ? {
            name: rec.name,
            path: rec.path,
            sha256: rec.sha256,
            format: rec.format,
            latent_channels: rec.latent_channels ?? null,
            scaling_factor: rec.scaling_factor ?? null,
          }
        : null,
    }
  } else if (kind === 'text_encoder' || kind === 'text_encoder_primary' || kind === 'text_encoder_secondary' || kind === 'wan_text_encoder') {
    const rec = findTextEncoderRecord(value)
    const sha = store.resolveTextEncoderSha(value) || (rec?.sha256 ? String(rec.sha256) : undefined)
    if (kind === 'text_encoder_primary') title = 'Text encoder metadata (CLIP)'
    else if (kind === 'text_encoder_secondary') title = 'Text encoder metadata (T5)'
    else if (kind === 'wan_text_encoder') title = 'WAN text encoder metadata'
    else title = 'Text encoder metadata'
    subtitle = rec?.name ? String(rec.name) : value
    filePathForMetadata = rec?.path ? String(rec.path) : null
    out = {
      selection: value,
      sha256: sha || (isSha256(value) ? value.toLowerCase() : undefined),
      inventory: rec ? { name: rec.name, path: rec.path, sha256: rec.sha256 } : null,
    }
  } else if (kind === 'wan_high_model' || kind === 'wan_low_model') {
    const stage = kind === 'wan_high_model' ? 'high' : 'low'
    const rec = findWanGgufRecord(value, stage)
    const sha = store.resolveWanGgufSha(value) || (rec?.sha256 ? String(rec.sha256) : undefined)
    title = stage === 'high' ? 'WAN high model metadata' : 'WAN low model metadata'
    subtitle = rec?.name ? String(rec.name) : value
    filePathForMetadata = rec?.path ? String(rec.path) : null
    out = {
      selection: value,
      stage,
      sha256: sha || (isSha256(value) ? value.toLowerCase() : undefined),
      inventory: rec ? { name: rec.name, path: rec.path, sha256: rec.sha256, stage: rec.stage } : null,
    }
  } else {
    title = 'Metadata'
    subtitle = value
    out = { selection: value, kind: parsed.kind }
  }

  if (filePathForMetadata) {
    out = { ...out, metadata: { status: 'loading' } }
  }

	  metadataModalTitle.value = title
	  metadataModalSubtitle.value = subtitle
	  metadataModalPayload.value = out
	  showMetadataModal.value = true

	  if (kind === 'checkpoint') {
	    void (async () => {
	      try {
	        const payload = await fetchCheckpointMetadata(value)
	        metadataModalPayload.value = payload
	      } catch (error: unknown) {
	        const message = error instanceof Error ? error.message : String(error)
	        metadataModalPayload.value = {
	          selection: value,
	          metadata: { status: 'error', error: message },
	        }
	      }
	    })()
	    return
	  }

	  if (!filePathForMetadata) return

	  void (async () => {
	    try {
	      const res = await fetchFileMetadata(filePathForMetadata)
	      const current = metadataModalPayload.value
	      if (!isRecordObject(current)) return
	      const flat = res.flat
	      const nested = res.nested
	      const sizeBytes = extractSizeBytes(res.summary)

	      const filePatch: Record<string, unknown> = {}
	      if (sizeBytes !== null) {
	        const mb = sizeBytes / 1_000_000
	        const gb = sizeBytes / 1_000_000_000
	        filePatch['file.size.bytes'] = sizeBytes
	        filePatch['file.size.megabytes'] = Number(mb.toFixed(3))
	        filePatch['file.size.gigabytes'] = Number(gb.toFixed(3))
	      }

	      const metaOut: Record<string, unknown> = {
	        raw: isRecordObject(flat) ? flat : (res as unknown as Record<string, unknown>),
	        nested,
	      }
	      metadataModalPayload.value = {
	        ...current,
	        ...filePatch,
	        metadata: metaOut,
	      }
	    } catch (error: unknown) {
      const current = metadataModalPayload.value
      if (!isRecordObject(current)) return
      const message = error instanceof Error ? error.message : String(error)
      metadataModalPayload.value = {
        ...current,
        metadata: { status: 'error', error: message },
      }
    }
  })()
}

function fileInPaths(file: string, key: string): boolean {
  if (!file) return false
  const roots = pathsConfig.value[key] || []
  if (!roots.length) return false
  const fNorm = normalizePath(file)
  for (const root of roots) {
    const rNorm = normalizePath(root)
    if (!rNorm) continue
    // Absolute root: direct prefix match.
    if (fNorm === rNorm || fNorm.startsWith(rNorm + '/')) return true
    // Repo-relative root (e.g. 'models/*-tenc'): match by suffix segment.
    const rel = rNorm.startsWith('/') ? rNorm.slice(1) : rNorm
    if (fNorm.includes('/' + rel + '/') || fNorm.endsWith('/' + rel)) return true
  }
  return false
}

const filteredModelTitles = computed(() => filterModelTitlesForFamily(store.models, activeFamily.value, pathsConfig.value))

function isVaeForFamily(name: string, fam: string): boolean {
  const rec = inventoryVaes.value.find(v => v.name === name || v.path.endsWith('/' + name))
  const scale = rec?.scaling_factor ?? null
  const path = rec?.path ?? ''
  if (fam === 'sdxl') return (scale !== null) ? Math.abs(Number(scale) - 0.13025) < 1e-3 : /sdxl|xl/i.test(name)
  if (fam === 'sd15') return (scale !== null) ? Math.abs(Number(scale) - 0.18215) < 5e-3 : /sd1|1\.5|sd15|v1-5/i.test(name)
  if (fam === 'flux1') return fileInPaths(path, 'flux1_vae')
  if (fam === 'chroma') return fileInPaths(path, 'flux1_vae')
  if (fam === 'zimage') return fileInPaths(path, 'zimage_vae') || fileInPaths(path, 'flux1_vae')  // Z Image uses same VAE as Flux.1
  return true
}

function withBuiltInVaeChoice(values: string[]): string[] {
  const out: string[] = ['built-in']
  const seen = new Set<string>(['built-in'])
  for (const raw of values) {
    const value = String(raw || '').trim()
    if (!value) continue
    const lower = value.toLowerCase()
    if (lower === 'automatic' || lower === 'built in' || lower === 'built-in') continue
    if (seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
}

function canonicalizeVaeChoiceForActiveFamily(current: string, choices: readonly string[]): string | null {
  if (!Array.isArray(choices) || choices.length === 0) return null

  const rawCurrent = String(current || '').trim()
  if (!rawCurrent) {
    return choices.includes('built-in') ? 'built-in' : String(choices[0] || '')
  }
  if (choices.includes(rawCurrent)) return rawCurrent

  const currentLower = rawCurrent.toLowerCase()
  if (currentLower === 'automatic' || currentLower === 'built in' || currentLower === 'built-in') {
    return choices.includes('built-in') ? 'built-in' : String(choices[0] || '')
  }
  if (currentLower === 'none' && choices.includes('none')) {
    return 'none'
  }

  const currentSha = store.resolveVaeSha(rawCurrent)
  if (currentSha) {
    const normalizedCurrentSha = String(currentSha).trim().toLowerCase()
    for (const choice of choices) {
      const candidateSha = store.resolveVaeSha(choice)
      if (!candidateSha) continue
      if (String(candidateSha).trim().toLowerCase() === normalizedCurrentSha) {
        return choice
      }
    }
  }

  return choices.includes('built-in') ? 'built-in' : String(choices[0] || '')
}

const filteredVaeChoices = computed(() => {
  const fam = activeFamily.value
  if (fam === 'flux1' || fam === 'chroma') {
    return withBuiltInVaeChoice(inventoryVaes.value
      .filter((v) => typeof v.path === 'string' && fileInPaths(v.path, 'flux1_vae'))
      .map((v) => String(v.path || ''))
    )
  }
  if (fam === 'zimage') {
    return withBuiltInVaeChoice(inventoryVaes.value
      .filter((v) => typeof v.path === 'string' && (fileInPaths(v.path, 'zimage_vae') || fileInPaths(v.path, 'flux1_vae')))
      .map((v) => String(v.path || ''))
    )
  }
  const familyChoices = (store.vaeChoices.length ? store.vaeChoices : ['built-in']).filter((value) => {
    const normalized = String(value || '').trim().toLowerCase()
    if (normalized === 'automatic' || normalized === 'built in' || normalized === 'built-in') return true
    if (normalized === 'none') return true
    return isVaeForFamily(value, fam)
  })
  return withBuiltInVaeChoice(familyChoices)
})

watch(
  () => [activeFamily.value, store.currentVae, filteredVaeChoices.value, isQuicksettingsReady.value] as const,
  ([family, currentVae, choices, quicksettingsReady]) => {
    if (family === 'wan') return
    if (!quicksettingsReady) return
    const nextVae = canonicalizeVaeChoiceForActiveFamily(String(currentVae || ''), choices)
    if (!nextVae) return
    if (String(currentVae || '') === nextVae) return
    store.setVae(nextVae).catch((error) => {
      toastQuicksettingsError(error)
    })
  },
  { immediate: true },
)

const filteredTextEncoderChoices = computed(() => {
  const fam = activeFamily.value
  if (fam === 'flux1') {
    // For FLUX.1, derive choices from inventory.text_encoders constrained by flux1_tenc paths.
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'flux1_tenc'))
      .map((item) => `flux1/${item.path}`)
  }
  if (fam === 'chroma') {
    // Chroma uses a single T5 text encoder; roots are shared with Flux.1 (`flux1_tenc`).
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'flux1_tenc'))
      .map((item) => `chroma/${item.path}`)
  }
  if (fam === 'zimage') {
    // For Z Image, derive choices from inventory.text_encoders constrained by zimage_tenc paths.
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'zimage_tenc'))
      .map((item) => `zimage/${item.path}`)
  }
  const prefix = fam === 'wan' ? 'wan22/' : `${fam}/`
  return store.textEncoderChoices.filter((name) => typeof name === 'string' && typeof prefix === 'string' && name.startsWith(prefix))
})

const isModelTabRoute = computed(() => route.path.startsWith('/models/') && Boolean(routeTabId.value))
const activeImageTab = computed(() => {
  if (!isModelTabRoute.value) return null
  return asImageTab(activeModelTab.value)
})
const activeImageSurface = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return null
  return engineCaps.get(tab.type)
})
const canToggleInitImage = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  const surface = activeImageSurface.value
  if (!surface) return false
  return Boolean(surface.supports_img2img)
})
const canShowModeToggles = computed(() => canToggleInitImage.value)
const useInitImage = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return Boolean(tab.params.useInitImage)
})
const useMask = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return Boolean(tab.params.useMask)
})
const hasInitImage = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return String(tab.params.initImageData || '').trim().length > 0
})
const supportsInpaint = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return tab.type !== 'flux1'
})
const isActiveImageTabRunning = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return false
  return isGenerationRunningForTab(tab.id)
})
const inpaintToggleDisabled = computed(() => (
  isActiveImageTabRunning.value
  || !useInitImage.value
  || !hasInitImage.value
  || !supportsInpaint.value
))
const inpaintToggleTitle = computed(() => {
  if (isActiveImageTabRunning.value) return 'Cannot change INPAINT while generation is running.'
  if (!supportsInpaint.value) return 'INPAINT is not supported for Flux.1 img2img (Kontext) yet.'
  if (!useInitImage.value) return 'Enable IMG2IMG first.'
  if (!hasInitImage.value) return 'Select an init image first.'
  return 'Toggle INPAINT'
})

const activeWanTab = computed(() => asWanTab(activeModelTab.value))

function normalizeTextEncoderLabels(raw: unknown): string[] {
  if (!Array.isArray(raw)) return []
  return raw.map((it) => String(it || '').trim()).filter((it) => it.length > 0)
}

const effectiveTextEncoders = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return store.currentTextEncoders
  return normalizeTextEncoderLabels(tab.params.textEncoders)
})

const primaryTextEncoder = computed(() => effectiveTextEncoders.value[0] ?? '')

const flux1TextEncoders = computed(() => effectiveTextEncoders.value.filter((label) => typeof label === 'string' && label.startsWith('flux1/')))
const flux1TextEncoderPrimary = computed(() => flux1TextEncoders.value[0] ?? '')
const flux1TextEncoderSecondary = computed(() => flux1TextEncoders.value[1] ?? '')

const primaryTeAutomaticLabel = 'Built-in'
const secondaryTeAutomaticLabel = 'Secondary (optional)'

const effectiveCheckpoint = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return store.currentModel
  const ckpt = String(tab.params.checkpoint || '').trim()
  if (ckpt) return ckpt
  return filteredModelTitles.value[0] ?? ''
})

const CODEX_REPO_URL = 'https://github.com/sangoi-exe/stable-diffusion-webui-codex'

const zimageTurbo = computed<boolean>(() => {
  const tab = activeImageTab.value
  if (!tab || tab.type !== 'zimage') return true
  const raw = tab.params.zimageTurbo
  return typeof raw === 'boolean' ? raw : true
})

const zimageTurboLocked = ref(false)
let zimageVariantDetectToken = 0

function _trustedZImageVariantFromCheckpointMeta(payload: unknown): 'turbo' | 'base' | null {
  if (!isRecordObject(payload)) return null
  const metadata = payload.metadata
  if (!isRecordObject(metadata)) return null
  const raw = metadata.raw
  if (!isRecordObject(raw)) return null

  const codexRepo = String(raw['codex.repository'] ?? '').trim()
  const codexBy = String(raw['codex.quantized_by'] ?? '').trim()
  if (!codexRepo || !codexBy) return null
  if (codexRepo !== CODEX_REPO_URL) return null

  const variant = String(raw['codex.zimage.variant'] ?? '').trim().toLowerCase()
  if (variant === 'turbo' || variant === 'base') return variant
  return null
}

watch(
  () => [activeFamily.value, activeImageTab.value?.id ?? '', effectiveCheckpoint.value] as const,
  async ([family, tabId, checkpoint]) => {
    zimageTurboLocked.value = false
    if (family !== 'zimage') return
    if (!tabId) return
    if (!checkpoint) return

    const token = ++zimageVariantDetectToken
    try {
      const meta = await fetchCheckpointMetadata(checkpoint)
      if (token !== zimageVariantDetectToken) return
      const variant = _trustedZImageVariantFromCheckpointMeta(meta)
      if (!variant) return
      zimageTurboLocked.value = true

      const turbo = variant === 'turbo'
      const tab = activeImageTab.value
      if (!tab || tab.type !== 'zimage') return
      const current = typeof tab.params.zimageTurbo === 'boolean' ? Boolean(tab.params.zimageTurbo) : true
      if (current !== turbo) {
        await updateImageTabParams(tab.id, { zimageTurbo: turbo })
        qsToast(`Z-Image: Turbo is ${turbo ? 'ON' : 'OFF'} (from model metadata).`)
      }
    } catch {
      // Non-fatal: if metadata can't be read, keep the toggle user-controlled.
    }
  },
  { immediate: true },
)

watch(
  () => [activeImageTab.value?.id ?? '', filteredModelTitles.value] as const,
  ([tabId, models]) => {
    if (!tabId) return
    const tab = activeImageTab.value
    if (!tab) return
    const ckpt = String(tab.params.checkpoint || '').trim()
    if (ckpt) return
    const first = models[0]
    if (!first) return
    updateImageTabParams(tab.id, { checkpoint: first }).catch((error) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  },
  { immediate: true },
)

async function initQuicksettings(
  options?: { forceInventoryRefresh?: boolean; forceModelsRefresh?: boolean },
  controls?: { includeStoreInit?: boolean },
): Promise<void> {
  isQuicksettingsReady.value = false
  isLoadingQuicksettings.value = true
  try {
    if (controls?.includeStoreInit !== false) {
      await store.init()
    }
    if (options?.forceModelsRefresh === true) {
      await store.refreshModelsList()
    }
    await Promise.all([
      loadPaths(),
      loadInventory({ forceRefresh: options?.forceInventoryRefresh === true }),
    ])
    isQuicksettingsReady.value = true
  } finally {
    isLoadingQuicksettings.value = false
  }
}

function parseInventoryTaskResult(event: TaskEvent): InventoryResponse | null {
  if (event.type !== 'result') return null
  const payload = event as unknown as Record<string, unknown>
  const direct = payload.inventory
  if (isRecordObject(direct)) {
    return direct as unknown as InventoryResponse
  }
  const info = payload.info
  if (!isRecordObject(info)) return null
  const nested = info.inventory
  if (!isRecordObject(nested)) return null
  return nested as unknown as InventoryResponse
}

async function runAsyncInventoryRefreshTask(): Promise<InventoryResponse> {
  const { task_id } = await startModelInventoryRefreshTask()
  return await new Promise<InventoryResponse>((resolve, reject) => {
    let settled = false
    let resolvedInventory: InventoryResponse | null = null
    let unsubscribe: (() => void) | null = null
    const settle = (fn: () => void): void => {
      if (settled) return
      settled = true
      try { unsubscribe?.() } catch (_) { /* ignore */ }
      unsubscribe = null
      fn()
    }

    unsubscribe = subscribeTask(
      task_id,
      (event) => {
        if (event.type === 'error') {
          settle(() => reject(new Error(String(event.message || 'inventory refresh task failed'))))
          return
        }
        if (event.type === 'result') {
          const parsed = parseInventoryTaskResult(event)
          if (parsed) resolvedInventory = parsed
          return
        }
        if (event.type === 'end') {
          if (resolvedInventory) {
            settle(() => resolve(resolvedInventory as InventoryResponse))
            return
          }
          settle(() => reject(new Error('inventory refresh task completed without inventory payload')))
        }
      },
      (err) => {
        settle(() => reject(err instanceof Error ? err : new Error(String(err))))
      },
    )
  })
}

async function refreshAll(): Promise<void> {
  if (isLoadingQuicksettings.value) return
  isLoadingQuicksettings.value = true
  try {
    await Promise.all([store.refreshModelsList(), loadPaths()])
    const refreshedInventory = await runAsyncInventoryRefreshTask()
    cacheModelInventorySnapshot(refreshedInventory)
    applyInventorySnapshot(refreshedInventory)
  } catch (error) {
    toastQuicksettingsError(error)
  } finally {
    isLoadingQuicksettings.value = false
  }
}

const wanHighDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    const stage = String(g.stage || 'unknown').trim().toLowerCase()
    if (stage !== 'high' && stage !== 'unknown') continue
    const path = String(g.path || '').trim()
    if (!path) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

const wanLowDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    const stage = String(g.stage || 'unknown').trim().toLowerCase()
    if (stage !== 'low' && stage !== 'unknown') continue
    const path = String(g.path || '').trim()
    if (!path) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

type WanModelMode = 'i2v_14b' | 't2v_14b' | 'i2v_5b' | 't2v_5b'
const WAN_LIGHTX2V_I2V_14B_FLOW_SHIFT = 5.0

function resolveWanFlowShiftForMode(mode: WanModelMode, lightx2v: boolean): number | null {
  if (!lightx2v) return null
  if (mode === 'i2v_14b') return WAN_LIGHTX2V_I2V_14B_FLOW_SHIFT
  return null
}

function patchWanStageFlowShift(stage: WanStageParams, flowShift: number | null): WanStageParams {
  const next: WanStageParams = { ...stage }
  if (flowShift === null) {
    if (
      typeof next.flowShift === 'number'
      && Number.isFinite(next.flowShift)
      && Math.abs(next.flowShift - WAN_LIGHTX2V_I2V_14B_FLOW_SHIFT) < 1e-9
    ) {
      delete next.flowShift
    }
    return next
  }
  next.flowShift = flowShift
  return next
}

function finiteStageFlowShift(stage: WanStageParams): number | undefined {
  if (typeof stage.flowShift !== 'number') return undefined
  if (!Number.isFinite(stage.flowShift)) return undefined
  return stage.flowShift
}

let syncingWanFlowShiftPolicy = false

async function ensureWanFlowShiftPolicy(): Promise<void> {
  if (syncingWanFlowShiftPolicy) return
  const tab = activeWanTab.value
  if (!tab) return
  const flowShift = resolveWanFlowShiftForMode(wanModelMode.value, Boolean(tab.params.lightx2v))
  const nextHigh = patchWanStageFlowShift(tab.params.high, flowShift)
  const nextLow = patchWanStageFlowShift(tab.params.low, flowShift)
  if (
    finiteStageFlowShift(tab.params.high) === finiteStageFlowShift(nextHigh)
    && finiteStageFlowShift(tab.params.low) === finiteStageFlowShift(nextLow)
  ) {
    return
  }
  syncingWanFlowShiftPolicy = true
  try {
    await tabsStore.updateParams(tab.id, { high: nextHigh, low: nextLow })
  } finally {
    syncingWanFlowShiftPolicy = false
  }
}

function _wanRepoForMode(mode: WanModelMode): string {
  if (mode === 't2v_14b') return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
  if (mode === 'i2v_5b' || mode === 't2v_5b') return 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
  return 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
}

const wanModelMode = computed<WanModelMode>(() => {
  const tab = activeWanTab.value
  if (!tab) return 't2v_14b'
  const video = tab.params.video
  const rawAssets = tab.params.assets || { metadata: '', textEncoder: '', vae: '' }
  const meta = String(rawAssets.metadata || '').trim().toLowerCase()
  const is5b = meta.includes('ti2v-5b') || meta.includes('5b')
  const kind = video?.useInitImage ? 'i2v' : 't2v'

  if (is5b) return kind === 'i2v' ? 'i2v_5b' : 't2v_5b'
  if (kind === 'i2v') return 'i2v_14b'
  return 't2v_14b'
})

const wanLightx2v = computed(() => {
  const tab = activeWanTab.value
  if (!tab) return false
  return Boolean(tab.params.lightx2v)
})

const wanHighModel = computed(() => {
  const tab = activeWanTab.value
  if (!tab) return ''
  return tab.params.high?.modelDir || ''
})

const wanLowModel = computed(() => {
  const tab = activeWanTab.value
  if (!tab) return ''
  return tab.params.low?.modelDir || ''
})

function currentWanAssets(): WanAssetsParams {
  const base: WanAssetsParams = { metadata: '', textEncoder: '', vae: '' }
  const tab = activeWanTab.value
  if (!tab) return base
  const raw = tab.params.assets
  return raw ? { ...base, ...raw } : base
}

const wanTextEncoder = computed(() => currentWanAssets().textEncoder || '')
const wanVae = computed(() => currentWanAssets().vae || '')

const wanTextEncoderChoices = computed(() => {
  // WAN22 GGUF requires an explicit TE weights file (.safetensors or .gguf). Prefer concrete
  // files under the configured wan22_tenc roots (paths.json) rather than root labels from a dedicated endpoint.
  return inventoryTextEncoders.value
    .filter((item) => {
      const path = typeof item.path === 'string' ? item.path : ''
      if (!path) return false
      const lower = path.toLowerCase()
      if (!lower.endsWith('.safetensors') && !lower.endsWith('.gguf')) return false
      return fileInPaths(path, 'wan22_tenc')
    })
    .map((item) => `wan22/${item.path}`)
})

const wanVaeChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of inventoryVaes.value) {
    const path = String(item.path || '')
    if (!path) continue
    if (!fileInPaths(path, 'wan22_vae')) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

// Event handlers
function toastQuicksettingsError(error: unknown): void {
  qsToast(error instanceof Error ? error.message : String(error))
}

function updateImageTabParams(tabId: string, patch: Partial<ImageBaseParams>): Promise<void> {
  return tabsStore.updateParams(tabId, patch as Partial<Record<string, unknown>>)
}

async function onModelChange(value: string): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (tab) {
      await updateImageTabParams(tab.id, { checkpoint: String(value || '') })
      return
    }
    await store.setModel(value)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onVaeChange(value: string): Promise<void> {
  try {
    await store.setVae(value)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onUseInitImageChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab) return
    const patch: Partial<ImageBaseParams> = { useInitImage: Boolean(value) }
    if (!value) {
      patch.initImageData = ''
      patch.initImageName = ''
      patch.useMask = false
      patch.maskImageData = ''
      patch.maskImageName = ''
    }
    await updateImageTabParams(tab.id, patch)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onUseMaskChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab) return
    if (isActiveImageTabRunning.value) return
    if (tab.type === 'flux1') return
    if (!useInitImage.value) return
    if (!hasInitImage.value) return
    const patch: Partial<ImageBaseParams> = { useMask: Boolean(value) }
    if (!value) {
      patch.maskImageData = ''
      patch.maskImageName = ''
    }
    await updateImageTabParams(tab.id, patch)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const [family, ...rest] = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (!family || rest.length === 0) return value
  const basename = rest[rest.length - 1] || rest[0]
  return `${family}/${basename}`
}

async function updateFlux1TextEncoders(primary: string, secondary: string): Promise<void> {
  const tab = activeImageTab.value
  const fluxLabels: string[] = []
  const p = primary.trim()
  const s = secondary.trim()
  if (p) fluxLabels.push(p)
  if (s && s !== p) fluxLabels.push(s)
  if (tab) {
    await updateImageTabParams(tab.id, { textEncoders: fluxLabels })
    return
  }
  const all = store.currentTextEncoders.slice()
  const other = all.filter((label) => !String(label).startsWith('flux1/'))
  const next = [...other, ...fluxLabels]
  await store.setTextEncoders(next)
}

function onPrimaryTextEncoderChange(value: string): void {
  const fam = activeFamily.value
  if (fam === 'flux1') {
    const primary = value || ''
    const secondary = flux1TextEncoderSecondary.value || ''
    updateFlux1TextEncoders(primary, secondary).catch((error) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  } else {
    const tab = activeImageTab.value
    const payload = value ? [value] : []
    if (tab) {
      updateImageTabParams(tab.id, { textEncoders: payload }).catch((error) => {
        qsToast(error instanceof Error ? error.message : String(error))
      })
      return
    }
    store.setTextEncoders(payload).catch((error) => {
      qsToast(error instanceof Error ? error.message : String(error))
    })
  }
}

function onSecondaryTextEncoderChange(value: string): void {
  const fam = activeFamily.value
  if (fam !== 'flux1') return
  const primary = flux1TextEncoderPrimary.value || ''
  const secondary = value || ''
  updateFlux1TextEncoders(primary, secondary).catch((error) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onSmartOffloadChange(value: boolean): void {
  store.setSmartOffload(value).catch((error) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onSmartFallbackChange(value: boolean): void {
  store.setSmartFallback(value).catch((error) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onSmartCacheChange(value: boolean): void {
  store.setSmartCache(value).catch((error) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

function onCoreStreamingChange(value: boolean): void {
  store.setCoreStreaming(value).catch((error) => {
    qsToast(error instanceof Error ? error.message : String(error))
  })
}

async function onObliterateVram(): Promise<void> {
  if (isObliteratingVram.value) return
  isObliteratingVram.value = true
  try {
    const result = await fetchObliterateVram()
    console.info('[QuickSettingsBar] obliterate-vram result', result)
    if (Array.isArray(result.warnings) && result.warnings.length > 0) {
      console.warn('[QuickSettingsBar] obliterate-vram warnings', result.warnings)
    }
    if (!result.ok) {
      console.error('[QuickSettingsBar] obliterate-vram failed', {
        internal_failures: result.internal_failures,
        external_failures: result.external?.failures ?? [],
      })
      throw new Error(result.message || 'Obliterate VRAM finished with failures.')
    }
    const killedCount = Array.isArray(result.external?.terminated_pids)
      ? result.external.terminated_pids.length
      : 0
    const detectedCount = Array.isArray(result.external?.detected_processes)
      ? result.external.detected_processes.length
      : 0
    const disabledExternalKill = Array.isArray(result.warnings)
      && result.warnings.includes('external_gpu_termination_disabled_by_default')
    if (killedCount > 0) {
      qsToast(`Obliterate VRAM done. Killed ${killedCount} external GPU process(es).`)
      return
    }
    if (disabledExternalKill && detectedCount > 0) {
      qsToast(`Obliterate VRAM done (internal only). ${detectedCount} external GPU process(es) detected.`)
      return
    }
    qsToast('Obliterate VRAM done.')
  } catch (error) {
    qsToast(error instanceof Error ? error.message : String(error))
  } finally {
    isObliteratingVram.value = false
  }
}

async function onWanModeChange(value: string): Promise<void> {
  try {
    const tab = activeWanTab.value
    if (!tab) return

    const raw = String(value || '').trim().toLowerCase()
    const nextMode: WanModelMode =
      raw === 'i2v_14b' ? 'i2v_14b'
        : raw === 't2v_14b' ? 't2v_14b'
          : raw === 'i2v_5b' ? 'i2v_5b'
            : raw === 't2v_5b' ? 't2v_5b'
                : 't2v_14b'

    const currentVideo = tab.params.video
    const videoPatch: Record<string, unknown> = {}
    if (nextMode === 't2v_14b' || nextMode === 't2v_5b') {
      videoPatch.useInitImage = false
      videoPatch.initImageData = ''
      videoPatch.initImageName = ''
    } else {
      // i2v
      videoPatch.useInitImage = true
    }

    const currentAssets = currentWanAssets()
    const nextAssets = { ...currentAssets, metadata: _wanRepoForMode(nextMode) }
    const flowShift = resolveWanFlowShiftForMode(nextMode, Boolean(tab.params.lightx2v))
    const nextHigh = patchWanStageFlowShift(tab.params.high, flowShift)
    const nextLow = patchWanStageFlowShift(tab.params.low, flowShift)

    await tabsStore.updateParams(
      tab.id,
      {
        video: { ...currentVideo, ...videoPatch },
        assets: nextAssets,
        high: nextHigh,
        low: nextLow,
      },
    )
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanLightx2vChange(value: boolean): Promise<void> {
  try {
    const tab = activeWanTab.value
    if (!tab) return
    const nextLightx2v = Boolean(value)
    const flowShift = resolveWanFlowShiftForMode(wanModelMode.value, nextLightx2v)
    const nextHigh = patchWanStageFlowShift(tab.params.high, flowShift)
    const nextLow = patchWanStageFlowShift(tab.params.low, flowShift)
    await tabsStore.updateParams(tab.id, { lightx2v: nextLightx2v, high: nextHigh, low: nextLow })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

watch(
  () => {
    const tab = activeWanTab.value
    if (!tab) return null
    return {
      tabId: tab.id,
      mode: wanModelMode.value,
      lightx2v: Boolean(tab.params.lightx2v),
      highFlowShift: finiteStageFlowShift(tab.params.high),
      lowFlowShift: finiteStageFlowShift(tab.params.low),
    }
  },
  () => {
    void ensureWanFlowShiftPolicy()
  },
  { immediate: true },
)

async function onZImageTurboChange(value: boolean): Promise<void> {
  try {
    const tab = activeImageTab.value
    if (!tab || tab.type !== 'zimage') return
    if (zimageTurboLocked.value) {
      qsToast('Z-Image: Turbo variant is fixed by model metadata.')
      return
    }

    const currentSteps = Number(tab.params.steps)
    const currentCfg = Number(tab.params.cfgScale)

    const turbo = Boolean(value)
    const patch: Record<string, unknown> = { zimageTurbo: turbo }

    // Apply variant-recommended defaults only when the user is still on the previous variant's defaults.
    // Turbo defaults: steps≈9, distilled guidance≈1.0. Base defaults: steps≈30, CFG≈4.0.
    if (turbo) {
      if (Number.isFinite(currentSteps) && (currentSteps === 30)) patch.steps = 9
      if (Number.isFinite(currentCfg) && Math.abs(currentCfg - 4.0) < 1e-6) patch.cfgScale = 1.0
    } else {
      if (Number.isFinite(currentSteps) && (currentSteps === 8 || currentSteps === 9)) patch.steps = 30
      if (Number.isFinite(currentCfg) && Math.abs(currentCfg - 1.0) < 1e-6) patch.cfgScale = 4.0
    }

    await updateImageTabParams(tab.id, patch)
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanHighModelChange(value: string): Promise<void> {
  try {
    const tab = activeWanTab.value
    if (!tab) return
    const current = tab.params.high || {}
    await tabsStore.updateParams(tab.id, { high: { ...current, modelDir: value } })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanLowModelChange(value: string): Promise<void> {
  try {
    const tab = activeWanTab.value
    if (!tab) return
    const current = tab.params.low || {}
    await tabsStore.updateParams(tab.id, { low: { ...current, modelDir: value } })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanTextEncoderChange(value: string): Promise<void> {
  try {
    const tab = activeWanTab.value
    if (!tab) return
    const current = currentWanAssets()
    await tabsStore.updateParams(tab.id, { assets: { ...current, textEncoder: value } })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

async function onWanVaeChange(value: string): Promise<void> {
  try {
    const tab = activeWanTab.value
    if (!tab) return
    const current = currentWanAssets()
    await tabsStore.updateParams(tab.id, { assets: { ...current, vae: value } })
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function onWanGuidedGen(): void {
  const tab = activeWanTab.value
  if (!tab) return
  window.dispatchEvent(new CustomEvent('codex-wan-guided-gen', { detail: { tabId: tab.id } }))
}

function openAddPathModal(options: {
  title: string
  label: string
  key: string
  kind: AddPathTargetKind
  placeholder?: string
}): void {
  addPathModalTitle.value = options.title
  addPathModalLabel.value = options.label
  addPathModalTargetKey.value = options.key
  addPathModalTargetKind.value = options.kind
  addPathModalPlaceholder.value = String(options.placeholder || '')
  showAddPathModal.value = true
}

function onAddCheckpointPath(): void {
  const prefix = enginePrefixForFamily(activeFamily.value)
  openAddPathModal({
    title: 'Add Checkpoint Directory',
    label: 'Checkpoint path',
    key: `${prefix}_ckpt`,
    kind: 'checkpoint',
  })
}

function onAddVaePath(): void {
  const prefix = enginePrefixForFamily(activeFamily.value)
  openAddPathModal({
    title: 'Add VAE Directory',
    label: 'VAE path',
    key: `${prefix}_vae`,
    kind: 'vae',
  })
}

function onAddTencPath(): void {
  const prefix = enginePrefixForFamily(activeFamily.value)
  openAddPathModal({
    title: 'Add Text Encoder Directory',
    label: 'Text encoder path',
    key: `${prefix}_tenc`,
    kind: 'text_encoder',
  })
}

async function onAddPathModalAdded(payload: { addedCount: number }): Promise<void> {
  if (!payload || !Number.isFinite(payload.addedCount) || payload.addedCount <= 0) return
  try {
    await refreshAll()
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function onAddPathModalError(message: string): void {
  const text = String(message || '').trim()
  if (!text) return
  qsToast(text)
}

function openPathInputModal(
  options: { title: string; label: string; placeholder?: string; initialValue?: string },
  apply: (value: string) => Promise<void>,
): void {
  pathInputModalTitle.value = options.title
  pathInputModalLabel.value = options.label
  pathInputModalPlaceholder.value = options.placeholder || ''
  pathInputModalValue.value = String(options.initialValue || '')
  pathInputApply = apply
  showPathInputModal.value = true
  void nextTick(() => {
    pathInputEl.value?.focus()
    pathInputEl.value?.select()
  })
}

function closePathInputModal(): void {
  showPathInputModal.value = false
  pathInputApply = null
  pathInputModalValue.value = ''
}

async function confirmPathInputModal(): Promise<void> {
  const apply = pathInputApply
  if (!apply) {
    closePathInputModal()
    return
  }
  const trimmed = pathInputModalValue.value.trim()
  if (!trimmed) {
    qsToast('Path is required.')
    return
  }
  try {
    await apply(trimmed)
    closePathInputModal()
  } catch (error) {
    toastQuicksettingsError(error)
  }
}

function onWanBrowseModels(): void {
  openAddPathModal({
    title: 'Add WAN Model Directory',
    label: 'WAN model path',
    key: 'wan22_ckpt',
    kind: 'checkpoint',
  })
}

async function onWanBrowseTe(): Promise<void> {
  openPathInputModal(
    {
      title: 'WAN Text Encoder',
      label: 'WAN Text Encoder (.safetensors or .gguf) path or sha256',
      initialValue: wanTextEncoder.value,
    },
    async (value) => {
      const normalized = value.replace(/\\+/g, '/')
      const stored = normalized.startsWith('wan22/') || !normalized.startsWith('/') ? normalized : `wan22/${normalized}`
      await onWanTextEncoderChange(stored)
    },
  )
}

async function onWanBrowseVae(): Promise<void> {
  openPathInputModal(
    {
      title: 'WAN VAE',
      label: 'WAN VAE path or sha256',
      initialValue: wanVae.value,
    },
    async (value) => {
      await onWanVaeChange(value)
    },
  )
}

function openOverrides(): void {
  showOverridesModal.value = true
}

onMounted(() => {
  window.addEventListener('resize', syncAdvancedHeight)
  requestAnimationFrame(syncAdvancedHeight)
  bootstrap
    .runRequired('Failed to initialize QuickSettings', async () => {
      await Promise.all([
        initQuicksettings(undefined, { includeStoreInit: false }),
        presets.init(currentTab()),
      ])
    })
    .catch(() => {
      // Fatal state is already set by bootstrap store.
    })
})

onBeforeUnmount(() => {
  cancelAdvancedAnimation()
  showAddPathModal.value = false
  closePathInputModal()
  window.removeEventListener('resize', syncAdvancedHeight)
})

watch(() => route.path, async () => {
  try {
    await presets.init(currentTab())
    await loadInventory()
  } catch (error) {
    toastQuicksettingsError(error)
  }
})

// random seed button removed from quicksettings; presets applied elsewhere
</script>
