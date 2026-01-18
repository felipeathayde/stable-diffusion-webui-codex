<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared QuickSettings top bar for Model Tabs (SD/Flux/Chroma/ZImage/WAN).
Loads `/api/options`, `/api/models`, `/api/models/inventory`, and `/api/paths`, then filters/presents per-family selectors (models/TE/VAE)
and emits/commits overrides (device/engine/memory flags) used by generation payload builders.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsBar` (component): Main QuickSettings SFC; includes “advanced” UI, per-family subcomponents, and selector filtering logic.
- `cancelAdvancedAnimation` (function): Cancels in-flight advanced-row animations (used by toggling/resize logic).
- `easeOutCubic` (function): Easing helper used for advanced-row animations.
- `syncAdvancedHeight` (function): Measures/synchronizes advanced-row height for smooth expand/collapse transitions.
- `toggleAdvancedRow` (function): Toggles the advanced row (uses animation helpers and persisted UI state).
- `currentTab` (function): Determines the current tab kind (`txt2img`/`img2img`/`txt2vid`/`img2vid`) from routing/state.
- `TabFamily` (type): Normalized model family identifiers used for per-family UI filtering (`sd15`/`sdxl`/`flux1`/`chroma`/`wan`/`zimage`).
- `normalizeTabFamily` (function): Normalizes unknown inputs to a `TabFamily` (or `null`).
- `tabFamilyFromStorage` (function): Loads persisted per-tab family from local storage (used to keep UI consistent on reload).
- `normalizePath` (function): Normalizes paths for stable comparisons (slash/case handling).
- `MetadataKind` (type): Discriminant for inline metadata popups (checkpoint/TE/VAE/WAN stage).
- `onShowMetadata` (function): Resolves selection metadata and opens a modal.
- `fileInPaths` (function): Checks whether a file path belongs to the configured roots for a key from `/api/paths` (drives selector filtering).
- `modelMatchesFamily` (function): Determines whether a checkpoint/inventory entry matches a given family (combines heuristics + `fileInPaths`).
- `isVaeForFamily` (function): Filters VAE entries to those relevant for the current family.
- `normalizeTextEncoderLabels` (function): Normalizes raw TE values into a stable label list (used for Flux/WAN multi-TE cases).
- `WanAssetsParams` (type): Minimal WAN assets triple used for payload building (metadata dir + TE + VAE).
- `currentWanAssets` (function): Builds `WanAssetsParams` from current UI selections (used by WAN payload generation).
- `textEncoderLabel` (function): Converts raw TE selector values into a canonical label (handles WAN-style prefixes).
- `onPrimaryTextEncoderChange` (function): Applies primary text-encoder selection changes (and triggers dependent updates).
- `onSecondaryTextEncoderChange` (function): Applies secondary text-encoder selection changes (Flux/Kontext dual-encoder workflows).
- `onUnetDtypeChange` (function): Updates UNet dtype selection in quicksettings/store.
- `onGpuWeightsChange` (function): Updates GPU weights slider value for relevant engines.
- `onAttentionChange` (function): Updates attention implementation selection (backend/runtime policy).
- `onSmartOffloadChange` (function): Updates Smart Offload toggle (impacts per-request memory behavior).
- `onSmartFallbackChange` (function): Updates Smart Fallback toggle (best-effort OOM fallback behavior).
- `onSmartCacheChange` (function): Updates Smart Cache toggle (conditioning caching behavior).
- `onCoreStreamingChange` (function): Updates core streaming toggle (runtime streaming behavior).
- `onWanModeChange` (function): Updates WAN mode selection and derived controls.
- `onWanGuidedGen` (function): Opens WAN guided generation flow (UI navigation/CTA).
- `enginePrefixForFamily` (function): Maps a `TabFamily` to the engine prefix used in options/labels.
- `dedupePaths` (function): Deduplicates path lists for selector options.
- `promptForPath` (function): Prompts the user for a path update when needed (path config UX).
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
          @browseHigh="onWanBrowseHigh"
          @browseLow="onWanBrowseLow"
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
          :vae="store.currentVae"
          :vae-choices="filteredVaeChoices"
          :text-encoder="primaryTextEncoder"
          :text-encoder-choices="filteredTextEncoderChoices"
          @update:checkpoint="onModelChange"
          @update:vae="onVaeChange"
          @update:textEncoder="onPrimaryTextEncoderChange"
          @addCheckpointPath="onAddCheckpointPath"
          @addVaePath="onAddVaePath"
          @addTencPath="onAddTencPath"
          @showMetadata="onShowMetadata"
        />
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
          :mode="store.currentMode"
          :mode-choices="filteredModeChoices"
          :checkpoint="effectiveCheckpoint"
          :checkpoints="filteredModelTitles"
          :vae="store.currentVae"
          :vae-choices="filteredVaeChoices"
          :text-encoder="primaryTextEncoder"
          :text-encoder-choices="filteredTextEncoderChoices"
          text-encoder-automatic-label="Built-in"
          :show-text-encoder="activeFamily !== 'sd15' && activeFamily !== 'sdxl'"
          @update:mode="onModeChange"
          @update:checkpoint="onModelChange"
          @update:vae="onVaeChange"
          @update:textEncoder="onPrimaryTextEncoderChange"
          @addCheckpointPath="onAddCheckpointPath"
          @addVaePath="onAddVaePath"
          @showMetadata="onShowMetadata"
        />
        <div class="quicksettings-group qs-group-models">
          <label class="label-muted">Models</label>
          <div class="qs-row">
            <button class="btn qs-btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
          </div>
        </div>
      </template>
    </div>

    <div ref="advancedRowEl" class="quicksettings-advanced-collapse" :data-state="advancedOpen ? 'open' : 'closed'">
      <div ref="advancedRowInnerEl" class="quicksettings-row quicksettings-row--advanced-inner">
        <div class="quicksettings-group qs-group-attention">
          <label class="label-muted">Attention Backend</label>
          <div class="qs-row">
            <select class="select-md" :value="store.currentAttention" @change="onAttentionChange(($event.target as HTMLSelectElement).value)">
              <option v-for="opt in store.attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            </select>
          </div>
        </div>

        <div class="quicksettings-group qs-group-perf qs-group-perf-vram">
          <label class="label-muted">GPU VRAM (MB)</label>
          <div class="qs-row">
            <input
              class="ui-input"
              type="number"
              :min="0"
              :max="store.gpuTotalMb"
              :value="store.gpuWeightsMb"
              @change="onGpuWeightsChange(Number(($event.target as HTMLInputElement).value))"
            />
          </div>
        </div>

        <QuickSettingsPerf
          :smart-offload="store.smartOffload"
          :smart-fallback="store.smartFallback"
          :smart-cache="store.smartCache"
          :core-streaming="store.coreStreaming"
          @update:smartOffload="onSmartOffloadChange"
          @update:smartFallback="onSmartFallbackChange"
          @update:smartCache="onSmartCacheChange"
          @update:coreStreaming="onCoreStreamingChange"
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
    <AssetMetadataModal v-model="showMetadataModal" :title="metadataModalTitle" :subtitle="metadataModalSubtitle" :payload="metadataModalPayload" />
  </section>
</template>


<script setup lang="ts">
import { onBeforeUnmount, onMounted, computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useUiBlocksStore } from '../stores/ui_blocks'
import { MODEL_TABS_STORAGE_KEY, useModelTabsStore } from '../stores/model_tabs'
import { fetchCheckpointMetadata, fetchFileMetadata, fetchModelInventory, refreshModelInventory, fetchPaths, updatePaths } from '../api/client'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import QuickSettingsBase from './quicksettings/QuickSettingsBase.vue'
import QuickSettingsPerf from './quicksettings/QuickSettingsPerf.vue'
import QuickSettingsWan from './quicksettings/QuickSettingsWan.vue'
import QuickSettingsFlux from './quicksettings/QuickSettingsFlux.vue'
import QuickSettingsChroma from './quicksettings/QuickSettingsChroma.vue'
import QuickSettingsZImage from './quicksettings/QuickSettingsZImage.vue'
import QuickSettingsOverridesModal from './modals/QuickSettingsOverridesModal.vue'
import AssetMetadataModal from './modals/AssetMetadataModal.vue'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const uiBlocks = useUiBlocksStore()
const tabsStore = useModelTabsStore()
const pathsConfig = ref<Record<string, string[]>>({})
type InventoryVae = { name: string; path: string; sha256?: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }
type InventoryWanGguf = { name: string; path: string; sha256?: string; stage: string }
type InventoryTextEncoder = { name: string; path: string; sha256?: string }
const inventoryVaes = ref<InventoryVae[]>([])
const inventoryWan = ref<InventoryWanGguf[]>([])
const inventoryTextEncoders = ref<InventoryTextEncoder[]>([])
const engineCaps = useEngineCapabilitiesStore()
const showOverridesModal = ref(false)
const showMetadataModal = ref(false)
const metadataModalTitle = ref('Metadata')
const metadataModalSubtitle = ref('')
const metadataModalPayload = ref<unknown>(null)
const isLoadingQuicksettings = ref(false)
const advancedOpen = ref(true)
const advancedRowEl = ref<HTMLElement | null>(null)
const advancedRowInnerEl = ref<HTMLElement | null>(null)
const advancedAnimating = ref(false)
let advancedRafId: number | null = null

function cancelAdvancedAnimation(): void {
  if (advancedRafId !== null) cancelAnimationFrame(advancedRafId)
  advancedRafId = null
}

function easeOutCubic(t: number): number {
  const clamped = Math.min(1, Math.max(0, t))
  return 1 - Math.pow(1 - clamped, 3)
}

function syncAdvancedHeight(): void {
  if (!advancedOpen.value) return
  if (advancedAnimating.value) return
  const el = advancedRowEl.value
  const inner = advancedRowInnerEl.value
  if (!el || !inner) return
  const nextHeight = inner.getBoundingClientRect().height
  if (nextHeight > 0) el.style.height = `${nextHeight}px`
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

type TabFamily = 'sd15' | 'sdxl' | 'flux1' | 'chroma' | 'wan' | 'zimage'

function normalizeTabFamily(value: unknown): TabFamily | null {
  const raw = String(value || '').trim().toLowerCase()
  if (raw === 'wan22' || raw === 'wan22_14b' || raw === 'wan22_5b') return 'wan'
  if (raw === 'flux1_chroma') return 'chroma'
  if (raw === 'sd15' || raw === 'sdxl' || raw === 'flux1' || raw === 'chroma' || raw === 'wan' || raw === 'zimage') return raw as TabFamily
  return null
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

function tabFamilyFromStorage(tabId: string): TabFamily | null {
  if (!tabId) return null
  try {
    const raw = localStorage.getItem(MODEL_TABS_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as { tabs?: unknown[] }
    const list = Array.isArray(parsed.tabs) ? parsed.tabs : []
    const match = list.find((t) => String((t as any)?.id || '') === tabId) as any
    return normalizeTabFamily(match?.type)
  } catch {
    return null
  }
}

let routeActiveSyncToken = 0
watch(routeTabId, async (tabId) => {
  const token = ++routeActiveSyncToken
  if (!tabId) return
  if (!tabsStore.tabs.length) {
    try {
      await tabsStore.load()
    } catch {
      // ignore; store handles bootstrap fallback
    }
  }
  if (token !== routeActiveSyncToken) return
  tabsStore.setActive(tabId)
}, { immediate: true })

const activeFamily = computed<TabFamily>(() => {
  if (route.path.startsWith('/models/') && routeTabId.value) {
    const type = normalizeTabFamily(activeModelTab.value?.type) || tabFamilyFromStorage(routeTabId.value)
    if (type) return type
  }

  // Fallback to global engine selection
  const eng = (store.currentEngine || '').toLowerCase()
  if (eng === 'flux1_chroma' || eng === 'chroma') return 'chroma'
  if (eng.startsWith('flux1')) return 'flux1'
  if (eng.startsWith('sdxl')) return 'sdxl'
  if (eng.startsWith('wan')) return 'wan'
  if (eng.startsWith('zimage')) return 'zimage'

  return 'sd15'
})
const semanticEngine = computed<string>(() => {
  // Prefer semantic engine from UI blocks when available (video tabs etc.).
  if (uiBlocks.semanticEngine) return uiBlocks.semanticEngine
  // Fallback to global Codex engine selection.
  return store.currentEngine || 'sd15'
})

async function loadInventory(options?: { forceRefresh?: boolean }): Promise<void> {
  try {
    const inv = options?.forceRefresh ? await refreshModelInventory() : await fetchModelInventory()
    inventoryVaes.value = inv.vaes
    inventoryWan.value = (inv.wan22?.gguf ?? []).map((g: any) => ({
      name: String(g.name),
      path: String(g.path),
      sha256: typeof g?.sha256 === 'string' ? String(g.sha256) : undefined,
      stage: String(g.stage || 'unknown'),
    }))
    // Text encoder files are available via inventory for future use (e.g., Flux overrides).
    inventoryTextEncoders.value = inv.text_encoders ?? []
  } catch (e) {
    inventoryVaes.value = []
    inventoryWan.value = []
    inventoryTextEncoders.value = []
  }
}

async function loadPaths(): Promise<void> {
  try {
    const res = await fetchPaths()
    pathsConfig.value = (res.paths || {}) as Record<string, string[]>
  } catch (e) {
    pathsConfig.value = {}
  }
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

function findModelByTitle(title: string): any | undefined {
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

function onShowMetadata(payload: { kind: MetadataKind; value: string }): void {
  const kind = String((payload as any)?.kind || '').trim() as MetadataKind
  const value = String((payload as any)?.value || '').trim()
  if (!kind || !value) return

  let title = 'Metadata'
  let subtitle = ''
  let out: unknown = null
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
    out = { selection: value, kind }
  }

  if (filePathForMetadata && typeof out === 'object' && out !== null) {
    ;(out as any).metadata = { status: 'loading' }
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
	      } catch (e: any) {
	        metadataModalPayload.value = {
	          selection: value,
	          metadata: { status: 'error', error: String(e?.message || e) },
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
	      if (typeof current !== 'object' || current === null) return
	      const flat = (res as any)?.flat
	      const nested = (res as any)?.nested
	      const sizeBytes = (res as any)?.summary?.file?.size_bytes

	      const filePatch: Record<string, unknown> = {}
	      if (typeof sizeBytes === 'number' && Number.isFinite(sizeBytes) && sizeBytes >= 0) {
	        const mb = sizeBytes / 1_000_000
	        const gb = sizeBytes / 1_000_000_000
	        filePatch['file.size.bytes'] = sizeBytes
	        filePatch['file.size.megabytes'] = Number(mb.toFixed(3))
	        filePatch['file.size.gigabytes'] = Number(gb.toFixed(3))
	      }

	      const metaOut: Record<string, unknown> = {
	        raw: flat && typeof flat === 'object' ? flat : (res as any),
	        nested: nested && typeof nested === 'object' ? nested : undefined,
	      }
	      metadataModalPayload.value = {
	        ...(current as any),
	        ...filePatch,
	        metadata: metaOut,
	      }
	    } catch (e: any) {
      const current = metadataModalPayload.value
      if (typeof current !== 'object' || current === null) return
      metadataModalPayload.value = {
        ...(current as any),
        metadata: { status: 'error', error: String(e?.message || e) },
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

function modelMatchesFamily(
  meta: Record<string, unknown> | undefined,
  title: string,
  file: string,
  family: TabFamily,
): boolean {
  const prefix = enginePrefixForFamily(family)
  const key = `${prefix}_ckpt`
  const roots = pathsConfig.value[key] || []
  if (roots.length) return fileInPaths(file, key)

  const fam = String((meta?.['family'] as string) || (meta?.['model_family'] as string) || '').toLowerCase()
  const t = (title || '').toLowerCase(); const f = (file || '').toLowerCase()
  if (fam) return fam.includes(family)
  if (family === 'sdxl') return t.includes('sdxl') || f.includes('sdxl')
  if (family === 'sd15') return t.includes('1.5') || t.includes('sd15') || f.includes('sd15') || f.includes('v1-5')
  if (family === 'flux1') return false
  if (family === 'chroma') return t.includes('chroma') || f.includes('chroma')
  if (family === 'wan') return t.includes('wan') || f.includes('wan')
  if (family === 'zimage') return t.includes('zimage') || t.includes('z-image') || t.includes('z_image') || f.includes('zimage') || f.includes('z-image') || f.includes('z_image')
  return true
}

const filteredModels = computed(() => {
  const fam = activeFamily.value
  return store.models.filter(m => modelMatchesFamily(m.metadata as Record<string, unknown> | undefined, m.title, m.filename, fam))
})
const filteredModelTitles = computed(() => filteredModels.value.map(m => m.title))

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

const filteredVaeChoices = computed(() => {
  const fam = activeFamily.value
  if (fam === 'flux1' || fam === 'chroma') {
    return inventoryVaes.value
      .filter((v) => typeof v.path === 'string' && fileInPaths(v.path, 'flux1_vae'))
      .map((v) => String(v.path || ''))
  }
  if (fam === 'zimage') {
    return inventoryVaes.value
      .filter((v) => typeof v.path === 'string' && (fileInPaths(v.path, 'zimage_vae') || fileInPaths(v.path, 'flux1_vae')))
      .map((v) => String(v.path || ''))
  }
  return (store.vaeChoices.length ? store.vaeChoices : ['Automatic']).filter(v => v === 'Automatic' || isVaeForFamily(v, fam))
})

const filteredModeChoices = computed(() => {
  const fam = activeFamily.value
  const base = store.modeChoices
  if (fam === 'sdxl') return base.filter(m => ['Normal','Lightning','Turbo'].includes(m))
  if (fam === 'sd15') return base.filter(m => ['Normal','LCM','Turbo'].includes(m))
  if (fam === 'flux1' || fam === 'chroma') return base.filter(m => ['Normal'].includes(m))
  return base
})

const filteredUnetDtypeChoices = computed(() => {
  const fam = activeFamily.value
  const base = store.unetDtypeChoices
  if (fam === 'flux1' || fam === 'chroma') return base.filter(x => /Automatic|float8|fp16/i.test(x))
  return base
})

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
  const tab = activeModelTab.value
  if (!tab || tab.type === 'wan') return null
  return tab
})

function normalizeTextEncoderLabels(raw: unknown): string[] {
  if (!Array.isArray(raw)) return []
  return raw.map((it) => String(it || '').trim()).filter((it) => it.length > 0)
}

const effectiveTextEncoders = computed(() => {
  const tab = activeImageTab.value
  if (!tab) return store.currentTextEncoders
  return normalizeTextEncoderLabels((tab.params as any)?.textEncoders)
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
  const ckpt = String((tab.params as any)?.checkpoint || '').trim()
  if (ckpt) return ckpt
  return filteredModelTitles.value[0] ?? ''
})

watch(
  () => [activeImageTab.value?.id ?? '', filteredModelTitles.value] as const,
  ([tabId, models]) => {
    if (!tabId) return
    const tab = activeImageTab.value
    if (!tab) return
    const ckpt = String((tab.params as any)?.checkpoint || '').trim()
    if (ckpt) return
    const first = models[0]
    if (!first) return
    void tabsStore.updateParams(tab.id, { checkpoint: first } as any)
  },
  { immediate: true },
)

async function initQuicksettings(options?: { forceInventoryRefresh?: boolean; forceModelsRefresh?: boolean }): Promise<void> {
  isLoadingQuicksettings.value = true
  try {
    await store.init()
    if (options?.forceModelsRefresh === true) {
      await store.refreshModelsList()
    }
    await Promise.all([
      loadPaths(),
      loadInventory({ forceRefresh: options?.forceInventoryRefresh === true }),
    ])
  } finally {
    isLoadingQuicksettings.value = false
  }
}

async function refreshAll(): Promise<void> {
  await initQuicksettings({ forceInventoryRefresh: true, forceModelsRefresh: true })
}

const wanHighDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    if (g.stage !== 'high') continue
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
    if (g.stage !== 'low') continue
    const path = String(g.path || '').trim()
    if (!path) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

type WanModelMode = 'i2v_14b' | 't2v_14b' | 'i2v_5b' | 't2v_5b' | 'v2v_14b'

function _wanRepoForMode(mode: WanModelMode): string {
  if (mode === 't2v_14b') return 'Wan-AI/Wan2.2-T2V-A14B-Diffusers'
  if (mode === 'i2v_5b' || mode === 't2v_5b') return 'Wan-AI/Wan2.2-TI2V-5B-Diffusers'
  return 'Wan-AI/Wan2.2-I2V-A14B-Diffusers'
}

const wanModelMode = computed<WanModelMode>(() => {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return 't2v_14b'
  const video = (tab.params as any).video as { useInitVideo?: boolean; useInitImage?: boolean } | undefined
  const rawAssets = ((tab.params as any).assets as WanAssetsParams | undefined) || { metadata: '', textEncoder: '', vae: '' }
  const meta = String(rawAssets.metadata || '').trim().toLowerCase()
  const is5b = meta.includes('ti2v-5b') || meta.includes('5b')
  const kind = video?.useInitVideo ? 'v2v' : (video?.useInitImage ? 'i2v' : 't2v')

  if (is5b) return kind === 'i2v' ? 'i2v_5b' : 't2v_5b'
  if (kind === 'v2v') return 'v2v_14b'
  if (kind === 'i2v') return 'i2v_14b'
  return 't2v_14b'
})

const wanLightx2v = computed(() => {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return false
  return Boolean((tab.params as any)?.lightx2v)
})

const wanHighModel = computed(() => {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return ''
  const high = (tab.params as any).high as { modelDir?: string } | undefined
  return high?.modelDir || ''
})

const wanLowModel = computed(() => {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return ''
  const low = (tab.params as any).low as { modelDir?: string } | undefined
  return low?.modelDir || ''
})

type WanAssetsParams = { metadata: string; textEncoder: string; vae: string }

function currentWanAssets(): WanAssetsParams {
  const base: WanAssetsParams = { metadata: '', textEncoder: '', vae: '' }
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return base
  const raw = (tab.params as any).assets as WanAssetsParams | undefined
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
async function onModeChange(value: string): Promise<void> {
  await store.setMode(value)
}

async function onModelChange(value: string): Promise<void> {
  const tab = activeImageTab.value
  if (tab) {
    await tabsStore.updateParams(tab.id, { checkpoint: String(value || '') } as any)
    return
  }
  await store.setModel(value)
}

async function onVaeChange(value: string): Promise<void> {
  await store.setVae(value)
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
    await tabsStore.updateParams(tab.id, { textEncoders: fluxLabels } as any)
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
    void updateFlux1TextEncoders(primary, secondary)
  } else {
    const tab = activeImageTab.value
    const payload = value ? [value] : []
    if (tab) {
      void tabsStore.updateParams(tab.id, { textEncoders: payload } as any)
      return
    }
    void store.setTextEncoders(payload)
  }
}

function onSecondaryTextEncoderChange(value: string): void {
  const fam = activeFamily.value
  if (fam !== 'flux1') return
  const primary = flux1TextEncoderPrimary.value || ''
  const secondary = value || ''
  void updateFlux1TextEncoders(primary, secondary)
}

function onUnetDtypeChange(value: string): void {
  void store.setUnetDtype(value)
}

function onGpuWeightsChange(value: number): void {
  const v = Number(value)
  if (Number.isFinite(v)) void store.setGpuWeightsMb(v)
}

function onAttentionChange(value: string): void {
  void store.setAttentionBackend(value)
}

function onSmartOffloadChange(value: boolean): void {
  void store.setSmartOffload(value)
}

function onSmartFallbackChange(value: boolean): void {
  void store.setSmartFallback(value)
}

function onSmartCacheChange(value: boolean): void {
  void store.setSmartCache(value)
}

function onCoreStreamingChange(value: boolean): void {
  void store.setCoreStreaming(value)
}

async function onWanModeChange(value: string): Promise<void> {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return

  const raw = String(value || '').trim().toLowerCase()
  const nextMode: WanModelMode =
    raw === 'i2v_14b' ? 'i2v_14b'
      : raw === 't2v_14b' ? 't2v_14b'
        : raw === 'i2v_5b' ? 'i2v_5b'
          : raw === 't2v_5b' ? 't2v_5b'
            : raw === 'v2v_14b' ? 'v2v_14b'
              : 't2v_14b'

  const currentVideo = ((tab.params as any).video as any) || {}
  const videoPatch: Record<string, unknown> = {}
  if (nextMode === 't2v_14b' || nextMode === 't2v_5b') {
    videoPatch.useInitVideo = false
    videoPatch.initVideoName = ''
    videoPatch.initVideoPath = ''
    videoPatch.useInitImage = false
    videoPatch.initImageData = ''
    videoPatch.initImageName = ''
  } else if (nextMode === 'v2v_14b') {
    videoPatch.useInitVideo = true
    videoPatch.useInitImage = false
    videoPatch.initImageData = ''
    videoPatch.initImageName = ''
  } else {
    // i2v
    videoPatch.useInitVideo = false
    videoPatch.initVideoName = ''
    videoPatch.initVideoPath = ''
    videoPatch.useInitImage = true
  }

  const currentAssets = currentWanAssets()
  const nextAssets = { ...currentAssets, metadata: _wanRepoForMode(nextMode) }

  await tabsStore.updateParams(
    tab.id,
    {
      video: { ...currentVideo, ...videoPatch },
      assets: nextAssets,
    } as any,
  )
}

async function onWanLightx2vChange(value: boolean): Promise<void> {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return
  await tabsStore.updateParams(tab.id, { lightx2v: Boolean(value) } as any)
}

async function onWanHighModelChange(value: string): Promise<void> {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return
  const current = (tab.params as any).high || {}
  await tabsStore.updateParams(tab.id, { high: { ...current, modelDir: value } })
}

async function onWanLowModelChange(value: string): Promise<void> {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return
  const current = (tab.params as any).low || {}
  await tabsStore.updateParams(tab.id, { low: { ...current, modelDir: value } })
}

async function onWanTextEncoderChange(value: string): Promise<void> {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return
  const current = currentWanAssets()
  await tabsStore.updateParams(tab.id, { assets: { ...current, textEncoder: value } })
}

async function onWanVaeChange(value: string): Promise<void> {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return
  const current = currentWanAssets()
  await tabsStore.updateParams(tab.id, { assets: { ...current, vae: value } })
}

function onWanGuidedGen(): void {
  const tab = activeModelTab.value
  if (!tab || tab.type !== 'wan') return
  window.dispatchEvent(new CustomEvent('codex-wan-guided-gen', { detail: { tabId: tab.id } }))
}

function enginePrefixForFamily(fam: TabFamily): 'sd15' | 'sdxl' | 'flux1' | 'wan22' | 'zimage' {
  if (fam === 'wan') return 'wan22'
  if (fam === 'chroma') return 'flux1'
  return fam
}

async function onAddCheckpointPath(): Promise<void> {
  try {
    const res = await fetchPaths()
    const raw = (res.paths || {}) as Record<string, string[]>
    const fam = activeFamily.value
    const prefix = enginePrefixForFamily(fam)
    const key = `${prefix}_ckpt`
    const current = (raw[key] || []) as string[]
    const next = window.prompt('Add checkpoint directory (server path)', '')
    if (!next) return
    const trimmed = next.trim()
    if (!trimmed) return
    const paths = dedupePaths([...current, trimmed])
    const payload: Record<string, string[]> = {}
    for (const [k, v] of Object.entries(raw)) {
      if (k === 'checkpoints' || k === 'vae' || k === 'lora' || k === 'text_encoders') continue
      payload[k] = Array.isArray(v) ? [...v] : []
    }
    payload[key] = paths
    await updatePaths(payload)
  } catch (e) {
    console.error('[quicksettings] failed to add checkpoint path', e)
  }
}

async function onAddVaePath(): Promise<void> {
  try {
    const res = await fetchPaths()
    const raw = (res.paths || {}) as Record<string, string[]>
    const fam = activeFamily.value
    const prefix = enginePrefixForFamily(fam)
    const key = `${prefix}_vae`
    const current = (raw[key] || []) as string[]
    const next = window.prompt('Add VAE directory (server path)', '')
    if (!next) return
    const trimmed = next.trim()
    if (!trimmed) return
    const paths = dedupePaths([...current, trimmed])
    const payload: Record<string, string[]> = {}
    for (const [k, v] of Object.entries(raw)) {
      if (k === 'checkpoints' || k === 'vae' || k === 'lora' || k === 'text_encoders') continue
      payload[k] = Array.isArray(v) ? [...v] : []
    }
    payload[key] = paths
    await updatePaths(payload)
  } catch (e) {
    console.error('[quicksettings] failed to add VAE path', e)
  }
}

async function onAddTencPath(): Promise<void> {
  try {
    const res = await fetchPaths()
    const raw = (res.paths || {}) as Record<string, string[]>
    const fam = activeFamily.value
    const prefix = enginePrefixForFamily(fam)
    const key = `${prefix}_tenc`
    const current = (raw[key] || []) as string[]
    const next = window.prompt('Add Text Encoder directory (server path)', '')
    if (!next) return
    const trimmed = next.trim()
    if (!trimmed) return
    const paths = dedupePaths([...current, trimmed])
    const payload: Record<string, string[]> = {}
    for (const [k, v] of Object.entries(raw)) {
      if (k === 'checkpoints' || k === 'vae' || k === 'lora' || k === 'text_encoders') continue
      payload[k] = Array.isArray(v) ? [...v] : []
    }
    payload[key] = paths
    await updatePaths(payload)
  } catch (e) {
    console.error('[quicksettings] failed to add text encoder path', e)
  }
}

function dedupePaths(list: string[]): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const p of list) {
    const key = p.replace(/\\+/g, '/').replace(/\/$/, '')
    if (!seen.has(key)) {
      seen.add(key)
      out.push(p)
    }
  }
  return out
}

function promptForPath(label: string, current: string): string | null {
  const next = window.prompt(label, current || '')
  if (next === null) return null
  const trimmed = next.trim()
  return trimmed || null
}

async function onWanBrowseHigh(): Promise<void> {
  const path = promptForPath('WAN High model (.gguf) path or sha256', wanHighModel.value)
  if (path) await onWanHighModelChange(path)
}

async function onWanBrowseLow(): Promise<void> {
  const path = promptForPath('WAN Low model (.gguf) path or sha256', wanLowModel.value)
  if (path) await onWanLowModelChange(path)
}

async function onWanBrowseTe(): Promise<void> {
  const current = wanTextEncoder.value
  const next = promptForPath('WAN Text Encoder (.safetensors or .gguf) path or sha256', current)
  if (next === null) return
  const normalized = next.replace(/\\+/g, '/')
  // Keep the stored value consistent with the dropdown labels (wan22/<abs_path>).
  const stored = normalized.startsWith('wan22/') || !normalized.startsWith('/') ? normalized : `wan22/${normalized}`
  await onWanTextEncoderChange(stored)
}

async function onWanBrowseVae(): Promise<void> {
  const current = wanVae.value
  const next = promptForPath('WAN VAE path or sha256', current)
  if (next !== null) await onWanVaeChange(next)
}

function openOverrides(): void {
  showOverridesModal.value = true
}

onMounted(() => {
  void initQuicksettings()
  void presets.init(currentTab())
  void engineCaps.init()
  window.addEventListener('resize', syncAdvancedHeight)
  requestAnimationFrame(syncAdvancedHeight)
})

onBeforeUnmount(() => {
  cancelAdvancedAnimation()
  window.removeEventListener('resize', syncAdvancedHeight)
})

watch(() => route.path, async () => {
  await presets.init(currentTab())
  await loadInventory()
})

// random seed button removed from quicksettings; presets applied elsewhere
</script>
