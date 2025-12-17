<template>
  <section :class="['quicksettings', { 'quicksettings-loading': isLoadingQuicksettings }]">
    <!-- WAN-specific quicksettings -->
    <template v-if="activeFamily === 'wan'">
      <QuickSettingsWan
        :mode="wanMode"
        :model-format="wanModelFormat"
        :high-model="wanHighModel"
        :high-choices="wanHighDirChoices"
        :low-model="wanLowModel"
        :low-choices="wanLowDirChoices"
        :metadata-dir="wanMetadataDir"
        :metadata-choices="wanMetadataChoices"
        :text-encoder="wanTextEncoder"
        :text-encoder-choices="wanTextEncoderChoices"
        :vae="wanVae"
        :vae-choices="wanVaeChoices"
        :unet-dtype="store.currentUnetDtype"
        :unet-dtype-choices="filteredUnetDtypeChoices"
        :gpu-weights-mb="store.gpuWeightsMb"
        :gpu-total-mb="store.gpuTotalMb"
        :attention-backend="store.currentAttention"
        :attention-choices="store.attentionChoices"
        @update:mode="onWanModeChange"
        @update:modelFormat="onWanModelFormatChange"
        @update:highModel="onWanHighModelChange"
        @update:lowModel="onWanLowModelChange"
        @update:metadataDir="onWanMetadataDirChange"
        @update:textEncoder="onWanTextEncoderChange"
        @update:vae="onWanVaeChange"
        @update:unetDtype="onUnetDtypeChange"
        @update:gpuWeightsMb="onGpuWeightsChange"
        @update:attentionBackend="onAttentionChange"
        @browseHigh="onWanBrowseHigh"
        @browseLow="onWanBrowseLow"
        @browseMetadata="onWanBrowseMetadata"
        @browseTe="onWanBrowseTe"
        @browseVae="onWanBrowseVae"
        @openOverrides="openOverrides"
        @guidedGen="onWanGuidedGen"
      />
      <QuickSettingsOverridesModal v-model="showOverridesModal" />
    </template>

    <!-- Flux-specific quicksettings -->
    <template v-else-if="activeFamily === 'flux'">
      <QuickSettingsFlux
        :checkpoint="store.currentModel"
        :checkpoints="filteredModelTitles"
        :vae="store.currentVae"
        :vae-choices="filteredVaeChoices"
        :text-encoder-primary="fluxTextEncoderPrimary"
        :text-encoder-secondary="fluxTextEncoderSecondary"
        :text-encoder-choices="filteredTextEncoderChoices"
        :unet-dtype="store.currentUnetDtype"
        :unet-dtype-choices="filteredUnetDtypeChoices"
        :attention-backend="store.currentAttention"
        :attention-choices="store.attentionChoices"
        @update:checkpoint="onModelChange"
        @update:vae="onVaeChange"
        @update:textEncoderPrimary="onPrimaryTextEncoderChange"
        @update:textEncoderSecondary="onSecondaryTextEncoderChange"
        @update:unetDtype="onUnetDtypeChange"
        @update:attentionBackend="onAttentionChange"
        @addCheckpointPath="onAddCheckpointPath"
        @addVaePath="onAddVaePath"
        @addTencPath="onAddTencPath"
        @openOverrides="openOverrides"
      />
      <div class="quicksettings-group qs-group-models">
        <label class="label-muted">Models</label>
        <div class="qs-row">
          <button class="btn btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
        </div>
      </div>
      <QuickSettingsPerf
        :unet-dtype="store.currentUnetDtype"
        :unet-dtype-choices="filteredUnetDtypeChoices"
        :gpu-weights-mb="store.gpuWeightsMb"
        :gpu-total-mb="store.gpuTotalMb"
        :smart-offload="store.smartOffload"
        :smart-fallback="store.smartFallback"
        :smart-cache="store.smartCache"
        :core-streaming="store.coreStreaming"
        @update:unetDtype="onUnetDtypeChange"
        @update:gpuWeightsMb="onGpuWeightsChange"
        @update:smartOffload="onSmartOffloadChange"
        @update:smartFallback="onSmartFallbackChange"
        @update:smartCache="onSmartCacheChange"
        @update:coreStreaming="onCoreStreamingChange"
      />
      <QuickSettingsOverridesModal v-model="showOverridesModal" />
    </template>

    <!-- Z Image-specific quicksettings -->
    <template v-else-if="activeFamily === 'zimage'">
      <QuickSettingsZImage
        :checkpoint="store.currentModel"
        :checkpoints="filteredModelTitles"
        :vae="store.currentVae"
        :vae-choices="filteredVaeChoices"
        :text-encoder="primaryTextEncoder"
        :text-encoder-choices="filteredTextEncoderChoices"
        :unet-dtype="store.currentUnetDtype"
        :unet-dtype-choices="filteredUnetDtypeChoices"
        :attention-backend="store.currentAttention"
        :attention-choices="store.attentionChoices"
        @update:checkpoint="onModelChange"
        @update:vae="onVaeChange"
        @update:textEncoder="onPrimaryTextEncoderChange"
        @update:unetDtype="onUnetDtypeChange"
        @update:attentionBackend="onAttentionChange"
        @addCheckpointPath="onAddCheckpointPath"
        @addVaePath="onAddVaePath"
        @addTencPath="onAddTencPath"
        @openOverrides="openOverrides"
      />
      <div class="quicksettings-group qs-group-models">
        <label class="label-muted">Models</label>
        <div class="qs-row">
          <button class="btn btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
        </div>
      </div>
      <QuickSettingsPerf
        :unet-dtype="store.currentUnetDtype"
        :unet-dtype-choices="filteredUnetDtypeChoices"
        :gpu-weights-mb="store.gpuWeightsMb"
        :gpu-total-mb="store.gpuTotalMb"
        :smart-offload="store.smartOffload"
        :smart-fallback="store.smartFallback"
        :smart-cache="store.smartCache"
        :core-streaming="store.coreStreaming"
        @update:unetDtype="onUnetDtypeChange"
        @update:gpuWeightsMb="onGpuWeightsChange"
        @update:smartOffload="onSmartOffloadChange"
        @update:smartFallback="onSmartFallbackChange"
        @update:smartCache="onSmartCacheChange"
        @update:coreStreaming="onCoreStreamingChange"
      />
      <QuickSettingsOverridesModal v-model="showOverridesModal" />
    </template>

    <!-- Default (SD15/SDXL) quicksettings -->
    <template v-else>
      <QuickSettingsBase
        :mode="store.currentMode"
        :mode-choices="filteredModeChoices"
        :checkpoint="store.currentModel"
        :checkpoints="filteredModelTitles"
        :hide-checkpoint="hideCheckpoint"
        :vae="store.currentVae"
        :vae-choices="filteredVaeChoices"
        :text-encoder="primaryTextEncoder"
        :text-encoder-choices="filteredTextEncoderChoices"
        :attention-backend="store.currentAttention"
        :attention-choices="store.attentionChoices"
        text-encoder-automatic-label="Built-in"
        :show-text-encoder="activeFamily !== 'sd15' && activeFamily !== 'sdxl'"
        @update:mode="onModeChange"
        @update:checkpoint="onModelChange"
        @update:vae="onVaeChange"
        @update:textEncoder="onPrimaryTextEncoderChange"
        @update:attentionBackend="onAttentionChange"
        @addCheckpointPath="onAddCheckpointPath"
        @addVaePath="onAddVaePath"
        @openOverrides="openOverrides"
      />
      <div class="quicksettings-group qs-group-models">
        <label class="label-muted">Models</label>
        <div class="qs-row">
          <button class="btn btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh lists">Refresh</button>
        </div>
        <div v-if="currentPathsHint" class="qs-row qs-paths-hint">
          <small class="label-muted">{{ currentPathsHint }}</small>
        </div>
      </div>
      <QuickSettingsPerf
        :unet-dtype="store.currentUnetDtype"
        :unet-dtype-choices="filteredUnetDtypeChoices"
        :gpu-weights-mb="store.gpuWeightsMb"
        :gpu-total-mb="store.gpuTotalMb"
        :smart-offload="store.smartOffload"
        :smart-fallback="store.smartFallback"
        :smart-cache="store.smartCache"
        :core-streaming="store.coreStreaming"
        @update:unetDtype="onUnetDtypeChange"
        @update:gpuWeightsMb="onGpuWeightsChange"
        @update:smartOffload="onSmartOffloadChange"
        @update:smartFallback="onSmartFallbackChange"
        @update:smartCache="onSmartCacheChange"
        @update:coreStreaming="onCoreStreamingChange"
      />
      <QuickSettingsOverridesModal v-model="showOverridesModal" />
    </template>
  </section>
</template>


<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useUiBlocksStore } from '../stores/ui_blocks'
import { useModelTabsStore } from '../stores/model_tabs'
import { fetchModelInventory, refreshModelInventory, fetchPaths, updatePaths } from '../api/client'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import QuickSettingsBase from './quicksettings/QuickSettingsBase.vue'
import QuickSettingsPerf from './quicksettings/QuickSettingsPerf.vue'
import QuickSettingsWan from './quicksettings/QuickSettingsWan.vue'
import QuickSettingsFlux from './quicksettings/QuickSettingsFlux.vue'
import QuickSettingsZImage from './quicksettings/QuickSettingsZImage.vue'
import QuickSettingsOverridesModal from './modals/QuickSettingsOverridesModal.vue'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const uiBlocks = useUiBlocksStore()
const tabsStore = useModelTabsStore()
const pathsConfig = ref<Record<string, string[]>>({})
const inventoryVaes = ref<Array<{ name: string; path: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }>>([])
const inventoryWan = ref<Array<{ name: string; path: string; stage: string }>>([])
const inventoryTextEncoders = ref<Array<{ name: string; path: string }>>([])
const inventoryMetadata = ref<Array<{ name: string; path: string }>>([])
const engineCaps = useEngineCapabilitiesStore()
const showOverridesModal = ref(false)
const isLoadingQuicksettings = ref(false)

function currentTab(): 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' {
  const p = route.path
  if (p.startsWith('/img2img')) return 'img2img'
  if (p.startsWith('/txt2vid')) return 'txt2vid'
  if (p.startsWith('/img2vid')) return 'img2vid'
  return 'txt2img'
}

const activeFamily = computed<'sd15' | 'sdxl' | 'flux' | 'wan' | 'zimage'>(() => {
  const p = route.path
  // Dedicated inference surfaces override tab state
  if (p.startsWith('/flux')) return 'flux'
  if (p.startsWith('/sdxl')) return 'sdxl'
  if (p.startsWith('/zimage')) return 'zimage'

  // Model tabs: derive from active tab type
  if (p.startsWith('/models')) {
    const tabType = tabsStore.activeTab?.type
    if (tabType === 'sd15' || tabType === 'sdxl' || tabType === 'flux' || tabType === 'wan' || tabType === 'zimage') {
      return tabType as 'sd15' | 'sdxl' | 'flux' | 'wan' | 'zimage'
    }
  }

  // Fallback to global engine selection
  const eng = (store.currentEngine || '').toLowerCase()
  if (eng.startsWith('flux')) return 'flux'
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
      stage: String(g.stage || 'unknown'),
    }))
    // Text encoder files are available via inventory for future use (e.g., Flux overrides).
    inventoryTextEncoders.value = inv.text_encoders ?? []
    inventoryMetadata.value = inv.metadata ?? []
  } catch (e) {
    inventoryVaes.value = []
    inventoryWan.value = []
    inventoryTextEncoders.value = []
    inventoryMetadata.value = []
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
    // Repo-relative root (e.g. 'models/flux-tenc'): match by suffix segment.
    const rel = rNorm.startsWith('/') ? rNorm.slice(1) : rNorm
    if (fNorm.includes('/' + rel + '/') || fNorm.endsWith('/' + rel)) return true
  }
  return false
}

function modelMatchesFamily(meta: Record<string, unknown> | undefined, title: string, file: string, family: string): boolean {
  const fam = String((meta?.['family'] as string) || (meta?.['model_family'] as string) || '').toLowerCase()
  const t = (title || '').toLowerCase(); const f = (file || '').toLowerCase()
  if (fam) return fam.includes(family)
  if (family === 'sdxl') return t.includes('sdxl') || f.includes('sdxl')
  if (family === 'sd15') return t.includes('1.5') || t.includes('sd15') || f.includes('sd15') || f.includes('v1-5')
  if (family === 'flux') {
    if (fileInPaths(file, 'flux_ckpt')) return true
    return t.includes('flux') || f.includes('flux')
  }
  if (family === 'wan') return t.includes('wan') || f.includes('wan')
  if (family === 'zimage') {
    if (fileInPaths(file, 'zimage_ckpt')) return true
    return t.includes('zimage') || t.includes('z-image') || t.includes('z_image') || f.includes('zimage') || f.includes('z-image') || f.includes('z_image')
  }
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
  if (fam === 'flux') return fileInPaths(path, 'flux_vae')
  if (fam === 'zimage') return fileInPaths(path, 'zimage_vae') || fileInPaths(path, 'flux_vae')  // Z Image uses same VAE as Flux
  return true
}

const filteredVaeChoices = computed(() => {
  const fam = activeFamily.value
  return (store.vaeChoices.length ? store.vaeChoices : ['Automatic']).filter(v => v === 'Automatic' || isVaeForFamily(v, fam))
})

const filteredModeChoices = computed(() => {
  const fam = activeFamily.value
  const base = store.modeChoices
  if (fam === 'sdxl') return base.filter(m => ['Normal','Lightning','Turbo'].includes(m))
  if (fam === 'sd15') return base.filter(m => ['Normal','LCM','Turbo'].includes(m))
  if (fam === 'flux') return base.filter(m => ['Normal'].includes(m))
  return base
})

const filteredUnetDtypeChoices = computed(() => {
  const fam = activeFamily.value
  const base = store.unetDtypeChoices
  if (fam === 'flux') return base.filter(x => /Automatic|float8|fp16/i.test(x))
  return base
})

const filteredTextEncoderChoices = computed(() => {
  const fam = activeFamily.value
  if (fam === 'flux') {
    // For Flux, derive choices from inventory.text_encoders constrained by flux_tenc paths.
    return inventoryTextEncoders.value
      .filter((item) => typeof item.path === 'string' && fileInPaths(item.path, 'flux_tenc'))
      .map((item) => `flux/${item.path}`)
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

const primaryTextEncoder = computed(() => store.currentTextEncoders[0] ?? '')

const fluxTextEncoders = computed(() => store.currentTextEncoders.filter((label) => typeof label === 'string' && label.startsWith('flux/')))
const fluxTextEncoderPrimary = computed(() => fluxTextEncoders.value[0] ?? '')
const fluxTextEncoderSecondary = computed(() => fluxTextEncoders.value[1] ?? '')

const primaryTeAutomaticLabel = 'Built-in'
const secondaryTeAutomaticLabel = 'Secondary (optional)'

const currentPathsHint = computed(() => {
  const fam = activeFamily.value
  const prefix = enginePrefixForFamily(fam)
  const keys = [`${prefix}_ckpt`, `${prefix}_tenc`, `${prefix}_vae`]
  const parts: string[] = []
  for (const key of keys) {
    const vals = pathsConfig.value[key] || []
    if (vals.length) {
      parts.push(`${key}: ${vals.join(', ')}`)
    }
  }
  return parts.join(' | ')
})
const hideCheckpoint = computed(() => {
  const p = route.path
  // In model tabs (/models), the tab manages model dirs (e.g., WAN 2.2); hide checkpoint there.
  if (p.startsWith('/models')) return true
  const isVideo = p.startsWith('/txt2vid') || p.startsWith('/img2vid')
  return isVideo && uiBlocks.semanticEngine === 'wan22'
})

async function initQuicksettings(options?: { forceInventoryRefresh?: boolean }): Promise<void> {
  isLoadingQuicksettings.value = true
  try {
    await store.init()
    await Promise.all([
      loadPaths(),
      loadInventory({ forceRefresh: options?.forceInventoryRefresh === true }),
    ])
  } finally {
    isLoadingQuicksettings.value = false
  }
}

async function refreshAll(): Promise<void> {
  await initQuicksettings({ forceInventoryRefresh: true })
}

// WAN-specific helpers (directories derived from inventory)
function parentDir(path: string): string {
  const norm = path.replace(/\\/g, '/')
  const idx = norm.lastIndexOf('/')
  return idx >= 0 ? norm.slice(0, idx) : norm
}

const wanHighDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    if (g.stage !== 'high') continue
    const dir = parentDir(g.path)
    if (!seen.has(dir)) { seen.add(dir); out.push(dir) }
  }
  return out
})

const wanLowDirChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const g of inventoryWan.value) {
    if (g.stage !== 'low') continue
    const dir = parentDir(g.path)
    if (!seen.has(dir)) { seen.add(dir); out.push(dir) }
  }
  return out
})

const wanMode = computed(() => {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return 'txt2vid'
  const video = (tab.params as any).video as { useInitVideo?: boolean; useInitImage?: boolean } | undefined
  if (video?.useInitVideo) return 'vid2vid'
  if (video?.useInitImage) return 'img2vid'
  return 'txt2vid'
})

const wanModelFormat = computed(() => {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return 'auto'
  const raw = String((tab.params as any)?.modelFormat || 'auto').trim().toLowerCase()
  if (raw === 'diffusers' || raw === 'gguf') return raw
  return 'auto'
})

const wanHighModel = computed(() => {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return ''
  const high = (tab.params as any).high as { modelDir?: string } | undefined
  return high?.modelDir || ''
})

const wanLowModel = computed(() => {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return ''
  const low = (tab.params as any).low as { modelDir?: string } | undefined
  return low?.modelDir || ''
})

type WanAssetsParams = { metadata: string; textEncoder: string; vae: string }

function currentWanAssets(): WanAssetsParams {
  const base: WanAssetsParams = { metadata: '', textEncoder: '', vae: '' }
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return base
  const raw = (tab.params as any).assets as WanAssetsParams | undefined
  return raw ? { ...base, ...raw } : base
}

const wanTextEncoder = computed(() => currentWanAssets().textEncoder || '')
const wanVae = computed(() => currentWanAssets().vae || '')
const wanMetadataDir = computed(() => currentWanAssets().metadata || '')

const wanMetadataChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const item of inventoryMetadata.value) {
    const name = String(item.name || '')
    // Keep the dropdown focused on WAN2.2 metadata repos; users can still Browse… for arbitrary paths.
    if (name && !name.startsWith('Wan-AI/')) continue
    const path = String(item.path || '').trim()
    if (!path) continue
    if (!seen.has(path)) { seen.add(path); out.push(path) }
  }
  return out
})

const wanTextEncoderChoices = computed(() => {
  // WAN22 GGUF requires an explicit TE weights file (.safetensors). Prefer concrete
  // files under the configured wan22_tenc roots rather than root labels from /api/text-encoders.
  return inventoryTextEncoders.value
    .filter((item) => typeof item.path === 'string' && item.path.toLowerCase().endsWith('.safetensors') && fileInPaths(item.path, 'wan22_tenc'))
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

async function updateFluxTextEncoders(primary: string, secondary: string): Promise<void> {
  const all = store.currentTextEncoders.slice()
  const other = all.filter((label) => !String(label).startsWith('flux/'))
  const fluxLabels: string[] = []
  const p = primary.trim()
  const s = secondary.trim()
  if (p) fluxLabels.push(p)
  if (s && s !== p) fluxLabels.push(s)
  const next = [...other, ...fluxLabels]
  await store.setTextEncoders(next)
}

function onPrimaryTextEncoderChange(value: string): void {
  const fam = activeFamily.value
  if (fam === 'flux') {
    const primary = value || ''
    const secondary = fluxTextEncoderSecondary.value || ''
    void updateFluxTextEncoders(primary, secondary)
  } else {
    const payload = value ? [value] : []
    void store.setTextEncoders(payload)
  }
}

function onSecondaryTextEncoderChange(value: string): void {
  const fam = activeFamily.value
  if (fam !== 'flux') return
  const primary = fluxTextEncoderPrimary.value || ''
  const secondary = value || ''
  void updateFluxTextEncoders(primary, secondary)
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

function onWanModeChange(value: string): void {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const raw = String(value || '').trim().toLowerCase()
  const mode = raw === 'vid2vid' ? 'vid2vid' : (raw === 'img2vid' ? 'img2vid' : 'txt2vid')
  window.dispatchEvent(new CustomEvent('codex-wan-mode-change', { detail: { tabId: tab.id, mode } }))
}

async function onWanModelFormatChange(value: string): Promise<void> {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const raw = String(value || '').trim().toLowerCase()
  const fmt = raw === 'diffusers' || raw === 'gguf' ? raw : 'auto'
  await tabsStore.updateParams(tab.id, { modelFormat: fmt } as any)
}

async function onWanHighModelChange(value: string): Promise<void> {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const current = (tab.params as any).high || {}
  await tabsStore.updateParams(tab.id, { high: { ...current, modelDir: value } })
}

async function onWanLowModelChange(value: string): Promise<void> {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const current = (tab.params as any).low || {}
  await tabsStore.updateParams(tab.id, { low: { ...current, modelDir: value } })
}

async function onWanMetadataDirChange(value: string): Promise<void> {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const current = currentWanAssets()
  await tabsStore.updateParams(tab.id, { assets: { ...current, metadata: value } })
}

async function onWanTextEncoderChange(value: string): Promise<void> {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const current = currentWanAssets()
  await tabsStore.updateParams(tab.id, { assets: { ...current, textEncoder: value } })
}

async function onWanVaeChange(value: string): Promise<void> {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  const current = currentWanAssets()
  await tabsStore.updateParams(tab.id, { assets: { ...current, vae: value } })
}

function onWanGuidedGen(): void {
  const tab = tabsStore.activeTab
  if (!tab || tab.type !== 'wan') return
  window.dispatchEvent(new CustomEvent('codex-wan-guided-gen', { detail: { tabId: tab.id } }))
}

function enginePrefixForFamily(fam: 'sd15' | 'sdxl' | 'flux' | 'wan' | 'zimage'): 'sd15' | 'sdxl' | 'flux' | 'wan22' | 'zimage' {
  if (fam === 'wan') return 'wan22'
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
  const path = promptForPath('WAN High model directory (server path)', wanHighModel.value)
  if (path) await onWanHighModelChange(path)
}

async function onWanBrowseLow(): Promise<void> {
  const path = promptForPath('WAN Low model directory (server path)', wanLowModel.value)
  if (path) await onWanLowModelChange(path)
}

async function onWanBrowseMetadata(): Promise<void> {
  const path = promptForPath('WAN Metadata directory (server path)', wanMetadataDir.value)
  if (path) await onWanMetadataDirChange(path)
}

async function onWanBrowseTe(): Promise<void> {
  const current = wanTextEncoder.value
  const next = promptForPath('WAN Text Encoder identifier or path', current)
  if (next === null) return
  const normalized = next.replace(/\\+/g, '/')
  // Keep the stored value consistent with the dropdown labels (wan22/<abs_path>).
  const stored = normalized.startsWith('wan22/') || !normalized.startsWith('/') ? normalized : `wan22/${normalized}`
  await onWanTextEncoderChange(stored)
}

async function onWanBrowseVae(): Promise<void> {
  const current = wanVae.value
  const next = promptForPath('WAN VAE identifier or path', current)
  if (next !== null) await onWanVaeChange(next)
}

function openOverrides(): void {
  showOverridesModal.value = true
}

onMounted(() => {
  void initQuicksettings()
  void presets.init(currentTab())
  void engineCaps.init()
})

watch(() => route.path, async () => {
  await presets.init(currentTab())
  await loadInventory()
})

// random seed button removed from quicksettings; presets applied elsewhere
</script>
