<template>
  <section class="quicksettings">
    <!-- Engine-specific quicksettings surface -->
    <QuickSettingsWan
      v-if="activeFamily === 'wan'"
      :high-model="wanHighModel"
      :high-choices="wanHighDirChoices"
      :low-model="wanLowModel"
      :low-choices="wanLowDirChoices"
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
      @update:highModel="onWanHighModelChange"
      @update:lowModel="onWanLowModelChange"
      @update:textEncoder="onWanTextEncoderChange"
      @update:vae="onWanVaeChange"
      @update:unetDtype="onUnetDtypeChange"
      @update:gpuWeightsMb="onGpuWeightsChange"
      @update:attentionBackend="onAttentionChange"
      @browseHigh="onWanBrowseHigh"
      @browseLow="onWanBrowseLow"
      @browseTe="onWanBrowseTe"
      @browseVae="onWanBrowseVae"
      @openOverrides="openOverrides"
    />

    <QuickSettingsBase
      v-else
      :mode="store.currentMode"
      :mode-choices="filteredModeChoices"
      :checkpoint="store.currentModel"
      :checkpoints="filteredModelTitles"
      :hide-checkpoint="hideCheckpoint"
      :vae="store.currentVae"
      :vae-choices="filteredVaeChoices"
      :text-encoder="store.currentTextEncoders[0] ?? ''"
      :text-encoder-choices="filteredTextEncoderChoices"
      :unet-dtype="store.currentUnetDtype"
      :unet-dtype-choices="filteredUnetDtypeChoices"
      :gpu-weights-mb="store.gpuWeightsMb"
      :gpu-total-mb="store.gpuTotalMb"
      :attention-backend="store.currentAttention"
      :attention-choices="store.attentionChoices"
      :smart-offload="store.smartOffload"
      :smart-fallback="store.smartFallback"
      :smart-cache="store.smartCache"
      text-encoder-automatic-label="Built-in"
      :show-text-encoder="activeFamily !== 'sd15' && activeFamily !== 'sdxl'"
      @update:mode="onModeChange"
      @update:checkpoint="onModelChange"
      @update:vae="onVaeChange"
      @update:textEncoder="onTextEncoderChange"
      @update:unetDtype="onUnetDtypeChange"
      @update:gpuWeightsMb="onGpuWeightsChange"
      @update:attentionBackend="onAttentionChange"
      @update:smartOffload="onSmartOffloadChange"
      @update:smartFallback="onSmartFallbackChange"
      @update:smartCache="onSmartCacheChange"
      @addCheckpointPath="onAddCheckpointPath"
      @addVaePath="onAddVaePath"
      @openOverrides="openOverrides"
    />

    <!-- Right-most refresh button spanning to the end -->
    <div class="quicksettings-group quicksettings-right">
      <label class="label-muted">Models</label>
      <div class="qs-row">
        <button class="btn btn-secondary qs-refresh-btn" type="button" @click="refreshAll" title="Refresh checkpoint, VAE and text encoder lists">Refresh</button>
      </div>
      <div v-if="currentPathsHint" class="qs-row qs-paths-hint">
        <small class="label-muted">{{ currentPathsHint }}</small>
      </div>
    </div>

    <QuickSettingsOverridesModal v-model="showOverridesModal" />
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue'
import { useRoute } from 'vue-router'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useUiBlocksStore } from '../stores/ui_blocks'
import { useModelTabsStore } from '../stores/model_tabs'
import { fetchModelInventory, fetchPaths, updatePaths } from '../api/client'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'
import QuickSettingsBase from './quicksettings/QuickSettingsBase.vue'
import QuickSettingsWan from './quicksettings/QuickSettingsWan.vue'
import QuickSettingsOverridesModal from './modals/QuickSettingsOverridesModal.vue'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const uiBlocks = useUiBlocksStore()
const tabsStore = useModelTabsStore()
const pathsConfig = ref<Record<string, string[]>>({})
const inventoryVaes = ref<Array<{ name: string; path: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }>>([])
const inventoryWan = ref<Array<{ name: string; path: string; stage: string }>>([])
const engineCaps = useEngineCapabilitiesStore()
const showOverridesModal = ref(false)

function currentTab(): 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' {
  const p = route.path
  if (p.startsWith('/img2img')) return 'img2img'
  if (p.startsWith('/txt2vid')) return 'txt2vid'
  if (p.startsWith('/img2vid')) return 'img2vid'
  return 'txt2img'
}

const activeFamily = computed<'sd15' | 'sdxl' | 'flux' | 'wan'>(() => {
  const tabType = tabsStore.activeTab?.type
  if (tabType === 'sd15' || tabType === 'sdxl' || tabType === 'flux' || tabType === 'wan') {
    return tabType
  }

  const p = route.path
  if (p.startsWith('/flux')) return 'flux'
  if (p.startsWith('/sdxl')) return 'sdxl'

  const eng = (store.currentEngine || '').toLowerCase()
  if (eng.startsWith('flux')) return 'flux'
  if (eng.startsWith('sdxl')) return 'sdxl'
  if (eng.startsWith('wan')) return 'wan'

  return 'sd15'
})
const semanticEngine = computed<string>(() => {
  // Prefer semantic engine from UI blocks when available (video tabs etc.).
  if (uiBlocks.semanticEngine) return uiBlocks.semanticEngine
  // Fallback to global Codex engine selection.
  return store.currentEngine || 'sd15'
})

async function loadInventory(): Promise<void> {
  try {
    const inv = await fetchModelInventory()
    inventoryVaes.value = inv.vaes
    inventoryWan.value = (inv.wan22?.gguf ?? []).map((g: any) => ({
      name: String(g.name),
      path: String(g.path),
      stage: String(g.stage || 'unknown'),
    }))
  } catch (e) {
    inventoryVaes.value = []
    inventoryWan.value = []
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
    if (fNorm === rNorm || fNorm.startsWith(rNorm + '/')) return true
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
  if (fam === 'flux') {
    if (fileInPaths(path, 'flux_vae')) return true
    return (scale !== null) ? Math.abs(Number(scale) - 0.3611) < 1e-3 : /flux/i.test(name)
  }
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
  const prefix = fam === 'wan' ? 'wan22/' : `${fam}/`
  return store.textEncoderChoices.filter((name) => typeof name === 'string' && name.startsWith(prefix))
})

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

async function refreshAll(): Promise<void> { await store.init() }

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

const wanTextEncoderChoices = computed(() => {
  // Limit WAN choices to WAN22-specific labels (family prefix).
  return store.textEncoderChoices.filter((name) => typeof name === 'string' && name.startsWith('wan22/'))
})

const wanVaeChoices = computed(() => filteredVaeChoices.value)

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

function onTextEncoderChange(value: string): void {
  const payload = value ? [value] : []
  void store.setTextEncoders(payload)
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

function enginePrefixForFamily(fam: 'sd15' | 'sdxl' | 'flux' | 'wan'): 'sd15' | 'sdxl' | 'flux' | 'wan22' {
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

async function onWanBrowseTe(): Promise<void> {
  const current = wanTextEncoder.value
  const next = promptForPath('WAN Text Encoder identifier or path', current)
  if (next !== null) await onWanTextEncoderChange(next)
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
  void store.init()
  void presets.init(currentTab())
  void loadInventory()
  void loadPaths()
  void engineCaps.init()
})

watch(() => route.path, async () => {
  await presets.init(currentTab())
  await loadInventory()
})

// random seed button removed from quicksettings; presets applied elsewhere
</script>
