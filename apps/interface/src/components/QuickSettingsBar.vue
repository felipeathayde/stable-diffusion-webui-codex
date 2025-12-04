<template>
  <section class="quicksettings">
    <div class="quicksettings-group">
      <label class="label-muted" title="Presets for UI + defaults; does not change the underlying engine semantics">Model UI</label>
      <div class="qs-row">
        <select class="select-md" :value="selectedPreset" @change="onPresetChange">
          <option v-for="name in presetChoices" :key="name" :value="name">{{ name }}</option>
        </select>
      </div>
    </div>
    <div class="quicksettings-group">
      <label class="label-muted">Mode</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentMode" @change="onModeChange">
          <option v-for="m in filteredModeChoices" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
    </div>
    <div class="quicksettings-group" v-if="!hideCheckpoint">
      <label class="label-muted">Checkpoint</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentModel" @change="onModelChange">
          <option v-for="model in filteredModels" :key="model.title" :value="model.title">
            {{ model.title }}
          </option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">VAE</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentVae" @change="onVaeChange">
          <option v-for="v in filteredVaeChoices" :key="v" :value="v">{{ v }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Text Encoder</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentTextEncoders[0] ?? ''" @change="onTextEncoderChange">
          <option value="">Automatic</option>
          <option v-for="te in store.textEncoderChoices" :key="te" :value="te">{{ te }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Diffusion in Low Bits</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentUnetDtype" @change="onUnetDtypeChange">
          <option v-for="opt in filteredUnetDtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">GPU Weights (MB)</label>
      <div class="qs-row">
        <input class="ui-input" type="number" :min="0" :max="store.gpuTotalMb" :value="store.gpuWeightsMb" @change="onGpuWeightsChange" />
      </div>
  </div>

    <div class="quicksettings-group">
      <label class="label-muted">Attention Backend</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentAttention" @change="onAttentionChange">
          <option v-for="opt in store.attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
  </div>

    <div class="quicksettings-group">
      <label class="label-muted">Device</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentDevice" @change="(e:any)=>store.setDevice((e.target as HTMLSelectElement).value)">
          <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Core dtype</label>
      <div class="qs-row">
        <select class="select-md" :value="store.coreDtype" @change="(e:any)=>store.setCoreDtype((e.target as HTMLSelectElement).value)">
          <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
      <label class="label-muted">TE dtype</label>
      <div class="qs-row">
        <select class="select-md" :value="store.teDtype" @change="(e:any)=>store.setTeDtype((e.target as HTMLSelectElement).value)">
          <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
      <label class="label-muted">VAE dtype</label>
      <div class="qs-row">
        <select class="select-md" :value="store.vaeDtype" @change="(e:any)=>store.setVaeDtype((e.target as HTMLSelectElement).value)">
          <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Core device</label>
      <div class="qs-row">
        <select class="select-md" :value="store.coreDevice" @change="(e:any)=>store.setCoreDevice((e.target as HTMLSelectElement).value)">
          <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
          <option value="auto">auto</option>
        </select>
      </div>
      <label class="label-muted">TE device</label>
      <div class="qs-row">
        <select class="select-md" :value="store.teDevice" @change="(e:any)=>store.setTeDevice((e.target as HTMLSelectElement).value)">
          <option value="auto">auto</option>
          <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
      <label class="label-muted">VAE device</label>
      <div class="qs-row">
        <select class="select-md" :value="store.vaeDevice" @change="(e:any)=>store.setVaeDevice((e.target as HTMLSelectElement).value)">
          <option value="auto">auto</option>
          <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>

    <!-- Right-most refresh button spanning to the end -->
    <div class="quicksettings-group quicksettings-right">
      <label class="label-muted">Models</label>
      <div class="qs-row">
        <button class="btn btn-sm btn-secondary" type="button" @click="refreshAll" title="Refresh checkpoint, VAE and text encoder lists">Refresh</button>
      </div>
    </div>
  
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { useUiPresetsStore } from '../stores/ui_presets'
import { useRoute } from 'vue-router'
import { useUiBlocksStore } from '../stores/ui_blocks'
import { useModelTabsStore } from '../stores/model_tabs'
import { fetchModelInventory } from '../api/client'
import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'

const store = useQuicksettingsStore()
const presets = useUiPresetsStore()
const route = useRoute()
const selectedPreset = ref('')
const uiBlocks = useUiBlocksStore()
const tabsStore = useModelTabsStore()
const inventoryVaes = ref<Array<{ name: string; path: string; format: string; latent_channels?: number | null; scaling_factor?: number | null }>>([])
const engineCaps = useEngineCapabilitiesStore()

onMounted(() => {
  void store.init()
  void presets.init(currentTab())
  void loadInventory()
  void engineCaps.init()
})

watch(() => route.path, async () => {
  await presets.init(currentTab())
  selectedPreset.value = ''
  await loadInventory()
})

function currentTab(): 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' {
  const p = route.path
  if (p.startsWith('/img2img')) return 'img2img'
  if (p.startsWith('/txt2vid')) return 'txt2vid'
  if (p.startsWith('/img2vid')) return 'img2vid'
  return 'txt2img'
}

const presetChoices = computed(() => presets.namesFor(currentTab()))
const activeFamily = computed<'sd15' | 'sdxl' | 'flux' | 'wan'>(() => tabsStore.activeTab?.type ?? 'sd15')
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
  } catch (e) {
    inventoryVaes.value = []
  }
}

function modelMatchesFamily(meta: Record<string, unknown> | undefined, title: string, file: string, family: string): boolean {
  const fam = String((meta?.['family'] as string) || (meta?.['model_family'] as string) || '').toLowerCase()
  const t = (title || '').toLowerCase(); const f = (file || '').toLowerCase()
  if (fam) return fam.includes(family)
  if (family === 'sdxl') return t.includes('sdxl') || f.includes('sdxl')
  if (family === 'sd15') return t.includes('1.5') || t.includes('sd15') || f.includes('sd15') || f.includes('v1-5')
  if (family === 'flux') return t.includes('flux') || f.includes('flux')
  if (family === 'wan') return t.includes('wan') || f.includes('wan')
  return true
}

const filteredModels = computed(() => {
  const fam = activeFamily.value
  return store.models.filter(m => modelMatchesFamily(m.metadata as Record<string, unknown> | undefined, m.title, m.filename, fam))
})

function isVaeForFamily(name: string, fam: string): boolean {
  const rec = inventoryVaes.value.find(v => v.name === name || v.path.endsWith('/' + name))
  const scale = rec?.scaling_factor ?? null
  if (fam === 'sdxl') return (scale !== null) ? Math.abs(Number(scale) - 0.13025) < 1e-3 : /sdxl|xl/i.test(name)
  if (fam === 'sd15') return (scale !== null) ? Math.abs(Number(scale) - 0.18215) < 5e-3 : /sd1|1\.5|sd15|v1-5/i.test(name)
  if (fam === 'flux') return (scale !== null) ? Math.abs(Number(scale) - 0.3611) < 1e-3 : /flux/i.test(name)
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
const hideCheckpoint = computed(() => {
  const p = route.path
  // In model tabs (/models), the tab manages model dirs (e.g., WAN 2.2); hide checkpoint there.
  if (p.startsWith('/models')) return true
  const isVideo = p.startsWith('/txt2vid') || p.startsWith('/img2vid')
  return isVideo && uiBlocks.semanticEngine === 'wan22'
})

async function onPresetChange(event: Event): Promise<void> {
  const title = (event.target as HTMLSelectElement).value
  selectedPreset.value = title
  await presets.applyByTitle(title, currentTab())
  // Refresh quicksettings to reflect applied checkpoint/options
  await store.init()
}

function onModelChange(event: Event): void {
  void store.setModel((event.target as HTMLSelectElement).value)
}

// sampler/scheduler/seed handlers removed from quicksettings

function onModeChange(event: Event): void {
  void store.setMode((event.target as HTMLSelectElement).value)
}

async function refreshAll(): Promise<void> { await store.init() }

function onVaeChange(event: Event): void {
  void store.setVae((event.target as HTMLSelectElement).value)
}

function onTextEncoderChange(event: Event): void {
  const select = event.target as HTMLSelectElement
  const value = select.value
  // Backend ainda espera array; enviamos [] para Automático (vazio) ou [value]
  const payload = value ? [value] : []
  void store.setTextEncoders(payload)
}

function onUnetDtypeChange(event: Event): void {
  void store.setUnetDtype((event.target as HTMLSelectElement).value)
}

function onGpuWeightsChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  void store.setGpuWeightsMb(Number.isFinite(v) ? v : store.gpuWeightsMb)
}

function onAttentionChange(event: Event): void {
  void store.setAttentionBackend((event.target as HTMLSelectElement).value)
}

// random seed button removed from quicksettings
</script>
