<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tools view (GGUF converter + CodexPack v1 packer + file browser modal).
Starts GGUF conversion jobs (`/api/tools/convert-gguf`) and CodexPack packing jobs (`/api/tools/codexpack/pack-v1`), polls job status, and
provides a modal file browser to pick config/weights/output paths without manual typing.

Symbols (top-level; keep in sync; no ghosts):
- `ToolsTab` (component): Tools page SFC; owns GGUF converter form state and the file browser modal.
- `FloatDtypeGroup` (interface): Float dtype override group returned by `/api/tools/gguf-converter/presets`.
- `GGUFConverterModelComponent` (interface): Convertible component entry (config dir + profile hints).
- `GGUFConverterModelMetadata` (interface): Vendored model metadata entry returned by `/api/tools/gguf-converter/presets`.
- `GGUFForm` (interface): GGUF converter form state (model metadata + denoiser + quant/mixed + overwrite + Comfy Layout).
- `CodexPackForm` (interface): CodexPack v1 packer form state (base GGUF path + output folder + overwrite).
- `ConversionStatus` (interface): Polled conversion job status payload (progress + current tensor + error).
- `BrowserItem` (interface): Single file browser entry (file/directory + optional size).
- `BrowserData` (interface): File browser listing payload (current path + items).
- `formatComponentLabel` (function): Formats component options in the selector.
- `loadModelMetadata` (function): Loads vendored model metadata + float groups for the selector.
- `startConversion` (function): Starts a conversion job and begins polling.
- `cancelConversion` (function): Requests cancellation of the current conversion job (cooperative).
- `pollStatus` (function): Polls job status and stops polling when complete/error/cancelled.
- `startCodexpackPack` (function): Starts a CodexPack v1 packing job and begins polling.
- `pollCodexpackPackStatus` (function): Polls CodexPack v1 packing job status.
- `browseForSafetensors` (function): Opens the file browser in safetensors selection mode.
- `browseForOutputDir` (function): Opens the file browser in output-folder selection mode.
- `browseForBaseGguf` (function): Opens the file browser in base GGUF selection mode.
- `browseForCodexpackOutputDir` (function): Opens the file browser in CodexPack output-folder selection mode.
- `openFileBrowser` (function): Opens the modal and loads the current path listing.
- `closeFileBrowser` (function): Closes the modal.
- `loadBrowserPath` (function): Fetches the directory listing for the current browser path.
- `goToParent` (function): Navigates the browser up one directory.
- `selectItem` (function): Selects a browser row item.
- `openItem` (function): Opens a directory (or confirms selection for files).
- `confirmSelection` (function): Applies the selected path to the active form field and closes the modal.
- `formatSize` (function): Formats byte sizes for display.
- `mixedSupported` (computed): Whether the selected quantization supports mixed variants.
- `effectiveQuantization` (computed): Derived quantization name sent to the API (base type + Mixed toggle).
- `outputFileName` (computed): Generated output filename derived from the safetensors path (base `.gguf`).
- `outputFullPath` (computed): Output full path (folder + generated filename).
- `codexpackPackFileName` (computed): Generated CodexPack output filename derived from the selected base GGUF filename.
- `codexpackPackFullPath` (computed): CodexPack output full path (folder + generated filename).
-->

<template>
  <section class="panel-stack cdx-tools">
    <div class="panel">
      <div class="panel-header">Tools</div>
      <div class="panel-body">
        <p class="subtitle">Utilities for model conversion and management</p>

        <div class="gen-card cdx-tools-card">
          <div>
            <div class="h3">GGUF Converter</div>
            <p class="caption">Convert Safetensors weights to GGUF format</p>
          </div>

          <div class="field">
            <label class="label-muted">Model Metadata (vendored Hugging Face)</label>
            <select class="select-md" v-model="ggufForm.modelId" :disabled="isConverting || metadataLoading">
              <option value="" disabled>Select a vendored model…</option>
              <option v-for="m in modelMetadata" :key="m.id" :value="m.id">{{ m.label }}</option>
            </select>
            <p class="caption">
              Uses the vendored Hugging Face mirror under <code>apps/backend/huggingface/**</code>.
            </p>
            <p v-if="metadataLoading" class="caption">Loading vendored model metadata…</p>
            <p v-if="metadataError" class="cdx-tools-error">{{ metadataError }}</p>
          </div>

          <div v-if="selectedModel" class="field">
            <label class="label-muted">Denoiser</label>
            <select class="select-md" v-model="ggufForm.componentId" :disabled="isConverting">
              <option v-for="c in selectedModel.components" :key="c.id" :value="c.id">{{ formatComponentLabel(c) }}</option>
            </select>
            <p class="caption">Uses the vendored config directory for the selected model.</p>
          </div>

          <div class="field">
            <label class="label-muted">Safetensors File or Folder</label>
            <div class="row-inline">
              <input class="ui-input cdx-tools-grow" type="text" v-model="ggufForm.safetensorsPath" placeholder="Path to .safetensors file, index.json, or folder" :disabled="isConverting" />
              <button class="btn-icon" type="button" @click="browseForSafetensors" :disabled="isConverting" aria-label="Browse for safetensors file">…</button>
            </div>
            <p class="caption">For sharded weights, select the folder that contains <code>*.safetensors.index.json</code>.</p>
          </div>

	          <div class="field">
	            <label class="label-muted">Quantization</label>
              <div class="row-inline">
	              <select class="select-md cdx-tools-grow" v-model="ggufForm.quantization" :disabled="isConverting">
	                <optgroup label="Float (no quant)">
	                  <option value="F16">F16 — float16</option>
	                  <option value="F32">F32 — float32</option>
	                </optgroup>
	                <optgroup label="K-quants">
	                  <option value="Q8_0">Q8_0 — 8-bit</option>
	                  <option value="Q6_K">Q6_K — 6-bit K</option>
	                  <option value="Q5_K">Q5_K — 5-bit K</option>
	                  <option value="Q4_K">Q4_K — 4-bit K</option>
	                  <option value="Q3_K">Q3_K — 3-bit K</option>
	                  <option value="Q2_K">Q2_K — 2-bit K</option>
	                </optgroup>
	                <optgroup label="Legacy">
	                  <option value="Q5_1">Q5_1 — 5-bit legacy</option>
	                  <option value="Q5_0">Q5_0 — 5-bit legacy</option>
	                  <option value="Q4_1">Q4_1 — 4-bit legacy</option>
	                  <option value="Q4_0">Q4_0 — 4-bit legacy</option>
	                </optgroup>
	                <optgroup label="Experimental">
	                  <option value="IQ4_NL">IQ4_NL — 4-bit IQ (NL)</option>
	                </optgroup>
	              </select>
		            <button
		              :class="['btn', 'qs-toggle-btn', ggufForm.mixed ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
		              type="button"
		              :aria-pressed="ggufForm.mixed"
		              :disabled="isConverting || !mixedSupported"
		              title="Enable mixed policy when available (e.g., Q5_K → Q5_K_M, Q4_K → Q4_K_M)"
		              @click="ggufForm.mixed = !ggufForm.mixed"
		            >
		              Mixed
		            </button>
		            <button
		              v-if="ggufForm.mixed && mixedSupported"
		              :class="[
		                'btn',
		                'qs-toggle-btn',
		                ggufForm.mixedFloatDtype === 'auto' ? 'qs-toggle-btn--off' : 'qs-toggle-btn--on',
		              ]"
		              type="button"
		              :disabled="isConverting"
		              title="Mixed float dtype (cycles AUTO → FP16 → FP32)"
		              @click="cycleMixedFloatDtype"
		            >
		              {{ mixedFloatDtypeLabel }}
		            </button>
              </div>
	            <p class="caption">
	              Mixed enables mixed quant variants when available. Float dtype cycles AUTO/FP16/FP32.
	            </p>
	          </div>

          <div class="field">
            <label class="label-muted">Output Folder</label>
            <div class="row-inline">
              <input
                class="ui-input cdx-tools-grow"
                type="text"
                v-model="ggufForm.outputDir"
                placeholder="Output folder path"
                :disabled="isConverting"
              />
              <button class="btn-icon" type="button" @click="browseForOutputDir" :disabled="isConverting" aria-label="Browse for output folder">…</button>
            </div>
	            <p class="caption">Output file name is generated automatically: <code>{{ outputFileName }}</code></p>
	            <div class="row-inline cdx-tools-actions">
                    <button
                      :class="['btn', 'qs-toggle-btn', ggufForm.overwrite ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                      type="button"
                      :aria-pressed="ggufForm.overwrite"
                      :disabled="isConverting"
                      title="Allow overwriting the output file if it already exists"
                      @click="ggufForm.overwrite = !ggufForm.overwrite"
                    >
                      Overwrite
                    </button>
                    <button
                      :class="['btn', 'qs-toggle-btn', ggufForm.comfyLayout ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
                      type="button"
                      :aria-pressed="ggufForm.comfyLayout"
                      :disabled="isConverting"
                      title="When on, denoiser exports are mapped to the Comfy/Codex key layout."
                      @click="ggufForm.comfyLayout = !ggufForm.comfyLayout"
                    >
                      Comfy Layout
                    </button>
                  </div>
                  <p class="caption">Overwrite: when off, conversion fails if the output file already exists.</p>
                  <p class="caption">Comfy Layout: enable for Codex runtime; turn off to preserve source key names.</p>
                </div>

	          <div class="row-inline cdx-tools-actions">
	            <button class="btn btn-md btn-primary" type="button" @click="startConversion" :disabled="!canConvert || isConverting">
	              <span v-if="!isConverting">Convert to GGUF</span>
	              <span v-else>Converting…</span>
	            </button>
            <button
              v-if="isConverting && currentJobId"
              class="btn btn-md btn-secondary"
              type="button"
              :disabled="conversionStatus?.status === 'cancelling'"
              @click="cancelConversion"
            >
              Cancel
            </button>
          </div>

          <div v-if="conversionStatus" class="panel-progress">
            <div class="cdx-tools-progress-head">
              <span class="cdx-tools-status" :data-status="conversionStatus.status">{{ conversionStatus.status }}</span>
              <span v-if="conversionStatus.current_tensor" class="cdx-tools-current-tensor">{{ conversionStatus.current_tensor }}</span>
            </div>
            <progress class="cdx-tools-progress" :value="conversionStatus.progress" max="100"></progress>
            <div class="caption">{{ Math.round(conversionStatus.progress) }}%</div>
            <div v-if="conversionStatus.error" class="cdx-tools-error">{{ conversionStatus.error }}</div>
          </div>
        </div>
      </div>
    </div>

    <div class="gen-card cdx-tools-card">
      <div>
        <div class="h3">CodexPack v1 Packer</div>
        <p class="caption">Pack an existing base GGUF into a CodexPack GGUF</p>
      </div>

      <div class="field">
        <label class="label-muted">Base GGUF File</label>
        <div class="row-inline">
          <input
            class="ui-input cdx-tools-grow"
            type="text"
            v-model="codexpackForm.srcGgufPath"
            placeholder="Path to base .gguf file"
            :disabled="isPackingCodexpack || isConverting"
          />
          <button
            class="btn-icon"
            type="button"
            @click="browseForBaseGguf"
            :disabled="isPackingCodexpack || isConverting"
            aria-label="Browse for base GGUF file"
          >
            …
          </button>
        </div>
        <p class="caption">
          Requires a Codex base GGUF with metadata: <code>codex.converter.comfy_layout=true</code>,
          <code>codex.zimage.variant=base</code>, <code>gguf.quantization=Q4_K</code>.
        </p>
      </div>

      <div class="field">
        <label class="label-muted">Output Folder</label>
        <div class="row-inline">
          <input
            class="ui-input cdx-tools-grow"
            type="text"
            v-model="codexpackForm.outputDir"
            placeholder="Output folder path"
            :disabled="isPackingCodexpack || isConverting"
          />
          <button
            class="btn-icon"
            type="button"
            @click="browseForCodexpackOutputDir"
            :disabled="isPackingCodexpack || isConverting"
            aria-label="Browse for CodexPack output folder"
          >
            …
          </button>
        </div>
        <p class="caption">Output file name is generated automatically: <code>{{ codexpackPackFileName }}</code></p>
      </div>

      <div class="row-inline cdx-tools-actions">
        <button
          :class="['btn', 'qs-toggle-btn', codexpackForm.overwrite ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :aria-pressed="codexpackForm.overwrite"
          :disabled="isPackingCodexpack || isConverting"
          title="Allow overwriting the output file if it already exists"
          @click="codexpackForm.overwrite = !codexpackForm.overwrite"
        >
          Overwrite
        </button>
      </div>

      <div class="row-inline cdx-tools-actions">
        <button
          class="btn btn-md btn-primary"
          type="button"
          @click="startCodexpackPack"
          :disabled="!canPackCodexpack || isPackingCodexpack || isConverting"
        >
          <span v-if="!isPackingCodexpack">Pack GGUF → CodexPack</span>
          <span v-else>Packing…</span>
        </button>
      </div>

      <div v-if="codexpackPackStatus" class="panel-progress">
        <div class="cdx-tools-progress-head">
          <span class="cdx-tools-status" :data-status="codexpackPackStatus.status">{{ codexpackPackStatus.status }}</span>
        </div>
        <progress class="cdx-tools-progress" :value="codexpackPackStatus.progress" max="100"></progress>
        <div class="caption">{{ Math.round(codexpackPackStatus.progress) }}%</div>
        <div v-if="codexpackPackStatus.error" class="cdx-tools-error">{{ codexpackPackStatus.error }}</div>
      </div>
    </div>

    <Modal v-model="showFileBrowser" :title="browserTitle">
      <div class="cdx-tools-pathbar">
        <button class="btn btn-sm btn-secondary" type="button" @click="goToParent" :disabled="!browserData.parent">Up</button>
        <input class="ui-input cdx-tools-grow" type="text" v-model="browserPath" @keyup.enter="loadBrowserPath" />
        <button class="btn btn-sm btn-secondary" type="button" @click="loadBrowserPath">Go</button>
      </div>
      <div class="cdx-tools-file-list">
        <div
          v-for="item in browserItems"
          :key="item.name"
          class="cdx-tools-file-item"
          :class="{ 'is-selected': selectedItem && selectedItem.name === item.name && selectedItem.type === item.type }"
          :data-type="item.type"
          @click="selectItem(item)"
          @dblclick="openItem(item)"
        >
          <span aria-hidden="true">{{ item.type === 'directory' ? '📁' : '📄' }}</span>
          <span class="cdx-tools-file-name">{{ item.name }}</span>
          <span v-if="item.size" class="cdx-tools-file-size">{{ formatSize(item.size) }}</span>
        </div>
      </div>

      <template #footer>
        <button class="btn btn-md btn-secondary" type="button" @click="closeFileBrowser">Cancel</button>
        <button
          class="btn btn-md btn-primary"
          type="button"
          @click="confirmSelection"
          :disabled="browserMode !== 'output_dir' && browserMode !== 'codexpack_output_dir' && !selectedItem"
        >
          Select
        </button>
      </template>
    </Modal>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted, watch } from 'vue'
import Modal from '../components/ui/Modal.vue'

interface FloatDtypeGroup {
  id: string
  label: string
  patterns: string[]
}

interface GGUFConverterModelComponent {
  id: string
  label: string
  config_dir: string
  kind: string
  profile_id: string | null
  profile_id_comfy: string | null
  profile_id_native: string | null
}

interface GGUFConverterModelMetadata {
  id: string
  label: string
  org: string
  repo: string
  components: GGUFConverterModelComponent[]
}

interface GGUFForm {
  modelId: string
  componentId: string
  safetensorsPath: string
  quantization: string
  mixed: boolean
  mixedFloatDtype: 'auto' | 'F16' | 'F32'
  outputDir: string
  overwrite: boolean
  comfyLayout: boolean
}

interface CodexPackForm {
  srcGgufPath: string
  outputDir: string
  overwrite: boolean
}

interface ConversionStatus {
  status: string
  progress: number
  current_tensor: string
  error: string | null
}

interface BrowserItem {
  name: string
  type: 'file' | 'directory'
  size?: number
}

interface BrowserData {
  path: string
  exists: boolean
  parent: string
  items: BrowserItem[]
}

const modelMetadata = ref<GGUFConverterModelMetadata[]>([])
const floatGroupsByProfileId = ref<Record<string, FloatDtypeGroup[]>>({})
const metadataLoading = ref(false)
const metadataError = ref<string | null>(null)

const ggufForm = ref<GGUFForm>({
  modelId: '',
  componentId: '',
  safetensorsPath: '',
  quantization: 'Q5_K',
  mixed: true,
  mixedFloatDtype: 'auto',
  outputDir: '',
  overwrite: false,
  comfyLayout: true,
})

const conversionStatus = ref<ConversionStatus | null>(null)
const currentJobId = ref<string | null>(null)
const pollInterval = ref<number | null>(null)

const codexpackForm = ref<CodexPackForm>({
  srcGgufPath: '',
  outputDir: '',
  overwrite: false,
})
const codexpackPackStatus = ref<ConversionStatus | null>(null)
const codexpackJobId = ref<string | null>(null)
const codexpackPollInterval = ref<number | null>(null)

// File browser
const showFileBrowser = ref(false)
const browserPath = ref('')
const browserData = ref<BrowserData>({ path: '', exists: false, parent: '', items: [] })
const browserMode = ref<'safetensors' | 'output_dir' | 'codexpack_output_dir' | 'base_gguf'>('safetensors')
const selectedItem = ref<BrowserItem | null>(null)

const selectedModel = computed(() => modelMetadata.value.find((m) => m.id === ggufForm.value.modelId) ?? null)
const selectedComponent = computed(() => {
  const model = selectedModel.value
  if (!model) return null
  return model.components.find((c) => c.id === ggufForm.value.componentId) ?? null
})

const effectiveProfileId = computed(() => {
  const component = selectedComponent.value
  if (component) {
    if (component.profile_id) return component.profile_id
    return ggufForm.value.comfyLayout ? component.profile_id_comfy : component.profile_id_native
  }
  return null
})

const floatGroups = computed(() => {
  const pid = effectiveProfileId.value
  if (!pid) return [] as FloatDtypeGroup[]
  return floatGroupsByProfileId.value[pid] || []
})

function formatComponentLabel(component: GGUFConverterModelComponent): string {
  const kind = String(component.kind || '')
  const base =
    kind === 'flux_transformer' || kind === 'zimage_transformer' || kind === 'wan22_transformer'
      ? 'Denoiser'
      : kind || 'Component'
  const suffix =
    component.label && !['root', 'denoiser'].includes(component.label) ? ` (${component.label})` : ''
  return `${base}${suffix}`
}

const isConverting = computed(() => {
  if (!currentJobId.value) return false
  const status = conversionStatus.value?.status
  if (!status) return true
  return !['complete', 'error', 'cancelled'].includes(status)
})

const isPackingCodexpack = computed(() => {
  if (!codexpackJobId.value) return false
  const status = codexpackPackStatus.value?.status
  if (!status) return true
  return !['complete', 'error', 'cancelled'].includes(status)
})

const canConvert = computed(() => {
  return Boolean(selectedComponent.value && ggufForm.value.safetensorsPath && ggufForm.value.outputDir)
})

const canPackCodexpack = computed(() => {
  return Boolean(codexpackForm.value.srcGgufPath && codexpackForm.value.outputDir)
})

const mixedSupported = computed(() => {
  const q = String(ggufForm.value.quantization || '').trim()
  return q === 'Q5_K' || q === 'Q4_K'
})

const mixedFloatDtypeLabel = computed(() => {
  const v = ggufForm.value.mixedFloatDtype
  if (v === 'auto') return 'AUTO'
  if (v === 'F16') return 'FP16'
  if (v === 'F32') return 'FP32'
  return String(v).toUpperCase()
})

function cycleMixedFloatDtype() {
  const order: Array<'auto' | 'F16' | 'F32'> = ['auto', 'F16', 'F32']
  const current = ggufForm.value.mixedFloatDtype
  const idx = order.indexOf(current)
  ggufForm.value.mixedFloatDtype = order[(idx + 1) % order.length]
}

const browserTitle = computed(() => {
  if (browserMode.value === 'safetensors') return 'Choose Weights'
  if (browserMode.value === 'base_gguf') return 'Choose GGUF'
  if (browserMode.value === 'output_dir' || browserMode.value === 'codexpack_output_dir') return 'Choose Output Folder'
  return 'Browse Files'
})

const browserItems = computed(() => {
  if (browserMode.value === 'output_dir' || browserMode.value === 'codexpack_output_dir') {
    return browserData.value.items.filter((it) => it.type === 'directory')
  }
  return browserData.value.items
})

function _sanitizeOutputStem(raw: string): string {
  const s = String(raw || '').trim()
  if (!s) return 'model'
  // Keep stable/portable: collapse whitespace and remove weird separators.
  const cleaned = s.replace(/[^A-Za-z0-9._-]+/g, '_').replace(/^_+|_+$/g, '')
  return cleaned || 'model'
}

function _basename(path: string): string {
  const p = String(path || '').replace(/[\\/]+$/, '').replace(/\\/g, '/')
  const parts = p.split('/')
  return parts[parts.length - 1] || ''
}

function _dirname(path: string): string {
  const raw = String(path || '').trim()
  if (!raw) return ''
  const p = raw.replace(/[\\/]+$/, '')
  const idx = Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\'))
  if (idx <= 0) return ''
  return p.slice(0, idx)
}

function _joinPath(dir: string, file: string): string {
  const d = String(dir || '').trim()
  if (!d) return file
  const sep = d.includes('\\') && !d.includes('/') ? '\\' : '/'
  return d.replace(/[\\/]+$/, '') + sep + file
}

function _deriveOutputStem(): string {
  const raw = String(ggufForm.value.safetensorsPath || '').trim()
  if (!raw) return 'model'
  const name = _basename(raw)
  if (name.toLowerCase().endsWith('.safetensors.index.json')) {
    return name.slice(0, -'.safetensors.index.json'.length)
  }
  if (name.toLowerCase().endsWith('.safetensors')) {
    return name.slice(0, -'.safetensors'.length)
  }
  if (name.toLowerCase().endsWith('.index.json')) {
    return name.slice(0, -'.index.json'.length)
  }
  // If user picked a folder, use its leaf.
  return name
}

const effectiveQuantization = computed(() => {
  const q = String(ggufForm.value.quantization || 'F16').trim() || 'F16'
  if (!ggufForm.value.mixed || !mixedSupported.value) return q
  if (q === 'Q5_K') return 'Q5_K_M'
  if (q === 'Q4_K') return 'Q4_K_M'
  return q
})

const outputFileName = computed(() => {
  const stem = _sanitizeOutputStem(_deriveOutputStem())
  const quant = String(effectiveQuantization.value || 'F16').trim() || 'F16'
  const base = `${stem}-${quant}-Codex`
  return `${base}.gguf`
})

const outputFullPath = computed(() => _joinPath(ggufForm.value.outputDir, outputFileName.value))

const codexpackPackFileName = computed(() => {
  const src = String(codexpackForm.value.srcGgufPath || '').trim()
  if (!src) return ''
  const name = _basename(src)
  if (!name) return ''
  const lower = name.toLowerCase()
  if (lower.endsWith('.codexpack.gguf')) return name
  if (lower.endsWith('.gguf')) return name.slice(0, -'.gguf'.length) + '.codexpack.gguf'
  return name + '.codexpack.gguf'
})

const codexpackPackFullPath = computed(() => _joinPath(codexpackForm.value.outputDir, codexpackPackFileName.value))

async function loadModelMetadata() {
  metadataLoading.value = true
  try {
    const res = await fetch('/api/tools/gguf-converter/presets')
    const data = await res.json().catch(() => ({}))
    if (!res.ok) {
      throw new Error((data as any)?.detail || `${res.status} ${res.statusText}`)
    }

    const models = Array.isArray((data as any)?.models) ? ((data as any).models as GGUFConverterModelMetadata[]) : []
    modelMetadata.value = models

    const fg = (data as any)?.float_groups
    floatGroupsByProfileId.value =
      fg && typeof fg === 'object' && !Array.isArray(fg) ? (fg as Record<string, FloatDtypeGroup[]>) : {}

    if (!ggufForm.value.modelId && models.length > 0) {
      ggufForm.value.modelId = models[0].id
      ggufForm.value.componentId = models[0].components[0]?.id || ''
    }

    metadataError.value = null
  } catch (e: any) {
    modelMetadata.value = []
    floatGroupsByProfileId.value = {}
    metadataError.value = String(e?.message || e)
  } finally {
    metadataLoading.value = false
  }
}

watch(
  () => ggufForm.value.quantization,
  (q) => {
    if (!['Q5_K', 'Q4_K'].includes(String(q || '').trim())) {
      ggufForm.value.mixed = false
    }
  },
)

watch(
  () => ggufForm.value.modelId,
  () => {
    const model = selectedModel.value
    if (!model) {
      ggufForm.value.componentId = ''
      return
    }
    if (!model.components.find((c) => c.id === ggufForm.value.componentId)) {
      ggufForm.value.componentId = model.components[0]?.id || ''
    }
  },
)

async function startConversion() {
  try {
    const component = selectedComponent.value
    if (!component) {
      throw new Error('Select a vendored model + denoiser config first.')
    }
	    const payload: Record<string, any> = {
	      config_path: component.config_dir,
	      safetensors_path: ggufForm.value.safetensorsPath,
	      output_path: outputFullPath.value,
	      overwrite: ggufForm.value.overwrite,
	      comfy_layout: ggufForm.value.comfyLayout,
	      quantization: effectiveQuantization.value,
	    }

    const profileId = effectiveProfileId.value
    if (profileId) {
      payload.profile_id = profileId
    }

    if (ggufForm.value.mixed && mixedSupported.value && ggufForm.value.mixedFloatDtype !== 'auto') {
      const floatGroupOverrides: Record<string, string> = {}
      for (const group of floatGroups.value) {
        floatGroupOverrides[group.id] = ggufForm.value.mixedFloatDtype
      }
      if (Object.keys(floatGroupOverrides).length > 0) {
        payload.float_group_overrides = floatGroupOverrides
      }
    }

    const response = await fetch('/api/tools/convert-gguf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await response.json()

    if (!response.ok) {
      conversionStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: (data as any)?.detail || 'Unknown error',
      }
      return
    }

    currentJobId.value = (data as any).job_id
    conversionStatus.value = { status: 'pending', progress: 0, current_tensor: '', error: null }

    pollInterval.value = window.setInterval(pollStatus, 500)
  } catch (e: any) {
    conversionStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: String(e?.message || e),
    }
  }
}

async function cancelConversion() {
  if (!currentJobId.value) return
  try {
    const res = await fetch(`/api/tools/convert-gguf/${currentJobId.value}/cancel`, { method: 'POST' })
    if (!res.ok) {
      const data = await res.json().catch(() => ({}))
      throw new Error((data as any)?.detail || `${res.status} ${res.statusText}`)
    }
    if (conversionStatus.value) {
      conversionStatus.value = { ...conversionStatus.value, status: 'cancelling' }
    }
  } catch (e: any) {
    if (conversionStatus.value) {
      conversionStatus.value = { ...conversionStatus.value, error: String(e?.message || e) }
    } else {
      conversionStatus.value = { status: 'error', progress: 0, current_tensor: '', error: String(e?.message || e) }
    }
  }
}

async function pollStatus() {
  if (!currentJobId.value) return

  try {
    const response = await fetch(`/api/tools/convert-gguf/${currentJobId.value}`)
    const data = await response.json()

    conversionStatus.value = data

    if (data.status === 'complete' || data.status === 'error' || data.status === 'cancelled') {
      if (pollInterval.value) {
        clearInterval(pollInterval.value)
        pollInterval.value = null
      }
    }
  } catch (e) {
    // Ignore polling errors
  }
}

async function startCodexpackPack() {
  try {
    const src = String(codexpackForm.value.srcGgufPath || '').trim()
    const outDir = String(codexpackForm.value.outputDir || '').trim()
    if (!src) {
      throw new Error('Select a base GGUF file first.')
    }
    if (!outDir) {
      throw new Error('Select an output folder first.')
    }
    if (!codexpackPackFileName.value) {
      throw new Error('Could not derive output filename from the selected GGUF path.')
    }

    const payload: Record<string, any> = {
      src_gguf_path: src,
      output_path: codexpackPackFullPath.value,
      overwrite: codexpackForm.value.overwrite,
    }

    const response = await fetch('/api/tools/codexpack/pack-v1', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    const data = await response.json().catch(() => ({}))

    if (!response.ok) {
      codexpackPackStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: (data as any)?.detail || 'Unknown error',
      }
      return
    }

    codexpackJobId.value = (data as any).job_id
    codexpackPackStatus.value = { status: 'pending', progress: 0, current_tensor: '', error: null }

    if (codexpackPollInterval.value) {
      clearInterval(codexpackPollInterval.value)
      codexpackPollInterval.value = null
    }
    codexpackPollInterval.value = window.setInterval(pollCodexpackPackStatus, 500)
  } catch (e: any) {
    codexpackPackStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: String(e?.message || e),
    }
  }
}

async function pollCodexpackPackStatus() {
  if (!codexpackJobId.value) return

  try {
    const response = await fetch(`/api/tools/codexpack/pack-v1/${codexpackJobId.value}`)
    const data = await response.json()

    codexpackPackStatus.value = data

    if (data.status === 'complete' || data.status === 'error' || data.status === 'cancelled') {
      if (codexpackPollInterval.value) {
        clearInterval(codexpackPollInterval.value)
        codexpackPollInterval.value = null
      }
    }
  } catch (e) {
    // Ignore polling errors
  }
}

// File browser functions
function browseForSafetensors() {
  browserMode.value = 'safetensors'
  browserPath.value = ggufForm.value.safetensorsPath || ''
  openFileBrowser()
}

function browseForOutputDir() {
  browserMode.value = 'output_dir'
  browserPath.value = ggufForm.value.outputDir || ''
  openFileBrowser()
}

function browseForBaseGguf() {
  browserMode.value = 'base_gguf'
  browserPath.value = _dirname(codexpackForm.value.srcGgufPath) || ''
  openFileBrowser()
}

function browseForCodexpackOutputDir() {
  browserMode.value = 'codexpack_output_dir'
  browserPath.value = codexpackForm.value.outputDir || ''
  openFileBrowser()
}

async function openFileBrowser() {
  showFileBrowser.value = true
  selectedItem.value = null
  await loadBrowserPath()
}

function closeFileBrowser() {
  showFileBrowser.value = false
}

async function loadBrowserPath() {
  try {
    let ext = ''
    if (browserMode.value === 'safetensors') {
      ext = '.safetensors,.safetensors.index.json,.index.json'
    } else if (browserMode.value === 'base_gguf') {
      ext = '.gguf'
    }

    const response = await fetch(
      `/api/tools/browse-files?path=${encodeURIComponent(browserPath.value)}&extensions=${encodeURIComponent(ext)}`,
    )
    browserData.value = await response.json()
    browserPath.value = browserData.value.path
  } catch (e) {
    console.error('Failed to browse:', e)
  }
}

function goToParent() {
  if (browserData.value.parent) {
    browserPath.value = browserData.value.parent
    loadBrowserPath()
  }
}

function selectItem(item: BrowserItem) {
  selectedItem.value = item
}

function openItem(item: BrowserItem) {
  if (item.type === 'directory') {
    browserPath.value = browserPath.value.replace(/[/\\]$/, '') + '/' + item.name
    loadBrowserPath()
    selectedItem.value = null
  } else {
    confirmSelection()
  }
}

function confirmSelection() {
  if (!selectedItem.value && browserMode.value !== 'output_dir' && browserMode.value !== 'codexpack_output_dir') return

  if (browserMode.value === 'safetensors') {
    if (!selectedItem.value) return
    const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
    ggufForm.value.safetensorsPath = fullPath
  } else if (browserMode.value === 'base_gguf') {
    if (!selectedItem.value) return
    if (selectedItem.value.type !== 'file') return
    const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
    codexpackForm.value.srcGgufPath = fullPath
  } else if (browserMode.value === 'output_dir') {
    if (!selectedItem.value) {
      ggufForm.value.outputDir = browserPath.value
    } else if (selectedItem.value.type === 'directory') {
      const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
      ggufForm.value.outputDir = fullPath
    }
  } else if (browserMode.value === 'codexpack_output_dir') {
    if (!selectedItem.value) {
      codexpackForm.value.outputDir = browserPath.value
    } else if (selectedItem.value.type === 'directory') {
      const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
      codexpackForm.value.outputDir = fullPath
    }
  }

  closeFileBrowser()
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return bytes + ' B'
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB'
  if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB'
  return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB'
}

onMounted(() => {
  loadModelMetadata()
})

onUnmounted(() => {
  if (pollInterval.value) {
    clearInterval(pollInterval.value)
    pollInterval.value = null
  }
  if (codexpackPollInterval.value) {
    clearInterval(codexpackPollInterval.value)
    codexpackPollInterval.value = null
  }
})
	</script>
