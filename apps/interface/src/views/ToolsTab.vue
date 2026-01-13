<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tools view (GGUF converter + file browser modal).
Starts GGUF conversion jobs via `/api/tools/convert-gguf`, polls job status, and provides a modal file browser to pick config/weights/output paths
without manual typing.

Symbols (top-level; keep in sync; no ghosts):
- `ToolsTab` (component): Tools page SFC; owns GGUF converter form state and the file browser modal.
- `GGUFForm` (interface): GGUF converter form state (paths + quantization + overrides + overwrite).
- `ConversionStatus` (interface): Polled conversion job status payload (progress + current tensor + error).
- `BrowserItem` (interface): Single file browser entry (file/directory + optional size).
- `BrowserData` (interface): File browser listing payload (current path + items).
- `startConversion` (function): Starts a conversion job and begins polling.
- `cancelConversion` (function): Requests cancellation of the current conversion job (cooperative).
- `parseTensorTypeOverrides` (function): Parses textarea overrides into a normalized line list.
- `pollStatus` (function): Polls job status and stops polling when complete/error/cancelled.
- `browseForConfig` (function): Opens the file browser in config selection mode.
- `browseForSafetensors` (function): Opens the file browser in safetensors selection mode.
- `browseForOutputDir` (function): Opens the file browser in output-folder selection mode.
- `openFileBrowser` (function): Opens the modal and loads the current path listing.
- `closeFileBrowser` (function): Closes the modal.
- `loadBrowserPath` (function): Fetches the directory listing for the current browser path.
- `goToParent` (function): Navigates the browser up one directory.
- `selectItem` (function): Selects a browser row item.
- `openItem` (function): Opens a directory (or confirms selection for files).
- `confirmSelection` (function): Applies the selected path to the active form field and closes the modal.
- `formatSize` (function): Formats byte sizes for display.
- `outputFileName` (computed): Generated output filename `{base}-{quant}-Codex.gguf` derived from the safetensors path.
- `outputFullPath` (computed): Output full path (folder + generated filename).
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
            <p class="caption">Convert Safetensors text encoders to GGUF format</p>
          </div>

          <div class="field">
            <label class="label-muted">Config Folder</label>
            <div class="row-inline">
              <input class="ui-input cdx-tools-grow" type="text" v-model="ggufForm.configPath" placeholder="Path to folder with config.json" :disabled="isConverting" />
              <button class="btn-icon" type="button" @click="browseForConfig" :disabled="isConverting" aria-label="Browse for config folder">…</button>
            </div>
            <p class="caption">Folder containing config.json (e.g., text_encoder/)</p>
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
            <select class="select-md" v-model="ggufForm.quantization" :disabled="isConverting">
              <option value="F16">F16 - No quantization (full precision)</option>
              <option value="Q8_0">Q8_0 - 8-bit (max quality, larger)</option>
              <option value="Q6_K">Q6_K - 6-bit K-quant (high quality)</option>
              <option value="Q5_K_M">Q5_K_M - 5-bit K-quant (mixed, recommended)</option>
              <option value="Q5_K">Q5_K - 5-bit (~35% size)</option>
              <option value="Q5_1">Q5_1 - 5-bit legacy</option>
              <option value="Q5_0">Q5_0 - 5-bit legacy</option>
              <option value="Q4_K_M">Q4_K_M - 4-bit K-quant (mixed, safe default)</option>
              <option value="Q4_K">Q4_K - 4-bit (~25% size)</option>
              <option value="Q4_1">Q4_1 - 4-bit legacy</option>
              <option value="Q4_0">Q4_0 - 4-bit legacy</option>
              <option value="Q3_K">Q3_K - 3-bit K-quant (last resort)</option>
              <option value="Q2_K">Q2_K - 2-bit K-quant (extreme)</option>
              <option value="IQ4_NL">IQ4_NL - 4-bit IQ (experimental)</option>
            </select>
          </div>

          <div class="field">
            <details class="accordion">
              <summary>Advanced</summary>
              <div class="accordion-body">
                <div class="field">
                  <label class="label-muted">Tensor Overrides</label>
                  <textarea
                    v-model="ggufForm.tensorTypeOverrides"
                    class="ui-textarea cdx-tools-overrides"
                    placeholder="One per line: <regex>=<quant>\nExample:\nattn_q\\.weight$=Q8_0"
                    :disabled="isConverting"
                    rows="5"
                  ></textarea>
                  <p class="caption">Applied to both source and GGUF tensor names. Last match wins.</p>
                </div>
              </div>
            </details>
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
            <p class="caption">File name is generated automatically: <code>{{ outputFileName }}</code></p>
            <div class="row-inline">
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
              <span class="caption">When off, conversion fails if the output file already exists.</span>
            </div>
          </div>

          <div class="row-inline">
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
        <button class="btn btn-md btn-primary" type="button" @click="confirmSelection" :disabled="browserMode !== 'output_dir' && !selectedItem">
          Select
        </button>
      </template>
    </Modal>
  </section>
</template>

<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'
import Modal from '../components/ui/Modal.vue'

interface GGUFForm {
  configPath: string
  safetensorsPath: string
  quantization: string
  outputDir: string
  tensorTypeOverrides: string
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

const ggufForm = ref<GGUFForm>({
  configPath: '',
  safetensorsPath: '',
  quantization: 'Q5_K_M',
  outputDir: '',
  tensorTypeOverrides: '',
  overwrite: false,
})

const conversionStatus = ref<ConversionStatus | null>(null)
const currentJobId = ref<string | null>(null)
const pollInterval = ref<number | null>(null)

// File browser
const showFileBrowser = ref(false)
const browserPath = ref('')
const browserData = ref<BrowserData>({ path: '', exists: false, parent: '', items: [] })
const browserMode = ref<'config' | 'safetensors' | 'output_dir'>('config')
const selectedItem = ref<BrowserItem | null>(null)

const isConverting = computed(() => {
  if (!currentJobId.value) return false
  const status = conversionStatus.value?.status
  if (!status) return true
  return !['complete', 'error', 'cancelled'].includes(status)
})

const canConvert = computed(() => {
  return ggufForm.value.configPath && 
         ggufForm.value.safetensorsPath && 
         ggufForm.value.outputDir
})

const browserTitle = computed(() => {
  if (browserMode.value === 'config') return 'Choose Config Folder'
  if (browserMode.value === 'safetensors') return 'Choose Weights'
  if (browserMode.value === 'output_dir') return 'Choose Output Folder'
  return 'Browse Files'
})

const browserItems = computed(() => {
  if (browserMode.value === 'output_dir') {
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

const outputFileName = computed(() => {
  const stem = _sanitizeOutputStem(_deriveOutputStem())
  const quant = String(ggufForm.value.quantization || 'F16').trim() || 'F16'
  return `${stem}-${quant}-Codex.gguf`
})

const outputFullPath = computed(() => _joinPath(ggufForm.value.outputDir, outputFileName.value))

	async function startConversion() {
	  try {
	    const tensorTypeOverrides = parseTensorTypeOverrides(ggufForm.value.tensorTypeOverrides)
	    const response = await fetch('/api/tools/convert-gguf', {
	      method: 'POST',
	      headers: { 'Content-Type': 'application/json' },
	      body: JSON.stringify({
	        config_path: ggufForm.value.configPath,
	        safetensors_path: ggufForm.value.safetensorsPath,
	        output_path: outputFullPath.value,
	        overwrite: ggufForm.value.overwrite,
	        quantization: ggufForm.value.quantization,
	        tensor_type_overrides: tensorTypeOverrides,
	      }),
	    })
    
    const data = await response.json()
    
    if (!response.ok) {
      conversionStatus.value = {
        status: 'error',
        progress: 0,
        current_tensor: '',
        error: data.detail || 'Unknown error',
      }
      return
    }
    
    currentJobId.value = data.job_id
    conversionStatus.value = { status: 'pending', progress: 0, current_tensor: '', error: null }
    
	    // Start polling
	    pollInterval.value = window.setInterval(pollStatus, 500)
	  } catch (e: any) {
    conversionStatus.value = {
      status: 'error',
      progress: 0,
      current_tensor: '',
      error: e.message,
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
}

function parseTensorTypeOverrides(raw: string): string[] {
  return raw
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
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

// File browser functions
function browseForConfig() {
  browserMode.value = 'config'
  browserPath.value = ggufForm.value.configPath || ''
  openFileBrowser()
}

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
    const ext = browserMode.value === 'safetensors' ? '.safetensors' : ''
    
    const response = await fetch(`/api/tools/browse-files?path=${encodeURIComponent(browserPath.value)}&extensions=${ext}`)
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
  if (!selectedItem.value && browserMode.value !== 'output_dir') return
  
  if (browserMode.value === 'config') {
    if (!selectedItem.value) return
    const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
    ggufForm.value.configPath = selectedItem.value.type === 'directory' ? fullPath : browserPath.value
  } else if (browserMode.value === 'safetensors') {
    if (!selectedItem.value) return
    const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
    ggufForm.value.safetensorsPath = fullPath
  } else if (browserMode.value === 'output_dir') {
    if (!selectedItem.value) {
      ggufForm.value.outputDir = browserPath.value
    } else if (selectedItem.value.type === 'directory') {
      const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
      ggufForm.value.outputDir = fullPath
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

onUnmounted(() => {
  if (pollInterval.value) {
    clearInterval(pollInterval.value)
  }
})
</script>
