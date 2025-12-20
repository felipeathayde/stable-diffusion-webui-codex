<template>
  <div class="tools-tab">
    <div class="tools-header">
      <h2>🔧 Tools</h2>
      <p class="subtitle">Utilities for model conversion and management</p>
    </div>

    <!-- GGUF Converter Section -->
    <div class="tool-card">
      <div class="tool-header">
        <div class="tool-icon">📦</div>
        <div class="tool-info">
          <h3>GGUF Converter</h3>
          <p>Convert Safetensors text encoders to GGUF format</p>
        </div>
      </div>

      <div class="tool-body">
        <!-- Config Path -->
        <div class="form-group">
          <label>Config Folder</label>
          <div class="path-input">
            <input 
              type="text" 
              v-model="ggufForm.configPath" 
              placeholder="Path to folder with config.json"
              :disabled="isConverting"
            />
            <button class="btn-browse" @click="browseForConfig" :disabled="isConverting">
              📁
            </button>
          </div>
          <span class="help-text">Folder containing config.json (e.g., text_encoder/)</span>
        </div>

        <!-- Safetensors Path -->
        <div class="form-group">
          <label>Safetensors File</label>
          <div class="path-input">
            <input 
              type="text" 
              v-model="ggufForm.safetensorsPath" 
              placeholder="Path to .safetensors file"
              :disabled="isConverting"
            />
            <button class="btn-browse" @click="browseForSafetensors" :disabled="isConverting">
              📁
            </button>
          </div>
        </div>

        <!-- Quantization -->
        <div class="form-group">
          <label>Quantization</label>
          <select v-model="ggufForm.quantization" :disabled="isConverting">
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

        <!-- Advanced: per-tensor overrides -->
        <div class="form-group">
          <label>Tensor Overrides (advanced)</label>
          <textarea
            v-model="ggufForm.tensorTypeOverrides"
            class="tensor-overrides"
            placeholder="One per line: <regex>=<quant>\nExample:\nattn_q\\.weight$=Q8_0"
            :disabled="isConverting"
            rows="5"
          ></textarea>
          <span class="help-text">Applied to both source and GGUF tensor names. Last match wins.</span>
        </div>

        <!-- Output Path -->
        <div class="form-group">
          <label>Output File</label>
          <div class="path-input">
            <input 
              type="text" 
              v-model="ggufForm.outputPath" 
              placeholder="Output .gguf file path"
              :disabled="isConverting"
            />
            <button class="btn-browse" @click="browseForOutput" :disabled="isConverting">
              📁
            </button>
          </div>
        </div>

        <!-- Convert Button -->
        <div class="form-actions">
          <button 
            class="btn-primary" 
            @click="startConversion" 
            :disabled="!canConvert || isConverting"
          >
            <span v-if="!isConverting">🚀 Convert to GGUF</span>
            <span v-else>⏳ Converting...</span>
          </button>
        </div>

        <!-- Progress -->
        <div v-if="conversionStatus" class="progress-section">
          <div class="progress-info">
            <span class="status-badge" :class="conversionStatus.status">
              {{ conversionStatus.status }}
            </span>
            <span v-if="conversionStatus.current_tensor" class="current-tensor">
              {{ conversionStatus.current_tensor }}
            </span>
          </div>
          <div class="progress-bar">
            <div 
              class="progress-fill" 
              :style="{ width: conversionStatus.progress + '%' }"
            ></div>
          </div>
          <div class="progress-text">{{ Math.round(conversionStatus.progress) }}%</div>
          <div v-if="conversionStatus.error" class="error-message">
            ❌ {{ conversionStatus.error }}
          </div>
        </div>
      </div>
    </div>

    <!-- File Browser Modal -->
    <div v-if="showFileBrowser" class="modal-overlay" @click.self="closeFileBrowser">
      <div class="file-browser-modal">
        <div class="modal-header">
          <h3>📁 Browse Files</h3>
          <button class="btn-close" @click="closeFileBrowser">×</button>
        </div>
        <div class="current-path">
          <button @click="goToParent" :disabled="!browserData.parent">⬆️</button>
          <input type="text" v-model="browserPath" @keyup.enter="loadBrowserPath" />
          <button @click="loadBrowserPath">Go</button>
        </div>
        <div class="file-list">
          <div 
            v-for="item in browserData.items" 
            :key="item.name"
            class="file-item"
            :class="{ 'is-dir': item.type === 'directory' }"
            @click="selectItem(item)"
            @dblclick="openItem(item)"
          >
            <span class="item-icon">{{ item.type === 'directory' ? '📁' : '📄' }}</span>
            <span class="item-name">{{ item.name }}</span>
            <span v-if="item.size" class="item-size">{{ formatSize(item.size) }}</span>
          </div>
        </div>
        <div class="modal-footer">
          <button class="btn-secondary" @click="closeFileBrowser">Cancel</button>
          <button class="btn-primary" @click="confirmSelection" :disabled="!selectedItem">
            Select
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onUnmounted } from 'vue'

interface GGUFForm {
  configPath: string
  safetensorsPath: string
  quantization: string
  outputPath: string
  tensorTypeOverrides: string
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
  outputPath: '',
  tensorTypeOverrides: '',
})

const conversionStatus = ref<ConversionStatus | null>(null)
const currentJobId = ref<string | null>(null)
const pollInterval = ref<number | null>(null)

// File browser
const showFileBrowser = ref(false)
const browserPath = ref('')
const browserData = ref<BrowserData>({ path: '', exists: false, parent: '', items: [] })
const browserMode = ref<'config' | 'safetensors' | 'output'>('config')
const selectedItem = ref<BrowserItem | null>(null)

const isConverting = computed(() => {
  return conversionStatus.value?.status === 'loading_config' ||
         conversionStatus.value?.status === 'loading_weights' ||
         conversionStatus.value?.status === 'converting' ||
         conversionStatus.value?.status === 'verifying'
})

const canConvert = computed(() => {
  return ggufForm.value.configPath && 
         ggufForm.value.safetensorsPath && 
         ggufForm.value.outputPath
})

async function startConversion() {
  try {
    const tensorTypeOverrides = parseTensorTypeOverrides(ggufForm.value.tensorTypeOverrides)
    const response = await fetch('/api/tools/convert-gguf', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config_path: ggufForm.value.configPath,
        safetensors_path: ggufForm.value.safetensorsPath,
        output_path: ggufForm.value.outputPath,
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
    
    if (data.status === 'complete' || data.status === 'error') {
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

function browseForOutput() {
  browserMode.value = 'output'
  browserPath.value = ggufForm.value.outputPath || ''
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
    const ext = browserMode.value === 'safetensors' ? '.safetensors' : 
                browserMode.value === 'output' ? '.gguf' : ''
    
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
  if (!selectedItem.value) return
  
  const fullPath = browserPath.value.replace(/[/\\]$/, '') + '/' + selectedItem.value.name
  
  if (browserMode.value === 'config') {
    ggufForm.value.configPath = selectedItem.value.type === 'directory' ? fullPath : browserPath.value
  } else if (browserMode.value === 'safetensors') {
    ggufForm.value.safetensorsPath = fullPath
  } else if (browserMode.value === 'output') {
    ggufForm.value.outputPath = fullPath
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

<style scoped>
.tools-tab {
  padding: 24px;
  max-width: 800px;
  margin: 0 auto;
}

.tools-header {
  margin-bottom: 24px;
}

.tools-header h2 {
  margin: 0;
  font-size: 1.75rem;
}

.subtitle {
  color: #888;
  margin: 4px 0 0 0;
}

.tool-card {
  background: #1a1a2e;
  border-radius: 12px;
  border: 1px solid #2a2a4a;
  overflow: hidden;
}

.tool-header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 20px;
  background: linear-gradient(135deg, #16213e 0%, #1a1a2e 100%);
  border-bottom: 1px solid #2a2a4a;
}

.tool-icon {
  font-size: 2rem;
}

.tool-info h3 {
  margin: 0;
  font-size: 1.25rem;
}

.tool-info p {
  margin: 4px 0 0 0;
  color: #888;
  font-size: 0.875rem;
}

.tool-body {
  padding: 20px;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  color: #ccc;
}

.path-input {
  display: flex;
  gap: 8px;
}

.path-input input {
  flex: 1;
  padding: 10px 14px;
  background: #0f0f1a;
  border: 1px solid #333;
  border-radius: 8px;
  color: #fff;
  font-size: 0.875rem;
}

.path-input input:focus {
  outline: none;
  border-color: #5865f2;
}

.btn-browse {
  padding: 10px 14px;
  background: #2a2a4a;
  border: 1px solid #3a3a5a;
  border-radius: 8px;
  cursor: pointer;
  transition: background 0.2s;
}

.btn-browse:hover:not(:disabled) {
  background: #3a3a5a;
}

.help-text {
  display: block;
  margin-top: 4px;
  font-size: 0.75rem;
  color: #666;
}

select {
  width: 100%;
  padding: 10px 14px;
  background: #0f0f1a;
  border: 1px solid #333;
  border-radius: 8px;
  color: #fff;
  font-size: 0.875rem;
}

.tensor-overrides {
  width: 100%;
  padding: 10px 14px;
  background: #0f0f1a;
  border: 1px solid #333;
  border-radius: 8px;
  color: #fff;
  font-size: 0.875rem;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
  resize: vertical;
}

.tensor-overrides:focus {
  outline: none;
  border-color: #5865f2;
}

.form-actions {
  margin-top: 24px;
}

.btn-primary {
  width: 100%;
  padding: 14px;
  background: linear-gradient(135deg, #5865f2 0%, #4752c4 100%);
  border: none;
  border-radius: 8px;
  color: #fff;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.1s, box-shadow 0.2s;
}

.btn-primary:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(88, 101, 242, 0.4);
}

.btn-primary:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.progress-section {
  margin-top: 20px;
  padding: 16px;
  background: #0f0f1a;
  border-radius: 8px;
}

.progress-info {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.status-badge {
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.status-badge.pending { background: #333; }
.status-badge.loading_config,
.status-badge.loading_weights,
.status-badge.converting { background: #5865f2; }
.status-badge.complete { background: #3ba55c; }
.status-badge.error { background: #ed4245; }

.current-tensor {
  color: #888;
  font-size: 0.75rem;
  font-family: monospace;
}

.progress-bar {
  height: 8px;
  background: #1a1a2e;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #5865f2, #7983f5);
  transition: width 0.3s;
}

.progress-text {
  text-align: center;
  margin-top: 8px;
  font-size: 0.875rem;
  color: #888;
}

.error-message {
  margin-top: 12px;
  padding: 12px;
  background: rgba(237, 66, 69, 0.1);
  border: 1px solid rgba(237, 66, 69, 0.3);
  border-radius: 6px;
  color: #ed4245;
  font-size: 0.875rem;
}

/* File Browser Modal */
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.file-browser-modal {
  width: 600px;
  max-height: 80vh;
  background: #1a1a2e;
  border-radius: 12px;
  border: 1px solid #2a2a4a;
  display: flex;
  flex-direction: column;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #2a2a4a;
}

.modal-header h3 {
  margin: 0;
}

.btn-close {
  background: none;
  border: none;
  color: #888;
  font-size: 1.5rem;
  cursor: pointer;
}

.current-path {
  display: flex;
  gap: 8px;
  padding: 12px 20px;
  background: #0f0f1a;
}

.current-path input {
  flex: 1;
  padding: 8px 12px;
  background: #1a1a2e;
  border: 1px solid #333;
  border-radius: 6px;
  color: #fff;
}

.current-path button {
  padding: 8px 12px;
  background: #2a2a4a;
  border: 1px solid #3a3a5a;
  border-radius: 6px;
  color: #fff;
  cursor: pointer;
}

.file-list {
  flex: 1;
  overflow-y: auto;
  padding: 12px;
}

.file-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 10px 12px;
  border-radius: 6px;
  cursor: pointer;
}

.file-item:hover {
  background: #2a2a4a;
}

.file-item.is-dir {
  font-weight: 500;
}

.item-icon {
  font-size: 1.25rem;
}

.item-name {
  flex: 1;
}

.item-size {
  color: #666;
  font-size: 0.75rem;
}

.modal-footer {
  display: flex;
  justify-content: flex-end;
  gap: 12px;
  padding: 16px 20px;
  border-top: 1px solid #2a2a4a;
}

.btn-secondary {
  padding: 10px 20px;
  background: #2a2a4a;
  border: 1px solid #3a3a5a;
  border-radius: 8px;
  color: #fff;
  cursor: pointer;
}

.modal-footer .btn-primary {
  width: auto;
  padding: 10px 24px;
}
</style>
