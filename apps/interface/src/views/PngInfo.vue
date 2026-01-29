<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: PNG info inspection view.
Inspect uploaded PNG metadata, parse common infotext formats, and bridge extracted parameters into model tabs and workflow snapshots.

Symbols (top-level; keep in sync; no ghosts):
- `PngInfo` (component): PNG info route view component.
-->

<template>
  <section class="panels pnginfo-panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Drop PNG</div>
        <div class="panel-body">
          <Dropzone
            accept="image/png,.png"
            label="Drop a PNG here, or click to browse"
            hint="We extract text metadata server-side, then parse infotext locally."
            @select="onDropFiles"
            @rejected="onDropRejected"
          >
            <div class="pnginfo-dropzone-slot">
              <div v-if="previewDataUrl" class="pnginfo-preview">
                <img :src="previewDataUrl" alt="PNG preview" />
              </div>
              <div class="pnginfo-dropzone-meta">
                <div class="pnginfo-dropzone-title">
                  {{ selectedFile ? selectedFile.name : 'Drop a PNG here, or click to browse' }}
                </div>
                <div class="caption">
                  <span v-if="analysis">{{ analysis.width }}×{{ analysis.height }} px</span>
                  <span v-else>PNG only · no upload storage</span>
                </div>
              </div>
            </div>
          </Dropzone>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <ResultsCard :showGenerate="false" headerClass="three-cols results-sticky" headerRightClass="results-actions" title="PNG Info">
        <template #header-right>
          <select class="select-md pnginfo-select" v-model="targetTabId" :disabled="!compatibleTabs.length">
            <option value="" disabled>Select tab</option>
            <option v-for="t in compatibleTabs" :key="t.id" :value="t.id">
              {{ t.title }} ({{ t.type }})
            </option>
          </select>

          <select class="select-md pnginfo-select" v-model="targetMode" :disabled="!targetTab">
            <option value="txt2img">txt2img</option>
            <option value="img2img">img2img</option>
          </select>

          <button class="btn btn-sm btn-secondary" type="button" :disabled="!canSaveSnapshot" @click="saveSnapshot">
            {{ workflowBusy ? 'Saving…' : 'Save snapshot' }}
          </button>

          <button class="btn btn-sm btn-primary" type="button" :disabled="!canSendTo" @click="sendTo">
            {{ sendBusy ? 'Sending…' : 'Send to' }}
          </button>
        </template>

        <div v-if="notice" class="pnginfo-notice-row">
          <div class="caption">{{ notice }}</div>
          <RouterLink v-if="lastSentTabId" class="btn btn-sm btn-outline" :to="`/models/${lastSentTabId}`">Open tab</RouterLink>
        </div>

        <div v-if="error" class="panel-error">{{ error }}</div>

        <div v-else-if="!selectedFile" class="viewer-card">
          <div class="viewer-empty">Drop a PNG on the left to inspect metadata and parse infotext.</div>
        </div>

        <div v-else class="pnginfo-body">
          <div v-if="parseWarnings.length || mappingWarnings.length" class="pnginfo-warnings">
            <div class="pnginfo-warnings-title">Warnings</div>
            <ul class="pnginfo-warnings-list">
              <li v-for="(w, idx) in allWarnings" :key="idx">{{ w }}</li>
            </ul>
          </div>

          <div class="pnginfo-section">
            <div class="pnginfo-section-title">Infotext</div>
            <textarea
              v-model="infotext"
              class="ui-textarea h-prompt-sm"
              placeholder="Infotext (e.g. A1111 'parameters'). Edit to re-parse."
            />
          </div>

          <div class="pnginfo-grid">
            <div class="pnginfo-card">
              <div class="pnginfo-card-title">Parsed</div>
              <div v-if="!hasAnyParsedField" class="caption">No parsed fields yet. (Some PNGs only include provenance.)</div>
              <dl v-else class="pnginfo-kv">
                <template v-if="parsed.prompt.trim()">
                  <dt>Prompt</dt>
                  <dd class="pnginfo-kv-pre">{{ parsed.prompt }}</dd>
                </template>
                <template v-if="parsed.hasNegativePrompt">
                  <dt>Negative</dt>
                  <dd class="pnginfo-kv-pre">{{ parsed.negativePrompt }}</dd>
                </template>
                <template v-if="parsed.width && parsed.height">
                  <dt>Size</dt>
                  <dd>{{ parsed.width }}×{{ parsed.height }}</dd>
                </template>
                <template v-else-if="analysis && analysis.width && analysis.height">
                  <dt>Image</dt>
                  <dd>{{ analysis.width }}×{{ analysis.height }}</dd>
                </template>
                <template v-if="parsed.steps !== undefined">
                  <dt>Steps</dt>
                  <dd>{{ parsed.steps }}</dd>
                </template>
                <template v-if="parsed.cfgScale !== undefined">
                  <dt>CFG</dt>
                  <dd>{{ parsed.cfgScale }}</dd>
                </template>
                <template v-if="parsed.seed !== undefined">
                  <dt>Seed</dt>
                  <dd>{{ parsed.seed }}</dd>
                </template>
                <template v-if="mappedSampler && mappedScheduler">
                  <dt>Sampler / Scheduler</dt>
                  <dd>{{ mappedSampler }} / {{ mappedScheduler }}</dd>
                </template>
                <template v-else-if="parsed.sampler || parsed.scheduler">
                  <dt>Sampler / Scheduler</dt>
                  <dd class="caption">Not applied (incompatible or unknown).</dd>
                </template>
                <template v-if="parsed.clipSkip !== undefined">
                  <dt>CLIP Skip</dt>
                  <dd>{{ parsed.clipSkip }}</dd>
                </template>
                <template v-if="parsed.denoiseStrength !== undefined">
                  <dt>Denoise</dt>
                  <dd>{{ parsed.denoiseStrength }}</dd>
                </template>
              </dl>
            </div>

            <div class="pnginfo-card">
              <div class="pnginfo-card-title">Raw metadata</div>
              <div v-if="analysis && Object.keys(analysis.metadata || {}).length" class="pnginfo-metadata">
                <JsonTreeView :value="analysis.metadata" :default-open-depth="1" :max-depth="8" />
              </div>
              <div v-else class="caption">No text chunks found.</div>
            </div>
          </div>
        </div>
      </ResultsCard>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { analyzePngInfo, fetchSamplers, fetchSchedulers } from '../api/client'
import type { PngInfoAnalyzeResponse, SamplerInfo, SchedulerInfo } from '../api/types'
import { useResultsCard } from '../composables/useResultsCard'
import { useModelTabsStore } from '../stores/model_tabs'
import { useWorkflowsStore } from '../stores/workflows'
import { mapSamplerScheduler, parseInfotext, type ParsedInfotext } from '../utils/pnginfo'
import ResultsCard from '../components/results/ResultsCard.vue'
import Dropzone from '../components/ui/Dropzone.vue'
import JsonTreeView from '../components/ui/JsonTreeView.vue'

type TargetMode = 'txt2img' | 'img2img'

const tabs = useModelTabsStore()
const workflows = useWorkflowsStore()
const { notice, toast } = useResultsCard()

const selectedFile = ref<File | null>(null)
const previewDataUrl = ref('')
const analysis = ref<PngInfoAnalyzeResponse | null>(null)
const infotext = ref('')
const error = ref('')

const samplers = ref<SamplerInfo[]>([])
const schedulers = ref<SchedulerInfo[]>([])

const workflowBusy = ref(false)
const sendBusy = ref(false)
const lastSentTabId = ref<string>('')

const compatibleTabs = computed(() => tabs.orderedTabs.filter(t => t.type !== 'wan'))
const targetTabId = ref('')
const targetTab = computed(() => compatibleTabs.value.find(t => t.id === targetTabId.value) || null)
const targetMode = ref<TargetMode>('txt2img')

const parsedResult = computed(() => parseInfotext(infotext.value))
const parsed = computed<ParsedInfotext>(() => parsedResult.value.parsed)
const parseWarnings = computed(() => parsedResult.value.warnings)

const mappingResult = computed(() =>
  mapSamplerScheduler(parsed.value.sampler, parsed.value.scheduler, samplers.value, schedulers.value),
)
const mappedSampler = computed(() => mappingResult.value.sampler || '')
const mappedScheduler = computed(() => mappingResult.value.scheduler || '')
const mappingWarnings = computed(() => mappingResult.value.warnings)
const allWarnings = computed(() => [...parseWarnings.value, ...mappingWarnings.value])

const hasAnyParsedField = computed(() => {
  const p = parsed.value
  return Boolean(
    p.prompt.trim()
      || p.hasNegativePrompt
      || p.steps !== undefined
      || p.cfgScale !== undefined
      || p.seed !== undefined
      || p.width !== undefined
      || p.height !== undefined
      || p.sampler
      || p.scheduler
      || p.clipSkip !== undefined
      || p.denoiseStrength !== undefined,
  )
})

const canSaveSnapshot = computed(() => Boolean(selectedFile.value && targetTab.value) && !workflowBusy.value)
const canSendTo = computed(() => {
  if (!selectedFile.value) return false
  if (!targetTab.value) return false
  if (sendBusy.value) return false
  if (targetMode.value === 'img2img' && !previewDataUrl.value) return false
  return true
})

function readFileAsDataURL(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve(String(reader.result))
    reader.onerror = () => reject(reader.error)
    reader.readAsDataURL(file)
  })
}

function metadataValue(metadata: Record<string, string> | undefined | null, key: string): string {
  if (!metadata) return ''
  const direct = metadata[key]
  if (typeof direct === 'string' && direct.trim()) return direct
  const want = key.toLowerCase()
  for (const [k, v] of Object.entries(metadata)) {
    if (k.toLowerCase() === want && typeof v === 'string' && v.trim()) return v
  }
  return ''
}

async function analyzeSelectedFile(): Promise<void> {
  if (!selectedFile.value) return
  error.value = ''
  analysis.value = null
  lastSentTabId.value = ''

  try {
    const res = await analyzePngInfo(selectedFile.value)
    analysis.value = res
    const params = metadataValue(res.metadata, 'parameters')
    if (params) {
      infotext.value = params
    } else if (!infotext.value.trim()) {
      infotext.value = ''
    }
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
}

async function onDropFiles(files: File[]): Promise<void> {
  const file = files[0]
  if (!file) return

  selectedFile.value = file
  error.value = ''
  lastSentTabId.value = ''

  try {
    previewDataUrl.value = await readFileAsDataURL(file)
  } catch (err) {
    previewDataUrl.value = ''
    error.value = err instanceof Error ? err.message : String(err)
    return
  }

  await analyzeSelectedFile()
}

function onDropRejected(payload: { reason: string; files: File[] }): void {
  const list = payload.files.map(f => f.name).join(', ')
  error.value = list ? `${payload.reason} (${list})` : payload.reason
}

function buildImageParamsPatch(options: { mode: TargetMode; includeInitImage: boolean }): { patch: Record<string, unknown>; warnings: string[] } {
  const p = parsed.value
  const patch: Record<string, unknown> = {}

  if (p.prompt.trim()) patch.prompt = p.prompt
  if (p.hasNegativePrompt) patch.negativePrompt = p.negativePrompt

  const width = p.width ?? analysis.value?.width
  const height = p.height ?? analysis.value?.height
  if (Number.isFinite(width) && Number.isFinite(height) && Number(width) > 0 && Number(height) > 0) {
    patch.width = Math.trunc(Number(width))
    patch.height = Math.trunc(Number(height))
  }

  if (p.steps !== undefined) patch.steps = p.steps
  if (p.cfgScale !== undefined) patch.cfgScale = p.cfgScale
  if (p.seed !== undefined) patch.seed = p.seed
  if (p.clipSkip !== undefined) patch.clipSkip = p.clipSkip
  if (p.denoiseStrength !== undefined) patch.denoiseStrength = p.denoiseStrength

  if (mappingResult.value.sampler && mappingResult.value.scheduler) {
    patch.sampler = mappingResult.value.sampler
    patch.scheduler = mappingResult.value.scheduler
  }

  if (options.mode === 'txt2img') {
    patch.useInitImage = false
    patch.initImageData = ''
    patch.initImageName = ''
  } else if (options.includeInitImage) {
    patch.useInitImage = true
    patch.initImageData = previewDataUrl.value
    patch.initImageName = selectedFile.value?.name || ''
  } else {
    patch.useInitImage = false
    patch.initImageData = ''
    patch.initImageName = ''
  }

  return { patch, warnings: allWarnings.value }
}

async function saveSnapshot(): Promise<void> {
  if (!targetTab.value) return
  if (!selectedFile.value) return

  workflowBusy.value = true
  try {
    const { patch } = buildImageParamsPatch({ mode: 'txt2img', includeInitImage: false })
    await workflows.createSnapshot({
      name: `${selectedFile.value.name} — ${new Date().toLocaleString()}`,
      source_tab_id: targetTab.value.id,
      type: targetTab.value.type,
      engine_semantics: targetTab.value.type === 'wan' ? 'wan22' : targetTab.value.type,
      params_snapshot: patch,
    })
    toast('Snapshot saved to Workflows.')
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  } finally {
    workflowBusy.value = false
  }
}

async function sendTo(): Promise<void> {
  if (!targetTab.value) return
  if (!selectedFile.value) return

  sendBusy.value = true
  try {
    await tabs.load()
    const { patch } = buildImageParamsPatch({ mode: targetMode.value, includeInitImage: true })
    await tabs.updateParams(targetTab.value.id, patch)
    lastSentTabId.value = targetTab.value.id
    toast(`Sent to ${targetTab.value.title}.`)
  } catch (err) {
    toast(err instanceof Error ? err.message : String(err))
  } finally {
    sendBusy.value = false
  }
}

onMounted(async () => {
  try {
    await tabs.load()
  } catch {
    // tabs store may bootstrap from localStorage; ignore here.
  }

  try {
    const [samp, sched] = await Promise.all([fetchSamplers(), fetchSchedulers()])
    samplers.value = samp.samplers
    schedulers.value = sched.schedulers
  } catch (err) {
    error.value = err instanceof Error ? err.message : String(err)
  }
})

watch([compatibleTabs, () => tabs.activeTab], () => {
  if (targetTabId.value && compatibleTabs.value.some(t => t.id === targetTabId.value)) return
  const active = tabs.activeTab
  if (active && active.type !== 'wan') {
    targetTabId.value = active.id
    return
  }
  targetTabId.value = compatibleTabs.value[0]?.id ?? ''
}, { immediate: true })
</script>
