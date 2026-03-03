<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reusable quicksettings add-path modal (scan + add-to-library).
Provides add-path workflows for checkpoint/VAE/text-encoder path keys by scanning a user-supplied path (no hash on scan),
then adding selected/all files with SHA computed only at add-time and row-overlay byte progress (spinner fallback when byte totals are unavailable).

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsAddPathModal` (component): Modal for scanning and adding model files into a target paths.json key.
- `sanitizePathInput` (function): Sanitizes path input (trim, quote removal, slash normalization, repeated separator collapse).
- `scanCandidates` (function): Calls backend scan endpoint and populates candidate rows (no SHA).
- `addOne` (function): Adds one candidate file to library key and records per-row SHA/result state.
- `addAllSequential` (function): Adds all scanned candidates sequentially with visible per-row progress.
- `normalizeSizeBytes` (function): Validates optional backend size metadata for honest progress calculations.
- `formatBytes` (function): Formats byte counts for UI progress/status labels.
- `planAddAllRun` (function): Builds one sequential add-all plan (pending rows + optional aggregate byte total).
- `sumKnownBytesForEntries` (function): Sums known row sizes across one add-all run for byte-aware progress planning.
- `sumProcessedKnownBytes` (function): Sums known row sizes for already-processed run entries.
- `rowSizeLabel` (function): Formats per-row size status (`Size …` / `Size unavailable`).
- `rowHasProgress` (function): Enables row overlay progress only for the active add-all row when totals are known.
- `rowProgressStyle` (function): Binds row overlay percentage via CSS custom property.
-->

<template>
  <Modal v-model="open" :title="title" panel-class="qs-add-path-modal-panel" :show-footer="false">
    <div class="qs-add-path-modal">
      <label class="label-muted" for="qs-add-path-input">{{ label }}</label>
      <div class="qs-add-path-input-row">
        <input
          id="qs-add-path-input"
          ref="pathInputEl"
          class="ui-input"
          type="text"
          v-model="pathInput"
          :placeholder="placeholderExample"
          @keydown.enter.prevent="scanCandidates"
        />
        <button class="btn btn-sm btn-secondary" type="button" :disabled="!canScan" @click="scanCandidates">
          <span v-if="scanLoading" class="qs-add-path-spinner" aria-hidden="true"></span>
          {{ scanLoading ? 'Scanning…' : 'Scan' }}
        </button>
      </div>

      <div class="qs-add-path-actions">
        <button class="btn btn-sm btn-secondary" type="button" :disabled="!canAddAll" @click="addAllSequential">
          <span>{{ addAllRunning ? 'Adding' : 'Add whole folder' }}</span>
          <span v-if="addAllRunning" class="qs-add-path-ellipsis" aria-hidden="true"></span>
        </button>
      </div>

      <p v-if="scanError" class="panel-error">Error: {{ scanError }}</p>
      <p v-else-if="scanned && scanResults.length === 0 && !scanLoading" class="caption">No supported files found.</p>

      <div v-if="scanResults.length > 0" class="panel-section modal-list-section qs-add-path-list-section">
        <div class="qs-add-path-list" role="list">
          <div
            v-for="(item, index) in scanResults"
            :key="item.path"
            :class="[
              'qs-add-path-row',
              rowState(item).error ? 'is-error' : '',
              rowState(item).adding ? 'is-adding' : '',
              rowHasProgress(item) ? 'has-progress' : '',
            ]"
            :style="rowProgressStyle(item)"
            role="listitem"
          >
            <div class="qs-add-path-row__name" :title="item.path">
              <div class="qs-add-path-row__title">{{ displayName(item) }}</div>
              <div class="qs-add-path-row__size">{{ rowSizeLabel(item) }}</div>
              <div v-if="rowState(item).error" class="qs-add-path-row__status qs-add-path-row__status--error">{{ rowState(item).error }}</div>
              <div
                v-else-if="rowState(item).done && rowState(item).sha256"
                class="qs-add-path-row__status"
                :title="`SHA256: ${rowState(item).sha256}`"
              >
                {{ rowState(item).addedToLibrary ? 'Added' : 'Already in library' }} · SHA {{ shortSha(rowState(item).sha256) }}
              </div>
              <div v-else-if="rowState(item).done && rowState(item).alreadyInLibrary" class="qs-add-path-row__status">
                Already in library
              </div>
            </div>
            <div class="qs-add-path-row__actions">
              <button
                class="btn btn-sm btn-outline"
                type="button"
                :disabled="isRowActionDisabled(item)"
                @click="addOne(item, index)"
              >
                <span>{{ rowActionLabel(item) }}</span>
                <span v-if="rowState(item).adding" class="qs-add-path-ellipsis" aria-hidden="true"></span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'

import { addModelPathItem, scanModelPath } from '../../api/client'
import type { ModelPathLibraryKind, ModelPathScanItem } from '../../api/types'
import Modal from '../ui/Modal.vue'

interface RowStatus {
  adding: boolean
  done: boolean
  addedToLibrary: boolean
  alreadyInLibrary: boolean
  sha256: string
  shortHash: string
  error: string
  sizeBytes: number | null
}

const props = withDefaults(defineProps<{
  modelValue: boolean
  title: string
  label: string
  targetKey: string
  targetKind: ModelPathLibraryKind
  placeholder?: string
}>(), {
  placeholder: '',
})

const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'added', payload: { addedCount: number }): void
  (e: 'error', message: string): void
}>()

const open = computed({
  get: () => props.modelValue,
  set: (value: boolean) => emit('update:modelValue', value),
})

const pathInput = ref('')
const pathInputEl = ref<HTMLInputElement | null>(null)
const scanLoading = ref(false)
const scanError = ref('')
const scanned = ref(false)
const scanResults = ref<ModelPathScanItem[]>([])
const rowStatuses = ref<Record<string, RowStatus>>({})
const addAllRunning = ref(false)
const addAllIndex = ref(0)
const addAllActivePath = ref('')
const addAllPlannedTotalBytes = ref<number | null>(null)
const addAllProcessedBytes = ref(0)

const placeholderExample = computed(() => {
  const explicit = String(props.placeholder || '').trim()
  if (explicit) return explicit

  const windows = isWindowsClient()
  const suffix = props.targetKind === 'checkpoint'
    ? 'checkpoints'
    : (props.targetKind === 'vae' ? 'vae' : 'text_encoders')
  if (windows) return `C:\\models\\${suffix}`
  return `/home/user/models/${suffix}`
})

const sanitizedInput = computed(() => sanitizePathInput(pathInput.value))
const hasRowAddInFlight = computed(() => scanResults.value.some((item) => rowState(item).adding))
const canScan = computed(() => !scanLoading.value && !addAllRunning.value && Boolean(sanitizedInput.value))
const pendingAddCount = computed(() => scanResults.value.filter((item) => {
  const state = rowState(item)
  return !(state.done && !state.error)
}).length)
const canAddAll = computed(() => !scanLoading.value && !addAllRunning.value && !hasRowAddInFlight.value && pendingAddCount.value > 0)
const hasAddAllByteTotal = computed(() => typeof addAllPlannedTotalBytes.value === 'number')
const addAllProgressPercent = computed(() => {
  const total = addAllPlannedTotalBytes.value
  if (total === null) return 0
  if (total <= 0) return 100
  const raw = (addAllProcessedBytes.value / total) * 100
  return Math.min(100, Math.max(0, Math.trunc(raw * 10) / 10))
})

watch(open, (isOpen) => {
  if (!isOpen) {
    resetState()
    return
  }
  void nextTick(() => {
    pathInputEl.value?.focus()
    pathInputEl.value?.select()
  })
})

function resetState(): void {
  pathInput.value = ''
  scanError.value = ''
  scanned.value = false
  scanLoading.value = false
  scanResults.value = []
  rowStatuses.value = {}
  addAllRunning.value = false
  addAllIndex.value = 0
  addAllActivePath.value = ''
  addAllPlannedTotalBytes.value = null
  addAllProcessedBytes.value = 0
}

function isWindowsClient(): boolean {
  if (typeof navigator === 'undefined') return false
  const uaData = (navigator as Navigator & { userAgentData?: { platform?: string } }).userAgentData
  const platform = String(uaData?.platform || navigator.platform || navigator.userAgent || '').toLowerCase()
  return platform.includes('win')
}

function sanitizePathInput(raw: string): string {
  let value = String(raw ?? '').trim()
  if (!value) return ''

  while ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'"))) {
    value = value.slice(1, -1).trim()
    if (!value) return ''
  }

  value = value.replace(/\\/g, '/')

  const hasUncPrefix = value.startsWith('//')
  const driveMatch = value.match(/^[A-Za-z]:/)
  if (driveMatch) {
    const drive = driveMatch[0]
    let rest = value.slice(drive.length)
    rest = rest.replace(/\/+/g, '/')
    if (rest && !rest.startsWith('/')) rest = `/${rest}`
    return `${drive}${rest}`.trim()
  }

  value = value.replace(/\/+/g, '/')
  if (hasUncPrefix) value = `//${value.replace(/^\/+/, '')}`
  if (value.length > 1 && /\/$/.test(value) && !/^[A-Za-z]:\/$/.test(value)) {
    value = value.replace(/\/+$/, '')
  }
  return value.trim()
}

function displayName(item: ModelPathScanItem): string {
  const base = String(item.name || '').trim()
  const ext = String(item.ext || '').trim().toLowerCase()
  if (base && ext && base.toLowerCase().endsWith(ext)) {
    return base.slice(0, base.length - ext.length)
  }
  return base || item.path
}

function shortSha(sha: string): string {
  const normalized = String(sha || '').trim().toLowerCase()
  if (normalized.length <= 10) return normalized
  return normalized.slice(0, 10)
}

function normalizeSizeBytes(raw: unknown, filePath: string): number | null {
  if (raw === undefined || raw === null) return null
  if (typeof raw !== 'number' || !Number.isFinite(raw) || raw < 0 || !Number.isInteger(raw)) {
    throw new Error(`invalid size_bytes for ${filePath}: ${String(raw)}`)
  }
  return raw
}

function formatBytes(value: number): string {
  if (!Number.isFinite(value) || value <= 0) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let amount = value
  let unitIndex = 0
  while (amount >= 1024 && unitIndex < units.length - 1) {
    amount /= 1024
    unitIndex += 1
  }
  const digits = unitIndex <= 1 ? 0 : 1
  return `${amount.toFixed(digits)} ${units[unitIndex]}`
}

function rowSizeLabel(item: ModelPathScanItem): string {
  const sizeBytes = rowState(item).sizeBytes
  if (sizeBytes === null) return 'Size unavailable'
  return `Size ${formatBytes(sizeBytes)}`
}

function rowState(item: ModelPathScanItem): RowStatus {
  const existing = rowStatuses.value[item.path]
  if (existing) return existing
  const alreadyInLibrary = item.already_in_library === true
  const created: RowStatus = {
    adding: false,
    done: alreadyInLibrary,
    addedToLibrary: false,
    alreadyInLibrary,
    sha256: '',
    shortHash: '',
    error: '',
    sizeBytes: normalizeSizeBytes(item.size_bytes, item.path),
  }
  rowStatuses.value[item.path] = created
  return created
}

function sumKnownBytesForEntries(entries: Array<{ item: ModelPathScanItem; index: number }>): { knownEntries: number; totalBytes: number } {
  let totalBytes = 0
  let knownEntries = 0
  for (const entry of entries) {
    const sizeBytes = rowState(entry.item).sizeBytes
    if (sizeBytes === null) continue
    if (!Number.isFinite(sizeBytes) || !Number.isInteger(sizeBytes) || sizeBytes < 0) {
      throw new Error(`invalid size_bytes for ${entry.item.path}: ${String(sizeBytes)}`)
    }
    totalBytes += sizeBytes
    knownEntries += 1
  }
  return { knownEntries, totalBytes }
}

function sumProcessedKnownBytes(entries: Array<{ item: ModelPathScanItem; index: number }>, processedCount: number): number {
  let processedBytes = 0
  const cappedCount = Math.max(0, Math.min(entries.length, processedCount))
  for (let index = 0; index < cappedCount; index += 1) {
    const sizeBytes = rowState(entries[index].item).sizeBytes
    if (sizeBytes === null) continue
    if (!Number.isFinite(sizeBytes) || !Number.isInteger(sizeBytes) || sizeBytes < 0) {
      throw new Error(`invalid size_bytes for ${entries[index].item.path}: ${String(sizeBytes)}`)
    }
    processedBytes += sizeBytes
  }
  return processedBytes
}

function planAddAllRun(): { entries: Array<{ item: ModelPathScanItem; index: number }>; totalBytes: number | null } {
  const entries: Array<{ item: ModelPathScanItem; index: number }> = []
  for (let index = 0; index < scanResults.value.length; index += 1) {
    const item = scanResults.value[index]
    const state = rowState(item)
    if (state.done && !state.error) continue
    entries.push({ item, index })
  }
  if (entries.length === 0) return { entries, totalBytes: null }
  const totals = sumKnownBytesForEntries(entries)
  if (totals.knownEntries <= 0) return { entries, totalBytes: null }
  return { entries, totalBytes: totals.totalBytes }
}

function rowHasProgress(item: ModelPathScanItem): boolean {
  const state = rowState(item)
  if (!state.adding) return false
  if (!addAllRunning.value) return true
  return addAllActivePath.value === item.path
}

function rowProgressStyle(item: ModelPathScanItem): Record<string, string> {
  if (!rowHasProgress(item)) return {}
  if (addAllRunning.value && hasAddAllByteTotal.value) {
    return { '--qs-row-progress': `${addAllProgressPercent.value}%` }
  }
  return { '--qs-row-progress': '100%' }
}

function isRowActionDisabled(item: ModelPathScanItem): boolean {
  const state = rowState(item)
  if (state.adding) return true
  if (addAllRunning.value) return true
  return state.done && !state.error
}

function rowActionLabel(item: ModelPathScanItem): string {
  const state = rowState(item)
  if (state.adding) return 'Adding'
  if (state.done && !state.error) {
    return state.addedToLibrary ? 'Added' : 'Already in library'
  }
  return 'Add to library'
}

async function scanCandidates(): Promise<void> {
  if (!canScan.value) return
  const sanitized = sanitizedInput.value
  if (!sanitized) return

  pathInput.value = sanitized
  scanLoading.value = true
  scanError.value = ''
  scanned.value = true
  scanResults.value = []
  rowStatuses.value = {}

  try {
    const response = await scanModelPath({
      path: sanitized,
      key: props.targetKey,
      kind: props.targetKind,
    })
    const normalizedItems = response.items.map((item) => ({
      ...item,
      size_bytes: normalizeSizeBytes(item.size_bytes, item.path),
      already_in_library: item.already_in_library === true,
    }))
    scanResults.value = [...normalizedItems].sort((left, right) => {
      const byName = left.name.localeCompare(right.name)
      if (byName !== 0) return byName
      return left.path.localeCompare(right.path)
    })
    const next: Record<string, RowStatus> = {}
    for (const item of scanResults.value) {
      const alreadyInLibrary = item.already_in_library === true
      next[item.path] = {
        adding: false,
        done: alreadyInLibrary,
        addedToLibrary: false,
        alreadyInLibrary,
        sha256: '',
        shortHash: '',
        error: '',
        sizeBytes: normalizeSizeBytes(item.size_bytes, item.path),
      }
    }
    rowStatuses.value = next
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    scanError.value = message
    emit('error', message)
  } finally {
    scanLoading.value = false
  }
}

async function addOne(item: ModelPathScanItem, index: number, options?: { silent?: boolean }): Promise<{ ok: boolean; added: boolean }> {
  const state = rowState(item)
  state.adding = true
  state.error = ''

  try {
    const response = await addModelPathItem({
      key: props.targetKey,
      kind: props.targetKind,
      path: item.path,
    })
    state.done = true
    state.addedToLibrary = Boolean(response.item.added)
    state.alreadyInLibrary = response.item.already_in_library === true || !state.addedToLibrary
    state.sha256 = String(response.item.sha256 || '')
    state.shortHash = String(response.item.short_hash || '')
    const responseSize = normalizeSizeBytes(response.item.size_bytes, item.path)
    if (responseSize !== null) state.sizeBytes = responseSize
    if (state.addedToLibrary && !options?.silent) {
      emit('added', { addedCount: 1 })
    }
    return { ok: true, added: state.addedToLibrary }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    state.error = message
    if (!options?.silent) {
      emit('error', message)
    }
    return { ok: false, added: false }
  } finally {
    state.adding = false
    if (!addAllRunning.value) {
      addAllIndex.value = Math.max(addAllIndex.value, index + 1)
    }
  }
}

async function addAllSequential(): Promise<void> {
  if (!canAddAll.value) return
  let runPlan: ReturnType<typeof planAddAllRun>
  try {
    runPlan = planAddAllRun()
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    scanError.value = message
    emit('error', message)
    return
  }
  if (runPlan.entries.length === 0) return

  addAllRunning.value = true
  addAllIndex.value = 0
  addAllActivePath.value = ''
  addAllPlannedTotalBytes.value = runPlan.totalBytes
  addAllProcessedBytes.value = 0

  let addedCount = 0
  try {
    for (let runIndex = 0; runIndex < runPlan.entries.length; runIndex += 1) {
      const { item, index } = runPlan.entries[runIndex]
      const state = rowState(item)
      addAllIndex.value = runIndex + 1
      addAllActivePath.value = item.path
      const result = await addOne(item, index, { silent: true })
      if (addAllPlannedTotalBytes.value === null) {
        const upgradedTotals = sumKnownBytesForEntries(runPlan.entries)
        if (upgradedTotals.knownEntries > 0) {
          addAllPlannedTotalBytes.value = upgradedTotals.totalBytes
          addAllProcessedBytes.value = sumProcessedKnownBytes(runPlan.entries, runIndex + 1)
        } else {
          addAllProcessedBytes.value = 0
        }
      } else {
        addAllProcessedBytes.value = sumProcessedKnownBytes(runPlan.entries, runIndex + 1)
      }
      if (addAllPlannedTotalBytes.value === null) {
        const invariantTotals = sumKnownBytesForEntries(runPlan.entries)
        if (invariantTotals.knownEntries > 0) {
          throw new Error('add-all progress invariant failed: known size entries present but total bytes unresolved')
        }
      }
      if (result.added) addedCount += 1
      if (!result.ok) {
        if (state.error) emit('error', state.error)
      }
    }
    if (runPlan.totalBytes !== null) {
      addAllProcessedBytes.value = runPlan.totalBytes
    }
  } finally {
    addAllRunning.value = false
    addAllActivePath.value = ''
  }

  if (addedCount > 0) {
    emit('added', { addedCount })
  }
}

watch(
  () => props.targetKey,
  () => {
    if (!open.value) return
    resetState()
  },
)

watch(
  () => props.targetKind,
  () => {
    if (!open.value) return
    resetState()
  },
)

watch(pathInput, (value) => {
  if (!value) return
  const sanitized = sanitizePathInput(value)
  if (sanitized !== value) {
    pathInput.value = sanitized
  }
})
</script>
