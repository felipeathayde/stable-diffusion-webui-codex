<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reusable quicksettings add-path modal (scan + add-to-library).
Provides add-path workflows for checkpoint/VAE/text-encoder path keys by scanning a user-supplied path (no hash on scan),
then adding selected/all files with SHA computed only at add-time.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsAddPathModal` (component): Modal for scanning and adding model files into a target paths.json key.
- `sanitizePathInput` (function): Sanitizes path input (trim, quote removal, slash normalization, repeated separator collapse).
- `scanCandidates` (function): Calls backend scan endpoint and populates candidate rows (no SHA).
- `addOne` (function): Adds one candidate file to library key and records per-row SHA/result state.
- `addAllSequential` (function): Adds all scanned candidates sequentially with visible per-row progress.
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
          {{ addAllRunning ? `Adding ${addAllIndex}/${scanResults.length}…` : 'Add whole folder' }}
        </button>
      </div>

      <p v-if="scanError" class="panel-error">Error: {{ scanError }}</p>
      <p v-else-if="scanned && scanResults.length === 0 && !scanLoading" class="caption">No supported files found.</p>

      <div v-if="scanResults.length > 0" class="panel-section modal-list-section qs-add-path-list-section">
        <table class="qs-add-path-table">
          <tbody>
            <tr
              v-for="(item, index) in scanResults"
              :key="item.path"
              :class="['qs-add-path-row', rowState(item).error ? 'is-error' : '']"
            >
              <td class="qs-add-path-row__name" :title="item.path">
                <div class="qs-add-path-row__title">{{ displayName(item) }}</div>
                <div v-if="rowState(item).error" class="qs-add-path-row__status qs-add-path-row__status--error">{{ rowState(item).error }}</div>
                <div
                  v-else-if="rowState(item).done && rowState(item).sha256"
                  class="qs-add-path-row__status"
                  :title="`SHA256: ${rowState(item).sha256}`"
                >
                  {{ rowState(item).addedToLibrary ? 'Added' : 'Already in library' }} · SHA {{ shortSha(rowState(item).sha256) }}
                </div>
              </td>
              <td class="qs-add-path-row__actions">
                <button
                  class="btn btn-sm btn-outline"
                  type="button"
                  :disabled="isRowActionDisabled(item)"
                  @click="addOne(item, index)"
                >
                  {{ rowActionLabel(item, index) }}
                </button>
              </td>
            </tr>
          </tbody>
        </table>
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
  sha256: string
  shortHash: string
  error: string
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
const canAddAll = computed(() => !scanLoading.value && !addAllRunning.value && !hasRowAddInFlight.value && scanResults.value.length > 0)

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

function rowState(item: ModelPathScanItem): RowStatus {
  const existing = rowStatuses.value[item.path]
  if (existing) return existing
  const created: RowStatus = {
    adding: false,
    done: false,
    addedToLibrary: false,
    sha256: '',
    shortHash: '',
    error: '',
  }
  rowStatuses.value[item.path] = created
  return created
}

function isRowActionDisabled(item: ModelPathScanItem): boolean {
  const state = rowState(item)
  if (state.adding) return true
  if (addAllRunning.value) return true
  return state.done && !state.error
}

function rowActionLabel(item: ModelPathScanItem, index: number): string {
  const state = rowState(item)
  if (state.adding) {
    if (addAllRunning.value && addAllActivePath.value === item.path) {
      return `Adding ${index + 1}/${scanResults.value.length}…`
    }
    return 'Adding…'
  }
  if (state.done && !state.error) {
    return state.addedToLibrary ? 'Added' : 'Already added'
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
    scanResults.value = [...response.items].sort((left, right) => {
      const byName = left.name.localeCompare(right.name)
      if (byName !== 0) return byName
      return left.path.localeCompare(right.path)
    })
    const next: Record<string, RowStatus> = {}
    for (const item of scanResults.value) {
      next[item.path] = {
        adding: false,
        done: false,
        addedToLibrary: false,
        sha256: '',
        shortHash: '',
        error: '',
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
    state.sha256 = String(response.item.sha256 || '')
    state.shortHash = String(response.item.short_hash || '')
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
  addAllRunning.value = true
  addAllIndex.value = 0
  addAllActivePath.value = ''

  let addedCount = 0
  try {
    for (let index = 0; index < scanResults.value.length; index += 1) {
      const item = scanResults.value[index]
      addAllIndex.value = index + 1
      addAllActivePath.value = item.path
      const result = await addOne(item, index, { silent: true })
      if (result.added) addedCount += 1
      if (!result.ok) {
        const state = rowState(item)
        if (state.error) emit('error', state.error)
      }
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
