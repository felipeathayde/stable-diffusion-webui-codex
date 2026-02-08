<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Metadata viewer modal for model/asset selections.
Displays a read-only JSON view of the resolved metadata payload for the current selection (checkpoint / text encoder / VAE / WAN stage),
with an optional toggle to switch between raw and nested views for file metadata payloads.

Symbols (top-level; keep in sync; no ghosts):
- `AssetMetadataModal` (component): Modal for displaying a JSON metadata payload.
- `showTreePayload` (const): Computed gate selecting nested-tree view when payload is object-like and beautify mode applies.
-->

<template>
  <Modal v-model="open" :title="title">
    <div v-if="subtitle" class="cdx-metadata-modal__subtitle">
      <code class="cdx-metadata-modal__subtitle-code">{{ subtitle }}</code>
    </div>

    <div v-if="!hasPayload" class="card text-sm">
      No metadata available for this selection.
    </div>

    <div v-else-if="showTreePayload" class="card text-sm cdx-metadata-modal__viewer">
      <div class="cdx-metadata-modal__overlay">
        <button
          class="btn-icon cdx-metadata-modal__icon-btn"
          type="button"
          aria-label="Collapse all"
          title="Collapse all"
          @click="collapseAll"
        >
          <svg class="cdx-metadata-modal__icon" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 15L12 9L18 15" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
        <button
          class="btn-icon cdx-metadata-modal__icon-btn"
          type="button"
          aria-label="Expand all"
          title="Expand all"
          @click="expandAll"
        >
          <svg class="cdx-metadata-modal__icon" width="14" height="14" viewBox="0 0 24 24" aria-hidden="true">
            <path d="M6 9L12 15L18 9" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" />
          </svg>
        </button>
        <button
          v-if="supportsBeautifyToggle"
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', beautify ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :aria-pressed="beautify"
          :title="beautify ? 'Beautify: ON (nested/organized)' : 'Beautify: OFF (raw/flat)'"
          @click="beautify = !beautify"
        >
          Beautify
        </button>
      </div>

      <div class="cdx-metadata-modal__json cdx-json-scroll">
        <JsonTreeView :value="displayPayload" :expand-all-signal="expandAllSignal" :collapse-all-signal="collapseAllSignal" />
      </div>
    </div>

    <pre v-else class="card text-sm cdx-metadata-modal__json cdx-json-scroll"><code>{{ pretty }}</code></pre>

    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="open = false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import Modal from '../ui/Modal.vue'
import JsonTreeView from '../ui/JsonTreeView.vue'

const props = defineProps<{
  modelValue: boolean
  title: string
  subtitle?: string
  payload?: unknown
}>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void }>()

const open = computed({
  get: () => props.modelValue,
  set: (v: boolean) => emit('update:modelValue', v),
})

const beautify = ref(true)
const expandAllSignal = ref(0)
const collapseAllSignal = ref(0)

function expandAll(): void {
  expandAllSignal.value += 1
}

function collapseAll(): void {
  collapseAllSignal.value += 1
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

const supportsBeautifyToggle = computed(() => {
  const payload = props.payload
  if (!isPlainObject(payload)) return false
  const meta = (payload as any).metadata
  if (!isPlainObject(meta)) return false
  const raw = (meta as any).raw
  const nested = (meta as any).nested
  return isPlainObject(raw) && isPlainObject(nested)
})

const displayPayload = computed(() => {
  const payload = props.payload
  if (!supportsBeautifyToggle.value) return payload
  if (!isPlainObject(payload)) return payload

  const meta = (payload as any).metadata
  if (!isPlainObject(meta)) return payload

  const raw = (meta as any).raw
  const nested = (meta as any).nested
  if (!isPlainObject(raw) || !isPlainObject(nested)) return payload

  const common: Record<string, unknown> = {}
  if (typeof (meta as any).path === 'string') common.path = (meta as any).path
  if (typeof (meta as any).kind === 'string') common.kind = (meta as any).kind
  if ((meta as any).summary && typeof (meta as any).summary === 'object') common.summary = (meta as any).summary

  const view = beautify.value ? nested : raw
  return { ...(payload as any), metadata: { ...common, ...(view as any) } }
})

const hasPayload = computed(() => {
  const payload = displayPayload.value
  if (payload === null || payload === undefined) return false
  if (typeof payload === 'string') return payload.trim().length > 0
  if (Array.isArray(payload)) return payload.length > 0
  if (typeof payload === 'object') return Object.keys(payload as Record<string, unknown>).length > 0
  return true
})

const pretty = computed(() => {
  const payload = displayPayload.value
  if (payload === null || payload === undefined) return ''
  if (typeof payload === 'string') return payload
  try {
    return JSON.stringify(payload, null, 2)
  } catch {
    return String(payload)
  }
})

const isTreePayload = computed(() => {
  const payload = displayPayload.value
  return payload !== null && payload !== undefined && typeof payload === 'object'
})

const showTreePayload = computed(() => {
  if (!isTreePayload.value) return false
  if (!supportsBeautifyToggle.value) return true
  return beautify.value
})
</script>
