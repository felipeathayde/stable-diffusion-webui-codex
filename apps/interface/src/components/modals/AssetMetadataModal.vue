<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Metadata viewer modal for model/asset selections.
Displays a read-only JSON view of the resolved metadata payload for the current selection (checkpoint / text encoder / VAE / WAN stage).

Symbols (top-level; keep in sync; no ghosts):
- `AssetMetadataModal` (component): Modal for displaying a JSON metadata payload.
-->

<template>
  <Modal v-model="open" :title="title">
    <p v-if="subtitle" class="subtitle">{{ subtitle }}</p>

    <div v-if="!hasPayload" class="card text-sm">
      No metadata available for this selection.
    </div>

    <pre v-else class="card text-sm"><code>{{ pretty }}</code></pre>

    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="open = false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import Modal from '../ui/Modal.vue'

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

const hasPayload = computed(() => {
  const payload = props.payload
  if (payload === null || payload === undefined) return false
  if (typeof payload === 'string') return payload.trim().length > 0
  if (Array.isArray(payload)) return payload.length > 0
  if (typeof payload === 'object') return Object.keys(payload as Record<string, unknown>).length > 0
  return true
})

const pretty = computed(() => {
  const payload = props.payload
  if (payload === null || payload === undefined) return ''
  if (typeof payload === 'string') return payload
  try {
    return JSON.stringify(payload, null, 2)
  } catch {
    return String(payload)
  }
})
</script>

