<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA picker + token insertion modal.
Fetches LoRAs via the backend API, filters by search query, and emits `<lora:filename:weight>` tokens targeting positive/negative prompt inputs.

Symbols (top-level; keep in sync; no ghosts):
- `LoraModal` (component): Modal for browsing LoRAs and emitting insertion tokens.
- `filtered` (const): Filtered LoRA list based on the current search query.
- `loadItems` (function): Loads LoRA inventory (cached or forced refresh) into the modal list.
- `resolveTokenName` (function): Resolves token filename from inventory row data.
- `normalizeInsertWeight` (function): Normalizes user-entered LoRA weight to a finite numeric value.
- `refreshList` (function): Forces a backend inventory refresh and reloads the modal list.
- `insert` (function): Emits a formatted LoRA token from a selected inventory row into a prompt target.
-->

<template>
  <Modal v-model="open" title="LoRA Selector">
    <div class="form-grid">
      <div>
        <label class="label-muted">Search</label>
        <input class="ui-input" v-model="q" placeholder="type to filter..." />
      </div>
      <div>
        <label class="label-muted">Weight</label>
        <input class="ui-input" type="number" step="0.1" min="0" v-model.number="weight" />
      </div>
    </div>
    <div class="toolbar">
      <button class="btn btn-sm btn-secondary" type="button" :disabled="loading" @click="refreshList">
        {{ loading ? 'Refreshing…' : 'Refresh' }}
      </button>
      <span class="caption">{{ filtered.length }} / {{ items.length }} LoRAs</span>
    </div>
    <p v-if="loadError" class="caption">Error: {{ loadError }}</p>
    <div class="panel-section modal-list-section">
      <ul class="list" role="listbox">
        <li v-for="item in filtered" :key="item.path || item.name" class="list-item clickable">
          <div class="flex items-center justify-between">
            <span>{{ item.name }}</span>
            <span class="lora-modal-actions">
              <button class="btn btn-sm btn-secondary" type="button" title="Insert into Prompt" @click.stop="insert(item, 'positive')">+</button>
              <button class="btn btn-sm btn-outline" type="button" title="Insert into Negative Prompt" @click.stop="insert(item, 'negative')">-</button>
            </span>
          </div>
        </li>
      </ul>
    </div>
    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="open=false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Modal from '../ui/Modal.vue'
import { fetchModelInventory, refreshModelInventory } from '../../api/client'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void; (e:'insert', payload: { token: string; target: 'positive' | 'negative' }): void }>()
const open = computed({ get: () => props.modelValue, set: (v: boolean) => emit('update:modelValue', v) })

interface LoraItem {
  name: string
  path: string
}
const items = ref<LoraItem[]>([])
const q = ref('')
const weight = ref(0.8)
const loading = ref(false)
const loaded = ref(false)
const loadError = ref('')

const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  return items.value.filter(n => n.name.toLowerCase().includes(query))
})
watch(
  open,
  async (isOpen) => {
    if (!isOpen) return
    if (loaded.value) return
    await loadItems(false)
  },
  { immediate: true },
)

async function loadItems(refresh: boolean): Promise<void> {
  if (loading.value) return
  loading.value = true
  loadError.value = ''
  try {
    const inv = refresh ? await refreshModelInventory() : await fetchModelInventory()
    items.value = (inv.loras || []) as LoraItem[]
    loaded.value = true
  } catch (error) {
    items.value = []
    loaded.value = true
    loadError.value = error instanceof Error ? error.message : String(error)
  } finally {
    loading.value = false
  }
}

async function refreshList(): Promise<void> {
  await loadItems(true)
}

function resolveTokenName(item: LoraItem): string {
  const name = String(item.name || '').trim()
  if (name) return name
  const normalizedPath = String(item.path || '').trim().replace(/\\+/g, '/')
  return normalizedPath ? (normalizedPath.split('/').pop() || '').trim() : ''
}

function normalizeInsertWeight(rawWeight: unknown): number {
  const numeric = Number(rawWeight)
  if (!Number.isFinite(numeric)) return 1.0
  return numeric
}

function insert(item: LoraItem, target: 'positive' | 'negative'): void {
  const tokenName = resolveTokenName(item)
  if (!tokenName) {
    loadError.value = `LoRA '${item.name || item.path}' has no valid filename; refresh and retry.`
    return
  }
  const resolvedWeight = normalizeInsertWeight(weight.value)
  const t = `<lora:${tokenName}:${resolvedWeight.toFixed(2)}>`
  emit('insert', { token: t, target })
}
</script>
