<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: LoRA picker + token insertion modal.
Fetches LoRAs via the backend API, filters by search query, and emits `<lora:filename:weight>` tokens targeting positive/negative prompt inputs.
Refreshes quicksettings LoRA SHA mappings from the same inventory payload used by the modal list.

Symbols (top-level; keep in sync; no ghosts):
- `LoraModal` (component): Modal for browsing LoRAs and emitting insertion tokens.
- `filtered` (const): Filtered LoRA list based on the current search query.
- `loadItems` (function): Loads LoRA inventory (cached or forced refresh) into the modal list.
- `resolveTokenName` (function): Resolves token filename from inventory row data.
- `normalizeInsertWeight` (function): Normalizes user-entered LoRA weight to a finite numeric value.
- `refreshList` (function): Forces a backend inventory refresh, updates quicksettings LoRA SHA map, and reloads the modal list.
- `toggleInsert` (function): Toggles add/remove insertion state for a LoRA token target (`positive`/`negative`).
-->

<template>
  <Modal v-model="open" title="LoRA Selector" panel-class="lora-modal-panel" :show-footer="false">
    <div class="lora-modal-toolbar">
      <div class="lora-modal-field lora-modal-field--search">
        <label class="label-muted">Search</label>
        <input class="ui-input" v-model="q" placeholder="type to filter..." />
      </div>
      <div class="lora-modal-field lora-modal-field--weight">
        <label class="label-muted">Weight</label>
        <input class="ui-input lora-modal-weight-input" type="number" step="0.05" min="0" v-model.number="weight" />
      </div>
      <button class="btn btn-sm btn-secondary lora-modal-refresh-btn" type="button" :disabled="loading" @click="refreshList">
        {{ loading ? 'Refreshing…' : 'Refresh' }}
      </button>
      <span class="caption lora-modal-count">{{ filtered.length }} / {{ items.length }} LoRAs</span>
    </div>
    <p v-if="loadError" class="panel-error">Error: {{ loadError }}</p>
    <div class="panel-section modal-list-section lora-modal-list-section">
      <ul class="lora-modal-list" role="listbox">
        <li v-for="item in filtered" :key="item.path || item.name" class="lora-modal-item">
          <span class="lora-modal-item__name" :title="item.name">{{ item.name }}</span>
          <span class="lora-modal-item__actions">
            <button
              :class="['btn', 'btn-sm', 'btn-secondary', 'lora-modal-action', 'lora-modal-action--positive', { 'is-active': isSelected(item, 'positive') }]"
              type="button"
              :aria-pressed="isSelected(item, 'positive') ? 'true' : 'false'"
              :title="isSelected(item, 'positive') ? 'Remove from Prompt' : 'Insert into Prompt'"
              @click.stop="toggleInsert(item, 'positive')"
            >
              {{ isSelected(item, 'positive') ? 'Prompt ✓' : 'Prompt' }}
            </button>
            <button
              :class="['btn', 'btn-sm', 'btn-outline', 'lora-modal-action', 'lora-modal-action--negative', { 'is-active': isSelected(item, 'negative') }]"
              type="button"
              :aria-pressed="isSelected(item, 'negative') ? 'true' : 'false'"
              :title="isSelected(item, 'negative') ? 'Remove from Negative Prompt' : 'Insert into Negative Prompt'"
              @click.stop="toggleInsert(item, 'negative')"
            >
              {{ isSelected(item, 'negative') ? 'Negative ✓' : 'Negative' }}
            </button>
          </span>
        </li>
      </ul>
    </div>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Modal from '../ui/Modal.vue'
import { fetchModelInventory, refreshModelInventory } from '../../api/client'
import { useQuicksettingsStore } from '../../stores/quicksettings'

type PromptTarget = 'positive' | 'negative'
type InsertAction = 'add' | 'remove'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{
  (e: 'update:modelValue', value: boolean): void
  (e: 'insert', payload: { token: string; target: PromptTarget; action: InsertAction }): void
}>()
const open = computed({ get: () => props.modelValue, set: (v: boolean) => emit('update:modelValue', v) })
const quicksettings = useQuicksettingsStore()

interface LoraItem {
  name: string
  path: string
}
const items = ref<LoraItem[]>([])
const q = ref('')
const weight = ref(1.0)
const loading = ref(false)
const loaded = ref(false)
const loadError = ref('')
const selectedPositive = ref<Record<string, string>>({})
const selectedNegative = ref<Record<string, string>>({})

const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  return items.value.filter(n => n.name.toLowerCase().includes(query))
})
watch(
  open,
  async (isOpen) => {
    if (!isOpen) {
      selectedPositive.value = {}
      selectedNegative.value = {}
      return
    }
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
    quicksettings.hydrateLoraShaMap(inv)
    items.value = ((inv.loras || []) as LoraItem[]).slice().sort((left, right) => left.name.localeCompare(right.name))
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

function selectionKey(item: LoraItem): string {
  const byPath = String(item.path || '').trim()
  if (byPath) return `path:${byPath}`
  return `name:${String(item.name || '').trim()}`
}

function isSelected(item: LoraItem, target: PromptTarget): boolean {
  const key = selectionKey(item)
  if (target === 'negative') return Boolean(selectedNegative.value[key] || '')
  return Boolean(selectedPositive.value[key] || '')
}

function selectedToken(item: LoraItem, target: PromptTarget): string {
  const key = selectionKey(item)
  if (target === 'negative') return String(selectedNegative.value[key] || '')
  return String(selectedPositive.value[key] || '')
}

function setSelected(item: LoraItem, target: PromptTarget, token: string): void {
  const key = selectionKey(item)
  if (target === 'negative') {
    selectedNegative.value[key] = token
    return
  }
  selectedPositive.value[key] = token
}

function buildToken(item: LoraItem): string {
  const tokenName = resolveTokenName(item)
  if (!tokenName) {
    loadError.value = `LoRA '${item.name || item.path}' has no valid filename; refresh and retry.`
    return ''
  }
  const resolvedWeight = normalizeInsertWeight(weight.value)
  return `<lora:${tokenName}:${resolvedWeight.toFixed(2)}>`
}

function emitInsert(token: string, target: PromptTarget, action: InsertAction): void {
  if (!token) return
  emit('insert', { token, target, action })
}

function toggleInsert(item: LoraItem, target: PromptTarget): void {
  const currentToken = selectedToken(item, target)
  if (currentToken) {
    emitInsert(currentToken, target, 'remove')
    setSelected(item, target, '')
    return
  }

  const nextToken = buildToken(item)
  if (!nextToken) return
  emitInsert(nextToken, target, 'add')
  setSelected(item, target, nextToken)
}
</script>
