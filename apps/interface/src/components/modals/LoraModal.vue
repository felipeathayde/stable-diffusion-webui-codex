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
- `loadItems` (function): Loads cached LoRA inventory into the modal list.
- `resolveTokenName` (function): Resolves token filename from inventory row data.
- `normalizeInsertWeight` (function): Normalizes user-entered LoRA weight to a finite numeric value.
- `parseInventoryTaskResult` (function): Extracts inventory payloads from refresh task `result` SSE events.
- `runAsyncInventoryRefreshTask` (function): Starts and awaits async inventory refresh task completion.
- `cancelActiveRefreshTask` (function): Cancels any in-flight refresh task SSE subscription.
- `refreshList` (function): Runs async inventory refresh and applies LoRA/quicksettings updates on terminal success.
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
import { computed, onBeforeUnmount, ref, watch } from 'vue'
import Modal from '../ui/Modal.vue'
import { cacheModelInventorySnapshot, fetchModelInventory, startModelInventoryRefreshTask, subscribeTask } from '../../api/client'
import type { InventoryResponse, TaskEvent } from '../../api/types'
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
let activeRefreshCancel: (() => void) | null = null
let refreshAbortRequested = false

const REFRESH_CANCELLED_MESSAGE = 'LoRA inventory refresh cancelled'

const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  return items.value.filter(n => n.name.toLowerCase().includes(query))
})
watch(
  open,
  async (isOpen) => {
    if (!isOpen) {
      cancelActiveRefreshTask()
      selectedPositive.value = {}
      selectedNegative.value = {}
      return
    }
    if (loaded.value) return
    await loadItems()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  cancelActiveRefreshTask()
})

function isRecordObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function parseInventoryTaskResult(event: TaskEvent): InventoryResponse | null {
  if (event.type !== 'result') return null
  const payload = event as unknown as Record<string, unknown>
  const direct = payload.inventory
  if (isRecordObject(direct)) return direct as unknown as InventoryResponse
  const info = payload.info
  if (!isRecordObject(info)) return null
  const nested = info.inventory
  if (!isRecordObject(nested)) return null
  return nested as unknown as InventoryResponse
}

function sortedLoraItems(inv: Pick<InventoryResponse, 'loras'>): LoraItem[] {
  if (!Array.isArray(inv.loras)) {
    throw new Error('LoRA inventory payload is missing required loras[] array')
  }
  return (inv.loras as LoraItem[]).slice().sort((left, right) => left.name.localeCompare(right.name))
}

function applyInventorySnapshot(inv: InventoryResponse): void {
  quicksettings.hydrateLoraShaMap(inv)
  items.value = sortedLoraItems(inv)
  loaded.value = true
}

async function loadItems(): Promise<void> {
  if (loading.value) return
  loading.value = true
  loadError.value = ''
  try {
    const inv = await fetchModelInventory()
    applyInventorySnapshot(inv)
  } catch (error) {
    items.value = []
    loaded.value = true
    loadError.value = error instanceof Error ? error.message : String(error)
  } finally {
    loading.value = false
  }
}

function cancelActiveRefreshTask(): void {
  refreshAbortRequested = true
  const cancel = activeRefreshCancel
  if (!cancel) return
  activeRefreshCancel = null
  try {
    cancel()
  } catch (_) {
    // Ignore cancellation callback failures.
  }
}

async function runAsyncInventoryRefreshTask(): Promise<InventoryResponse> {
  const { task_id } = await startModelInventoryRefreshTask()
  if (refreshAbortRequested || !open.value) {
    throw new Error(REFRESH_CANCELLED_MESSAGE)
  }
  return await new Promise<InventoryResponse>((resolve, reject) => {
    let settled = false
    let unsubscribe: (() => void) | null = null
    let resolvedInventory: InventoryResponse | null = null

    const settle = (fn: () => void): void => {
      if (settled) return
      settled = true
      try { unsubscribe?.() } catch (_) { /* ignore */ }
      unsubscribe = null
      if (activeRefreshCancel === cancel) {
        activeRefreshCancel = null
      }
      fn()
    }

    const cancel = (): void => {
      settle(() => reject(new Error(REFRESH_CANCELLED_MESSAGE)))
    }
    activeRefreshCancel = cancel

    unsubscribe = subscribeTask(
      task_id,
      (event) => {
        if (event.type === 'error') {
          settle(() => reject(new Error(String(event.message || 'LoRA inventory refresh task failed'))))
          return
        }
        if (event.type === 'result') {
          const parsed = parseInventoryTaskResult(event)
          if (!parsed) {
            settle(() => reject(new Error('LoRA inventory refresh task result missing inventory payload')))
            return
          }
          if (!Array.isArray(parsed.loras)) {
            settle(() => reject(new Error('LoRA inventory refresh task result payload missing inventory.loras[]')))
            return
          }
          resolvedInventory = parsed
          return
        }
        if (event.type === 'end') {
          if (!resolvedInventory) {
            settle(() => reject(new Error('LoRA inventory refresh task completed without inventory payload')))
            return
          }
          settle(() => resolve(resolvedInventory as InventoryResponse))
        }
      },
      (err) => {
        settle(() => reject(err instanceof Error ? err : new Error(String(err))))
      },
    )
  })
}

async function refreshList(): Promise<void> {
  if (loading.value) return
  refreshAbortRequested = false
  loading.value = true
  loadError.value = ''
  try {
    const refreshedInventory = await runAsyncInventoryRefreshTask()
    cacheModelInventorySnapshot(refreshedInventory)
    applyInventorySnapshot(refreshedInventory)
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error)
    if (message !== REFRESH_CANCELLED_MESSAGE) {
      loadError.value = message
    }
  } finally {
    loading.value = false
  }
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
