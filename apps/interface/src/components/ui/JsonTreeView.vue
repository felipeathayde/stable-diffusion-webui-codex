<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Collapsible JSON tree viewer.
Renders a VS Code–style expandable/collapsible JSON view using `<details>/<summary>`.

Symbols (top-level; keep in sync; no ghosts):
- `JsonTreeView` (component): Render an interactive tree for JSON-like data.
-->

<template>
  <div v-if="kind === 'object' && isMaxDepth" class="cdx-json-line">
    <span v-if="hasName" class="cdx-json-key">"{{ nameText }}"</span>
    <span v-if="hasName" class="cdx-json-punct">: </span>
    <span class="cdx-json-punct">{…}</span>
    <span v-if="!isLast" class="cdx-json-punct">,</span>
  </div>

  <div v-else-if="kind === 'array' && isMaxDepth" class="cdx-json-line">
    <span v-if="hasName" class="cdx-json-key">"{{ nameText }}"</span>
    <span v-if="hasName" class="cdx-json-punct">: </span>
    <span class="cdx-json-punct">[…] </span>
    <span v-if="!isLast" class="cdx-json-punct">,</span>
  </div>

  <details v-else-if="kind === 'object'" class="cdx-json-node" :open="isOpen" @toggle="onToggle">
    <summary class="cdx-json-summary">
      <span v-if="hasName" class="cdx-json-key">"{{ nameText }}"</span>
      <span v-if="hasName" class="cdx-json-punct">: </span>
      <span class="cdx-json-punct">{</span>
      <span class="cdx-json-preview">
        <span class="cdx-json-punct">…</span><span class="cdx-json-punct">}</span
        ><span v-if="!isLast" class="cdx-json-punct">,</span>
      </span>
    </summary>

    <div class="cdx-json-children">
      <JsonTreeView
        v-for="(entry, idx) in objectEntries"
        :key="entry.key"
        :value="entry.value"
        :name="entry.key"
        :depth="depth + 1"
        :is-last="idx === objectEntries.length - 1"
        :default-open-depth="defaultOpenDepth"
        :max-depth="maxDepth"
        :max-items="maxItems"
        :expand-all-signal="expandAllSignal"
        :collapse-all-signal="collapseAllSignal"
      />
      <div v-if="objectHasMore" class="cdx-json-more">…</div>
    </div>

    <div class="cdx-json-tail">
      <span class="cdx-json-punct">}</span><span v-if="!isLast" class="cdx-json-punct">,</span>
    </div>
  </details>

  <details v-else-if="kind === 'array'" class="cdx-json-node" :open="isOpen" @toggle="onToggle">
    <summary class="cdx-json-summary">
      <span v-if="hasName" class="cdx-json-key">"{{ nameText }}"</span>
      <span v-if="hasName" class="cdx-json-punct">: </span>
      <span class="cdx-json-punct">[</span>
      <span class="cdx-json-preview">
        <span class="cdx-json-punct">…]</span><span v-if="!isLast" class="cdx-json-punct">,</span>
      </span>
    </summary>

    <div class="cdx-json-children">
      <JsonTreeView
        v-for="(item, idx) in arrayItems"
        :key="idx"
        :value="item"
        :depth="depth + 1"
        :is-last="idx === arrayItems.length - 1"
        :default-open-depth="defaultOpenDepth"
        :max-depth="maxDepth"
        :max-items="maxItems"
        :expand-all-signal="expandAllSignal"
        :collapse-all-signal="collapseAllSignal"
      />
      <div v-if="arrayHasMore" class="cdx-json-more">…</div>
    </div>

    <div class="cdx-json-tail">
      <span class="cdx-json-punct">]</span><span v-if="!isLast" class="cdx-json-punct">,</span>
    </div>
  </details>

  <div v-else class="cdx-json-line">
    <span v-if="hasName" class="cdx-json-key">"{{ nameText }}"</span>
    <span v-if="hasName" class="cdx-json-punct">: </span>
    <span :class="valueClass">{{ valueText }}</span>
    <span v-if="!isLast" class="cdx-json-punct">,</span>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

defineOptions({ name: 'JsonTreeView' })

const props = withDefaults(
  defineProps<{
    value: unknown
    name?: string | number
    depth?: number
    isLast?: boolean
    defaultOpenDepth?: number
    maxDepth?: number
    maxItems?: number
    expandAllSignal?: number
    collapseAllSignal?: number
  }>(),
  {
    depth: 0,
    isLast: true,
    defaultOpenDepth: 1,
    maxDepth: 24,
    maxItems: 200,
    expandAllSignal: 0,
    collapseAllSignal: 0,
  },
)

const depth = computed(() => Math.max(0, Number(props.depth) || 0))
const defaultOpenDepth = computed(() => Math.max(0, Number(props.defaultOpenDepth) || 0))
const maxDepth = computed(() => Math.max(0, Number(props.maxDepth) || 0))
const maxItems = computed(() => Math.max(0, Number(props.maxItems) || 0))
const expandAllSignal = computed(() => Math.max(0, Number(props.expandAllSignal) || 0))
const collapseAllSignal = computed(() => Math.max(0, Number(props.collapseAllSignal) || 0))

const hasName = computed(() => props.name !== undefined && props.name !== null && String(props.name).length > 0)
const nameText = computed(() => String(props.name ?? ''))

const kind = computed(() => {
  const v = props.value
  if (v === null) return 'null'
  if (Array.isArray(v)) return 'array'
  if (typeof v === 'object') return 'object'
  if (typeof v === 'string') return 'string'
  if (typeof v === 'number') return 'number'
  if (typeof v === 'boolean') return 'boolean'
  return 'other'
})

const isContainer = computed(() => kind.value === 'object' || kind.value === 'array')
const isMaxDepth = computed(() => maxDepth.value > 0 && depth.value >= maxDepth.value)
const isOpen = ref(depth.value < defaultOpenDepth.value)

function onToggle(event: Event): void {
  const target = event.target
  if (!target || !(target instanceof HTMLDetailsElement)) return
  isOpen.value = Boolean(target.open)
}

watch(expandAllSignal, (value, prev) => {
  if (value === prev) return
  if (!isContainer.value || isMaxDepth.value) return
  isOpen.value = true
})

watch(collapseAllSignal, (value, prev) => {
  if (value === prev) return
  if (!isContainer.value || isMaxDepth.value) return
  isOpen.value = false
})

type Entry = { key: string; value: unknown }

const objectEntries = computed<Entry[]>(() => {
  if (kind.value !== 'object') return []
  if (props.value === null || typeof props.value !== 'object') return []
  const keys = Object.keys(props.value as Record<string, unknown>)
  const limit = maxItems.value > 0 ? maxItems.value : keys.length
  return keys.slice(0, limit).map((key) => ({ key, value: (props.value as any)[key] }))
})

const objectHasMore = computed(() => {
  if (kind.value !== 'object') return false
  if (props.value === null || typeof props.value !== 'object') return false
  const keys = Object.keys(props.value as Record<string, unknown>)
  return maxItems.value > 0 && keys.length > maxItems.value
})

const arrayItems = computed<unknown[]>(() => {
  if (kind.value !== 'array') return []
  const items = Array.isArray(props.value) ? props.value : []
  const limit = maxItems.value > 0 ? maxItems.value : items.length
  return items.slice(0, limit)
})

const arrayHasMore = computed(() => {
  if (kind.value !== 'array') return false
  const items = Array.isArray(props.value) ? props.value : []
  return maxItems.value > 0 && items.length > maxItems.value
})

const valueText = computed(() => {
  const v = props.value
  if (v === null) return 'null'
  if (typeof v === 'string') return JSON.stringify(v)
  if (typeof v === 'number') return Number.isFinite(v) ? String(v) : 'null'
  if (typeof v === 'boolean') return v ? 'true' : 'false'
  if (v === undefined) return 'undefined'
  try {
    return JSON.stringify(v)
  } catch {
    return String(v)
  }
})

const valueClass = computed(() => {
  const v = props.value
  if (v === null) return 'cdx-json-null'
  if (typeof v === 'string') return 'cdx-json-string'
  if (typeof v === 'number') return 'cdx-json-number'
  if (typeof v === 'boolean') return 'cdx-json-boolean'
  if (v === undefined) return 'cdx-json-null'
  return 'cdx-json-string'
})
</script>
