<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Home dependency status panel with backend-driven OK/ERROR rows across semantic engines.
Renders backend dependency checks grouped by engine with binary indicators and actionable messages so Home can expose one
canonical readiness surface for all engines without frontend guesswork.

Symbols (top-level; keep in sync; no ghosts):
- `DependencyCheckPanel` (component): Displays backend dependency checks grouped by semantic engine.
- `EngineDependencyChecks` (type): Map of semantic engine id -> dependency status payload.
- `entries` (computed): Sorted engine entries rendered by the panel.
- `allReady` (computed): Global readiness flag across all rendered engines.
- `expandedByEngine` (ref): Per-engine expanded/collapsed state for dependency rows.
- `isExpanded` (function): Returns whether an engine group is currently expanded.
- `toggleExpanded` (function): Flips the expanded/collapsed state for an engine group.
-->

<template>
  <section class="panel dependency-check-panel">
    <div class="panel-header">
      <span>{{ title }}</span>
      <span v-if="entries.length > 0" :class="['dependency-overall', allReady ? 'is-ready' : 'is-error']">
        {{ allReady ? 'OK' : 'ERROR' }}
      </span>
    </div>
    <div class="panel-body">
      <p v-if="error" class="caption dependency-error-message">
        {{ error }}
      </p>
      <p v-else-if="loading" class="caption">
        Loading dependency checks…
      </p>
      <p v-else-if="entries.length === 0" class="caption">
        Dependency checks are unavailable.
      </p>
      <ul v-else class="dependency-engine-list">
        <li v-for="entry in entries" :key="entry.engine" class="dependency-engine-group">
          <div class="dependency-engine-header">
            <button
              class="dependency-collapse-btn"
              type="button"
              :aria-expanded="isExpanded(entry.engine)"
              :aria-label="isExpanded(entry.engine) ? `Collapse ${entry.label}` : `Expand ${entry.label}`"
              @click="toggleExpanded(entry.engine)"
            >
              <span class="dependency-engine-label">{{ entry.label }}</span>
              <span class="dependency-collapse-icon" :data-expanded="isExpanded(entry.engine) ? '1' : '0'">
                {{ isExpanded(entry.engine) ? '▾' : '▸' }}
              </span>
            </button>
            <span :class="['dependency-overall', entry.status.ready ? 'is-ready' : 'is-error']">
              {{ entry.status.ready ? 'OK' : 'ERROR' }}
            </span>
          </div>
          <ul v-if="isExpanded(entry.engine) && entry.status.checks.length > 0" class="dependency-check-list">
            <li
              v-for="row in entry.status.checks"
              :key="`${entry.engine}:${row.id}`"
              class="dependency-check-row"
            >
              <div class="dependency-check-text">
                <div class="dependency-check-label">{{ row.label }}</div>
                <p class="dependency-check-message">{{ row.message }}</p>
              </div>
              <span
                :class="['dependency-light', row.ok ? 'is-ready' : 'is-error']"
                :title="row.ok ? 'Check OK' : 'Check failed'"
                :aria-label="row.ok ? 'Check OK' : 'Check failed'"
              />
            </li>
          </ul>
          <p v-else-if="isExpanded(entry.engine)" class="caption dependency-empty-checks">
            No dependency rows reported for {{ entry.label }}.
          </p>
        </li>
      </ul>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { EngineDependencyStatus } from '../api/types'

type EngineDependencyChecks = Record<string, EngineDependencyStatus>

const props = withDefaults(defineProps<{
  statuses: EngineDependencyChecks | null | undefined
  labels?: Record<string, string>
  title?: string
  loading?: boolean
  error?: string
}>(), {
  labels: () => ({}),
  title: 'Dependency Check',
  loading: false,
  error: '',
})

const entries = computed(() =>
  Object.entries(props.statuses ?? {})
    .map(([engine, status]) => ({
      engine,
      status,
      label: String(props.labels?.[engine] || engine),
    }))
    .sort((left, right) => left.label.localeCompare(right.label, undefined, { sensitivity: 'base' })),
)

const allReady = computed(() => entries.value.every((entry) => entry.status.ready))

const expandedByEngine = ref<Record<string, boolean>>({})

watch(
  entries,
  (next) => {
    const current = expandedByEngine.value
    const updated: Record<string, boolean> = {}
    for (const entry of next) {
      if (Object.prototype.hasOwnProperty.call(current, entry.engine)) {
        updated[entry.engine] = current[entry.engine]
        continue
      }
      updated[entry.engine] = true
    }
    expandedByEngine.value = updated
  },
  { immediate: true },
)

function isExpanded(engine: string): boolean {
  return expandedByEngine.value[engine] !== false
}

function toggleExpanded(engine: string): void {
  expandedByEngine.value[engine] = !isExpanded(engine)
}

</script>
