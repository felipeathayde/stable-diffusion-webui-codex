<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Embedded XYZ sweep card.
Provides XYZ controls/results as an embeddable card (for Generation Parameters), reusing the shared XYZ store
with per-axis toggles and sampler/scheduler autofill actions.
Run/stop ownership stays in the shared RUN card (this card does not render local stop buttons).

Symbols (top-level; keep in sync; no ghosts):
- `XyzSweepCard` (component): Embedded XYZ controls + results card.
- `controlsLocked` (const): Read-only lock state while an XYZ run is active.
- `toggleEnabled` (function): Toggles the XYZ master enabled state.
- `showFillAllButton` (function): Returns whether sampler/scheduler autofill is available for the selected axis param.
- `fillAxisValues` (function): Fills an axis values input with all available samplers/schedulers.
- `toDataUrl` (function): Converts generated image payloads to `data:` URLs.
- `label` (function): Formats axis values for display.
- `cellKey` (function): Builds stable keys for result cells.
-->

<template>
  <div class="gen-card xyz-card">
    <WanSubHeader
      title="XYZ workflow"
      :clickable="true"
      :disabled="controlsLocked"
      :aria-pressed="store.enabled"
      :aria-expanded="store.enabled"
      @header-click="toggleEnabled"
    >
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', store.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="store.enabled"
        :disabled="controlsLocked"
        @click.stop="toggleEnabled"
      >
        {{ store.enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </WanSubHeader>
    <div v-if="store.enabled" class="xyz-card-body">
      <div class="xyz-grid-config">
        <div :class="['xyz-axis-card', { 'xyz-axis-card--disabled': !store.xEnabled }]">
          <div class="axis-header">
            <span class="axis-title">X type</span>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', store.xEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="store.xEnabled"
              :disabled="controlsLocked"
              @click="store.xEnabled = !store.xEnabled"
            >
              {{ store.xEnabled ? 'On' : 'Off' }}
            </button>
          </div>
          <div class="axis-select-row">
            <select class="select-md" :disabled="controlsLocked || !store.xEnabled" v-model="store.xParam">
              <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
            </select>
            <button
              v-if="showFillAllButton(store.xParam)"
              class="btn btn-sm btn-outline"
              type="button"
              :disabled="controlsLocked || !store.xEnabled"
              @click="fillAxisValues('x')"
            >
              All
            </button>
          </div>
          <label class="label-muted">X values</label>
          <textarea class="ui-textarea h-prompt-sm" :disabled="controlsLocked || !store.xEnabled" v-model="store.xValuesText" placeholder="e.g. 6,7,8"></textarea>
        </div>

        <div :class="['xyz-axis-card', { 'xyz-axis-card--disabled': !store.yEnabled }]">
          <div class="axis-header">
            <span class="axis-title">Y type</span>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', store.yEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="store.yEnabled"
              :disabled="controlsLocked"
              @click="store.yEnabled = !store.yEnabled"
            >
              {{ store.yEnabled ? 'On' : 'Off' }}
            </button>
          </div>
          <div class="axis-select-row">
            <select class="select-md" :disabled="controlsLocked || !store.yEnabled" v-model="store.yParam">
              <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
            </select>
            <button
              v-if="showFillAllButton(store.yParam)"
              class="btn btn-sm btn-outline"
              type="button"
              :disabled="controlsLocked || !store.yEnabled"
              @click="fillAxisValues('y')"
            >
              All
            </button>
          </div>
          <label class="label-muted">Y values</label>
          <textarea class="ui-textarea h-prompt-sm" :disabled="controlsLocked || !store.yEnabled" v-model="store.yValuesText" placeholder="optional"></textarea>
        </div>

        <div :class="['xyz-axis-card', { 'xyz-axis-card--disabled': !store.zEnabled }]">
          <div class="axis-header">
            <span class="axis-title">Z type</span>
            <button
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', store.zEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :aria-pressed="store.zEnabled"
              :disabled="controlsLocked"
              @click="store.zEnabled = !store.zEnabled"
            >
              {{ store.zEnabled ? 'On' : 'Off' }}
            </button>
          </div>
          <div class="axis-select-row">
            <select class="select-md" :disabled="controlsLocked || !store.zEnabled" v-model="store.zParam">
              <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
            </select>
            <button
              v-if="showFillAllButton(store.zParam)"
              class="btn btn-sm btn-outline"
              type="button"
              :disabled="controlsLocked || !store.zEnabled"
              @click="fillAxisValues('z')"
            >
              All
            </button>
          </div>
          <label class="label-muted">Z values</label>
          <textarea class="ui-textarea h-prompt-sm" :disabled="controlsLocked || !store.zEnabled" v-model="store.zValuesText" placeholder="optional"></textarea>
        </div>
      </div>

      <div class="toolbar">
        <span class="caption">Run XYZ from the main Generate button while XYZ is enabled.</span>
        <span v-if="store.progress.total" class="caption">
          {{ store.progress.completed }} / {{ store.progress.total }} done · {{ store.progress.current }}
        </span>
      </div>

      <div v-if="store.errorMessage" class="panel-error">
        {{ store.errorMessage }}
      </div>

      <div v-if="store.cells.length" class="xyz-results">
        <div v-for="group in store.groupedByZ" :key="group.label" class="xyz-group">
          <div class="xyz-group-header">
            <span class="h3">Z: {{ group.label }}</span>
            <span class="caption">{{ group.rows.length }} cells</span>
          </div>
          <div class="xyz-grid">
            <div v-for="cell in group.rows" :key="cellKey(cell)" class="xyz-cell" :data-status="cell.status">
              <div class="xyz-cell-thumb">
                <template v-if="cell.image">
                  <img :src="toDataUrl(cell.image)" alt="XYZ result" />
                </template>
                <template v-else>
                  <span class="caption">{{ cell.status }}</span>
                </template>
              </div>
              <div class="xyz-cell-meta">
                <span class="xyz-chip">X: {{ label(cell.x) }}</span>
                <span class="xyz-chip">Y: {{ label(cell.y) }}</span>
              </div>
              <p v-if="cell.error" class="caption error-text">{{ cell.error }}</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { AXIS_OPTIONS, labelOf, type AxisParam } from '../utils/xyz'
import { useXyzStore } from '../stores/xyz'
import type { GeneratedImage } from '../api/types'
import WanSubHeader from './wan/WanSubHeader.vue'
import { computed } from 'vue'

type AxisKey = 'x' | 'y' | 'z'

const props = withDefaults(defineProps<{
  samplers?: string[]
  schedulers?: string[]
}>(), {
  samplers: () => [],
  schedulers: () => [],
})

const store = useXyzStore()
const axisOptions = AXIS_OPTIONS
const controlsLocked = computed(() => store.status === 'running')

function toggleEnabled(): void {
  if (controlsLocked.value) return
  store.enabled = !store.enabled
}

function axisValuesForParam(param: AxisParam): string[] {
  if (param === 'sampler') return props.samplers
  if (param === 'scheduler') return props.schedulers
  return []
}

function showFillAllButton(param: AxisParam): boolean {
  return axisValuesForParam(param).length > 0
}

function normalizedCsv(values: string[]): string {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of values) {
    const value = String(raw || '').trim()
    if (!value) continue
    const key = value.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    out.push(value)
  }
  return out.join(',')
}

function fillAxisValues(axis: AxisKey): void {
  if (axis === 'x') {
    store.xValuesText = normalizedCsv(axisValuesForParam(store.xParam))
    return
  }
  if (axis === 'y') {
    store.yValuesText = normalizedCsv(axisValuesForParam(store.yParam))
    return
  }
  store.zValuesText = normalizedCsv(axisValuesForParam(store.zParam))
}

function toDataUrl(image: GeneratedImage): string {
  return `data:image/${image.format};base64,${image.data}`
}

function label(value: unknown): string {
  return labelOf(value as any)
}

function cellKey(cell: { x: unknown; y: unknown; z: unknown }): string {
  return `${label(cell.z)}-${label(cell.y)}-${label(cell.x)}`
}
</script>
