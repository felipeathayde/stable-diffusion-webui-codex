<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Embedded XYZ sweep card.
Provides XYZ controls/results as an embeddable card (for Generation Parameters), reusing the shared XYZ store.

Symbols (top-level; keep in sync; no ghosts):
- `XyzSweepCard` (component): Embedded XYZ controls + results card.
- `onRun` (function): Starts the XYZ run via store.
- `toDataUrl` (function): Converts generated image payloads to `data:` URLs.
- `label` (function): Formats axis values for display.
- `cellKey` (function): Builds stable keys for result cells.
-->

<template>
  <div class="gen-card xyz-card">
    <div class="row-split">
      <span class="label-muted">Script</span>
      <span class="caption">X/Y/Z plot</span>
    </div>

    <div class="xyz-grid-config">
      <div class="xyz-axis-card">
        <div class="axis-header">X type</div>
        <select class="select-md" v-model="store.xParam">
          <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
        </select>
        <label class="label-muted">X values</label>
        <textarea class="ui-textarea h-prompt-sm" v-model="store.xValuesText" placeholder="e.g. 6,7,8"></textarea>
      </div>

      <div class="xyz-axis-card">
        <div class="axis-header">Y type</div>
        <select class="select-md" v-model="store.yParam">
          <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
        </select>
        <label class="label-muted">Y values</label>
        <textarea class="ui-textarea h-prompt-sm" v-model="store.yValuesText" placeholder="optional"></textarea>
      </div>

      <div class="xyz-axis-card">
        <div class="axis-header">Z type</div>
        <select class="select-md" v-model="store.zParam">
          <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
        </select>
        <label class="label-muted">Z values</label>
        <textarea class="ui-textarea h-prompt-sm" v-model="store.zValuesText" placeholder="optional"></textarea>
      </div>
    </div>

    <div class="toolbar">
      <button class="btn btn-sm btn-primary" type="button" :disabled="store.status === 'running'" @click="onRun">Run XYZ</button>
      <button class="btn btn-sm btn-outline" type="button" :disabled="store.status !== 'running'" @click="() => store.stop('after_current')">
        Stop after current
      </button>
      <button class="btn btn-sm btn-destructive" type="button" :disabled="store.status !== 'running'" @click="() => store.stop('immediate')">
        Stop now
      </button>
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
              <span class="chip">X: {{ label(cell.x) }}</span>
              <span class="chip">Y: {{ label(cell.y) }}</span>
            </div>
            <p v-if="cell.error" class="caption error-text">{{ cell.error }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { AXIS_OPTIONS, labelOf } from '../utils/xyz'
import { useXyzStore } from '../stores/xyz'
import type { GeneratedImage } from '../api/types'

const store = useXyzStore()
const axisOptions = AXIS_OPTIONS

function onRun(): void {
  void store.run()
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

