<template>
  <section class="panels xyz-panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header"><span>XYZ sweep</span></div>
        <div class="panel-body xyz-body">
          <p class="caption">Run batched txt2img sweeps varying up to three parameters (X/Y/Z). Uses current SDXL settings as the base payload.</p>

          <div class="xyz-grid-config">
            <div class="xyz-axis-card">
              <div class="axis-header">X axis</div>
              <label class="label-muted">Parameter</label>
              <select class="select-md" v-model="store.xParam">
                <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
              </select>
              <label class="label-muted">Values (comma or newline)</label>
              <textarea class="ui-textarea" rows="4" v-model="store.xValuesText" placeholder="e.g. 6,7,8"></textarea>
              <p class="caption">Required. Each value will be swept across columns.</p>
            </div>

            <div class="xyz-axis-card">
              <div class="axis-header">Y axis</div>
              <label class="label-muted">Parameter</label>
              <select class="select-md" v-model="store.yParam">
                <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
              </select>
              <label class="label-muted">Values (comma or newline)</label>
              <textarea class="ui-textarea" rows="4" v-model="store.yValuesText" placeholder="optional"></textarea>
              <p class="caption">Optional. Leave empty to sweep only X.</p>
            </div>

            <div class="xyz-axis-card">
              <div class="axis-header">Z axis</div>
              <label class="label-muted">Parameter</label>
              <select class="select-md" v-model="store.zParam">
                <option v-for="opt in axisOptions" :key="opt.id" :value="opt.id">{{ opt.label }}</option>
              </select>
              <label class="label-muted">Values (comma or newline)</label>
              <textarea class="ui-textarea" rows="4" v-model="store.zValuesText" placeholder="optional"></textarea>
              <p class="caption">Optional. Each Z value becomes a separate grid.</p>
            </div>
          </div>

          <div class="toolbar">
            <button class="btn btn-primary btn-md" type="button" :disabled="store.status === 'running'" @click="onRun">Run XYZ</button>
            <div class="toolbar">
              <button class="btn btn-outline btn-sm" type="button" :disabled="store.status !== 'running'" @click="() => store.stop('after_current')">Stop after current</button>
              <button class="btn btn-destructive btn-sm" type="button" :disabled="store.status !== 'running'" @click="() => store.stop('immediate')">Stop now</button>
            </div>
            <span class="caption" v-if="store.progress.total">{{ store.progress.completed }} / {{ store.progress.total }} done · {{ store.progress.current }}</span>
          </div>
          <div v-if="store.errorMessage" class="panel-error">{{ store.errorMessage }}</div>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header"><span>Results</span></div>
        <div class="panel-body xyz-results" v-if="store.cells.length">
          <div v-for="group in store.groupedByZ" :key="group.label" class="xyz-group">
            <div class="xyz-group-header">
              <span class="h3">Z: {{ group.label }}</span>
              <span class="caption">{{ group.rows.length }} cells</span>
            </div>
            <div class="xyz-grid">
              <div v-for="cell in group.rows" :key="cellKey(cell)" class="xyz-cell" :data-status="cell.status">
                <div class="xyz-cell-thumb">
                  <template v-if="cell.image">
                    <img :src="toDataUrl(cell.image)" alt="result" />
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
        <div v-else class="panel-body">
          <p class="caption">No runs yet. Configure axes and press Run.</p>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
// tags: xyz, sweeps, grid
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
