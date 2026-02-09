<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared tile controls UI for upscaler-based stages (standalone `/upscale` and hires-fix).
Renders tile preset buttons (128/256/512/768), overlap input, a `min_tile` control, and an explicit "fallback on OOM" toggle.

Symbols (top-level; keep in sync; no ghosts):
- `UpscalerTileControls` (component): Presentational tile config controls; emits normalized updates for tile size/overlap/min tile/fallback.
- `setTileSize` (function): Sets tile size and clamps overlap to `< tile`.
- `setOverlap` (function): Sets overlap and clamps to `[0, tile-1]`.
- `setMinTile` (function): Sets min tile and clamps to `[1, tile]`.
- `presetButtonClass` (function): Returns preset button classes for the selected visual variant.
-->

<template>
  <div>
    <div class="toolbar">
      <button
        v-for="preset in presets"
        :key="preset"
        :class="presetButtonClass(preset)"
        type="button"
        :aria-pressed="tileSizeIsSelected(preset)"
        :disabled="disabled"
        @click="setTileSize(preset)"
      >
        {{ preset }}
      </button>
    </div>

    <div class="toolbar">
      <span class="caption">Overlap</span>
      <NumberStepperInput
        :modelValue="overlap"
        :min="0"
        :max="Math.max(0, tileSize - 1)"
        :step="4"
        :nudgeStep="4"
        size="sm"
        inputClass="cdx-input-w-xs"
        :disabled="disabled"
        updateOnInput
        @update:modelValue="setOverlap"
      />
      <span class="caption">Min tile</span>
      <NumberStepperInput
        :modelValue="minTile"
        :min="1"
        :max="Math.max(1, tileSize)"
        :step="8"
        :nudgeStep="8"
        size="sm"
        inputClass="cdx-input-w-xs"
        :disabled="disabled"
        updateOnInput
        @update:modelValue="setMinTile"
      />
      <button
        class="btn qs-toggle-btn qs-toggle-btn--sm"
        :class="fallbackOnOom ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off'"
        type="button"
        :disabled="disabled"
        title="When enabled, the backend halves tile size on OOM until min tile; otherwise it raises immediately."
        @click="emit('update:fallbackOnOom', !fallbackOnOom)"
      >
        Fallback on OOM
      </button>
    </div>
  </div>
</template>

<script setup lang="ts">
import NumberStepperInput from './NumberStepperInput.vue'

const presets = [128, 256, 512, 768] as const

const props = withDefaults(defineProps<{
  tileSize: number
  overlap: number
  minTile: number
  fallbackOnOom: boolean
  disabled?: boolean
  presetVariant?: 'toggle' | 'resolution'
}>(), {
  presetVariant: 'toggle',
})

const emit = defineEmits<{
  (e: 'update:tileSize', value: number): void
  (e: 'update:overlap', value: number): void
  (e: 'update:minTile', value: number): void
  (e: 'update:fallbackOnOom', value: boolean): void
}>()

function setTileSize(value: number): void {
  const v = Math.trunc(Number(value))
  if (!Number.isFinite(v) || v <= 0) return
  emit('update:tileSize', v)
  const nextOverlap = Math.max(0, Math.min(v - 1, Math.trunc(Number(props.overlap))))
  if (Number.isFinite(nextOverlap) && nextOverlap !== Math.trunc(Number(props.overlap))) {
    emit('update:overlap', nextOverlap)
  }
}

function presetButtonClass(preset: number): string[] {
  if (props.presetVariant === 'resolution') {
    return ['btn', 'btn-sm', 'btn-outline']
  }
  return ['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', tileSizeIsSelected(preset) ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']
}

function tileSizeIsSelected(preset: number): boolean {
  return Math.trunc(Number(props.tileSize)) === Math.trunc(Number(preset))
}

function setOverlap(value: number): void {
  const v = Math.trunc(Number(value))
  if (!Number.isFinite(v)) return
  const max = Math.max(0, Math.trunc(Number(props.tileSize)) - 1)
  emit('update:overlap', Math.max(0, Math.min(max, v)))
}

function setMinTile(value: number): void {
  const v = Math.trunc(Number(value))
  if (!Number.isFinite(v) || v <= 0) return
  const max = Math.max(1, Math.trunc(Number(props.tileSize)))
  emit('update:minTile', Math.max(1, Math.min(max, v)))
}
</script>
