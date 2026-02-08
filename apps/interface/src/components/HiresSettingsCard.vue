<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Hires (second pass) settings panel.
Renders hires controls (scale/denoise/steps/upscaler + Spandrel tile config, including min tile and OOM fallback preference) and optional
second-pass swap-model settings when enabled.
Upscaler values are stable ids (`latent:*` / `spandrel:*`), not legacy display labels.

Symbols (top-level; keep in sync; no ghosts):
- `HiresSettingsCard` (component): Hires settings block for supported image tabs.
- `toggle` (function): Toggles the hires enabled state.
-->

<template>
  <div class="gen-card hires-card">
    <div class="row-split">
      <span class="label-muted">Hires (second pass)</span>
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="enabled"
        @click="toggle"
      >
        {{ enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </div>
    <div v-if="enabled" class="hr-grid">
      <div class="field">
        <SliderField
          label="Scale"
          :modelValue="scale"
          :min="1"
          :max="4"
          :step="0.1"
          :inputStep="0.1"
          :disabled="disabled || !enabled"
          :showButtons="false"
          inputClass="cdx-input-w-xs"
          @update:modelValue="(v) => emit('update:scale', v)"
        />
        <p class="hr-hint" v-if="targetWidth && targetHeight">Target ~ {{ targetWidth }}×{{ targetHeight }}</p>
      </div>
      <div class="field">
        <SliderField
          label="Denoise"
          :modelValue="denoise"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :disabled="disabled || !enabled"
          :showButtons="false"
          inputClass="cdx-input-w-xs"
          @update:modelValue="(v) => emit('update:denoise', v)"
        />
      </div>
      <div class="field">
        <label class="label-muted">Hires steps</label>
        <input
          class="ui-input ui-input-sm"
          type="number"
          min="0"
          :value="steps"
          :disabled="disabled || !enabled"
          @change="onStepsChange"
        />
        <p class="hr-hint">0 = reuse base steps</p>
      </div>
      <div class="field">
        <label class="label-muted">Upscaler</label>
        <select class="select-md" :value="upscaler" :disabled="disabled || !enabled || upscalersLoading" @change="onUpscalerChange">
          <option v-if="upscalersLoading" :value="upscaler">Loading…</option>
          <option v-else-if="upscaler && !isUpscalerKnown" :value="upscaler">Invalid selection: {{ upscaler }}</option>
          <option v-else value="" disabled>Select</option>
          <optgroup v-if="spandrelUpscalers.length" label="Spandrel (pixel SR)">
            <option v-for="u in spandrelUpscalers" :key="u.id" :value="u.id">{{ u.label }}</option>
          </optgroup>
          <optgroup v-if="latentUpscalers.length" label="Latent">
            <option v-for="u in latentUpscalers" :key="u.id" :value="u.id">{{ u.label }}</option>
          </optgroup>
        </select>
        <p class="hr-hint" v-if="upscalersError">Error: {{ upscalersError }}</p>
        <p class="hr-hint" v-else-if="upscaler && !isUpscalerKnown">Select an upscaler id from `GET /api/upscalers`.</p>
      </div>
      <div class="field hr-field--full">
        <label class="label-muted">Tile</label>
        <UpscalerTileControls
          :tileSize="tileConfig.tile"
          :overlap="tileConfig.overlap"
          :minTile="minTile"
          :fallbackOnOom="fallbackOnOom"
          :disabled="disabled || !enabled || !isSpandrelSelected"
          @update:tileSize="onTileSize"
          @update:overlap="onTileOverlap"
          @update:minTile="(v) => emit('update:minTile', v)"
          @update:fallbackOnOom="(v) => emit('update:fallbackOnOom', v)"
        />
        <p class="hr-hint" v-if="upscaler && !isSpandrelSelected">Tile settings apply to Spandrel (pixel SR) upscalers only.</p>
      </div>
    </div>
    <div v-if="enabled && showRefiner" class="hr-refiner">
      <RefinerSettingsCard
        label="Second-Pass Swap Model"
        :dense="true"
        :model-choices="refinerModelChoices"
        v-model:enabled="refinerEnabled"
        v-model:swapAtStep="refinerSwapAtStep"
        v-model:cfg="refinerCfg"
        v-model:seed="refinerSeed"
        v-model:model="refinerModel"
      />
      <p class="hr-hint">Swap uses step-pointer semantics in the second pass (switch model at the selected step).</p>
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: hires, settings, grid
import { computed } from 'vue'
import type { UpscalerDefinition, UpscalerKind } from '../api/types'
import RefinerSettingsCard from './RefinerSettingsCard.vue'
import SliderField from './ui/SliderField.vue'
import UpscalerTileControls from './ui/UpscalerTileControls.vue'

type TileConfigState = { tile: number; overlap: number }

const props = defineProps<{
  disabled?: boolean
  enabled: boolean
  denoise: number
  scale: number
  steps: number
  upscaler: string
  tile?: TileConfigState
  minTile?: number
  fallbackOnOom?: boolean
  upscalers?: UpscalerDefinition[]
  upscalersLoading?: boolean
  upscalersError?: string
  baseWidth?: number
  baseHeight?: number
  refinerEnabled?: boolean
  refinerSwapAtStep?: number
  refinerCfg?: number
  refinerSeed?: number
  refinerModel?: string
  refinerModelChoices?: string[]
}>()

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:denoise', value: number): void
  (e: 'update:scale', value: number): void
  (e: 'update:steps', value: number): void
  (e: 'update:upscaler', value: string): void
  (e: 'update:tile', value: TileConfigState): void
  (e: 'update:minTile', value: number): void
  (e: 'update:fallbackOnOom', value: boolean): void
  (e: 'update:refinerEnabled', value: boolean): void
  (e: 'update:refinerSwapAtStep', value: number): void
  (e: 'update:refinerCfg', value: number): void
  (e: 'update:refinerSeed', value: number): void
  (e: 'update:refinerModel', value: string): void
}>()

const disabled = computed(() => Boolean(props.disabled))
const fallbackOnOom = computed(() => props.fallbackOnOom ?? true)
const upscalersLoading = computed(() => Boolean(props.upscalersLoading))
const upscalersError = computed(() => String(props.upscalersError ?? '').trim())

const upscalers = computed(() => Array.isArray(props.upscalers) ? props.upscalers : [])
const spandrelUpscalers = computed(() => upscalers.value.filter((u) => u.kind === 'spandrel'))
const latentUpscalers = computed(() => upscalers.value.filter((u) => u.kind === 'latent'))
const isUpscalerKnown = computed(() => upscalers.value.some((u) => u.id === props.upscaler))
const selectedUpscalerKind = computed<UpscalerKind | null>(() => {
  const found = upscalers.value.find((u) => u.id === props.upscaler)
  if (found) return found.kind
  const id = String(props.upscaler || '')
  if (id.startsWith('spandrel:')) return 'spandrel'
  if (id.startsWith('latent:')) return 'latent'
  return null
})
const isSpandrelSelected = computed(() => selectedUpscalerKind.value === 'spandrel')

const tileConfig = computed<TileConfigState>(() => {
  const v = props.tile
  if (!v) return { tile: 256, overlap: 16 }
  const tile = Number.isFinite(v.tile) ? Math.max(1, Math.trunc(v.tile)) : 256
  const overlap = Number.isFinite(v.overlap) ? Math.max(0, Math.trunc(v.overlap)) : 16
  return { tile, overlap: Math.min(tile - 1, overlap) }
})

const minTile = computed(() => {
  const raw = props.minTile
  const v = (typeof raw === 'number' && Number.isFinite(raw)) ? Math.max(1, Math.trunc(raw)) : 128
  return Math.min(tileConfig.value.tile, v)
})

const targetWidth = computed(() => {
  if (!props.baseWidth || props.scale <= 1) return null
  return Math.round(props.baseWidth * props.scale)
})

const targetHeight = computed(() => {
  if (!props.baseHeight || props.scale <= 1) return null
  return Math.round(props.baseHeight * props.scale)
})

const showRefiner = computed(() => props.refinerEnabled !== undefined)
const refinerEnabled = computed({
  get: () => Boolean(props.refinerEnabled),
  set: (value: boolean) => emit('update:refinerEnabled', value),
})
const refinerSwapAtStep = computed({
  get: () => {
    const value = Number(props.refinerSwapAtStep)
    if (!Number.isFinite(value) || value < 1) return 1
    return Math.trunc(value)
  },
  set: (value: number) => emit('update:refinerSwapAtStep', value),
})
const refinerCfg = computed({
  get: () => Number.isFinite(props.refinerCfg) ? Number(props.refinerCfg) : 7,
  set: (value: number) => emit('update:refinerCfg', value),
})
const refinerSeed = computed({
  get: () => Number.isFinite(props.refinerSeed) ? Number(props.refinerSeed) : -1,
  set: (value: number) => emit('update:refinerSeed', value),
})
const refinerModel = computed({
  get: () => props.refinerModel ?? '',
  set: (value: string) => emit('update:refinerModel', value),
})
const refinerModelChoices = computed(() => Array.isArray(props.refinerModelChoices) ? props.refinerModelChoices : [])

function toggle(): void {
  emit('update:enabled', !props.enabled)
}

function onStepsChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:steps', Number.isNaN(v) || v < 0 ? 0 : v)
}

function onUpscalerChange(event: Event): void {
  emit('update:upscaler', (event.target as HTMLSelectElement).value)
}

function onTileSize(value: number): void {
  const v = Math.max(1, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  emit('update:tile', { tile: v, overlap: Math.min(v - 1, tileConfig.value.overlap) })
}

function onTileOverlap(value: number): void {
  const v = Math.max(0, Math.trunc(Number(value)))
  if (!Number.isFinite(v)) return
  emit('update:tile', { tile: tileConfig.value.tile, overlap: Math.min(tileConfig.value.tile - 1, v) })
}
</script>

<!-- styles in styles/components/hires-settings-card.css -->
