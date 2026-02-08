<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Swap-model configuration card (first pass or hires second pass).
Renders a compact enable switch and optional fields (checkpoint swap step + CFG/seed), emitting updates to parent views.

Symbols (top-level; keep in sync; no ghosts):
- `RefinerSettingsCard` (component): Swap-model settings panel component.
- `toggle` (function): Toggles `enabled` via `update:enabled`.
-->

<template>
  <div :class="['gen-card', 'refiner-card', { 'refiner-card--dense': dense } ]">
    <div class="row-split">
      <span class="label-muted">{{ label }}</span>
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="enabled"
        @click="toggle"
      >
        {{ enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </div>
    <div v-if="enabled" class="rf-grid">
      <div class="field rf-field--full">
        <label class="label-muted">Checkpoint Swap</label>
        <select class="select-md" :value="model" @change="onModelChange">
          <option value="">Keep current model</option>
          <option v-if="showCurrentModelOption" :value="model">{{ model }}</option>
          <option v-for="choice in normalizedModelChoices" :key="choice" :value="choice">{{ choice }}</option>
        </select>
      </div>
      <div class="field">
        <label class="label-muted">Swap At Step</label>
        <input class="ui-input ui-input-sm" type="number" min="1" :value="normalizedSwapAtStep" @change="onSwapAtStepChange" />
      </div>
      <div class="field">
        <label class="label-muted">CFG</label>
        <input class="ui-input ui-input-sm" type="number" step="0.1" :value="cfg" @change="onCfgChange" />
      </div>
      <div class="field">
        <label class="label-muted">Seed</label>
        <input class="ui-input ui-input-sm" type="number" :value="seed" @change="onSeedChange" />
        <p class="rf-hint">Use -1 for random</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: refiner, settings, grid
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  enabled: boolean
  swapAtStep: number
  cfg: number
  seed: number
  model?: string
  modelChoices?: string[]
  label?: string
  dense?: boolean
}>(), {
  label: 'Swap Model',
  dense: false,
})

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:swapAtStep', value: number): void
  (e: 'update:cfg', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'update:model', value: string): void
}>()

const normalizedModelChoices = computed(() => {
  const seen = new Set<string>()
  const out: string[] = []
  for (const raw of props.modelChoices || []) {
    const value = String(raw || '').trim()
    if (!value || seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
})

const showCurrentModelOption = computed(() => {
  const current = String(props.model || '').trim()
  if (!current) return false
  return !normalizedModelChoices.value.includes(current)
})

function toggle(): void {
  emit('update:enabled', !props.enabled)
}

const normalizedSwapAtStep = computed(() => {
  const v = Number(props.swapAtStep)
  if (!Number.isFinite(v) || v < 1) return 1
  return Math.trunc(v)
})

function onSwapAtStepChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:swapAtStep', Number.isNaN(v) || v < 1 ? 1 : Math.trunc(v))
}

function onCfgChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:cfg', Number.isNaN(v) ? props.cfg : v)
}

function onSeedChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:seed', Number.isNaN(v) ? props.seed : v)
}

function onModelChange(event: Event): void {
  emit('update:model', (event.target as HTMLSelectElement).value)
}
</script>

<!-- styles in styles/components/refiner-settings-card.css -->
