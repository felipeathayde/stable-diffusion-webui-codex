<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational SUPIR mode configuration card for native SDXL img2img/inpaint.
Renders the single nested `supir` owner with truthful tranche-1 controls only (`enabled`, variant, sampler, control/restoration scales, and color-fix mode)
plus an optional blocking-reason note, without owning API calls, readiness state, or tab-store logic.

Symbols (top-level; keep in sync; no ghosts):
- `SupirModeCard` (component): Dedicated SUPIR mode UI card for truthful SDXL img2img/inpaint surfaces.
- `COLOR_FIX_OPTIONS` (constant): Select options for the public SUPIR color-fix surface.
-->

<template>
  <div class="gen-card refiner-card">
    <WanSubHeader title="SUPIR Mode">
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', supir.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="supir.enabled"
        :disabled="toggleDisabled && !supir.enabled"
        @click="emit('patch:supir', { enabled: !supir.enabled })"
      >
        {{ supir.enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </WanSubHeader>

    <p class="caption hr-hint">Native SDXL img2img/inpaint restoration controls.</p>
    <p v-if="blockingReason" class="caption hr-hint">{{ blockingReason }}</p>

    <div class="rf-grid">
      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Variant</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="supir.variant"
            @change="emit('patch:supir', { variant: ($event.target as HTMLSelectElement).value as SupirModeFormState['variant'] })"
          >
            <option v-if="variantChoices.length === 0" value="">No SUPIR variants reported</option>
            <option
              v-for="choice in variantChoices"
              :key="choice.value"
              :value="choice.value"
              :disabled="!choice.available"
            >
              {{ choice.available ? choice.label : `${choice.label} (missing)` }}
            </option>
          </select>
        </div>

        <div class="gc-col field">
          <label class="label-muted">Sampler</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="supir.sampler"
            @change="emit('patch:supir', { sampler: ($event.target as HTMLSelectElement).value })"
          >
            <option v-if="samplerChoices.length === 0" value="">No SUPIR samplers reported</option>
            <option v-for="choice in samplerChoices" :key="choice" :value="choice">
              {{ choice }}
            </option>
          </select>
        </div>
      </div>

      <div class="gc-row">
        <SliderField
          class="gc-col gc-col--wide"
          label="Control Scale"
          :modelValue="supir.controlScale"
          :min="0.01"
          :max="2"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:supir', { controlScale: value })"
        />

        <SliderField
          class="gc-col gc-col--wide"
          label="Restoration Scale"
          :modelValue="supir.restorationScale"
          :min="0.01"
          :max="6"
          :step="0.05"
          :inputStep="0.05"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('patch:supir', { restorationScale: value })"
        />
      </div>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Color fix</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="supir.colorFix"
            @change="emit('patch:supir', { colorFix: ($event.target as HTMLSelectElement).value as SupirModeFormState['colorFix'] })"
          >
            <option v-for="choice in COLOR_FIX_OPTIONS" :key="choice.value" :value="choice.value">
              {{ choice.label }}
            </option>
          </select>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { SupirModeFormState } from '../stores/model_tabs'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

const COLOR_FIX_OPTIONS = [
  { value: 'None', label: 'None' },
  { value: 'AdaIN', label: 'AdaIN' },
  { value: 'Wavelet', label: 'Wavelet' },
] as const

defineProps<{
  disabled?: boolean
  toggleDisabled?: boolean
  supir: SupirModeFormState
  variantChoices: Array<{ value: string; label: string; available: boolean }>
  samplerChoices: string[]
  blockingReason?: string
}>()

const emit = defineEmits<{
  (e: 'patch:supir', value: Partial<SupirModeFormState>): void
}>()
</script>
