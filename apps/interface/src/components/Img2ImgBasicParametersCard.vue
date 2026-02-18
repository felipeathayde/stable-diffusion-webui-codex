<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Img2img-focused Basic Parameters card with hires-like structure.
Renders sampler/scheduler/steps, dimensions, resize-mode + upscaler controls, and seed/CFG/denoise
for init-image mode without hires-only prompt/checkpoint swap controls, plus optional advanced CFG/APG controls
gated by per-engine capabilities.

Symbols (top-level; keep in sync; no ghosts):
- `Img2ImgBasicParametersCard` (component): Img2img parameters card used when init image mode is active.
- `hasGuidanceSupport` (function): Returns whether a specific advanced-guidance control is supported by the active engine capability contract.
- `clampFloat` (function): Clamps a numeric value to a `[min,max]` range.
- `clampInt` (function): Clamps and truncates a numeric value to an integer range.
- `clampIntToStep` (function): Clamps and snaps an integer value to a step size.
- `onSeedChange` (function): Handles manual seed input changes and emits a normalized integer seed.
- `onResizeModeChange` (function): Emits normalized resize-mode updates from the resize-type select.
- `onUpscalerChange` (function): Emits upscaler selection updates.
- `patchGuidanceAdvanced` (function): Emits partial updates for nested advanced-guidance state.
- `swapWH` (function): Swaps width/height while respecting min/max and step constraints.
-->

<template>
  <div class="gen-card img2img-basic-card">
    <WanSubHeader title="Basic Parameters" />
    <div class="gc-stack">
      <div class="gc-row">
        <SamplerSelector
          class="gc-col"
          :samplers="samplers"
          :modelValue="sampler"
          label="Sampler"
          :allow-empty="false"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:sampler', value)"
        />
        <SchedulerSelector
          class="gc-col"
          :schedulers="schedulers"
          :modelValue="scheduler"
          label="Scheduler"
          :allow-empty="false"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:scheduler', value)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Steps"
          :modelValue="steps"
          :min="minSteps"
          :max="maxSteps"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:steps', clampInt(value, minSteps, maxSteps))"
        />
      </div>

      <div class="gc-row">
        <SliderField
          class="gc-col"
          label="Width"
          :modelValue="width"
          :min="minWidth"
          :max="maxWidth"
          :step="widthStep"
          :inputStep="widthInputStep"
          :nudgeStep="widthInputStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:width', clampIntToStep(value, minWidth, maxWidth, widthInputStep))"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="width"
              :min="minWidth"
              :max="maxWidth"
              :step="widthInputStep"
              :nudgeStep="widthInputStep"
              inputClass="cdx-input-w-md"
              :disabled="disabled"
              @update:modelValue="(value) => emit('update:width', clampIntToStep(value, minWidth, maxWidth, widthInputStep))"
            />
            <button class="btn-swap" type="button" :disabled="disabled" title="Swap width/height" @click="swapWH">
              <span class="btn-swap-icon" aria-hidden="true">⇵</span>
            </button>
            <button
              v-if="showInitImageDims"
              class="btn-swap"
              type="button"
              :disabled="disabled"
              title="Use init image dimensions"
              @click="emit('sync-init-image-dims')"
            >
              <span aria-hidden="true">🖼</span>
            </button>
          </template>
        </SliderField>

        <SliderField
          class="gc-col"
          label="Height"
          :modelValue="height"
          :min="minHeight"
          :max="maxHeight"
          :step="heightStep"
          :inputStep="heightInputStep"
          :nudgeStep="heightInputStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:height', clampIntToStep(value, minHeight, maxHeight, heightInputStep))"
        />
      </div>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Resize type</label>
          <select class="select-md" :disabled="disabled" :value="resizeModeValue" @change="onResizeModeChange">
            <option v-for="option in resizeModeOptions" :key="option.value" :value="option.value">
              {{ option.label }}
            </option>
          </select>
        </div>

        <div class="gc-col field">
          <label class="label-muted">Upscaler</label>
          <select
            class="select-md"
            :value="upscaler"
            :disabled="disabled || !isUpscalerResizeMode || upscalersLoading"
            @change="onUpscalerChange"
          >
            <option v-if="upscalersLoading" :value="upscaler">Loading…</option>
            <option v-else-if="upscaler && !isUpscalerKnown" :value="upscaler">Invalid selection: {{ upscaler }}</option>
            <option v-else value="" disabled>Select</option>
            <optgroup v-if="spandrelUpscalers.length" label="Spandrel (pixel SR)">
              <option v-for="entry in spandrelUpscalers" :key="entry.id" :value="entry.id">{{ entry.label }}</option>
            </optgroup>
            <optgroup v-if="latentUpscalers.length" label="Latent">
              <option v-for="entry in latentUpscalers" :key="entry.id" :value="entry.id">{{ entry.label }}</option>
            </optgroup>
          </select>
          <p class="hr-hint" v-if="upscalersError">Error: {{ upscalersError }}</p>
          <p class="hr-hint" v-else-if="isUpscalerResizeMode && upscaler && !isUpscalerKnown">Select an upscaler id from `GET /api/upscalers`.</p>
        </div>
      </div>

      <div class="gc-row">
        <div class="gc-col field">
          <label class="label-muted">Seed</label>
          <div class="number-with-controls w-full">
            <input class="ui-input ui-input-sm pad-right" type="number" :disabled="disabled" :value="seed" @change="onSeedChange" />
            <div class="stepper">
              <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="emit('random-seed')">🎲</button>
              <button class="step-btn" type="button" :disabled="disabled" title="Reuse seed" @click="emit('reuse-seed')">↺</button>
            </div>
          </div>
        </div>

        <div v-if="showClipSkip" class="gc-col field">
          <label class="label-muted">CLIP Skip</label>
          <NumberStepperInput
            :modelValue="clipSkip"
            :min="minClipSkip"
            :max="maxClipSkip"
            :step="1"
            :nudgeStep="1"
            inputClass="cdx-input-w-xs"
            :disabled="disabled"
            @update:modelValue="(value) => emit('update:clipSkip', clampInt(value, minClipSkip, maxClipSkip))"
          />
        </div>

        <SliderField
          class="gc-col gc-col--wide"
          :label="cfgLabel"
          :modelValue="cfgScale"
          :min="minCfg"
          :max="maxCfg"
          :step="cfgStep"
          :inputStep="cfgStep"
          :nudgeStep="cfgStep"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:cfgScale', clampFloat(value, minCfg, maxCfg))"
        >
          <template #right>
            <NumberStepperInput
              :modelValue="cfgScale"
              :min="minCfg"
              :max="maxCfg"
              :step="cfgStep"
              :nudgeStep="cfgStep"
              inputClass="cdx-input-w-md"
              :disabled="disabled"
              @update:modelValue="(value) => emit('update:cfgScale', clampFloat(value, minCfg, maxCfg))"
            />
            <button
              v-if="showGuidanceAdvancedToggle"
              :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', guidanceAdvanced.enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
              type="button"
              :disabled="disabled"
              title="Show advanced guidance controls"
              @click="patchGuidanceAdvanced({ enabled: !guidanceAdvanced.enabled })"
            >
              Advanced
            </button>
          </template>
        </SliderField>

        <SliderField
          class="gc-col"
          label="Denoise"
          :modelValue="denoiseStrength"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:denoiseStrength', clampFloat(value, 0, 1))"
        />
      </div>

      <div v-if="showGuidanceAdvancedRow" class="gc-row cfg-advanced-row">
        <div v-if="hasGuidanceSupport('apg_enabled')" class="gc-col field gc-col--compact">
          <label class="label-muted">APG</label>
          <button
            :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', guidanceAdvanced.apgEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
            type="button"
            :disabled="disabled"
            @click="patchGuidanceAdvanced({ apgEnabled: !guidanceAdvanced.apgEnabled })"
          >
            {{ guidanceAdvanced.apgEnabled ? 'On' : 'Off' }}
          </button>
        </div>

        <div v-if="hasGuidanceSupport('cfg_trunc_ratio')" class="gc-col field gc-col--compact">
          <label class="label-muted">CFG Trunc</label>
          <button
            :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', guidanceAdvanced.cfgTruncEnabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
            type="button"
            :disabled="disabled"
            @click="patchGuidanceAdvanced({ cfgTruncEnabled: !guidanceAdvanced.cfgTruncEnabled })"
          >
            {{ guidanceAdvanced.cfgTruncEnabled ? 'On' : 'Off' }}
          </button>
        </div>

        <SliderField
          v-if="hasGuidanceSupport('apg_start_step')"
          class="gc-col"
          label="APG Start"
          :modelValue="guidanceAdvanced.apgStartStep"
          :min="0"
          :max="maxSteps"
          :step="1"
          :inputStep="1"
          :nudgeStep="1"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !guidanceAdvanced.apgEnabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgStartStep: clampInt(v, 0, maxSteps) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_eta')"
          class="gc-col"
          label="APG Eta"
          :modelValue="guidanceAdvanced.apgEta"
          :min="-1"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !guidanceAdvanced.apgEnabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgEta: clampFloat(v, -1, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_momentum')"
          class="gc-col"
          label="APG Momentum"
          :modelValue="guidanceAdvanced.apgMomentum"
          :min="0"
          :max="0.99"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !guidanceAdvanced.apgEnabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgMomentum: clampFloat(v, 0, 0.99) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_norm_threshold')"
          class="gc-col"
          label="APG Norm"
          :modelValue="guidanceAdvanced.apgNormThreshold"
          :min="0"
          :max="40"
          :step="0.1"
          :inputStep="0.1"
          :nudgeStep="0.1"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !guidanceAdvanced.apgEnabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgNormThreshold: clampFloat(v, 0, 40) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('apg_rescale')"
          class="gc-col"
          label="APG Rescale"
          :modelValue="guidanceAdvanced.apgRescale"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !guidanceAdvanced.apgEnabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ apgRescale: clampFloat(v, 0, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('guidance_rescale')"
          class="gc-col"
          label="Guidance Rescale"
          :modelValue="guidanceAdvanced.guidanceRescale"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ guidanceRescale: clampFloat(v, 0, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('cfg_trunc_ratio')"
          class="gc-col"
          label="CFG Trunc Ratio"
          :modelValue="guidanceAdvanced.cfgTruncRatio"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          :nudgeStep="0.01"
          inputClass="cdx-input-w-md"
          :disabled="disabled || !guidanceAdvanced.cfgTruncEnabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ cfgTruncRatio: clampFloat(v, 0, 1) })"
        />

        <SliderField
          v-if="hasGuidanceSupport('renorm_cfg')"
          class="gc-col"
          label="Renorm CFG"
          :modelValue="guidanceAdvanced.renormCfg"
          :min="0"
          :max="4"
          :step="0.05"
          :inputStep="0.05"
          :nudgeStep="0.05"
          inputClass="cdx-input-w-md"
          :disabled="disabled"
          @update:modelValue="(v) => patchGuidanceAdvanced({ renormCfg: clampFloat(v, 0, 4) })"
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { GuidanceAdvancedCapabilities, SamplerInfo, SchedulerInfo, UpscalerDefinition } from '../api/types'
import type { GuidanceAdvancedParams } from '../stores/model_tabs'

import NumberStepperInput from './ui/NumberStepperInput.vue'
import SliderField from './ui/SliderField.vue'
import SamplerSelector from './SamplerSelector.vue'
import SchedulerSelector from './SchedulerSelector.vue'
import WanSubHeader from './wan/WanSubHeader.vue'
import {
  IMG2IMG_RESIZE_MODE_OPTIONS,
  normalizeImg2ImgResizeMode,
  type Img2ImgResizeMode,
} from '../utils/img2img_resize'

const DEFAULT_GUIDANCE_ADVANCED: GuidanceAdvancedParams = {
  enabled: false,
  apgEnabled: false,
  apgStartStep: 0,
  apgEta: 0,
  apgMomentum: 0,
  apgNormThreshold: 15,
  apgRescale: 0,
  guidanceRescale: 0,
  cfgTruncEnabled: false,
  cfgTruncRatio: 0.8,
  renormCfg: 0,
}

const props = withDefaults(defineProps<{
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  upscalers?: UpscalerDefinition[]
  upscalersLoading?: boolean
  upscalersError?: string
  sampler: string
  scheduler: string
  steps: number
  cfgScale: number
  denoiseStrength: number
  seed: number
  width: number
  height: number
  upscaler: string
  resizeMode: Img2ImgResizeMode
  disabled?: boolean
  minSteps?: number
  maxSteps?: number
  minCfg?: number
  maxCfg?: number
  cfgStep?: number
  cfgLabel?: string
  minWidth?: number
  maxWidth?: number
  minHeight?: number
  maxHeight?: number
  widthStep?: number
  widthInputStep?: number
  heightStep?: number
  heightInputStep?: number
  showInitImageDims?: boolean
  showClipSkip?: boolean
  clipSkip?: number
  minClipSkip?: number
  maxClipSkip?: number
  guidanceAdvanced?: GuidanceAdvancedParams
  guidanceSupport?: GuidanceAdvancedCapabilities | null
}>(), {
  disabled: false,
  upscalers: () => [],
  upscalersLoading: false,
  upscalersError: '',
  minSteps: 1,
  maxSteps: 150,
  minCfg: 0,
  maxCfg: 30,
  cfgStep: 0.5,
  cfgLabel: 'CFG',
  minWidth: 64,
  maxWidth: 8192,
  minHeight: 64,
  maxHeight: 8192,
  widthStep: 64,
  widthInputStep: 8,
  heightStep: 64,
  heightInputStep: 8,
  showInitImageDims: false,
  showClipSkip: false,
  clipSkip: 0,
  minClipSkip: 0,
  maxClipSkip: 12,
  guidanceAdvanced: () => ({
    enabled: false,
    apgEnabled: false,
    apgStartStep: 0,
    apgEta: 0,
    apgMomentum: 0,
    apgNormThreshold: 15,
    apgRescale: 0,
    guidanceRescale: 0,
    cfgTruncEnabled: false,
    cfgTruncRatio: 0.8,
    renormCfg: 0,
  }),
  guidanceSupport: null,
})

const emit = defineEmits<{
  (e: 'update:sampler', value: string): void
  (e: 'update:scheduler', value: string): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfgScale', value: number): void
  (e: 'update:denoiseStrength', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'update:width', value: number): void
  (e: 'update:height', value: number): void
  (e: 'update:upscaler', value: string): void
  (e: 'update:resizeMode', value: Img2ImgResizeMode): void
  (e: 'update:clipSkip', value: number): void
  (e: 'update:guidanceAdvanced', patch: Partial<GuidanceAdvancedParams>): void
  (e: 'random-seed'): void
  (e: 'reuse-seed'): void
  (e: 'sync-init-image-dims'): void
}>()

const minSteps = computed(() => Number.isFinite(props.minSteps) ? Math.trunc(Number(props.minSteps)) : 1)
const maxSteps = computed(() => Number.isFinite(props.maxSteps) ? Math.trunc(Number(props.maxSteps)) : 150)
const minCfg = computed(() => Number.isFinite(props.minCfg) ? Number(props.minCfg) : 0)
const maxCfg = computed(() => Number.isFinite(props.maxCfg) ? Number(props.maxCfg) : 30)
const cfgStep = computed(() => Number.isFinite(props.cfgStep) ? Number(props.cfgStep) : 0.5)
const minWidth = computed(() => Number.isFinite(props.minWidth) ? Math.trunc(Number(props.minWidth)) : 64)
const maxWidth = computed(() => Number.isFinite(props.maxWidth) ? Math.trunc(Number(props.maxWidth)) : 8192)
const minHeight = computed(() => Number.isFinite(props.minHeight) ? Math.trunc(Number(props.minHeight)) : 64)
const maxHeight = computed(() => Number.isFinite(props.maxHeight) ? Math.trunc(Number(props.maxHeight)) : 8192)
const widthStep = computed(() => Number.isFinite(props.widthStep) ? Math.trunc(Number(props.widthStep)) : 64)
const widthInputStep = computed(() => Number.isFinite(props.widthInputStep) ? Math.trunc(Number(props.widthInputStep)) : 8)
const heightStep = computed(() => Number.isFinite(props.heightStep) ? Math.trunc(Number(props.heightStep)) : 64)
const heightInputStep = computed(() => Number.isFinite(props.heightInputStep) ? Math.trunc(Number(props.heightInputStep)) : 8)
const minClipSkip = computed(() => Number.isFinite(props.minClipSkip) ? Math.trunc(Number(props.minClipSkip)) : 0)
const maxClipSkip = computed(() => Number.isFinite(props.maxClipSkip) ? Math.trunc(Number(props.maxClipSkip)) : 12)
const showClipSkip = computed(() => props.showClipSkip === true)
const guidanceAdvanced = computed(() => props.guidanceAdvanced ?? DEFAULT_GUIDANCE_ADVANCED)
const guidanceSupport = computed(() => props.guidanceSupport ?? null)
const showGuidanceAdvancedToggle = computed(() => {
  const support = guidanceSupport.value
  if (!support) return false
  return Object.values(support).some((flag) => flag === true)
})
const showGuidanceAdvancedRow = computed(() => showGuidanceAdvancedToggle.value && guidanceAdvanced.value.enabled)

const resizeModeOptions = IMG2IMG_RESIZE_MODE_OPTIONS
const resizeModeValue = computed<Img2ImgResizeMode>(() => normalizeImg2ImgResizeMode(props.resizeMode))
const isUpscalerResizeMode = computed(() => resizeModeValue.value === 'upscaler')

const upscalers = computed(() => Array.isArray(props.upscalers) ? props.upscalers : [])
const spandrelUpscalers = computed(() => upscalers.value.filter((entry) => entry.kind === 'spandrel'))
const latentUpscalers = computed(() => upscalers.value.filter((entry) => entry.kind === 'latent'))
const isUpscalerKnown = computed(() => upscalers.value.some((entry) => entry.id === props.upscaler))

function clampFloat(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(max, Math.max(min, value))
}

function clampInt(value: number, min: number, max: number): number {
  const numberValue = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, numberValue))
}

function clampIntToStep(value: number, min: number, max: number, step: number): number {
  const clamped = clampInt(value, min, max)
  if (!Number.isFinite(step) || step <= 0) return clamped
  const snapped = Math.round(clamped / step) * step
  return Math.min(max, Math.max(min, snapped))
}

function onSeedChange(event: Event): void {
  const raw = Number((event.target as HTMLInputElement).value)
  if (!Number.isFinite(raw)) return
  emit('update:seed', Math.trunc(raw))
}

function onResizeModeChange(event: Event): void {
  const value = (event.target as HTMLSelectElement).value
  emit('update:resizeMode', normalizeImg2ImgResizeMode(value))
}

function onUpscalerChange(event: Event): void {
  emit('update:upscaler', (event.target as HTMLSelectElement).value)
}

function hasGuidanceSupport(control: keyof GuidanceAdvancedCapabilities): boolean {
  return Boolean(guidanceSupport.value?.[control])
}

function patchGuidanceAdvanced(patch: Partial<GuidanceAdvancedParams>): void {
  emit('update:guidanceAdvanced', patch)
}

function swapWH(): void {
  emit('update:width', clampIntToStep(props.height, minWidth.value, maxWidth.value, widthInputStep.value))
  emit('update:height', clampIntToStep(props.width, minHeight.value, maxHeight.value, heightInputStep.value))
}
</script>

<!-- styles in styles/components/img2img-basic-parameters-card.css -->
