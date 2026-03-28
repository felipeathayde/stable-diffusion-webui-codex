<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational IP-Adapter configuration card for supported image tabs.
Provides a dedicated card with enable toggle, asset selectors, source-mode switching (`DIR|IMG`), same-as-init shortcut, optional reference-image dropzone, folder-source controls, and weight range sliders without owning store or API state.

Symbols (top-level; keep in sync; no ghosts):
- `IpAdapterCard` (component): Dedicated IP-Adapter UI card for supported image tabs.
- `SOURCE_MODE_OPTIONS` (constant): Segmented-control options for `DIR|IMG` source selection.
-->

<template>
  <div class="gen-card ip-adapter-card">
    <WanSubHeader title="IP-Adapter">
      <button
        :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', enabled ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
        type="button"
        :aria-pressed="enabled"
        @click="emit('update:enabled', !enabled)"
      >
        {{ enabled ? 'Enabled' : 'Disabled' }}
      </button>
    </WanSubHeader>

    <div v-if="enabled" class="ip-adapter-card__body">
      <div class="ip-adapter-card__top-row">
        <div class="field">
          <label class="label-muted">Source</label>
          <CompactSegmentedControl
            :modelValue="sourceMode"
            :options="SOURCE_MODE_OPTIONS"
            :disabled="disabled"
            ariaLabel="IP-Adapter source mode"
            @update:modelValue="(value) => emit('update:sourceMode', value as 'dir' | 'img')"
          />
        </div>

        <div v-if="img2imgMode" class="field ip-adapter-card__same-init-field">
          <label class="label-muted">Shortcut</label>
          <button
            :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', sameAsInit ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
            type="button"
            :aria-pressed="sameAsInit"
            :disabled="disabled || sourceMode !== 'img'"
            @click="emit('update:sameAsInit', !sameAsInit)"
          >
            Same as init image
          </button>
        </div>
      </div>

      <div class="gc-row ip-adapter-card__selectors">
        <div class="field gc-col gc-col--wide">
          <label class="label-muted">Adapter model</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="model"
            @change="emit('update:model', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">Select IP-Adapter model</option>
            <option v-for="choice in modelChoices" :key="choice.value" :value="choice.value">
              {{ choice.label }}
            </option>
          </select>
        </div>

        <div class="field gc-col gc-col--wide">
          <label class="label-muted">Image encoder</label>
          <select
            class="select-md"
            :disabled="disabled"
            :value="imageEncoder"
            @change="emit('update:imageEncoder', ($event.target as HTMLSelectElement).value)"
          >
            <option value="">Select image encoder</option>
            <option v-for="choice in imageEncoderChoices" :key="choice.value" :value="choice.value">
              {{ choice.label }}
            </option>
          </select>
        </div>
      </div>

      <div class="gc-row ip-adapter-card__sliders">
        <SliderField
          class="gc-col gc-col--wide"
          label="Weight"
          :modelValue="weight"
          :min="0"
          :max="2"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:weight', value)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="Start"
          :modelValue="startAt"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:startAt', value)"
        />
        <SliderField
          class="gc-col gc-col--wide"
          label="End"
          :modelValue="endAt"
          :min="0"
          :max="1"
          :step="0.01"
          :inputStep="0.01"
          inputClass="cdx-input-w-xs"
          :disabled="disabled"
          @update:modelValue="(value) => emit('update:endAt', value)"
        />
      </div>

      <p v-if="sourceMode === 'img' && sameAsInit" class="caption ip-adapter-card__hint">
        Uses the current init image as the IP-Adapter reference image.
      </p>

      <InitialImageCard
        v-else-if="sourceMode === 'img'"
        label="Reference Image"
        :src="referenceImageData"
        :has-image="Boolean(referenceImageData)"
        :disabled="disabled"
        :dropzone="true"
        :thumbnail="true"
        :zoomable="true"
        @set="(file) => emit('set:referenceImage', file)"
        @clear="() => emit('clear:referenceImage')"
        @rejected="(payload) => emit('reject:referenceImage', payload)"
      />

      <ImageFolderSourceFields
        v-else
        :folderPath="folderPath"
        :selectionMode="selectionMode"
        :count="count"
        :order="order"
        :sortBy="sortBy"
        :disabled="disabled"
        pathLabel="Folder path"
        pathPlaceholder="input/ip-adapter-source"
        countLabel="Reference images"
        @update:folderPath="(value) => emit('update:folderPath', value)"
        @update:selectionMode="(value) => emit('update:selectionMode', value)"
        @update:count="(value) => emit('update:count', value)"
        @update:order="(value) => emit('update:order', value)"
        @update:sortBy="(value) => emit('update:sortBy', value)"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import InitialImageCard from './InitialImageCard.vue'
import ImageFolderSourceFields from './ImageFolderSourceFields.vue'
import CompactSegmentedControl from './ui/CompactSegmentedControl.vue'
import SliderField from './ui/SliderField.vue'
import WanSubHeader from './wan/WanSubHeader.vue'

type SelectChoice = {
  value: string
  label: string
}

const SOURCE_MODE_OPTIONS = [
  { value: 'dir', label: 'DIR' },
  { value: 'img', label: 'IMG' },
] as const

withDefaults(defineProps<{
  disabled?: boolean
  enabled: boolean
  img2imgMode?: boolean
  model: string
  imageEncoder: string
  modelChoices: SelectChoice[]
  imageEncoderChoices: SelectChoice[]
  sourceMode: 'img' | 'dir'
  sameAsInit: boolean
  referenceImageData: string
  folderPath: string
  selectionMode: 'all' | 'count'
  count: number
  order: 'random' | 'sorted'
  sortBy: 'name' | 'size' | 'created_at' | 'modified_at'
  weight: number
  startAt: number
  endAt: number
}>(), {
  disabled: false,
  img2imgMode: false,
})

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:model', value: string): void
  (e: 'update:imageEncoder', value: string): void
  (e: 'update:sourceMode', value: 'img' | 'dir'): void
  (e: 'update:sameAsInit', value: boolean): void
  (e: 'update:folderPath', value: string): void
  (e: 'update:selectionMode', value: 'all' | 'count'): void
  (e: 'update:count', value: number): void
  (e: 'update:order', value: 'random' | 'sorted'): void
  (e: 'update:sortBy', value: 'name' | 'size' | 'created_at' | 'modified_at'): void
  (e: 'update:weight', value: number): void
  (e: 'update:startAt', value: number): void
  (e: 'update:endAt', value: number): void
  (e: 'set:referenceImage', value: File): void
  (e: 'clear:referenceImage'): void
  (e: 'reject:referenceImage', payload: { reason: string; files: File[] }): void
}>()
</script>
