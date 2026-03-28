<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Presentational folder-source controls shared by image automation surfaces.
Owns folder path input, `all|count` amount selection, `random|sorted` ordering, sort-key selection, and an optional crop toggle while delegating state to the parent.

Symbols (top-level; keep in sync; no ghosts):
- `ImageFolderSourceFields` (component): Shared folder-source configuration block.
- `onFolderPathInput` (function): Emits the folder path string as the user types.
-->

<template>
  <div class="cdx-image-source-fields">
    <div class="field">
      <label class="label-muted">{{ pathLabel }}</label>
      <input
        class="ui-input"
        type="text"
        :disabled="disabled"
        :value="folderPath"
        :placeholder="pathPlaceholder"
        @input="onFolderPathInput"
      />
    </div>

    <div class="cdx-image-source-fields__row">
      <div class="field">
        <label class="label-muted">Selection</label>
        <CompactSegmentedControl
          :modelValue="selectionMode"
          :options="selectionOptions"
          :disabled="disabled"
          ariaLabel="Folder selection mode"
          @update:modelValue="(value) => emit('update:selectionMode', value as 'all' | 'count')"
        />
      </div>

      <div class="field">
        <label class="label-muted">Order</label>
        <CompactSegmentedControl
          :modelValue="order"
          :options="orderOptions"
          :disabled="disabled"
          ariaLabel="Folder order mode"
          @update:modelValue="(value) => emit('update:order', value as 'random' | 'sorted')"
        />
      </div>

      <div v-if="showUseCrop" class="field cdx-image-source-fields__toggle-field">
        <label class="label-muted">Crop</label>
        <button
          :class="['btn', 'qs-toggle-btn', 'qs-toggle-btn--sm', useCrop ? 'qs-toggle-btn--on' : 'qs-toggle-btn--off']"
          type="button"
          :aria-pressed="useCrop"
          :disabled="disabled"
          @click="emit('toggle:useCrop')"
        >
          Use crop
        </button>
      </div>
    </div>

    <div class="cdx-image-source-fields__row">
      <div v-if="selectionMode === 'count'" class="field cdx-image-source-fields__count-field">
        <label class="label-muted">{{ countLabel }}</label>
        <NumberStepperInput
          :modelValue="count"
          :disabled="disabled"
          :min="1"
          :step="1"
          :nudgeStep="1"
          size="sm"
          inputClass="cdx-input-w-xs"
          @update:modelValue="(value) => emit('update:count', value)"
        />
      </div>

      <div v-if="order === 'sorted'" class="field cdx-image-source-fields__sort-field">
        <label class="label-muted">Sort by</label>
        <select
          class="select-md"
          :disabled="disabled"
          :value="sortBy"
          @change="emit('update:sortBy', ($event.target as HTMLSelectElement).value as 'name' | 'size' | 'created_at' | 'modified_at')"
        >
          <option value="name">Name</option>
          <option value="size">Size</option>
          <option value="created_at">Created</option>
          <option value="modified_at">Modified</option>
        </select>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import CompactSegmentedControl from './ui/CompactSegmentedControl.vue'
import NumberStepperInput from './ui/NumberStepperInput.vue'

const props = withDefaults(defineProps<{
  folderPath: string
  selectionMode: 'all' | 'count'
  count: number
  order: 'random' | 'sorted'
  sortBy: 'name' | 'size' | 'created_at' | 'modified_at'
  useCrop?: boolean
  showUseCrop?: boolean
  disabled?: boolean
  pathLabel?: string
  pathPlaceholder?: string
  countLabel?: string
}>(), {
  useCrop: false,
  showUseCrop: false,
  disabled: false,
  pathLabel: 'Folder path',
  pathPlaceholder: 'input/img2img-source',
  countLabel: 'Images to generate',
})

const emit = defineEmits<{
  (e: 'update:folderPath', value: string): void
  (e: 'update:selectionMode', value: 'all' | 'count'): void
  (e: 'update:count', value: number): void
  (e: 'update:order', value: 'random' | 'sorted'): void
  (e: 'update:sortBy', value: 'name' | 'size' | 'created_at' | 'modified_at'): void
  (e: 'toggle:useCrop'): void
}>()

const selectionOptions = computed(() => [
  { value: 'all', label: 'All' },
  { value: 'count', label: 'Count' },
])

const orderOptions = computed(() => [
  { value: 'random', label: 'Random' },
  { value: 'sorted', label: 'Order' },
])

function onFolderPathInput(event: Event): void {
  emit('update:folderPath', (event.target as HTMLInputElement).value)
}
</script>
