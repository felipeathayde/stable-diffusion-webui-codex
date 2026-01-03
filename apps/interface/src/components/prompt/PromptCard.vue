<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt panel wrapper with toolbars and modals.
Renders `PromptFields` and an optional toolbar for Negative prompt toggling, asset insertion (LoRA/TI), and styles creation/application.

Symbols (top-level; keep in sync; no ghosts):
- `PromptCard` (component): Prompt panel with prompt/negative fields, optional assets/styles controls, and insertion modals.
-->

<template>
  <div class="panel">
    <div class="panel-header">{{ title }}
      <div class="toolbar prompt-toolbar">
        <template v-if="enableAssets">
          <button class="btn btn-sm btn-secondary" type="button" @click="showTI = true">Textual Inversion</button>
          <button class="btn btn-sm btn-secondary" type="button" @click="showLora = true">LoRA</button>
        </template>

        <button
          v-if="supportsNegative && allowNegativeToggle"
          :class="['btn', 'btn-sm', showNegative ? 'btn-secondary' : 'btn-outline']"
          type="button"
          :title="showNegative ? 'Hide negative prompt' : 'Show negative prompt'"
          @click="toggleNegative"
        >
          Negative
        </button>

        <template v-if="enableStyles">
          <label class="label-muted styles-label">{{ stylesLabel }}</label>
          <div class="cdx-input-with-actions">
            <input class="ui-input styles-input" :list="styleListId" v-model="styleName" placeholder="Filter styles" />
            <datalist :id="styleListId">
              <option v-for="s in styleNames" :key="s" :value="s" />
            </datalist>
            <div class="cdx-input-actions">
              <button class="btn btn-sm btn-secondary" type="button" @click="showStyle = true">New</button>
              <button class="btn btn-sm btn-outline" type="button" @click="applyStyle(styleName)">Apply</button>
            </div>
          </div>
        </template>

        <template v-else-if="toolbarLabel">
          <label class="label-muted styles-label">{{ toolbarLabel }}</label>
        </template>
      </div>
    </div>

    <div class="panel-body">
      <div v-if="fieldsId" :id="fieldsId">
        <PromptFields v-model:prompt="innerPrompt" v-model:negative="innerNegative" :hide-negative="hideNegative" />
      </div>
      <PromptFields v-else v-model:prompt="innerPrompt" v-model:negative="innerNegative" :hide-negative="hideNegative" />

      <slot />
    </div>

    <LoraModal v-if="enableAssets" v-model="showLora" @insert="onInsertToken" />
    <TextualInversionModal v-if="enableAssets" v-model="showTI" @insert="onInsertToken" />
    <StyleEditorModal v-if="enableStyles" v-model="showStyle" @saved="onStyleSaved" />
  </div>
</template>

<script setup lang="ts">
import { computed, getCurrentInstance } from 'vue'

import { usePromptCard } from '../../composables/usePromptCard'
import LoraModal from '../modals/LoraModal.vue'
import StyleEditorModal from '../modals/StyleEditorModal.vue'
import TextualInversionModal from '../modals/TextualInversionModal.vue'
import PromptFields from './PromptFields.vue'

const props = withDefaults(defineProps<{
  prompt: string
  negative: string
  title?: string
  enableAssets?: boolean
  enableStyles?: boolean
  stylesLabel?: string
  toolbarLabel?: string
  defaultShowNegative?: boolean
  allowNegativeToggle?: boolean
  supportsNegative?: boolean
  fieldsId?: string
}>(), {
  title: 'Prompt',
  enableAssets: true,
  enableStyles: true,
  stylesLabel: 'Styles',
  toolbarLabel: '',
  defaultShowNegative: false,
  allowNegativeToggle: true,
  supportsNegative: true,
  fieldsId: '',
})

const emit = defineEmits<{
  (e: 'update:prompt', value: string): void
  (e: 'update:negative', value: string): void
}>()

const innerPrompt = computed({
  get: () => props.prompt,
  set: (value: string) => emit('update:prompt', value),
})

const innerNegative = computed({
  get: () => props.negative,
  set: (value: string) => emit('update:negative', value),
})

const {
  showNegative,
  hideNegative,
  toggleNegative,
  showLora,
  showTI,
  showStyle,
  styleName,
  styleNames,
  applyStyle,
  onInsertToken,
  onStyleSaved,
} = usePromptCard({
  prompt: innerPrompt,
  negative: innerNegative,
  defaultShowNegative: props.defaultShowNegative,
  supportsNegative: props.supportsNegative,
})

const instance = getCurrentInstance()
const styleListId = `style-list-${instance?.uid ?? Math.floor(Math.random() * 1_000_000_000)}`
</script>
