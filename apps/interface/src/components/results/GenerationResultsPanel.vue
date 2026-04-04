<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared generation results owner for image and video model tabs.
Provides the common Results card shell plus optional History, media-preview, and Generation Info sections, while callers supply family-specific actions and viewer content through explicit slots.

Symbols (top-level; keep in sync; no ghosts):
- `GenerationResultsPanel` (component): Shared results owner that wraps `ResultsCard` and renders the canonical section order for generation tabs.
-->

<template>
  <ResultsCard
    headerClass="three-cols"
    :headerRightClass="props.headerRightClass"
    :bodyClass="props.bodyClass"
    :showGenerate="false"
  >
    <template #header-right>
      <slot name="header-right" />
    </template>

    <div v-if="props.showHistory" class="gen-card">
      <div class="row-split">
        <span class="label-muted">{{ props.historyTitle }}</span>
        <div v-if="$slots['history-actions']" class="results-header-actions">
          <slot name="history-actions" />
        </div>
      </div>
      <div class="mt-2">
        <slot name="history">
          <div class="caption">{{ props.historyEmptyText }}</div>
        </slot>
      </div>
    </div>

    <div v-if="props.showMedia" class="gen-card">
      <div class="row-split">
        <span class="label-muted">{{ props.mediaTitle }}</span>
        <div v-if="$slots['media-actions']" class="results-header-actions">
          <slot name="media-actions" />
        </div>
      </div>
      <div class="mt-2">
        <slot name="media" />
      </div>
    </div>

    <slot name="viewer" />

    <div v-if="props.showInfo" class="gen-card">
      <div class="row-split">
        <span class="label-muted">{{ props.infoTitle }}</span>
        <div v-if="$slots['info-actions']" class="results-header-actions">
          <slot name="info-actions" />
        </div>
      </div>
      <div class="mt-2">
        <slot name="info" />
      </div>
    </div>

    <slot name="after" />
  </ResultsCard>
</template>

<script setup lang="ts">
import ResultsCard from './ResultsCard.vue'

const props = withDefaults(defineProps<{
  headerRightClass?: string
  bodyClass?: string
  showHistory?: boolean
  historyTitle?: string
  historyEmptyText?: string
  showMedia?: boolean
  mediaTitle?: string
  showInfo?: boolean
  infoTitle?: string
}>(), {
  headerRightClass: 'results-header-actions',
  bodyClass: '',
  showHistory: false,
  historyTitle: 'History',
  historyEmptyText: 'No runs yet.',
  showMedia: false,
  mediaTitle: 'Exported Video',
  showInfo: false,
  infoTitle: 'Generation Info',
})
</script>
