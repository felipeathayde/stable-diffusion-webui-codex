<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Compact “run summary” chips renderer.
Splits a summary string by a separator (default `·`) and renders each trimmed chunk as a chip for scannable run metadata.

Symbols (top-level; keep in sync; no ghosts):
- `RunSummaryChips` (component): Renders a summary string as a row of chips.
- `chips` (const): Computed chip list derived from `props.text` and `props.separator`.
-->

<template>
  <div class="caption cdx-chips-row">
    <span v-for="(chip, i) in chips" :key="i" class="chip">{{ chip }}</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  text: string
  separator?: string
}>(), {
  separator: '·',
})

const chips = computed(() => {
  return String(props.text ?? '')
    .split(props.separator)
    .map((s) => s.trim())
    .filter(Boolean)
})
</script>
