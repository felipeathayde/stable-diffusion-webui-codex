<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Reusable hover/focus tooltip primitive for compact UI hints.
Wraps trigger content and renders a floating tooltip panel with a title and multi-line body, shown on hover or keyboard focus.

Symbols (top-level; keep in sync; no ghosts):
- `HoverTooltip` (component): Wrapper component that displays a floating tooltip for its slot trigger.
- `tooltipLines` (computed): Normalized tooltip body lines (trimmed, non-empty).
-->

<template>
  <span class="cdx-hover-tooltip" tabindex="0">
    <slot />
    <span class="cdx-hover-tooltip__panel" role="tooltip">
      <span v-if="title" class="cdx-hover-tooltip__title">{{ title }}</span>
      <span
        v-for="(line, index) in tooltipLines"
        :key="`${line}-${index}`"
        class="cdx-hover-tooltip__line"
      >
        {{ line }}
      </span>
    </span>
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  title?: string
  content: string | readonly string[]
}>(), {
  title: '',
})

const tooltipLines = computed<string[]>(() => {
  if (typeof props.content === 'string') {
    return props.content
      .split('\n')
      .map((line: string) => line.trim())
      .filter((line: string) => line.length > 0)
  }
  return Array.from(props.content)
    .map((line: string) => line.trim())
    .filter((line: string) => line.length > 0)
})
</script>
