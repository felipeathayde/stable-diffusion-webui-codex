<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WAN section sub-header component.
Provides a small, consistent header row for WAN sections (e.g., Video/High/Low), with optional subtitle and right-side slot content.

Symbols (top-level; keep in sync; no ghosts):
- `WanSubHeader` (component): WAN section sub-header with slots for subtitle and right content.
-->

<template>
  <div class="wan-subheader">
    <div class="wan-subheader-left">
      <span class="wan-subheader-title">{{ title }}</span>
      <slot name="subtitle" />
    </div>
    <div v-if="$slots.default" class="wan-subheader-right">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>
