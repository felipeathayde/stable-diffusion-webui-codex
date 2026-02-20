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
- `onHeaderClick` (function): Emits `header-click` for clickable headers when the click target is non-interactive.
- `onHeaderKeydown` (function): Adds Enter/Space keyboard parity for clickable headers.
-->

<template>
  <div
    class="wan-subheader"
    :class="{ 'wan-subheader--clickable': clickable, 'wan-subheader--disabled': disabled }"
    :role="clickable ? 'button' : undefined"
    :tabindex="clickable && !disabled ? 0 : undefined"
    :aria-disabled="clickable ? disabled : undefined"
    :aria-pressed="clickable && ariaPressed !== null ? ariaPressed : undefined"
    :aria-expanded="clickable && ariaExpanded !== null ? ariaExpanded : undefined"
    @click="onHeaderClick"
    @keydown="onHeaderKeydown"
  >
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
const props = withDefaults(defineProps<{
  title: string
  clickable?: boolean
  disabled?: boolean
  ariaPressed?: boolean | null
  ariaExpanded?: boolean | null
}>(), {
  clickable: false,
  disabled: false,
  ariaPressed: null,
  ariaExpanded: null,
})

const emit = defineEmits<{
  (e: 'header-click'): void
}>()

const INTERACTIVE_HEADER_TARGET_SELECTOR = [
  'button',
  'a',
  'input',
  'select',
  'textarea',
  'summary',
  'label',
  '[role="button"]',
  '[data-wan-subheader-interactive]',
].join(', ')

function isInteractiveHeaderTarget(target: EventTarget | null, currentTarget: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false
  const interactiveTarget = target.closest(INTERACTIVE_HEADER_TARGET_SELECTOR)
  if (!interactiveTarget) return false
  if (currentTarget instanceof Element && interactiveTarget === currentTarget) return false
  return true
}

function onHeaderClick(event: MouseEvent): void {
  if (!props.clickable || props.disabled) return
  if (isInteractiveHeaderTarget(event.target, event.currentTarget)) return
  emit('header-click')
}

function onHeaderKeydown(event: KeyboardEvent): void {
  if (!props.clickable || props.disabled) return
  if (event.key !== 'Enter' && event.key !== ' ') return
  if (isInteractiveHeaderTarget(event.target, event.currentTarget)) return
  event.preventDefault()
  emit('header-click')
}
</script>
