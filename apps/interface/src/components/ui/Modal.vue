<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Generic modal shell component.
Teleports to `body`, provides a backdrop click-to-close behavior, and exposes header/body/footer slots for reusable modal dialogs.
Supports optional footer rendering and per-modal panel class overrides.

Symbols (top-level; keep in sync; no ghosts):
- `Modal` (component): Reusable modal shell used across the UI.
- `close` (function): Closes the modal by emitting `update:modelValue=false`.
-->

<template>
  <Teleport to="body">
    <div v-if="modelValue" class="modal-backdrop" @click.self="close">
      <div class="modal-panel" :class="panelClass" role="dialog" aria-modal="true">
        <header class="modal-header">
          <h3 class="h3">{{ title }}</h3>
          <button class="btn-icon" type="button" @click="close" aria-label="Close">✕</button>
        </header>
        <div class="modal-body">
          <slot />
        </div>
        <footer v-if="showFooter" class="modal-footer">
          <slot name="footer">
            <button class="btn btn-md btn-outline" type="button" @click="close">Close</button>
          </slot>
        </footer>
      </div>
    </div>
  </Teleport>
  
</template>

<script setup lang="ts">
withDefaults(
  defineProps<{
    title: string
    modelValue: boolean
    panelClass?: string
    showFooter?: boolean
  }>(),
  {
    panelClass: '',
    showFooter: true,
  },
)
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void }>()

function close(): void {
  emit('update:modelValue', false)
}
</script>
