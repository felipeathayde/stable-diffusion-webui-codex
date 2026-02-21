<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model-tab header controls (rename/enable/actions).
Provides a small header panel used by model tabs to rename, enable/disable, and run tab actions (duplicate/remove/load/unload/generate).

Symbols (top-level; keep in sync; no ghosts):
- `BaseTabHeader` (component): Header/actions control panel for a model tab.
- `onTitleInput` (function): Emits the updated title value (`rename`).
- `onToggle` (function): Emits the updated enabled state (`set-enabled`).
-->

<template>
  <div class="panel">
    <div class="panel-header">
      <input
        class="ui-input"
        type="text"
        :value="title"
        @input="onTitleInput"
        placeholder="Tab title"
        aria-label="Tab title"
      />
      <div class="panel-actions">
        <button class="btn btn-sm" type="button" @click="$emit('duplicate')">Duplicate</button>
        <button class="btn btn-sm btn-destructive" type="button" @click="$emit('remove')">Remove</button>
        <label class="switch-label">
          <input type="checkbox" :checked="enabled" @change="onToggle" />
          <span>Enabled</span>
        </label>
      </div>
    </div>
    <div class="panel-body">
      <div class="base-tab-header-actions-grid">
        <div class="base-tab-header-actions-left">
          <button class="btn btn-secondary" type="button" @click="$emit('load')">Load</button>
          <button class="btn base-tab-header-unload" type="button" @click="$emit('unload')">Unload</button>
        </div>
        <div class="base-tab-header-actions-center">
          <button class="btn" type="button" @click="$emit('send-to-workflows')">Send to Workflows</button>
        </div>
        <div class="base-tab-header-actions-right">
          <button class="btn btn-primary" type="button" @click="$emit('generate')">Generate</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ title: string; enabled: boolean }>()
const emit = defineEmits(['rename', 'duplicate', 'remove', 'set-enabled', 'load', 'unload', 'generate', 'send-to-workflows'])

function onTitleInput(e: Event): void {
  emit('rename', (e.target as HTMLInputElement).value)
}

function onToggle(e: Event): void {
  emit('set-enabled', (e.target as HTMLInputElement).checked)
}
</script>
