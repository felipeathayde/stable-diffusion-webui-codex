<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt style editor modal.
Creates a named style (prompt + negative) via the styles store and emits `saved` when the style is persisted.

Symbols (top-level; keep in sync; no ghosts):
- `StyleEditorModal` (component): Modal for creating/editing prompt styles.
- `onSave` (function): Persists the style via the store and closes the modal.
-->

<template>
  <div v-if="modelValue" class="modal">
    <div class="modal-panel">
      <div class="modal-header">
        <h3>Create Style</h3>
        <button class="btn btn-sm btn-ghost" @click="$emit('update:modelValue', false)">✕</button>
      </div>
      <div class="modal-body">
        <div class="space-y-3">
          <div>
            <label class="label-muted">Name</label>
            <input class="ui-input" v-model="name" placeholder="My style" />
          </div>
          <div>
            <label class="label-muted">Prompt</label>
            <textarea class="ui-textarea" rows="4" v-model="prompt" />
          </div>
          <div>
            <label class="label-muted">Negative</label>
            <textarea class="ui-textarea" rows="3" v-model="negative" />
          </div>
        </div>
      </div>
      <div class="modal-footer">
        <button class="btn btn-sm btn-outline" @click="$emit('update:modelValue', false)">Cancel</button>
        <button class="btn btn-sm btn-primary" :disabled="!name.trim()" @click="onSave">Save</button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { useStylesStore } from '../../stores/styles'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits(['update:modelValue','saved'])
const styles = useStylesStore()

const name = ref('')
const prompt = ref('')
const negative = ref('')

watch(() => props.modelValue, (v) => {
  if (v) { name.value=''; prompt.value=''; negative.value='' }
})

function onSave(): void {
  styles.create({ name: name.value.trim(), prompt: prompt.value, negative: negative.value })
  emit('saved', name.value.trim())
  emit('update:modelValue', false)
}
</script>

<!-- modal styles are provided globally in styles.css (.modal-backdrop/.modal-panel) -->
