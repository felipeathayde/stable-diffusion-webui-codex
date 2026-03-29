<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Checkpoint picker modal (QuickSettings models list).
Provides a searchable checkpoint list sourced from the quicksettings store and applies the selected checkpoint before closing; checkpoint-scoped
metadata/default derivation stays in stores/composables instead of being reimplemented in this modal.

Symbols (top-level; keep in sync; no ghosts):
- `CheckpointModal` (component): Modal for selecting a checkpoint from the known models list.
- `filtered` (const): Filtered model list based on the current search query.
- `apply` (function): Applies the selected model via the quicksettings store and closes the modal.
-->

<template>
  <Modal v-model="open" title="Choose Checkpoint">
    <div class="form-grid">
      <div>
        <label class="label-muted">Search</label>
        <input class="ui-input" type="text" v-model="q" placeholder="type to filter..." />
      </div>
      <div>
        <label class="label-muted">Current</label>
        <div class="card text-sm">{{ qs.currentModel || '(none)' }}</div>
      </div>
    </div>
    <div class="panel-section modal-list-section">
      <ul class="list" role="listbox">
        <li v-for="m in filtered" :key="m.title" class="cdx-list-item clickable" @click="apply(m.title)">
          <div class="flex items-center justify-between">
            <span>{{ m.title }}</span>
            <span class="caption opacity-70">{{ m.model_name }}</span>
          </div>
        </li>
      </ul>
    </div>
    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="open=false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import Modal from '../ui/Modal.vue'
import { useQuicksettingsStore } from '../../stores/quicksettings'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void }>()
const open = computed({ get: () => props.modelValue, set: (v: boolean) => emit('update:modelValue', v) })

const qs = useQuicksettingsStore()
const q = ref('')
const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  if (!query) return qs.models
  return qs.models.filter((m: { title: string; model_name: string }) => m.title.toLowerCase().includes(query) || m.model_name.toLowerCase().includes(query))
})

async function apply(title: string): Promise<void> {
  await qs.setModel(title)
  open.value = false
}
</script>
