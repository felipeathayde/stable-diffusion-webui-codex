<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Textual Inversion picker + insertion modal.
Fetches embeddings via the backend API, filters by search query, and emits TI tokens (optionally weighted) for prompt insertion.

Symbols (top-level; keep in sync; no ghosts):
- `TextualInversionModal` (component): Modal for selecting embeddings and emitting insertion tokens.
- `filtered` (const): Filtered embedding name list based on the current search query.
- `insert` (function): Formats and emits an embedding token for insertion into a prompt.
-->

<template>
  <Modal v-model="open" title="Textual Inversion">
    <div class="form-grid">
      <div>
        <label class="label-muted">Search</label>
        <input class="ui-input" v-model="q" placeholder="type to filter..." />
      </div>
      <div>
        <label class="label-muted">Weight</label>
        <input class="ui-input" type="number" step="0.1" min="0" v-model.number="weight" />
      </div>
    </div>
    <div class="panel-section modal-list-section">
      <ul class="list" role="listbox">
        <li v-for="name in filtered" :key="name" class="cdx-list-item clickable" @click="insert(name)">
          {{ name }}
        </li>
      </ul>
    </div>
    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="open=false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import Modal from '../ui/Modal.vue'
import { fetchEmbeddings } from '../../api/client'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void; (e:'insert', token: string): void }>()
const open = computed({ get: () => props.modelValue, set: (v: boolean) => emit('update:modelValue', v) })

const names = ref<string[]>([])
const q = ref('')
const weight = ref(1.0)
const loading = ref(false)
const loaded = ref(false)

const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  return names.value.filter(n => n.toLowerCase().includes(query))
})

watch(
  open,
  async (isOpen) => {
    if (!isOpen) return
    if (loaded.value || loading.value) return
    loading.value = true
    try {
      const res = await fetchEmbeddings()
      names.value = Object.keys(res.loaded || {}).sort((a, b) => a.localeCompare(b))
      loaded.value = true
    } catch {
      names.value = []
      loaded.value = true
    } finally {
      loading.value = false
    }
  },
  { immediate: true },
)

function insert(name: string): void {
  const t = weight.value && weight.value !== 1.0 ? `(${name}:${weight.value.toFixed(2)})` : name
  emit('insert', t)
}
</script>
