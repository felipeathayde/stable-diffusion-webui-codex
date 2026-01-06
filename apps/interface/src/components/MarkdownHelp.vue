<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Help content loader for small markdown/text docs.
Fetches the provided `src` URL and displays the raw content (or an error/loading state) inside views and panels.

Symbols (top-level; keep in sync; no ghosts):
- `MarkdownHelp` (component): Help content loader component.
- `load` (function): Fetches the help content for the current `src` prop and updates state.
-->

<template>
  <div class="help-markdown">
    <pre v-if="error" class="muted">Failed to load help: {{ error }}</pre>
    <pre v-else-if="!content" class="muted">Loading help…</pre>
    <pre v-else class="muted">{{ content }}</pre>
  </div>
</template>

<script setup lang="ts">
import { onMounted, watch, ref } from 'vue'

const props = defineProps<{
  src: string
}>()

const content = ref('')
const error = ref<string | null>(null)

async function load(): Promise<void> {
  if (!props.src) return
  try {
    error.value = null
    content.value = ''
    const res = await fetch(props.src)
    if (!res.ok) {
      throw new Error(`${res.status} ${res.statusText}`)
    }
    content.value = await res.text()
  } catch (e: any) {
    error.value = String(e?.message || e)
  }
}

onMounted(() => { void load() })
watch(() => props.src, () => { void load() })
</script>
