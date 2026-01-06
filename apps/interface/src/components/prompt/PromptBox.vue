<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt input box with label + token count.
Wraps `PromptEditor` with a label and a small whitespace-token counter used by generation views.

Symbols (top-level; keep in sync; no ghosts):
- `PromptBox` (component): Prompt editor wrapper with label + token-count badge.
- `countTokens` (function): Counts whitespace-delimited tokens in the current prompt text.
- `tokenCount` (const): Computed token count derived from the current prompt value.
-->

<template>
  <div class="prompt-box" :data-variant="variant">
    <div class="prompt-header">
      <span class="label-muted">{{ label }}</span>
    </div>
    <div class="prompt-editor-wrap">
      <span v-if="tokenCount > 0" class="ear-badge">{{ tokenCount }} tok</span>
      <PromptEditor v-model="inner" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import PromptEditor from './PromptEditor.vue'

const props = defineProps<{ label: string; modelValue: string; variant?: 'positive'|'negative' }>()
const emit = defineEmits<{ (e:'update:modelValue', v:string): void }>()

const inner = computed({
  get: () => props.modelValue || '',
  set: (v: string) => emit('update:modelValue', v),
})

function countTokens(text: string): number {
  const raw = String(text || '').trim()
  if (!raw) return 0
  return raw.split(/\s+/).filter(Boolean).length
}

const tokenCount = computed(() => countTokens(inner.value))
</script>

<!-- styles moved to styles/components/prompt-box.css -->
