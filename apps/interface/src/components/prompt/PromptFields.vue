<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt + negative prompt field pair.
Renders two `PromptBox` instances (prompt + negative prompt), supports hiding the negative prompt field via `hideNegative`,
and forwards optional `tokenEngine` context to each prompt field for backend token-count requests.

Symbols (top-level; keep in sync; no ghosts):
- `PromptFields` (component): Prompt field pair component used by generation views.
- `hideNegative` (const): Computed boolean controlling whether the negative prompt field is hidden.
- `tokenEngine` (const): Normalized token-engine id forwarded to both `PromptBox` instances.
-->

<template>
  <div class="prompt-fields">
    <div class="panel-section">
      <PromptBox label="Prompt" v-model="innerPrompt" :token-engine="tokenEngine" variant="positive" />
    </div>
    <div v-if="!hideNegative" class="panel-section">
      <PromptBox label="Negative Prompt" v-model="innerNegative" :token-engine="tokenEngine" variant="negative" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import PromptBox from './PromptBox.vue'

const props = defineProps<{ prompt: string; negative: string; hideNegative?: boolean; tokenEngine?: string }>()
const emit = defineEmits<{ 'update:prompt': [value: string]; 'update:negative': [value: string] }>()

const innerPrompt = computed({
  get: () => props.prompt,
  set: (value: string) => emit('update:prompt', value),
})

const innerNegative = computed({
  get: () => props.negative,
  set: (value: string) => emit('update:negative', value),
})

const hideNegative = computed(() => props.hideNegative === true)
const tokenEngine = computed(() => String(props.tokenEngine || '').trim())

// counters moved inside PromptBox
</script>

<!-- styles moved to styles/components/prompt-fields.css -->
