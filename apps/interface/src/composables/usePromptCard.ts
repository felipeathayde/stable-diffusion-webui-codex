/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared PromptCard state and helpers.
Encapsulates capability-gated negative prompt visibility, modal toggles (LoRA/TI/Style), style selection, and token insertion into prompt fields.

Symbols (top-level; keep in sync; no ghosts):
- `PromptInsertPayload` (type): Token insertion payload accepted by `onInsertToken`.
- `usePromptCard` (function): Returns reactive state and handlers for PromptCard UI.
*/

import { computed, ref, type Ref } from 'vue'

import { useStylesStore } from '../stores/styles'

export type PromptInsertPayload = string | { token: string; target?: 'positive' | 'negative' }

export function usePromptCard(options: {
  prompt: Ref<string>
  negative: Ref<string>
  supportsNegative?: boolean
}) {
  const stylesStore = useStylesStore()

  const supportsNegative = options.supportsNegative !== false
  const hideNegative = computed(() => !supportsNegative)

  const showLora = ref(false)
  const showTI = ref(false)
  const showStyle = ref(false)

  const styleName = ref('')
  const styleNames = computed(() => stylesStore.names())

  function appendToken(target: Ref<string>, token: string): void {
    if (!token) return
    target.value = (target.value ? target.value + ' ' : '') + token
  }

  function onInsertToken(payload: PromptInsertPayload): void {
    const token = typeof payload === 'string' ? payload : payload.token
    const target = typeof payload === 'string' ? 'positive' : (payload.target ?? 'positive')
    if (!token) return

    if (!supportsNegative) {
      appendToken(options.prompt, token)
      return
    }

    if (target === 'negative') {
      appendToken(options.negative, token)
      return
    }

    appendToken(options.prompt, token)
  }

  function applyStyle(name: string): void {
    const style = stylesStore.get(name)
    if (!style) return

    if (style.prompt) {
      appendToken(options.prompt, style.prompt)
    }
    if (style.negative) {
      appendToken(options.negative, style.negative)
    }
  }

  function onStyleSaved(_name: string): void {
    // styles list is reactive; nothing else to do here
  }

  return {
    hideNegative,
    showLora,
    showTI,
    showStyle,
    styleName,
    styleNames,
    applyStyle,
    onInsertToken,
    onStyleSaved,
  }
}
