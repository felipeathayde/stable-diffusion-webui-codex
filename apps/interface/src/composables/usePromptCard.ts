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
- `PromptInsertPayload` (type): Token insertion payload accepted by `onInsertToken` (`target` + optional `action` add/remove).
- `usePromptCard` (function): Returns reactive state and handlers for PromptCard UI.
*/

import { computed, ref, type Ref } from 'vue'

import { useStylesStore } from '../stores/styles'

export type PromptInsertPayload =
  | string
  | {
      token: string
      target?: 'positive' | 'negative'
      action?: 'add' | 'remove'
    }

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

  function tokenizePrompt(rawValue: string): string[] {
    return String(rawValue || '')
      .split(/\s+/)
      .map((token) => token.trim())
      .filter(Boolean)
  }

  function appendTokenUnique(target: Ref<string>, token: string): void {
    const trimmedToken = String(token || '').trim()
    if (!trimmedToken) return
    const tokens = tokenizePrompt(target.value)
    if (tokens.includes(trimmedToken)) return
    tokens.push(trimmedToken)
    target.value = tokens.join(' ')
  }

  function removeToken(target: Ref<string>, token: string): void {
    const trimmedToken = String(token || '').trim()
    if (!trimmedToken) return
    const tokens = tokenizePrompt(target.value).filter((current) => current !== trimmedToken)
    target.value = tokens.join(' ')
  }

  function onInsertToken(payload: PromptInsertPayload): void {
    const token = typeof payload === 'string' ? payload : payload.token
    const target = typeof payload === 'string' ? 'positive' : (payload.target ?? 'positive')
    const action = typeof payload === 'string' ? 'add' : (payload.action ?? 'add')
    if (!token) return

    if (!supportsNegative) {
      if (action === 'remove') {
        removeToken(options.prompt, token)
        return
      }
      appendTokenUnique(options.prompt, token)
      return
    }

    if (target === 'negative') {
      if (action === 'remove') {
        removeToken(options.negative, token)
        return
      }
      appendTokenUnique(options.negative, token)
      return
    }

    if (action === 'remove') {
      removeToken(options.prompt, token)
      return
    }
    appendTokenUnique(options.prompt, token)
  }

  function applyStyle(name: string): void {
    const style = stylesStore.get(name)
    if (!style) return

    if (style.prompt) {
      appendTokenUnique(options.prompt, style.prompt)
    }
    if (style.negative) {
      appendTokenUnique(options.negative, style.negative)
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
