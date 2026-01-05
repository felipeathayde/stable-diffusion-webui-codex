/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Server-driven UI blocks schema store.
Fetches `/api/ui/blocks` (optionally tab-scoped) and exposes the current block set/version for rendering parameter panels.

Symbols (top-level; keep in sync; no ghosts):
- `useUiBlocksStore` (store): Pinia store for UI block schemas (version/blocks/semantic_engine).
*/

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { UiBlock, UiBlocksResponse } from '../api/types'
import { fetchUiBlocks } from '../api/client'

export const useUiBlocksStore = defineStore('uiBlocks', () => {
  const version = ref<number>(0)
  const blocks = ref<UiBlock[]>([])
  const loaded = ref(false)
  const error = ref<string | null>(null)
  const semanticEngine = ref<string>('')

  async function init(tab?: string): Promise<void> {
    try {
      const res: UiBlocksResponse = await fetchUiBlocks(tab)
      version.value = res.version
      blocks.value = Array.isArray(res.blocks) ? res.blocks : []
      semanticEngine.value = String(res.semantic_engine || '')
      loaded.value = true
      error.value = null
    } catch (e: any) {
      error.value = String(e?.message || e)
      loaded.value = true
      blocks.value = []
      semanticEngine.value = ''
    }
  }

  return { version, blocks, loaded, error, semanticEngine, init }
})
