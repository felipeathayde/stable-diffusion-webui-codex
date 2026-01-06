/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Server-driven UI presets store.
Fetches `/api/ui/presets` and exposes preset lists with helpers to filter by tab and apply a preset via the backend.

Symbols (top-level; keep in sync; no ghosts):
- `useUiPresetsStore` (store): Pinia store for UI presets (init + namesFor + applyByTitle).
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { UiPreset, UiPresetsResponse } from '../api/types'
import { fetchUiPresets, applyUiPreset } from '../api/client'

export const useUiPresetsStore = defineStore('uiPresets', () => {
  const version = ref<number>(0)
  const presets = ref<UiPreset[]>([])
  const loaded = ref(false)
  const error = ref<string | null>(null)

  async function init(tab?: string): Promise<void> {
    try {
      const res: UiPresetsResponse = await fetchUiPresets(tab)
      version.value = res.version
      presets.value = Array.isArray(res.presets) ? res.presets : []
      loaded.value = true
      error.value = null
    } catch (e: any) {
      error.value = String(e?.message || e)
      loaded.value = true
      presets.value = []
    }
  }

  function namesFor(tab: string): string[] {
    const t = (tab || '').toLowerCase()
    return presets.value
      .filter((p) => !p.tabs || p.tabs.map(String).map((x) => x.toLowerCase()).includes(t))
      .map((p) => p.title)
  }

  function idByTitle(title: string): string | null {
    const p = presets.value.find((x) => x.title === title)
    return p ? p.id : null
  }

  async function applyByTitle(title: string, tab: string): Promise<void> {
    const id = idByTitle(title)
    if (!id) throw new Error(`Unknown preset: ${title}`)
    await applyUiPreset(id, tab)
  }

  return { version, presets, loaded, error, init, namesFor, applyByTitle }
})
