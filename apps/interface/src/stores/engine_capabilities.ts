/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pinia store for backend engine capability gating.
Fetches `/api/engines/capabilities` once and exposes a cached map so views/components can hide unsupported UI features per engine.

Symbols (top-level; keep in sync; no ghosts):
- `useEngineCapabilitiesStore` (store): Pinia store exposing engine capabilities, load state, and lookup helpers.
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { EngineCapabilitiesResponse, EngineCapabilities } from '../api/types'
import { fetchEngineCapabilities } from '../api/client'

export const useEngineCapabilitiesStore = defineStore('engineCapabilities', () => {
  const engines = ref<Record<string, EngineCapabilities>>({})
  const loaded = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)

  async function init(): Promise<void> {
    if (loaded.value || loading.value) return
    loading.value = true
    try {
      const res: EngineCapabilitiesResponse = await fetchEngineCapabilities()
      engines.value = res.engines ?? {}
      error.value = null
      loaded.value = true
    } catch (e: any) {
      // Keep UI permissive on failure: leave engines empty and surface error for diagnostics.
      error.value = String(e?.message || e)
    } finally {
      loading.value = false
    }
  }

  function get(engine: string | null | undefined): EngineCapabilities | null {
    if (!engine) return null
    return engines.value[engine] ?? null
  }

  const knownEngines = computed(() => Object.keys(engines.value))

  return {
    engines,
    knownEngines,
    loaded,
    loading,
    error,
    init,
    get,
  }
})
