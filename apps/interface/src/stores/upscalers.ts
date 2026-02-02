/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Upscalers store (inventory + shared preferences) for the WebUI.
Provides a cached local upscalers list from `GET /api/upscalers` and a persisted global "fallback on OOM" preference used by both hires-fix
and the standalone `/upscale` view.

Symbols (top-level; keep in sync; no ghosts):
- `useUpscalersStore` (store): Pinia store exposing `upscalers` inventory + `fallbackOnOom` preference and a `load()` action.
*/

import { defineStore } from 'pinia'
import { computed, ref, watch } from 'vue'
import type { UpscalerDefinition } from '../api/types'
import { fetchUpscalers, refreshUpscalers } from '../api/client'

const FALLBACK_ON_OOM_STORAGE_KEY = 'codex.upscale.tile.fallback_on_oom'

function parseStoredBool(raw: string | null): boolean | null {
  if (raw === null) return null
  const normalized = String(raw).trim().toLowerCase()
  if (!normalized) return null
  if (normalized === '0' || normalized === 'false' || normalized === 'no' || normalized === 'off') return false
  if (normalized === '1' || normalized === 'true' || normalized === 'yes' || normalized === 'on') return true
  return null
}

export const useUpscalersStore = defineStore('upscalers', () => {
  const upscalers = ref<UpscalerDefinition[]>([])
  const loading = ref(false)
  const error = ref('')

  const fallbackOnOom = ref(true)
  try {
    const parsed = parseStoredBool(localStorage.getItem(FALLBACK_ON_OOM_STORAGE_KEY))
    if (parsed !== null) fallbackOnOom.value = parsed
  } catch (err) {
    console.warn('[upscalers] failed to load fallback-on-oom from localStorage', err)
  }

  watch(
    fallbackOnOom,
    (value) => {
      try {
        localStorage.setItem(FALLBACK_ON_OOM_STORAGE_KEY, value ? '1' : '0')
      } catch (err) {
        console.warn('[upscalers] failed to persist fallback-on-oom to localStorage', err)
      }
    },
    { flush: 'post' },
  )

  const spandrelUpscalers = computed(() => upscalers.value.filter((u) => u.kind === 'spandrel'))
  const latentUpscalers = computed(() => upscalers.value.filter((u) => u.kind === 'latent'))

  async function load(opts: { refresh?: boolean } = {}): Promise<void> {
    const refresh = Boolean(opts.refresh)
    if (!refresh && upscalers.value.length > 0) return
    loading.value = true
    error.value = ''
    try {
      const res = refresh ? await refreshUpscalers() : await fetchUpscalers()
      upscalers.value = Array.isArray(res.upscalers) ? res.upscalers : []
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      loading.value = false
    }
  }

  return {
    // Inventory
    upscalers,
    loading,
    error,
    spandrelUpscalers,
    latentUpscalers,
    load,
    // Preferences
    fallbackOnOom,
  }
})

