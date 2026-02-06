/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for QuickSettings Anima path-family mapping and SHA resolution.
Locks that `anima_tenc` paths are surfaced as UI labels and `anima/<path>` aliases resolve for text encoders/VAEs.

Symbols (top-level; keep in sync; no ghosts):
- `quicksettings.test` (module): Regression coverage for Anima-aware quicksettings mappings.
*/

import { createPinia, setActivePinia } from 'pinia'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchModelInventory, fetchModels, fetchOptions, fetchPaths, refreshModels, updateOptions } from '../api/client'
import { useQuicksettingsStore } from './quicksettings'

vi.mock('../api/client', () => ({
  fetchModels: vi.fn(),
  refreshModels: vi.fn(),
  fetchModelInventory: vi.fn(),
  fetchPaths: vi.fn(),
  fetchOptions: vi.fn(),
  updateOptions: vi.fn(),
}))

const mockedFetchModels = vi.mocked(fetchModels)
const mockedRefreshModels = vi.mocked(refreshModels)
const mockedFetchModelInventory = vi.mocked(fetchModelInventory)
const mockedFetchPaths = vi.mocked(fetchPaths)
const mockedFetchOptions = vi.mocked(fetchOptions)
const mockedUpdateOptions = vi.mocked(updateOptions)

describe('useQuicksettingsStore anima mappings', () => {
  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: () => null,
      setItem: () => undefined,
      removeItem: () => undefined,
      clear: () => undefined,
      key: () => null,
      length: 0,
    } satisfies Storage)
    setActivePinia(createPinia())
    vi.clearAllMocks()

    mockedFetchModels.mockResolvedValue({ models: [], current: '' } as any)
    mockedRefreshModels.mockResolvedValue({ models: [], current: '' } as any)
    mockedFetchOptions.mockResolvedValue({ values: {} } as any)
    mockedUpdateOptions.mockResolvedValue({ ok: true } as any)
    mockedFetchPaths.mockResolvedValue({
      paths: {
        anima_tenc: ['models/anima-tenc'],
      },
    } as any)
    mockedFetchModelInventory.mockResolvedValue({
      text_encoders: [
        {
          name: 'qwen3_06b.safetensors',
          path: 'models/anima-tenc/qwen3_06b.safetensors',
          sha256: 'a'.repeat(64),
        },
      ],
      vaes: [
        {
          name: 'wan22_anima_vae.safetensors',
          path: 'models/anima-vae/wan22_anima_vae.safetensors',
          sha256: 'b'.repeat(64),
        },
      ],
      wan22: {},
    } as any)
  })

  afterEach(() => {
    vi.unstubAllGlobals()
  })

  it('surfaces anima text-encoder roots from /api/paths', async () => {
    const store = useQuicksettingsStore()
    await store.init()

    expect(store.textEncoderChoices).toContain('anima/models/anima-tenc')
  })

  it('resolves anima-prefixed text encoder and vae labels to SHA', async () => {
    const store = useQuicksettingsStore()
    await store.init()

    expect(store.resolveTextEncoderSha('anima/models/anima-tenc/qwen3_06b.safetensors')).toBe('a'.repeat(64))
    expect(store.resolveVaeSha('anima/models/anima-vae/wan22_anima_vae.safetensors')).toBe('b'.repeat(64))
  })
})
