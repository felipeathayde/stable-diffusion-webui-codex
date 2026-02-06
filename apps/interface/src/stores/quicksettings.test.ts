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
  let storage = new Map<string, string>()

  beforeEach(() => {
    vi.stubGlobal('localStorage', {
      getItem: (key: string) => storage.get(String(key)) ?? null,
      setItem: (key: string, value: string) => {
        storage.set(String(key), String(value))
      },
      removeItem: (key: string) => {
        storage.delete(String(key))
      },
      clear: () => {
        storage.clear()
      },
      key: (index: number) => Array.from(storage.keys())[index] ?? null,
      get length() {
        return storage.size
      },
    } satisfies Storage)
    setActivePinia(createPinia())
    vi.clearAllMocks()
    storage = new Map<string, string>()

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

  it('surfaces anima text-encoder file choices from inventory (not root folders)', async () => {
    const store = useQuicksettingsStore()
    await store.init()

    expect(store.textEncoderChoices).toContain('anima/models/anima-tenc/qwen3_06b.safetensors')
    expect(store.textEncoderChoices).not.toContain('anima/models/anima-tenc')
  })

  it('resolves anima-prefixed text encoder and vae labels to SHA', async () => {
    const store = useQuicksettingsStore()
    await store.init()

    expect(store.resolveTextEncoderSha('anima/models/anima-tenc/qwen3_06b.safetensors')).toBe('a'.repeat(64))
    expect(store.resolveVaeSha('anima/models/anima-vae/wan22_anima_vae.safetensors')).toBe('b'.repeat(64))
  })

  it('sanitizes stale root-folder text encoder overrides from localStorage', async () => {
    storage.set('codex.quicksettings.text_encoder_overrides', JSON.stringify(['anima/models/anima-tenc']))

    const store = useQuicksettingsStore()
    await store.init()

    expect(store.currentTextEncoders).toEqual([])
  })

  it('keeps valid file-root override entries when they resolve to SHA', async () => {
    storage.set(
      'codex.quicksettings.text_encoder_overrides',
      JSON.stringify(['anima/models/anima-tenc/qwen3_06b.safetensors']),
    )
    mockedFetchPaths.mockResolvedValueOnce({
      paths: {
        anima_tenc: ['models/anima-tenc/qwen3_06b.safetensors'],
      },
    } as any)

    const store = useQuicksettingsStore()
    await store.init()

    expect(store.currentTextEncoders).toEqual(['anima/models/anima-tenc/qwen3_06b.safetensors'])
    expect(store.resolveTextEncoderSha('anima/models/anima-tenc/qwen3_06b.safetensors')).toBe('a'.repeat(64))
  })

  it('matches family roots with trailing slash when deriving selectable TE files', async () => {
    mockedFetchPaths.mockResolvedValueOnce({
      paths: {
        anima_tenc: ['models/anima-tenc/'],
      },
    } as any)

    const store = useQuicksettingsStore()
    await store.init()

    expect(store.textEncoderChoices).toContain('anima/models/anima-tenc/qwen3_06b.safetensors')
  })

  it('matches absolute models root paths when deriving selectable TE files', async () => {
    mockedFetchPaths.mockResolvedValueOnce({
      paths: {
        anima_tenc: ['/home/lucas/work/stable-diffusion-webui-codex/models/anima-tenc'],
      },
    } as any)

    const store = useQuicksettingsStore()
    await store.init()

    expect(store.textEncoderChoices).toContain('anima/models/anima-tenc/qwen3_06b.safetensors')
  })

  it('fails loud when text encoder inventory load fails', async () => {
    const store = useQuicksettingsStore()
    store.textEncoderChoices = ['anima/models/anima-tenc/qwen3_06b.safetensors']
    store.textEncoderShaMap = new Map([['anima/models/anima-tenc/qwen3_06b.safetensors', 'a'.repeat(64)]])
    store.vaeShaMap = new Map([['anima/models/anima-vae/wan22_anima_vae.safetensors', 'b'.repeat(64)]])
    store.wanGgufShaMap = new Map([['wan/models/high.gguf', 'c'.repeat(64)]])

    mockedFetchModelInventory.mockRejectedValue(new Error('inventory down'))
    await expect(store.init()).rejects.toThrow('inventory down')

    expect(store.textEncoderChoices).toEqual(['anima/models/anima-tenc/qwen3_06b.safetensors'])
    expect(store.textEncoderShaMap.size).toBe(1)
    expect(store.vaeShaMap.size).toBe(1)
    expect(store.wanGgufShaMap.size).toBe(1)
  })
})
