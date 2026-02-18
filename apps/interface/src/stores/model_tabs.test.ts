/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for model-tab normalization, params persistence serialization invariants,
and localStorage persistence guardrails (light payload + quota handling) in the model-tabs store.
Locks fail-loud behavior for unknown tab types, validates capability-driven Anima tab auto-inclusion, and ensures
`updateParams` persistence handles nested reactive payload branches while still rejecting non-serializable payloads
without hanging deferred promises.

Symbols (top-level; keep in sync; no ghosts):
- `model_tabs.test` (module): Model tabs normalization/capabilities + params-persistence serialization tests.
*/

import { createPinia, setActivePinia } from 'pinia'
import { reactive } from 'vue'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { createTabApi, deleteTabApi, fetchTabs, reorderTabsApi, updateTabApi } from '../api/client'
import { defaultImageParamsForType, normalizeTabType, requiredTypesFromCapabilities, useModelTabsStore } from './model_tabs'

vi.mock('../api/client', () => ({
  fetchTabs: vi.fn(),
  createTabApi: vi.fn(),
  updateTabApi: vi.fn(),
  reorderTabsApi: vi.fn(),
  deleteTabApi: vi.fn(),
}))

const mockedFetchTabs = vi.mocked(fetchTabs)
const mockedCreateTabApi = vi.mocked(createTabApi)
const mockedUpdateTabApi = vi.mocked(updateTabApi)
const mockedReorderTabsApi = vi.mocked(reorderTabsApi)
const mockedDeleteTabApi = vi.mocked(deleteTabApi)

function makeStorageStub(): Storage {
  const storage = new Map<string, string>()
  return {
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
  }
}

function createQuotaExceededError(): DOMException {
  return new DOMException('Setting the value exceeded the quota.', 'QuotaExceededError')
}

function makeStorageStubThatThrowsQuotaOnce(): Storage {
  const storage = new Map<string, string>()
  let shouldThrowOnNextSet = true
  return {
    getItem: (key: string) => storage.get(String(key)) ?? null,
    setItem: (key: string, value: string) => {
      if (shouldThrowOnNextSet) {
        shouldThrowOnNextSet = false
        throw createQuotaExceededError()
      }
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
  }
}

function createSeededStore() {
  const store = useModelTabsStore()
  store.tabs = [{
    id: 'tab-1',
    type: 'sdxl',
    title: 'SDXL',
    order: 0,
    enabled: true,
    params: {
      prompt: '',
      negativePrompt: '',
      sampler: 'euler',
      scheduler: 'simple',
      checkpoint: '',
      vae: '',
      steps: 20,
      cfgScale: 7,
      width: 1024,
      height: 1024,
      seed: -1,
      batchSize: 1,
      batchCount: 1,
      img2imgResizeMode: 'just_resize',
      img2imgUpscaler: 'latent:bicubic-aa',
      useInitImage: false,
      initImageData: '',
      initImageName: '',
      denoiseStrength: 0.75,
      useMask: false,
      maskImageData: '',
      maskImageName: '',
      clipSkip: 1,
      zimageTurbo: false,
      highresUpscaler: '',
      upscalerTile: { enabled: false, tile: 512, overlap: 64, fallbackOnOom: false, minTile: 128 },
      hires: {
        enabled: false,
        denoise: 0.55,
        scale: 1.5,
        steps: 20,
        mode: 'latent',
        model: '',
        refiner: { enabled: false, swapAtStep: 1, cfg: 3.5, seed: -1 },
      },
      refiner: { enabled: false, swapAtStep: 1, cfg: 3.5, seed: -1, model: undefined },
      flux1TextEncoders: [],
      flux1AePath: '',
      zimageTencPath: '',
      zimageVaePath: '',
      zimageMode: 'int4',
      zimageResultInBrowser: false,
    },
    meta: {
      createdAt: '2026-02-08T00:00:00.000Z',
      updatedAt: '2026-02-08T00:00:00.000Z',
    },
  }] as any
  store.activeId = 'tab-1'
  return store
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.clearAllMocks()
  setActivePinia(createPinia())
  vi.stubGlobal('localStorage', makeStorageStub())
  mockedFetchTabs.mockResolvedValue({ tabs: [] } as any)
  mockedCreateTabApi.mockResolvedValue({ id: 'created-tab' } as any)
  mockedUpdateTabApi.mockResolvedValue({ ok: true } as any)
  mockedReorderTabsApi.mockResolvedValue({ ok: true } as any)
  mockedDeleteTabApi.mockResolvedValue({ ok: true } as any)
})

afterEach(() => {
  vi.useRealTimers()
  vi.unstubAllGlobals()
})

describe('normalizeTabType', () => {
  it('normalizes known aliases', () => {
    expect(normalizeTabType('wan22_5b')).toBe('wan')
    expect(normalizeTabType('flux1_chroma')).toBe('chroma')
    expect(normalizeTabType('anima')).toBe('anima')
  })

  it('fails loud on unknown type', () => {
    expect(() => normalizeTabType('mystery_engine')).toThrow(/Unsupported model tab type/)
  })
})

describe('requiredTypesFromCapabilities', () => {
  it('includes anima when capabilities expose anima', () => {
    const types = requiredTypesFromCapabilities({ anima: {}, sd15: {} })
    expect(types).toContain('anima')
  })

  it('does not include anima when capabilities omit anima', () => {
    const types = requiredTypesFromCapabilities({ sd15: {}, sdxl: {} })
    expect(types).not.toContain('anima')
  })
})

describe('defaultImageParamsForType', () => {
  it('uses hires scale default of 2 with resize overrides at 0', () => {
    const defaults = defaultImageParamsForType('sdxl')
    expect(defaults.hires.scale).toBe(2)
    expect(defaults.hires.resizeX).toBe(0)
    expect(defaults.hires.resizeY).toBe(0)
    expect(defaults.img2imgResizeMode).toBe('just_resize')
    expect(defaults.img2imgUpscaler).toBe('latent:bicubic-aa')
  })
})

describe('useModelTabsStore params persistence serialization', () => {
  it('serializes proxy patches and persists without structuredClone DataCloneError', async () => {
    const store = createSeededStore()
    const proxyPatch = reactive({ useInitImage: true, initImageName: 'init.png' }) as any

    const persistPromise = store.updateParams('tab-1', proxyPatch)
    await vi.advanceTimersByTimeAsync(300)
    await expect(persistPromise).resolves.toBeUndefined()

    expect(mockedUpdateTabApi).toHaveBeenCalledTimes(1)
    const payload = mockedUpdateTabApi.mock.calls[0]?.[1] as { params: Record<string, unknown> }
    expect(payload.params.useInitImage).toBe(true)
    expect(payload.params.initImageName).toBe('init.png')
  })

  it('serializes nested reactive branches in patch payloads', async () => {
    const store = createSeededStore()
    const currentHires = reactive((store.tabs[0] as any).params.hires) as any
    const patch = { hires: { ...currentHires, enabled: true } } as any

    const persistPromise = store.updateParams('tab-1', patch)
    await vi.advanceTimersByTimeAsync(300)
    await expect(persistPromise).resolves.toBeUndefined()

    expect(mockedUpdateTabApi).toHaveBeenCalledTimes(1)
    const payload = mockedUpdateTabApi.mock.calls[0]?.[1] as { params: Record<string, unknown> }
    const hires = payload.params.hires as Record<string, unknown>
    expect(hires.enabled).toBe(true)
    expect((hires.refiner as Record<string, unknown>).enabled).toBe(false)
  })

  it('fails loud with serialization_failure for non-serializable patch values', async () => {
    const store = createSeededStore()

    await expect(store.updateParams('tab-1', { bad: () => 'nope' } as any)).rejects.toMatchObject({
      code: 'serialization_failure',
    })
    expect(mockedUpdateTabApi).not.toHaveBeenCalled()
  })

  it('fails loud for non-plain patch payloads', async () => {
    const store = createSeededStore()
    class InvalidPatchPayload {
      readonly useInitImage = true
    }

    await expect(store.updateParams('tab-1', new InvalidPatchPayload() as any)).rejects.toMatchObject({
      code: 'serialization_failure',
    })
    expect(mockedUpdateTabApi).not.toHaveBeenCalled()
  })

  it('fails loud when structuredClone is unavailable', async () => {
    const store = createSeededStore()
    vi.stubGlobal('structuredClone', undefined)

    await expect(store.updateParams('tab-1', { useInitImage: true } as any)).rejects.toMatchObject({
      code: 'serialization_failure',
    })
    expect(mockedUpdateTabApi).not.toHaveBeenCalled()
  })

  it('rejects queued deferreds when flush serialization fails', async () => {
    const store = createSeededStore()

    const persistPromise = store.updateParams('tab-1', { useInitImage: true } as any)
    const persistedError = persistPromise.then(
      () => null,
      (error) => error,
    )
    ;(store.tabs[0] as any).params.poison = () => 'not-serializable'

    await vi.advanceTimersByTimeAsync(300)
    await expect(persistedError).resolves.toMatchObject({
      code: 'serialization_failure',
    })
    expect(mockedUpdateTabApi).not.toHaveBeenCalled()
  })
})

describe('useModelTabsStore local storage payload guard', () => {
  it('persists only light tab refs to localStorage (no params blob)', () => {
    const store = createSeededStore()
    ;(store.tabs[0] as any).params.initImageData = `data:image/png;base64,${'X'.repeat(1024)}`
    ;(store.tabs[0] as any).params.maskImageData = `data:image/png;base64,${'Y'.repeat(1024)}`

    store.save()

    const raw = localStorage.getItem('codex:model-tabs:v2')
    expect(raw).toBeTruthy()
    const parsed = JSON.parse(String(raw)) as { tabs?: unknown; activeId?: unknown }
    expect(parsed.activeId).toBe('tab-1')
    expect(parsed.tabs).toEqual([{ id: 'tab-1', type: 'sdxl' }])
    expect(String(raw)).not.toContain('initImageData')
    expect(String(raw)).not.toContain('maskImageData')
  })

  it('handles quota errors without throwing and retries light payload after clearing key', () => {
    vi.stubGlobal('localStorage', makeStorageStubThatThrowsQuotaOnce())
    const consoleWarn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    const store = createSeededStore()

    expect(() => store.save()).not.toThrow()
    const raw = localStorage.getItem('codex:model-tabs:v2')
    expect(raw).toBeTruthy()
    expect(JSON.parse(String(raw))).toEqual({
      tabs: [{ id: 'tab-1', type: 'sdxl' }],
      activeId: 'tab-1',
    })
    expect(consoleWarn).toHaveBeenCalled()
  })
})
