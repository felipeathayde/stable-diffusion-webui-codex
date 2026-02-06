/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for strict `/api/engines/capabilities` dependency-check contract parsing.
Locks that `engine_capabilities` fails loud on missing/invalid `dependency_checks` and exposes parsed readiness rows.

Symbols (top-level; keep in sync; no ghosts):
- `engine_capabilities.test` (module): Regression coverage for strict dependency-check parsing.
*/

import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { fetchEngineCapabilities } from '../api/client'
import { useEngineCapabilitiesStore } from './engine_capabilities'

vi.mock('../api/client', () => ({
  fetchEngineCapabilities: vi.fn(),
}))

const mockedFetchEngineCapabilities = vi.mocked(fetchEngineCapabilities)

function sd15Surface() {
  return {
    supports_txt2img: true,
    supports_img2img: true,
    supports_txt2vid: false,
    supports_img2vid: false,
    supports_hires: true,
    supports_refiner: false,
    supports_lora: true,
    supports_controlnet: false,
  }
}

describe('useEngineCapabilitiesStore dependency checks', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
    vi.clearAllMocks()
  })

  it('loads dependency checks when contract is valid', async () => {
    mockedFetchEngineCapabilities.mockResolvedValue({
      engines: { sd15: sd15Surface() },
      dependency_checks: {
        sd15: {
          ready: true,
          checks: [
            { id: 'capability_surface', label: 'Capability Surface', ok: true, message: 'ready' },
          ],
        },
      },
    } as any)

    const store = useEngineCapabilitiesStore()
    await store.init()

    expect(store.loaded).toBe(true)
    expect(store.get('sd15')?.supports_txt2img).toBe(true)
    expect(store.getDependencyStatus('sd15')?.ready).toBe(true)
    expect(store.firstDependencyError('sd15')).toBe('')
  })

  it('fails loud when dependency_checks are missing', async () => {
    mockedFetchEngineCapabilities.mockResolvedValue({
      engines: { sd15: sd15Surface() },
    } as any)

    const store = useEngineCapabilitiesStore()
    await expect(store.init()).rejects.toThrow(/dependency_checks/i)
    expect(store.loaded).toBe(false)
    expect(store.error).toMatch(/dependency_checks/i)
  })

  it('fails loud when ready flag conflicts with check rows', async () => {
    mockedFetchEngineCapabilities.mockResolvedValue({
      engines: { sd15: sd15Surface() },
      dependency_checks: {
        sd15: {
          ready: true,
          checks: [
            { id: 'checkpoint_inventory', label: 'Model Checkpoints', ok: false, message: 'missing checkpoint' },
          ],
        },
      },
    } as any)

    const store = useEngineCapabilitiesStore()
    await expect(store.init()).rejects.toThrow(/inconsistent/i)
    expect(store.loaded).toBe(false)
  })
})
