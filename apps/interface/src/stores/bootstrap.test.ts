/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Vitest coverage for bootstrap fatal-funnel invariants.
Locks that the bootstrap store preserves the first fatal root-cause details and does not overwrite them on later failures.

Symbols (top-level; keep in sync; no ghosts):
- `bootstrap.test` (module): Regression coverage for bootstrap fatal error invariants.
*/

import { createPinia, setActivePinia } from 'pinia'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { useBootstrapStore } from './bootstrap'

describe('useBootstrapStore fatal funnel', () => {
  beforeEach(() => {
    setActivePinia(createPinia())
  })

  it('keeps first fatal context/message immutable', () => {
    const errorSpy = vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const store = useBootstrapStore()

    store.reportFatal(new Error('first failure'), 'First context')
    store.reportFatal(new Error('second failure'), 'Second context')

    expect(store.status).toBe('fatal')
    expect(store.fatalContext).toBe('First context')
    expect(store.fatalMessage).toBe('first failure')
    errorSpy.mockRestore()
  })
})
