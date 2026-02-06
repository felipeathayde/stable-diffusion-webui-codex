/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Application bootstrap orchestrator and fatal-error funnel for the interface.
Owns required startup sequencing (engine capabilities, model tabs, quicksettings), enforces hard-fatal behavior on required failures,
and exposes a retry flow used by the root App fatal screen.

Symbols (top-level; keep in sync; no ghosts):
- `BootstrapStatus` (type): Global bootstrap state (`idle|loading|ready|fatal`).
- `useBootstrapStore` (store): Pinia store exposing bootstrap lifecycle + global error funnel helpers.
*/

import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import { useEngineCapabilitiesStore } from './engine_capabilities'
import { useModelTabsStore } from './model_tabs'
import { useQuicksettingsStore } from './quicksettings'

export type BootstrapStatus = 'idle' | 'loading' | 'ready' | 'fatal'

function normalizeErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message || error.name || 'Unknown error'
  if (typeof error === 'string') return error
  return String(error)
}

export const useBootstrapStore = defineStore('bootstrap', () => {
  const status = ref<BootstrapStatus>('idle')
  const fatalContext = ref<string>('')
  const fatalMessage = ref<string>('')
  let bootstrapPromise: Promise<void> | null = null
  let globalHandlersInstalled = false

  const isReady = computed(() => status.value === 'ready')
  const isLoading = computed(() => status.value === 'loading')
  const isFatal = computed(() => status.value === 'fatal')

  function reportFatal(error: unknown, context: string): void {
    const message = normalizeErrorMessage(error)
    if (isFatal.value) {
      console.error('[bootstrap] additional fatal error', { context, error })
      return
    }
    console.error('[bootstrap] fatal error', { context, error })
    fatalContext.value = String(context || 'Fatal error')
    fatalMessage.value = message
    status.value = 'fatal'
  }

  async function runRequired<T>(context: string, fn: () => Promise<T>): Promise<T> {
    try {
      return await fn()
    } catch (error: unknown) {
      reportFatal(error, context)
      throw error
    }
  }

  async function start(opts: { force?: boolean } = {}): Promise<void> {
    const force = Boolean(opts.force)
    if (!force && isReady.value) return
    if (bootstrapPromise) return bootstrapPromise

    if (force || status.value === 'fatal') {
      fatalContext.value = ''
      fatalMessage.value = ''
    }
    status.value = 'loading'

    const engineCaps = useEngineCapabilitiesStore()
    const tabsStore = useModelTabsStore()
    const quicksettingsStore = useQuicksettingsStore()

    bootstrapPromise = (async () => {
      await runRequired('Failed to load engine capabilities', async () => {
        await engineCaps.init({ force })
      })
      await Promise.all([
        runRequired('Failed to load model tabs', async () => {
          await tabsStore.load()
        }),
        runRequired('Failed to load quick settings', async () => {
          await quicksettingsStore.init()
        }),
      ])
      status.value = 'ready'
    })()

    try {
      await bootstrapPromise
    } finally {
      bootstrapPromise = null
    }
  }

  async function retry(): Promise<void> {
    await start({ force: true })
  }

  function installGlobalErrorHandlers(): void {
    if (globalHandlersInstalled) return
    globalHandlersInstalled = true

    window.addEventListener('error', (event: ErrorEvent) => {
      const reason = event.error ?? event.message ?? 'Unhandled runtime error'
      reportFatal(reason, 'Unhandled runtime error')
    })

    window.addEventListener('unhandledrejection', (event: PromiseRejectionEvent) => {
      reportFatal(event.reason, 'Unhandled promise rejection')
    })
  }

  return {
    status,
    fatalContext,
    fatalMessage,
    isReady,
    isLoading,
    isFatal,
    start,
    retry,
    reportFatal,
    runRequired,
    installGlobalErrorHandlers,
  }
})
