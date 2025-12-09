/**
 * Unified generation composable.
 * Provides generate(), progress, gallery, status for any engine type.
 * State is stored per-tab in model_tabs.
 */

import { computed, ref } from 'vue'
import { useModelTabsStore, type BaseTab, type ImageBaseParams } from '../stores/model_tabs'
import { useQuicksettingsStore } from '../stores/quicksettings'
import { getEngineConfig, type EngineType } from '../stores/engine_config'
import { buildTxt2ImgPayload, type Txt2ImgRequest } from '../api/payloads'
import { startTxt2Img, subscribeTask } from '../api/client'
import type { TaskEvent, GeneratedImage } from '../api/types'

export interface GenerationState {
  status: 'idle' | 'running' | 'done' | 'error'
  progress: {
    stage: string
    step: number
    totalSteps: number
  }
  gallery: GeneratedImage[]
  errorMessage: string
  taskId: string
  lastSeed: number | null
}

const DEFAULT_STATE: GenerationState = {
  status: 'idle',
  progress: { stage: 'none', step: 0, totalSteps: 0 },
  gallery: [],
  errorMessage: '',
  taskId: '',
  lastSeed: null,
}

// Per-tab generation state (keyed by tab ID)
const tabStates = new Map<string, GenerationState>()
const unsubscribers = new Map<string, () => void>()

function getTabState(tabId: string): GenerationState {
  if (!tabStates.has(tabId)) {
    tabStates.set(tabId, { ...DEFAULT_STATE })
  }
  return tabStates.get(tabId)!
}

export function useGeneration(tabId: string) {
  const modelTabs = useModelTabsStore()
  const quicksettings = useQuicksettingsStore()
  
  // Reactive state for this tab
  const state = ref(getTabState(tabId))
  
  // Tab info
  const tab = computed(() => modelTabs.tabs.find(t => t.id === tabId) as BaseTab | undefined)
  const params = computed(() => tab.value?.params as ImageBaseParams | undefined)
  const engineType = computed(() => tab.value?.type as EngineType | undefined)
  const engineConfig = computed(() => engineType.value ? getEngineConfig(engineType.value) : null)
  
  function stopStream(): void {
    const unsub = unsubscribers.get(tabId)
    if (unsub) {
      unsub()
      unsubscribers.delete(tabId)
    }
  }
  
  function resetProgress(): void {
    state.value.progress = { stage: 'none', step: 0, totalSteps: 0 }
  }
  
  async function generate(): Promise<void> {
    if (!tab.value || !params.value || !engineType.value) {
      state.value.status = 'error'
      state.value.errorMessage = 'Tab not found'
      return
    }
    
    stopStream()
    state.value.status = 'running'
    state.value.errorMessage = ''
    state.value.gallery = []
    resetProgress()
    
    const p = params.value
    const config = engineConfig.value!
    
    // Build extras based on engine capabilities
    const extras: Record<string, unknown> = {
      batch_size: 1,
      batch_count: 1,
    }
    
    // Add tenc_sha for engines that require it (like Z Image GGUF)
    if (config.capabilities.requiresTenc) {
      // Get first text encoder and look up its SHA
      const firstTenc = quicksettings.currentTextEncoders[0]
      if (firstTenc) {
        const sha = quicksettings.textEncoderShaMap.get(firstTenc)
        if (sha) {
          extras.tenc_sha = sha
        }
      }
    }
    
    let payload: Txt2ImgRequest
    try {
      payload = buildTxt2ImgPayload({
        prompt: p.prompt,
        negativePrompt: config.capabilities.usesNegativePrompt ? p.negativePrompt : '',
        width: p.width,
        height: p.height,
        steps: p.steps,
        guidanceScale: p.cfgScale,
        sampler: p.sampler || 'automatic',
        scheduler: p.scheduler || 'automatic',
        seed: p.seed,
        batchSize: 1,
        batchCount: 1,
        styles: [],
        device: (quicksettings.currentDevice || 'cpu') as any,
        engine: engineType.value,
        model: quicksettings.currentModel,
        smartOffload: quicksettings.smartOffload,
        smartFallback: quicksettings.smartFallback,
        smartCache: quicksettings.smartCache,
        extras,
      })
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
      return
    }
    
    try {
      const { task_id } = await startTxt2Img(payload)
      state.value.taskId = task_id
      state.value.progress.stage = 'submitted'
      
      const unsub = subscribeTask(task_id, handleTaskEvent)
      unsubscribers.set(tabId, unsub)
    } catch (error) {
      state.value.status = 'error'
      state.value.errorMessage = error instanceof Error ? error.message : String(error)
    }
  }
  
  function handleTaskEvent(event: TaskEvent): void {
    switch (event.type) {
      case 'progress':
        state.value.progress = {
          stage: event.stage,
          step: event.step ?? 0,
          totalSteps: event.total_steps ?? 0,
        }
        break
      case 'result':
        if ((event as any).images) {
          state.value.gallery = (event as any).images
        }
        if ((event as any).info?.seed !== undefined) {
          state.value.lastSeed = (event as any).info.seed
        }
        state.value.status = 'done'
        break
      case 'error':
        state.value.status = 'error'
        state.value.errorMessage = event.message
        stopStream()
        break
      case 'end':
        if (state.value.status !== 'error') {
          state.value.status = 'done'
        }
        stopStream()
        break
    }
  }
  
  // Expose reactive state and methods
  return {
    // State
    status: computed(() => state.value.status),
    progress: computed(() => state.value.progress),
    gallery: computed(() => state.value.gallery),
    errorMessage: computed(() => state.value.errorMessage),
    taskId: computed(() => state.value.taskId),
    lastSeed: computed(() => state.value.lastSeed),
    isRunning: computed(() => state.value.status === 'running'),
    
    // Tab info
    tab,
    params,
    engineType,
    engineConfig,
    
    // Actions
    generate,
    stopStream,
  }
}
