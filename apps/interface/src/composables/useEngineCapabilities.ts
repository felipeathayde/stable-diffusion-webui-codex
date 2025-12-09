/**
 * Composable for accessing engine capabilities in components.
 * Use this to conditionally show/hide fields based on engine type.
 */

import { computed } from 'vue'
import { getEngineConfig, getEngineCapabilities, type EngineType, type EngineCapabilities } from '../stores/engine_config'

export function useEngineCapabilities(engine: EngineType) {
  const config = computed(() => getEngineConfig(engine))
  const capabilities = computed(() => getEngineCapabilities(engine))
  
  // Convenience computed for common UI conditions
  const showNegativePrompt = computed(() => capabilities.value.usesNegativePrompt)
  const showCfg = computed(() => capabilities.value.usesCfg)
  const showDistilledCfg = computed(() => capabilities.value.usesDistilledCfg)
  const showTextEncoderSelector = computed(() => capabilities.value.requiresTenc)
  const showVaeSelector = computed(() => capabilities.value.requiresVae)
  const isVideoEngine = computed(() => capabilities.value.isVideoEngine)
  
  return {
    config,
    capabilities,
    showNegativePrompt,
    showCfg,
    showDistilledCfg,
    showTextEncoderSelector,
    showVaeSelector,
    isVideoEngine,
  }
}
