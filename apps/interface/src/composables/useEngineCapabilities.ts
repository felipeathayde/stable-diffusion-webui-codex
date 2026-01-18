/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Engine capability composable for UI gating.
Builds convenience computed flags from engine config/capabilities so components can show/hide fields based on engine semantics.

Symbols (top-level; keep in sync; no ghosts):
- `useEngineCapabilities` (function): Returns engine config, capabilities, and common UI gating computed flags.
*/

import { computed } from 'vue'
import { getEngineConfig, getEngineCapabilities, type EngineType } from '../stores/engine_config'

export function useEngineCapabilities(engine: EngineType) {
  const config = computed(() => getEngineConfig(engine))
  const capabilities = computed(() => getEngineCapabilities(engine))
  
  // Convenience computed for common UI conditions
  const showNegativePrompt = computed(() => capabilities.value.usesNegativePrompt)
  const showCfg = computed(() => capabilities.value.usesCfg)
  const showDistilledCfg = computed(() => capabilities.value.usesDistilledCfg)
  const isVideoEngine = computed(() => capabilities.value.isVideoEngine)
  
  return {
    config,
    capabilities,
    showNegativePrompt,
    showCfg,
    showDistilledCfg,
    isVideoEngine,
  }
}
