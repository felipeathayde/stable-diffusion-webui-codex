/**
 * Engine configuration - defaults and capabilities per engine type.
 * Used by dynamic tabs to set appropriate defaults.
 */

export type EngineType = 
  | 'sd15' 
  | 'sdxl' 
  | 'flux' 
  | 'zimage' 
  | 'chroma'
  | 'wan22_14b' 
  | 'wan22_5b'

export type TaskType = 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid'

export interface EngineCapabilities {
  // Supported tasks
  tasks: TaskType[]
  
  // CFG behavior
  usesCfg: boolean
  usesDistilledCfg: boolean
  usesNegativePrompt: boolean
  
  // Model requirements
  requiresTenc: boolean  // GGUF models require external text encoder
  requiresVae: boolean   // Some engines need explicit VAE
  
  // Video specific
  isVideoEngine: boolean
}

export interface EngineDefaults {
  width: number
  height: number
  steps: number
  cfg: number           // standard CFG (diffusion)
  distilledCfg?: number // distilled CFG (flow)
}

export interface EngineConfig {
  id: EngineType
  label: string
  capabilities: EngineCapabilities
  defaults: EngineDefaults
}

// =============================================================================
// Engine Configurations
// =============================================================================

const ENGINE_CONFIGS: Record<EngineType, EngineConfig> = {
  sd15: {
    id: 'sd15',
    label: 'SD 1.5',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      requiresTenc: false,
      requiresVae: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 512,
      height: 512,
      steps: 20,
      cfg: 7,
    },
  },
  
  sdxl: {
    id: 'sdxl',
    label: 'SDXL',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      requiresTenc: false,
      requiresVae: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 30,
      cfg: 7,
    },
  },
  
  flux: {
    id: 'flux',
    label: 'FLUX',
    capabilities: {
      tasks: ['txt2img', 'img2img'],
      usesCfg: false,
      usesDistilledCfg: true,
      usesNegativePrompt: false,
      requiresTenc: false,
      requiresVae: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 4,
      cfg: 1,
      distilledCfg: 3.5,
    },
  },
  
  zimage: {
    id: 'zimage',
    label: 'Z Image',
    capabilities: {
      tasks: ['txt2img'],
      usesCfg: false,
      usesDistilledCfg: true,
      usesNegativePrompt: false,
      requiresTenc: true,  // GGUF requires text encoder
      requiresVae: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 8,
      cfg: 1,
      distilledCfg: 1.0,
    },
  },
  
  chroma: {
    id: 'chroma',
    label: 'Chroma',
    capabilities: {
      tasks: ['txt2img'],
      usesCfg: false,
      usesDistilledCfg: true,
      usesNegativePrompt: false,
      requiresTenc: false,
      requiresVae: false,
      isVideoEngine: false,
    },
    defaults: {
      width: 1024,
      height: 1024,
      steps: 4,
      cfg: 1,
      distilledCfg: 3.5,
    },
  },
  
  wan22_14b: {
    id: 'wan22_14b',
    label: 'WAN 2.2 14B',
    capabilities: {
      tasks: ['txt2vid', 'img2vid'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      requiresTenc: false,
      requiresVae: false,
      isVideoEngine: true,
    },
    defaults: {
      width: 768,
      height: 432,
      steps: 30,
      cfg: 7,
    },
  },
  
  wan22_5b: {
    id: 'wan22_5b',
    label: 'WAN 2.2 5B',
    capabilities: {
      tasks: ['txt2vid', 'img2vid'],
      usesCfg: true,
      usesDistilledCfg: false,
      usesNegativePrompt: true,
      requiresTenc: false,
      requiresVae: false,
      isVideoEngine: true,
    },
    defaults: {
      width: 768,
      height: 432,
      steps: 30,
      cfg: 7,
    },
  },
}

// =============================================================================
// Exports
// =============================================================================

export function getEngineConfig(engine: EngineType): EngineConfig {
  return ENGINE_CONFIGS[engine]
}

export function getEngineDefaults(engine: EngineType): EngineDefaults {
  return ENGINE_CONFIGS[engine].defaults
}

export function getEngineCapabilities(engine: EngineType): EngineCapabilities {
  return ENGINE_CONFIGS[engine].capabilities
}

export function getAllEngines(): EngineConfig[] {
  return Object.values(ENGINE_CONFIGS)
}

export function getImageEngines(): EngineConfig[] {
  return getAllEngines().filter(e => !e.capabilities.isVideoEngine)
}

export function getVideoEngines(): EngineConfig[] {
  return getAllEngines().filter(e => e.capabilities.isVideoEngine)
}
