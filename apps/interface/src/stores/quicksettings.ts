/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: QuickSettings global store (models/samplers/schedulers/options + asset SHA selection).
Loads lists from `/api/*`, persists option changes via `/api/options`, and maintains SHA maps for VAEs/text encoders/WAN GGUF so UI selections
resolve to backend SHA-based assets (no raw-path inputs).

Symbols (top-level; keep in sync; no ghosts):
- `useQuicksettingsStore` (store): Pinia store that owns QuickSettings state + actions; includes nested loaders (`loadModels/loadVaes/...`),
  setters that call API updates, and resolvers that map UI labels → inventory SHA (`resolve*Sha` helpers).
*/

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ModelInfo, SamplerInfo, SchedulerInfo } from '../api/types'
import { fetchModels, refreshModels, fetchSamplers, fetchSchedulers, fetchOptions, updateOptions, fetchVaes, fetchTextEncoders, fetchMemory, fetchModelInventory } from '../api/client'

export const useQuicksettingsStore = defineStore('quicksettings', () => {
  const models = ref<ModelInfo[]>([])
  const samplers = ref<SamplerInfo[]>([])
  const schedulers = ref<SchedulerInfo[]>([])
  const currentModel = ref<string>('')
  const currentSampler = ref<string>('')
  const currentScheduler = ref<string>('')
  const currentSeed = ref<number | string>('-1')
  const currentEngine = ref<string>('sd15')
  const currentMode = ref<string>('Normal')
  const vaeChoices = ref<string[]>([])
  const currentVae = ref<string>('Automatic')
  const textEncoderChoices = ref<string[]>([])
  const currentTextEncoders = ref<string[]>([])
  // SHA256 lookup maps for text encoders and VAEs (populated from inventory)
  const textEncoderShaMap = ref<Map<string, string>>(new Map())
  const vaeShaMap = ref<Map<string, string>>(new Map())
  const wanGgufShaMap = ref<Map<string, string>>(new Map())
  const attentionChoices = ref<{ value: string; label: string }[]>([
    { value: 'torch-sdpa', label: 'Torch (SDPA)' },
    { value: 'xformers', label: 'xFormers' },
    { value: 'sage', label: 'SAGE' },
  ])
  const currentAttention = ref<string>('torch-sdpa')
  const deviceChoices = ref<{ value: string; label: string }[]>([
    { value: 'cuda', label: 'CUDA' },
    { value: 'cpu', label: 'CPU' },
    { value: 'mps', label: 'MPS' },
    { value: 'xpu', label: 'XPU' },
    { value: 'directml', label: 'DirectML' },
  ])
  const currentDevice = ref<string>('cuda')
  const coreDevice = ref<string>('auto')
  const teDevice = ref<string>('auto')
  const vaeDevice = ref<string>('auto')
  const dtypeChoices = ref<string[]>(['auto', 'fp16', 'bf16', 'fp32'])
  const coreDtype = ref<string>('auto')
  const teDtype = ref<string>('auto')
  const vaeDtype = ref<string>('auto')
  const unetDtypeChoices = ref<string[]>([
    'Automatic',
    'Automatic (fp16 LoRA)',
    'bnb-nf4',
    'bnb-nf4 (fp16 LoRA)',
    'float8-e4m3fn',
    'float8-e4m3fn (fp16 LoRA)',
    'bnb-fp4',
    'bnb-fp4 (fp16 LoRA)',
    'float8-e5m2',
    'float8-e5m2 (fp16 LoRA)'
  ])
  const currentUnetDtype = ref<string>('Automatic')
  const gpuTotalMb = ref<number>(12288)
  const gpuWeightsMb = ref<number>(12288)
  const smartOffload = ref<boolean>(false)
  const smartFallback = ref<boolean>(false)
  const smartCache = ref<boolean>(true)
  const coreStreaming = ref<boolean>(false)

  // Basic engine/mode options (sync with legacy fallback)
  const engineChoices = ref<string[]>(['sd15', 'sdxl', 'flux', 'kontext', 'svd', 'hunyuan_video', 'wan22'])
  const modeChoices = ref<string[]>(['Normal', 'LCM', 'Turbo', 'Lightning'])

  async function init(): Promise<void> {
    await Promise.all([
      loadModels(),
      loadSamplers(),
      loadSchedulers(),
      loadVaes(),
      loadTextEncoders(),
      loadMemory(),
      loadOptions(),
    ])
  }

  async function loadModels(): Promise<void> {
    const res = await fetchModels()
    models.value = res.models
    if (!currentModel.value && res.current) {
      currentModel.value = res.current
    }
  }

  async function refreshModelsList(): Promise<void> {
    const res = await refreshModels()
    models.value = res.models
    if (!currentModel.value && res.current) {
      currentModel.value = res.current
    }
  }

  async function loadSamplers(): Promise<void> {
    const res = await fetchSamplers()
    samplers.value = res.samplers
    if (!currentSampler.value && res.samplers.length > 0) {
      currentSampler.value = res.samplers[0].name
    }
  }

  async function loadSchedulers(): Promise<void> {
    const res = await fetchSchedulers()
    schedulers.value = res.schedulers
    if (!currentScheduler.value && res.schedulers.length > 0) {
      currentScheduler.value = res.schedulers[0].name
    }
  }

  async function loadOptions(): Promise<void> {
    const res = await fetchOptions()
    const opts = res.values
    if (typeof opts.sd_model_checkpoint === 'string') {
      currentModel.value = opts.sd_model_checkpoint
    }
    if (typeof opts.sampler_name === 'string') {
      currentSampler.value = opts.sampler_name
    }
    if (typeof opts.scheduler_name === 'string') {
      currentScheduler.value = opts.scheduler_name
    }
    if (typeof opts.seed === 'number' || typeof opts.seed === 'string') {
      currentSeed.value = opts.seed
    }
    if (typeof opts.codex_engine === 'string') {
      currentEngine.value = opts.codex_engine
      if (!engineChoices.value.includes(opts.codex_engine)) engineChoices.value.push(opts.codex_engine)
    }
    if (typeof opts.codex_mode === 'string') {
      currentMode.value = opts.codex_mode
      if (!modeChoices.value.includes(opts.codex_mode)) modeChoices.value.push(opts.codex_mode)
    }
    if (typeof opts.forge_selected_vae === 'string') {
      currentVae.value = opts.forge_selected_vae || 'Automatic'
    }
    if (Array.isArray(opts.forge_additional_modules)) {
      currentTextEncoders.value = (opts.forge_additional_modules as unknown[])
        .map((entry) => String(entry).trim())
        .filter((entry) => entry.length > 0)
    }
    if (typeof opts.forge_unet_storage_dtype === 'string') {
      currentUnetDtype.value = opts.forge_unet_storage_dtype
    }
    if (typeof (opts as any).codex_attention_backend === 'string') {
      currentAttention.value = (opts as any).codex_attention_backend
    }
    if (typeof (opts as any).codex_diffusion_device === 'string') {
      currentDevice.value = (opts as any).codex_diffusion_device
    }
    if (typeof (opts as any).codex_core_device === 'string') {
      coreDevice.value = (opts as any).codex_core_device
      currentDevice.value = coreDevice.value === 'auto' ? currentDevice.value : coreDevice.value
    }
    if (typeof (opts as any).codex_te_device === 'string') {
      teDevice.value = (opts as any).codex_te_device
    }
    if (typeof (opts as any).codex_vae_device === 'string') {
      vaeDevice.value = (opts as any).codex_vae_device
    }
    if (typeof (opts as any).codex_core_dtype === 'string') coreDtype.value = (opts as any).codex_core_dtype
    if (typeof (opts as any).codex_te_dtype === 'string') teDtype.value = (opts as any).codex_te_dtype
    if (typeof (opts as any).codex_vae_dtype === 'string') vaeDtype.value = (opts as any).codex_vae_dtype
    if (typeof opts.forge_inference_memory === 'number') {
      gpuWeightsMb.value = opts.forge_inference_memory
    }
    if (typeof (opts as any).codex_smart_offload === 'boolean') {
      smartOffload.value = (opts as any).codex_smart_offload
    }
    if (typeof (opts as any).codex_smart_fallback === 'boolean') {
      smartFallback.value = (opts as any).codex_smart_fallback
    }
    if (typeof (opts as any).codex_smart_cache === 'boolean') {
      smartCache.value = (opts as any).codex_smart_cache
    }
    if (typeof (opts as any).codex_core_streaming === 'boolean') {
      coreStreaming.value = (opts as any).codex_core_streaming
    }
  }

  async function setModel(title: string): Promise<void> {
    currentModel.value = title
    await updateOptions({ sd_model_checkpoint: title })
  }

  async function setSampler(name: string): Promise<void> {
    currentSampler.value = name
    await updateOptions({ sampler_name: name })
  }

  async function setScheduler(name: string): Promise<void> {
    currentScheduler.value = name
    await updateOptions({ scheduler_name: name })
  }

  async function setSeed(value: number | string): Promise<void> {
    currentSeed.value = value
    await updateOptions({ seed: value })
  }

  async function setEngine(name: string): Promise<void> {
    currentEngine.value = name
    await updateOptions({ codex_engine: name })
  }

  async function setMode(name: string): Promise<void> {
    currentMode.value = name
    await updateOptions({ codex_mode: name })
  }

  async function loadVaes(): Promise<void> {
    try {
      const res = await fetchVaes()
      vaeChoices.value = res.vaes
      if (res.current && !currentVae.value) currentVae.value = res.current
    } catch (e) {
      // API may not expose this yet; keep defaults
      vaeChoices.value = vaeChoices.value.length ? vaeChoices.value : ['Automatic', 'Built in', 'None']
    }
  }

  async function loadTextEncoders(): Promise<void> {
    try {
      const res = await fetchTextEncoders()
      textEncoderChoices.value = res.text_encoders
      if (Array.isArray(res.current) && currentTextEncoders.value.length === 0) {
        currentTextEncoders.value = res.current
      }
      // Also load inventory for SHA256 lookup
      try {
        const inv = await fetchModelInventory()
        const shaMap = new Map<string, string>()
        const prefixes = ['sd15', 'sdxl', 'flux', 'wan22', 'zimage']
        for (const te of inv.text_encoders || []) {
          const sha = typeof te.sha256 === 'string' ? te.sha256 : ''
          if (!sha) continue
          const name = typeof te.name === 'string' ? te.name : ''
          const rawPath = typeof te.path === 'string' ? te.path : ''
          const normPath = rawPath ? rawPath.replace(/\\+/g, '/') : ''
          const keys = new Set<string>()
          if (name) keys.add(name)
          if (rawPath) keys.add(rawPath)
          if (normPath) keys.add(normPath)
          const basename = normPath ? normPath.split('/').pop() : name
          if (basename) keys.add(basename)
          if (normPath) {
            for (const prefix of prefixes) {
              keys.add(`${prefix}/${normPath}`)
            }
          }
          for (const key of keys) {
            shaMap.set(key, sha)
          }
        }
        textEncoderShaMap.value = shaMap
        // Also populate VAE SHA map
        const vaeMap = new Map<string, string>()
        for (const v of inv.vaes || []) {
          const sha = typeof v.sha256 === 'string' ? v.sha256 : ''
          if (!sha) continue
          const name = typeof v.name === 'string' ? v.name : ''
          const rawPath = typeof v.path === 'string' ? v.path : ''
          const normPath = rawPath ? rawPath.replace(/\\+/g, '/') : ''
          const keys = new Set<string>()
          if (name) keys.add(name)
          if (rawPath) keys.add(rawPath)
          if (normPath) keys.add(normPath)
          const basename = normPath ? normPath.split('/').pop() : name
          if (basename) keys.add(basename)
          if (normPath) {
            for (const prefix of prefixes) {
              keys.add(`${prefix}/${normPath}`)
            }
          }
          for (const key of keys) {
            vaeMap.set(key, sha)
          }
        }
        vaeShaMap.value = vaeMap

        const wanMap = new Map<string, string>()
        const wanFiles = (inv as any)?.wan22?.gguf
        if (Array.isArray(wanFiles)) {
          for (const w of wanFiles) {
            const sha = typeof w?.sha256 === 'string' ? w.sha256 : ''
            if (!sha) continue
            const name = typeof w?.name === 'string' ? w.name : ''
            const rawPath = typeof w?.path === 'string' ? w.path : ''
            const normPath = rawPath ? rawPath.replace(/\\+/g, '/') : ''
            const keys = new Set<string>()
            if (name) keys.add(name)
            if (rawPath) keys.add(rawPath)
            if (normPath) keys.add(normPath)
            const basename = normPath ? normPath.split('/').pop() : name
            if (basename) keys.add(basename)
            for (const key of keys) {
              wanMap.set(key, sha)
            }
          }
        }
        wanGgufShaMap.value = wanMap
      } catch (_) {
        // Non-critical - SHA lookup will just be empty
      }
    } catch (e) {
      // Graceful when endpoint not present yet
      textEncoderChoices.value = textEncoderChoices.value.length ? textEncoderChoices.value : []
    }
  }

  function resolveTextEncoderSha(label: string | null | undefined): string | undefined {
    if (!label) return undefined
    const normalized = label.replace(/\\+/g, '/')
    const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
    const tail = normalized.split('/').pop() || ''
    return (
      textEncoderShaMap.value.get(normalized) ||
      textEncoderShaMap.value.get(withoutPrefix) ||
      textEncoderShaMap.value.get(tail)
    )
  }

  function resolveModelInfo(label: string | null | undefined): ModelInfo | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined
    if (models.value.length === 0) return undefined

    const normalized = raw.replace(/\\+/g, '/')
    const tail = normalized.split('/').pop() || ''
    const lower = raw.toLowerCase()
    const isHex = /^[0-9a-f]+$/.test(lower)
    const looksLikeShortSha = lower.length === 10 && isHex

    for (const model of models.value) {
      if (!model) continue
      const modelHash = String(model.hash || '').trim().toLowerCase()
      if (looksLikeShortSha && modelHash && modelHash === lower) return model

      if (raw === model.title || raw === model.name || raw === model.filename) return model

      const fileNorm = String(model.filename || '').replace(/\\+/g, '/')
      if (normalized && normalized === fileNorm) return model

      const fileTail = fileNorm.split('/').pop() || ''
      if (tail && (tail === model.title || tail === model.name || tail === fileTail)) return model
    }
    return undefined
  }

  function resolveModelSha(label: string | null | undefined): string | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined

    const lower = raw.toLowerCase()
    if ((lower.length === 10 || lower.length === 64) && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const model = resolveModelInfo(raw)
    const sha = model ? String(model.hash || '').trim() : ''
    return sha || undefined
  }

  function resolveVaeSha(label: string | null | undefined): string | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined

    const lower = raw.toLowerCase()
    if (lower === 'automatic' || lower === 'built in' || lower === 'built-in' || lower === 'none') {
      return undefined
    }
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const normalized = raw.replace(/\\+/g, '/')
    const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
    const tail = normalized.split('/').pop() || ''
    return (
      vaeShaMap.value.get(normalized) ||
      vaeShaMap.value.get(withoutPrefix) ||
      vaeShaMap.value.get(tail)
    )
  }

  function resolveWanGgufSha(label: string | null | undefined): string | undefined {
    const raw = String(label || '').trim()
    if (!raw) return undefined

    const lower = raw.toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    const normalized = raw.replace(/\\+/g, '/')
    const tail = normalized.split('/').pop() || ''
    return wanGgufShaMap.value.get(normalized) || wanGgufShaMap.value.get(tail)
  }

  function isModelGguf(label: string | null | undefined): boolean {
    const raw = String(label || '').trim()
    if (!raw) return false
    const lower = raw.replace(/\\+/g, '/').toLowerCase()
    if (lower.endsWith('.gguf')) return true

    const model = resolveModelInfo(raw)
    if (!model) return false
    const title = String(model.title || '').toLowerCase()
    const filename = String(model.filename || '').toLowerCase()
    return title.endsWith('.gguf') || filename.endsWith('.gguf')
  }

  async function setVae(label: string): Promise<void> {
    currentVae.value = label
    await updateOptions({ forge_selected_vae: label })
  }

  async function setTextEncoders(labels: string[]): Promise<void> {
    currentTextEncoders.value = labels.slice()
    // Look up SHA256 for each label from our inventory map
    const shas: string[] = []
    for (const label of labels) {
      const sha = resolveTextEncoderSha(label)
      if (sha) shas.push(sha)
    }
    // Labels may be backend text encoder roots or direct file paths; they are
    // persisted as-is so future overrides can resolve them centrally.
    // Also send SHA256 values for direct backend resolution.
    const opts: Record<string, unknown> = { forge_additional_modules: labels }
    if (shas.length > 0) {
      opts.forge_additional_modules_sha = shas
    }
    await updateOptions(opts)
  }

  async function loadMemory(): Promise<void> {
    try {
      const res = await fetchMemory()
      if (res.total_vram_mb && res.total_vram_mb > 0) {
        gpuTotalMb.value = res.total_vram_mb
        if (gpuWeightsMb.value > gpuTotalMb.value) gpuWeightsMb.value = gpuTotalMb.value
      }
    } catch (e) {
      // Optional endpoint; keep defaults
    }
  }

  async function setUnetDtype(name: string): Promise<void> {
    currentUnetDtype.value = name
    await updateOptions({ forge_unet_storage_dtype: name })
  }

  async function setAttentionBackend(value: string): Promise<void> {
    currentAttention.value = value
    await updateOptions({ codex_attention_backend: value })
  }

  async function setDevice(value: string): Promise<void> {
    currentDevice.value = value
    coreDevice.value = value
    await updateOptions({ codex_diffusion_device: value, codex_core_device: value })
  }

  async function setCoreDevice(value: string): Promise<void> {
    coreDevice.value = value
    await updateOptions({ codex_core_device: value })
  }

  async function setTeDevice(value: string): Promise<void> {
    teDevice.value = value
    await updateOptions({ codex_te_device: value })
  }

  async function setVaeDevice(value: string): Promise<void> {
    vaeDevice.value = value
    await updateOptions({ codex_vae_device: value })
  }

  async function setCoreDtype(value: string): Promise<void> {
    coreDtype.value = value
    await updateOptions({ codex_core_dtype: value })
  }

  async function setTeDtype(value: string): Promise<void> {
    teDtype.value = value
    await updateOptions({ codex_te_dtype: value })
  }

  async function setVaeDtype(value: string): Promise<void> {
    vaeDtype.value = value
    await updateOptions({ codex_vae_dtype: value })
  }

  async function setGpuWeightsMb(value: number): Promise<void> {
    gpuWeightsMb.value = value
    await updateOptions({ forge_inference_memory: value })
  }

  async function setSmartOffload(value: boolean): Promise<void> {
    smartOffload.value = value
    await updateOptions({ codex_smart_offload: value })
  }

  async function setSmartFallback(value: boolean): Promise<void> {
    smartFallback.value = value
    await updateOptions({ codex_smart_fallback: value })
  }

  async function setSmartCache(value: boolean): Promise<void> {
    smartCache.value = value
    await updateOptions({ codex_smart_cache: value })
  }

  async function setCoreStreaming(value: boolean): Promise<void> {
    coreStreaming.value = value
    await updateOptions({ codex_core_streaming: value })
  }

  return {
    models,
    samplers,
    schedulers,
    currentModel,
    currentSampler,
    currentScheduler,
    currentSeed,
    currentEngine,
    currentMode,
    vaeChoices,
    currentVae,
    textEncoderChoices,
    currentTextEncoders,
    engineChoices,
    modeChoices,
    attentionChoices,
    currentAttention,
    deviceChoices,
    currentDevice,
    init,
    refreshModelsList,
    setModel,
    setSampler,
    setScheduler,
    setSeed,
    setEngine,
    setMode,
    setVae,
    setTextEncoders,
    unetDtypeChoices,
    currentUnetDtype,
    setUnetDtype,
    setAttentionBackend,
    setDevice,
    gpuTotalMb,
    gpuWeightsMb,
    setGpuWeightsMb,
    coreDevice,
    teDevice,
    vaeDevice,
    coreDtype,
    teDtype,
    vaeDtype,
    dtypeChoices,
    setCoreDevice,
    setTeDevice,
    setVaeDevice,
    setCoreDtype,
    setTeDtype,
    setVaeDtype,
    smartOffload,
    smartFallback,
    smartCache,
    coreStreaming,
    setSmartOffload,
    setSmartFallback,
    setSmartCache,
    setCoreStreaming,
    // SHA maps for asset resolution
    textEncoderShaMap,
    resolveTextEncoderSha,
    resolveModelSha,
    resolveVaeSha,
    resolveWanGgufSha,
    isModelGguf,
    vaeShaMap,
    wanGgufShaMap,
  }
})
