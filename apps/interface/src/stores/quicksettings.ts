/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: QuickSettings global store (models/options + asset SHA selection).
Loads lists from `/api/*`, persists option changes via `/api/options`, and maintains SHA maps for VAEs/text encoders/WAN GGUF so UI selections
resolve to backend SHA-based assets (no raw-path inputs). Asset lists are sourced from `/api/models/inventory` and root config from `/api/paths`.

Symbols (top-level; keep in sync; no ghosts):
- `useQuicksettingsStore` (store): Pinia store that owns QuickSettings state + actions; includes nested loaders (`loadModels/loadVaes/...`),
  setters that call API updates, and resolvers that map UI labels → inventory SHA (`resolve*Sha` helpers).
*/

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ModelInfo } from '../api/types'
import { fetchModels, refreshModels, fetchOptions, updateOptions, fetchModelInventory, fetchPaths } from '../api/client'

const TEXT_ENCODER_OVERRIDES_STORAGE_KEY = 'codex.quicksettings.text_encoder_overrides'
const DEVICE_STORAGE_KEY = 'codex.quicksettings.device'
const VAE_STORAGE_KEY = 'codex.quicksettings.vae'

export const useQuicksettingsStore = defineStore('quicksettings', () => {
  const models = ref<ModelInfo[]>([])
  const currentModel = ref<string>('')
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
  const smartOffload = ref<boolean>(false)
  const smartFallback = ref<boolean>(false)
  const smartCache = ref<boolean>(true)
  const coreStreaming = ref<boolean>(false)

  function loadTextEncoderOverridesFromStorage(): void {
    try {
      const raw = localStorage.getItem(TEXT_ENCODER_OVERRIDES_STORAGE_KEY)
      if (!raw) return
      const parsed = JSON.parse(raw)
      if (!Array.isArray(parsed)) return
      currentTextEncoders.value = parsed
        .map((entry) => String(entry).trim())
        .filter((entry) => entry.length > 0)
    } catch (err) {
      console.warn('[quicksettings] failed to load text encoder overrides from localStorage', err)
    }
  }

  function saveTextEncoderOverridesToStorage(labels: string[]): void {
    try {
      localStorage.setItem(TEXT_ENCODER_OVERRIDES_STORAGE_KEY, JSON.stringify(labels))
    } catch (err) {
      console.warn('[quicksettings] failed to persist text encoder overrides to localStorage', err)
    }
  }

  function loadDeviceFromStorage(): void {
    try {
      const raw = localStorage.getItem(DEVICE_STORAGE_KEY)
      if (!raw) return
      const normalized = String(raw).trim().toLowerCase()
      if (!normalized) return
      if (deviceChoices.value.some((d) => d.value === normalized)) {
        currentDevice.value = normalized
      }
    } catch (err) {
      console.warn('[quicksettings] failed to load device from localStorage', err)
    }
  }

  function saveDeviceToStorage(device: string): void {
    try {
      localStorage.setItem(DEVICE_STORAGE_KEY, String(device))
    } catch (err) {
      console.warn('[quicksettings] failed to persist device to localStorage', err)
    }
  }

  function loadVaeFromStorage(): void {
    try {
      const raw = localStorage.getItem(VAE_STORAGE_KEY)
      if (!raw) return
      const normalized = String(raw).trim()
      if (!normalized) return
      currentVae.value = normalized
    } catch (err) {
      console.warn('[quicksettings] failed to load VAE selection from localStorage', err)
    }
  }

  function saveVaeToStorage(label: string): void {
    try {
      localStorage.setItem(VAE_STORAGE_KEY, String(label))
    } catch (err) {
      console.warn('[quicksettings] failed to persist VAE selection to localStorage', err)
    }
  }

  async function init(): Promise<void> {
    await Promise.all([
      loadModels(),
      loadVaes(),
      loadTextEncoders(),
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

  async function loadOptions(): Promise<void> {
    loadDeviceFromStorage()
    loadVaeFromStorage()
    loadTextEncoderOverridesFromStorage()

    const res = await fetchOptions()
    const opts = res.values
    if (typeof (opts as any).codex_attention_backend === 'string') {
      currentAttention.value = (opts as any).codex_attention_backend
    }
    if (typeof (opts as any).codex_core_device === 'string') {
      coreDevice.value = (opts as any).codex_core_device
      currentDevice.value = coreDevice.value === 'auto' ? currentDevice.value : coreDevice.value
      if (coreDevice.value !== 'auto') saveDeviceToStorage(coreDevice.value)
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
  }

  async function loadVaes(): Promise<void> {
    try {
      const inv = await fetchModelInventory()
      const seen = new Set<string>()
      const out: string[] = ['Automatic', 'Built in', 'None']
      for (const item of (inv as any)?.vaes || []) {
        const name = String(item?.name || '').trim()
        if (!name || seen.has(name)) continue
        seen.add(name)
        if (!out.includes(name)) out.push(name)
      }
      vaeChoices.value = out
    } catch (e) {
      // API may not expose this yet; keep defaults
      vaeChoices.value = vaeChoices.value.length ? vaeChoices.value : ['Automatic', 'Built in', 'None']
    }
  }

  async function loadTextEncoders(): Promise<void> {
    try {
      const res = await fetchPaths()
      const paths = ((res as any)?.paths || {}) as Record<string, string[]>
      const keys: Array<[string, string]> = [
        ['sd15', 'sd15_tenc'],
        ['sdxl', 'sdxl_tenc'],
        ['flux1', 'flux1_tenc'],
        ['wan22', 'wan22_tenc'],
        ['zimage', 'zimage_tenc'],
      ]
      const labels: string[] = []
      for (const [fam, key] of keys) {
        const entries = Array.isArray(paths[key]) ? paths[key] : []
        for (const raw of entries) {
          const p = String(raw || '').trim().replace(/\\+/g, '/')
          if (!p) continue
          labels.push(`${fam}/${p}`)
        }
      }
      textEncoderChoices.value = Array.from(new Set(labels)).sort()
      // Also load inventory for SHA256 lookup
      try {
        const inv = await fetchModelInventory()
        const shaMap = new Map<string, string>()
        const prefixes = ['sd15', 'sdxl', 'flux1', 'chroma', 'wan22', 'zimage']
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

    const lower = normalized.trim().toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

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

  function isModelCoreOnly(label: string | null | undefined): boolean {
    const raw = String(label || '').trim()
    if (!raw) return false

    const model = resolveModelInfo(raw)
    if (model && typeof (model as any).core_only === 'boolean') {
      return Boolean((model as any).core_only)
    }

    // Fallback for unknown/older model shapes: `.gguf` implies core-only.
    const lower = raw.replace(/\\+/g, '/').toLowerCase()
    return lower.endsWith('.gguf')
  }

  async function setVae(label: string): Promise<void> {
    currentVae.value = label
    saveVaeToStorage(label)
  }

  async function setTextEncoders(labels: string[]): Promise<void> {
    currentTextEncoders.value = labels.slice()
    saveTextEncoderOverridesToStorage(labels)
  }

  async function setAttentionBackend(value: string): Promise<void> {
    currentAttention.value = value
    await updateOptions({ codex_attention_backend: value })
  }

  async function setDevice(value: string): Promise<void> {
    currentDevice.value = value
    saveDeviceToStorage(value)
  }

  async function setCoreDevice(value: string): Promise<void> {
    coreDevice.value = value
    if (value !== 'auto') {
      currentDevice.value = value
      saveDeviceToStorage(value)
    }
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
    currentModel,
    vaeChoices,
    currentVae,
    textEncoderChoices,
    currentTextEncoders,
    attentionChoices,
    currentAttention,
    deviceChoices,
    currentDevice,
    init,
    refreshModelsList,
    setModel,
    setVae,
    setTextEncoders,
    setAttentionBackend,
    setDevice,
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
    isModelCoreOnly,
    vaeShaMap,
    wanGgufShaMap,
  }
})
