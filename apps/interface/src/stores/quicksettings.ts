/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: QuickSettings global store (models/options + asset SHA selection).
Loads lists from `/api/*`, persists option changes via `/api/options`, and maintains SHA maps for VAEs/text encoders/WAN GGUF so UI selections
resolve to backend SHA-based assets (no raw-path inputs). Also owns global component overrides (device + storage/compute dtype) applied via options,
and caches the current `/api/options` revision for generation payload contracts (`settings_revision`).
Text-encoder choices are sourced from inventory files constrained by `*_tenc` roots (not folder roots), and stale root-label overrides are
sanitized so `tenc_sha` resolution remains deterministic across families (including Anima).

Symbols (top-level; keep in sync; no ghosts):
- `useQuicksettingsStore` (store): Pinia store that owns QuickSettings state + actions; includes nested loaders (`loadModels/loadVaes/...`),
  setters that call API updates, and resolvers that map UI labels → inventory SHA (`resolve*Sha` helpers).
*/

import { defineStore } from 'pinia'
import { ref } from 'vue'
import type { ModelInfo } from '../api/types'
import { fetchModels, refreshModels, fetchOptions, updateOptions, fetchModelInventory, fetchPaths, getCachedOptionsRevision } from '../api/client'

const TEXT_ENCODER_OVERRIDES_STORAGE_KEY = 'codex.quicksettings.text_encoder_overrides'
const DEVICE_STORAGE_KEY = 'codex.quicksettings.device'
const VAE_STORAGE_KEY = 'codex.quicksettings.vae'

const TEXT_ENCODER_FAMILY_KEYS: Array<[string, string]> = [
  ['sd15', 'sd15_tenc'],
  ['sdxl', 'sdxl_tenc'],
  ['flux1', 'flux1_tenc'],
  ['anima', 'anima_tenc'],
  ['wan22', 'wan22_tenc'],
  ['zimage', 'zimage_tenc'],
]

const TEXT_ENCODER_PREFIXES = ['sd15', 'sdxl', 'flux1', 'anima', 'chroma', 'wan22', 'zimage']

function normalizeRevision(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return Math.max(0, Math.trunc(value))
  }
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    if (/^-?\d+$/.test(trimmed)) {
      return Math.max(0, Math.trunc(Number(trimmed)))
    }
  }
  return null
}

function normalizePath(raw: string): string {
  const normalized = String(raw || '').trim().replace(/\\+/g, '/')
  if (normalized.length <= 1) return normalized
  return normalized.replace(/\/+$/g, '')
}

function pathMatchesRoot(filePath: string, rootPath: string): boolean {
  if (!filePath || !rootPath) return false
  const fileNorm = normalizePath(filePath)
  const rootNorm = normalizePath(rootPath)
  if (!fileNorm || !rootNorm) return false
  const candidates = new Set<string>()
  candidates.add(rootNorm)
  if (rootNorm.startsWith('/')) {
    candidates.add(rootNorm.slice(1))
    const modelsIdx = rootNorm.lastIndexOf('/models/')
    if (modelsIdx >= 0) {
      candidates.add(rootNorm.slice(modelsIdx + 1))
    }
  }

  for (const candidate of candidates) {
    if (!candidate) continue
    if (fileNorm === candidate || fileNorm.startsWith(candidate + '/')) return true
    if (fileNorm.includes('/' + candidate + '/') || fileNorm.endsWith('/' + candidate)) return true
  }
  return false
}

function lookupTextEncoderShaFromMap(map: Map<string, string>, label: string): string | undefined {
  const normalized = normalizePath(label)
  if (!normalized) return undefined
  const withoutPrefix = normalized.includes('/') ? normalized.split('/').slice(1).join('/') : normalized
  const tail = normalized.split('/').pop() || ''
  return map.get(normalized) || map.get(withoutPrefix) || map.get(tail)
}

export const useQuicksettingsStore = defineStore('quicksettings', () => {
  const models = ref<ModelInfo[]>([])
  const currentModel = ref<string>('')
  const vaeChoices = ref<string[]>([])
  const currentVae = ref<string>('Automatic')
  const textEncoderChoices = ref<string[]>([])
  const currentTextEncoders = ref<string[]>([])
  const textEncoderRootLabels = ref<Set<string>>(new Set())
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
  const coreComputeDtype = ref<string>('auto')
  const teDtype = ref<string>('auto')
  const teComputeDtype = ref<string>('auto')
  const vaeDtype = ref<string>('auto')
  const vaeComputeDtype = ref<string>('auto')
  const smartOffload = ref<boolean>(false)
  const smartFallback = ref<boolean>(false)
  const smartCache = ref<boolean>(true)
  const coreStreaming = ref<boolean>(false)
  const settingsRevision = ref<number>(Math.max(0, Math.trunc(getCachedOptionsRevision())))
  const lastAppliedNowMessages = ref<string[]>([])
  const lastRestartRequiredMessages = ref<string[]>([])

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

  function sanitizeTextEncoderOverrides(): void {
    if (textEncoderRootLabels.value.size === 0 && textEncoderShaMap.value.size === 0) return
    const next: string[] = []
    const seen = new Set<string>()
    const canValidateSha = textEncoderShaMap.value.size > 0

    for (const entry of currentTextEncoders.value) {
      const raw = String(entry || '').trim()
      if (!raw) continue
      const normalized = normalizePath(raw)
      if (!normalized) continue
      const lower = normalized.toLowerCase()
      const isSha = lower.length === 64 && /^[0-9a-f]+$/.test(lower)
      if (isSha) {
        if (!seen.has(lower)) {
          seen.add(lower)
          next.push(lower)
        }
        continue
      }

      const resolvedSha = lookupTextEncoderShaFromMap(textEncoderShaMap.value, normalized)
      if (textEncoderRootLabels.value.has(normalized) && !resolvedSha) continue

      if (!canValidateSha || resolvedSha) {
        if (!seen.has(normalized)) {
          seen.add(normalized)
          next.push(normalized)
        }
      }
    }

    const changed =
      next.length !== currentTextEncoders.value.length ||
      next.some((label, index) => label !== currentTextEncoders.value[index])
    if (changed) {
      currentTextEncoders.value = next
      saveTextEncoderOverridesToStorage(next)
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

  function syncSettingsRevisionFromCache(): void {
    const cached = normalizeRevision(getCachedOptionsRevision())
    if (cached !== null && cached > settingsRevision.value) {
      settingsRevision.value = cached
    }
  }

  function applySettingsRevision(value: unknown): void {
    const normalized = normalizeRevision(value)
    if (normalized !== null) settingsRevision.value = normalized
  }

  function getSettingsRevision(): number {
    syncSettingsRevisionFromCache()
    return settingsRevision.value
  }

  async function refreshSettingsRevision(fallbackRevision?: number): Promise<number> {
    try {
      const res = await fetchOptions()
      applySettingsRevision((res as any).revision ?? (res.values as any)?.codex_options_revision)
      syncSettingsRevisionFromCache()
    } catch (error) {
      const fallback = normalizeRevision(fallbackRevision)
      if (fallback !== null) {
        settingsRevision.value = fallback
      } else {
        throw error
      }
    }
    return settingsRevision.value
  }

  async function applyOptionUpdate(payload: Record<string, unknown>): Promise<void> {
    const response = await updateOptions(payload)
    applySettingsRevision((response as any).revision)
    const appliedNowRaw = (response as any).applied_now
    const restartRequiredRaw = (response as any).restart_required
    lastAppliedNowMessages.value = Array.isArray(appliedNowRaw) ? appliedNowRaw.map((item) => String(item)) : []
    lastRestartRequiredMessages.value = Array.isArray(restartRequiredRaw) ? restartRequiredRaw.map((item) => String(item)) : []
    syncSettingsRevisionFromCache()
  }

  function clearOptionApplyMessages(): void {
    lastAppliedNowMessages.value = []
    lastRestartRequiredMessages.value = []
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
    sanitizeTextEncoderOverrides()

    const res = await fetchOptions()
    const opts = res.values
    applySettingsRevision((res as any).revision ?? (opts as any)?.codex_options_revision)
    syncSettingsRevisionFromCache()
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
    if (typeof (opts as any).codex_core_compute_dtype === 'string') coreComputeDtype.value = (opts as any).codex_core_compute_dtype
    if (typeof (opts as any).codex_te_dtype === 'string') teDtype.value = (opts as any).codex_te_dtype
    if (typeof (opts as any).codex_te_compute_dtype === 'string') teComputeDtype.value = (opts as any).codex_te_compute_dtype
    if (typeof (opts as any).codex_vae_dtype === 'string') vaeDtype.value = (opts as any).codex_vae_dtype
    if (typeof (opts as any).codex_vae_compute_dtype === 'string') vaeComputeDtype.value = (opts as any).codex_vae_compute_dtype
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
  }

  async function loadTextEncoders(): Promise<void> {
    const [pathsRes, inv] = await Promise.all([fetchPaths(), fetchModelInventory()])
    const paths = ((pathsRes as any)?.paths || {}) as Record<string, string[]>

    const rootsByFamily = new Map<string, string[]>()
    const rootLabels = new Set<string>()
    for (const [family, key] of TEXT_ENCODER_FAMILY_KEYS) {
      const roots = (Array.isArray(paths[key]) ? paths[key] : [])
        .map((entry) => normalizePath(String(entry || '')))
        .filter((entry) => entry.length > 0)
      rootsByFamily.set(family, roots)
      for (const root of roots) {
        rootLabels.add(`${family}/${root}`)
      }
    }
    textEncoderRootLabels.value = rootLabels

    const labels = new Set<string>()
    const shaMap = new Map<string, string>()
    for (const te of inv.text_encoders || []) {
      const sha = typeof te.sha256 === 'string' ? te.sha256.trim().toLowerCase() : ''
      if (!sha || !/^[0-9a-f]{64}$/.test(sha)) continue
      const name = typeof te.name === 'string' ? te.name.trim() : ''
      const rawPath = typeof te.path === 'string' ? te.path.trim() : ''
      const normPath = normalizePath(rawPath)
      const basename = normPath ? normPath.split('/').pop() || '' : ''

      const matchedFamilies: string[] = []
      if (normPath) {
        for (const [family] of TEXT_ENCODER_FAMILY_KEYS) {
          const roots = rootsByFamily.get(family) || []
          if (roots.some((root) => pathMatchesRoot(normPath, root))) {
            matchedFamilies.push(family)
          }
        }
      }
      for (const family of matchedFamilies) {
        labels.add(`${family}/${normPath}`)
      }

      const mapKeys = new Set<string>()
      if (name) mapKeys.add(name)
      if (rawPath) mapKeys.add(rawPath)
      if (normPath) mapKeys.add(normPath)
      if (basename) mapKeys.add(basename)
      if (normPath) {
        for (const prefix of TEXT_ENCODER_PREFIXES) {
          mapKeys.add(`${prefix}/${normPath}`)
        }
      }
      for (const key of mapKeys) {
        shaMap.set(key, sha)
      }
    }
    textEncoderChoices.value = Array.from(labels).sort()
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
        for (const prefix of TEXT_ENCODER_PREFIXES) {
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
    sanitizeTextEncoderOverrides()
  }

  function resolveTextEncoderSha(label: string | null | undefined): string | undefined {
    if (!label) return undefined
    const normalized = label.replace(/\\+/g, '/')

    const lower = normalized.trim().toLowerCase()
    if (lower.length === 64 && /^[0-9a-f]+$/.test(lower)) {
      return lower
    }

    return lookupTextEncoderShaFromMap(textEncoderShaMap.value, normalized)
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
    await applyOptionUpdate({ codex_attention_backend: value })
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
    await applyOptionUpdate({ codex_core_device: value })
  }

  async function setTeDevice(value: string): Promise<void> {
    teDevice.value = value
    await applyOptionUpdate({ codex_te_device: value })
  }

  async function setVaeDevice(value: string): Promise<void> {
    vaeDevice.value = value
    await applyOptionUpdate({ codex_vae_device: value })
  }

  async function setCoreDtype(value: string): Promise<void> {
    coreDtype.value = value
    await applyOptionUpdate({ codex_core_dtype: value })
  }

  async function setCoreComputeDtype(value: string): Promise<void> {
    coreComputeDtype.value = value
    await applyOptionUpdate({ codex_core_compute_dtype: value })
  }

  async function setTeDtype(value: string): Promise<void> {
    teDtype.value = value
    await applyOptionUpdate({ codex_te_dtype: value })
  }

  async function setTeComputeDtype(value: string): Promise<void> {
    teComputeDtype.value = value
    await applyOptionUpdate({ codex_te_compute_dtype: value })
  }

  async function setVaeDtype(value: string): Promise<void> {
    vaeDtype.value = value
    await applyOptionUpdate({ codex_vae_dtype: value })
  }

  async function setVaeComputeDtype(value: string): Promise<void> {
    vaeComputeDtype.value = value
    await applyOptionUpdate({ codex_vae_compute_dtype: value })
  }

  async function setSmartOffload(value: boolean): Promise<void> {
    smartOffload.value = value
    await applyOptionUpdate({ codex_smart_offload: value })
  }

  async function setSmartFallback(value: boolean): Promise<void> {
    smartFallback.value = value
    await applyOptionUpdate({ codex_smart_fallback: value })
  }

  async function setSmartCache(value: boolean): Promise<void> {
    smartCache.value = value
    await applyOptionUpdate({ codex_smart_cache: value })
  }

  async function setCoreStreaming(value: boolean): Promise<void> {
    coreStreaming.value = value
    await applyOptionUpdate({ codex_core_streaming: value })
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
    coreComputeDtype,
    teDtype,
    teComputeDtype,
    vaeDtype,
    vaeComputeDtype,
    dtypeChoices,
    setCoreDevice,
    setTeDevice,
    setVaeDevice,
    setCoreDtype,
    setCoreComputeDtype,
    setTeDtype,
    setTeComputeDtype,
    setVaeDtype,
    setVaeComputeDtype,
    smartOffload,
    smartFallback,
    smartCache,
    coreStreaming,
    settingsRevision,
    lastAppliedNowMessages,
    lastRestartRequiredMessages,
    setSmartOffload,
    setSmartFallback,
    setSmartCache,
    setCoreStreaming,
    getSettingsRevision,
    refreshSettingsRevision,
    clearOptionApplyMessages,
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
