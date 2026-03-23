/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Pinia store for backend engine capability gating.
Fetches `/api/engines/capabilities` and exposes cached capability + family + asset-contract + backend-owned dependency-check maps so views/components can gate
UI features, required asset selection, family-specific behavior, readiness indicators, family-scoped sampler/scheduler filtering, and the LTX-only
execution-profile/default surface from a single contract surface.

Symbols (top-level; keep in sync; no ghosts):
- `asEngineDependencyCheckRow` (function): Validates/coerces one dependency-check row from unknown payload data.
- `asEngineDependencyStatus` (function): Validates/coerces one dependency status payload per semantic engine.
- `parseDependencyChecks` (function): Parses strict `dependency_checks` map from capabilities response.
- `parseEngineIdToSemanticMap` (function): Parses strict `engine_id_to_semantic_engine` map from capabilities response.
- `asLtxExecutionSurface` (function): Parses the optional nested LTX execution-profile/default surface from one engine capability row.
- `filterSamplersForFamilyCapabilities` (function): Applies family `supported_samplers`/`excluded_samplers` constraints to executable sampler rows.
- `filterSchedulersForFamilyCapabilities` (function): Applies family `supported_schedulers`/`excluded_schedulers` constraints to executable scheduler rows.
- `filterSchedulersForSampler` (function): Filters scheduler rows by sampler `allowed_schedulers` compatibility.
- `normalizeSamplerSchedulerSelection` (function): Resolves a valid sampler/scheduler pair against executable catalogs + family + sampler compatibility constraints.
- `parseFamilyCapabilities` (function): Parses strict `families` capability map from capabilities response.
- `useEngineCapabilitiesStore` (store): Pinia store exposing engine capabilities, load state, and lookup helpers (including `getLtxExecutionSurface(...)`).
*/

import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type {
  EngineAssetContract,
  EngineAssetContractVariants,
  EngineCapabilitiesResponse,
  EngineCapabilities,
  FamilyCapabilities,
  EngineDependencyStatus,
  EngineDependencyCheckRow,
  LtxExecutionSurface,
  SamplerInfo,
  SchedulerInfo,
} from '../api/types'
import { fetchEngineCapabilities } from '../api/client'
import {
  KNOWN_ENGINE_IDS,
  normalizeSemanticEngine,
  resolveSemanticEngineForEngineId,
  type SamplingDefaults,
} from '../utils/engine_taxonomy'

const CAPABILITIES_CONTRACT_ERROR_PREFIX = "Invalid '/api/engines/capabilities' response:"

type FamilySamplingListKey =
  | 'supported_samplers'
  | 'supported_schedulers'
  | 'excluded_samplers'
  | 'excluded_schedulers'

function toUniqueNonEmptyStrings(values: Array<string | null | undefined>): string[] {
  const out: string[] = []
  const seen = new Set<string>()
  for (const rawValue of values) {
    const value = String(rawValue || '').trim()
    if (!value || seen.has(value)) continue
    seen.add(value)
    out.push(value)
  }
  return out
}

function parseFamilySamplingList(
  row: Record<string, unknown>,
  family: string,
  field: FamilySamplingListKey,
): string[] | null | undefined {
  const raw = row[field]
  if (typeof raw === 'undefined') return undefined
  if (raw === null) return null
  if (!Array.isArray(raw)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-array '${field}'.`)
  }
  const normalized: string[] = []
  const seen = new Set<string>()
  for (const [index, entry] of raw.entries()) {
    if (typeof entry !== 'string') {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-string '${field}[${index}]'.`,
      )
    }
    const value = entry.trim()
    if (!value) {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has empty '${field}[${index}]'.`,
      )
    }
    if (seen.has(value)) continue
    seen.add(value)
    normalized.push(value)
  }
  return normalized
}

function capabilitySet(values: string[] | null | undefined): Set<string> | null {
  if (!Array.isArray(values)) return null
  if (values.length === 0) return null
  return new Set(values)
}

function asLtxExecutionSurface(
  value: unknown,
  engine: string,
): LtxExecutionSurface | null {
  if (value == null) return null
  if (typeof value !== 'object' || Array.isArray(value)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has non-object 'ltx_execution_surface'.`)
  }
  const row = value as Record<string, unknown>
  const allowedExecutionProfiles = row.allowed_execution_profiles
  const defaultExecutionProfile = row.default_execution_profile
  const defaultStepsByProfile = row.default_steps_by_profile
  const defaultGuidanceByProfile = row.default_guidance_scale_by_profile
  if (!Array.isArray(allowedExecutionProfiles) || allowedExecutionProfiles.some((entry) => typeof entry !== 'string' || !entry.trim())) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.allowed_execution_profiles'.`)
  }
  if (typeof defaultExecutionProfile !== 'string' || !defaultExecutionProfile.trim()) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.default_execution_profile'.`)
  }
  if (defaultStepsByProfile === null || typeof defaultStepsByProfile !== 'object' || Array.isArray(defaultStepsByProfile)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.default_steps_by_profile'.`)
  }
  if (defaultGuidanceByProfile === null || typeof defaultGuidanceByProfile !== 'object' || Array.isArray(defaultGuidanceByProfile)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid 'ltx_execution_surface.default_guidance_scale_by_profile'.`)
  }
  const stepsOut: Record<string, number> = {}
  for (const [profile, rawValue] of Object.entries(defaultStepsByProfile as Record<string, unknown>)) {
    const normalizedProfile = String(profile || '').trim()
    if (!normalizedProfile) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has empty LTX profile key in 'default_steps_by_profile'.`)
    }
    if (typeof rawValue !== 'number' || !Number.isFinite(rawValue) || !Number.isInteger(rawValue) || rawValue <= 0) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid default steps for LTX profile '${normalizedProfile}'.`)
    }
    stepsOut[normalizedProfile] = rawValue
  }
  const guidanceOut: Record<string, number> = {}
  for (const [profile, rawValue] of Object.entries(defaultGuidanceByProfile as Record<string, unknown>)) {
    const normalizedProfile = String(profile || '').trim()
    if (!normalizedProfile) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has empty LTX profile key in 'default_guidance_scale_by_profile'.`)
    }
    if (typeof rawValue !== 'number' || !Number.isFinite(rawValue) || rawValue < 0) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine '${engine}' has invalid default guidance for LTX profile '${normalizedProfile}'.`)
    }
    guidanceOut[normalizedProfile] = rawValue
  }
  return {
    allowed_execution_profiles: allowedExecutionProfiles.map((entry) => String(entry).trim()),
    default_execution_profile: defaultExecutionProfile.trim(),
    default_steps_by_profile: stepsOut,
    default_guidance_scale_by_profile: guidanceOut,
  }
}

export function filterSamplersForFamilyCapabilities(
  samplers: SamplerInfo[],
  familyCapabilities: FamilyCapabilities | null | undefined,
): SamplerInfo[] {
  const supported = capabilitySet(familyCapabilities?.supported_samplers)
  const excluded = capabilitySet(familyCapabilities?.excluded_samplers)
  return samplers.filter((entry) => {
    if (supported && !supported.has(entry.name)) return false
    if (excluded && excluded.has(entry.name)) return false
    return true
  })
}

export function filterSchedulersForFamilyCapabilities(
  schedulers: SchedulerInfo[],
  familyCapabilities: FamilyCapabilities | null | undefined,
): SchedulerInfo[] {
  const supported = capabilitySet(familyCapabilities?.supported_schedulers)
  const excluded = capabilitySet(familyCapabilities?.excluded_schedulers)
  return schedulers.filter((entry) => {
    if (supported && !supported.has(entry.name)) return false
    if (excluded && excluded.has(entry.name)) return false
    return true
  })
}

export function filterSchedulersForSampler(
  schedulers: SchedulerInfo[],
  sampler: SamplerInfo | null | undefined,
): SchedulerInfo[] {
  if (!sampler) return schedulers.slice()
  const allowed = Array.isArray(sampler.allowed_schedulers)
    ? sampler.allowed_schedulers.map((entry) => String(entry || '').trim()).filter((entry) => entry.length > 0)
    : []
  if (allowed.length === 0) return schedulers.slice()
  const allowedSet = new Set(allowed)
  return schedulers.filter((entry) => allowedSet.has(entry.name))
}

export function normalizeSamplerSchedulerSelection(opts: {
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  familyCapabilities: FamilyCapabilities | null | undefined
  sampler: string | null | undefined
  scheduler: string | null | undefined
  preferredSamplers?: Array<string | null | undefined>
  preferredSchedulers?: Array<string | null | undefined>
}): { sampler: string; scheduler: string } | null {
  const familySamplers = filterSamplersForFamilyCapabilities(opts.samplers, opts.familyCapabilities)
  const familySchedulers = filterSchedulersForFamilyCapabilities(opts.schedulers, opts.familyCapabilities)
  if (familySamplers.length === 0 || familySchedulers.length === 0) return null

  const samplerByName = new Map(familySamplers.map((entry) => [entry.name, entry]))
  const samplerCandidates = toUniqueNonEmptyStrings([
    opts.sampler,
    ...(opts.preferredSamplers ?? []),
    ...familySamplers.map((entry) => entry.name),
  ])

  for (const candidateSampler of samplerCandidates) {
    const samplerSpec = samplerByName.get(candidateSampler)
    if (!samplerSpec) continue
    const allowedSchedulers = filterSchedulersForSampler(familySchedulers, samplerSpec)
    if (allowedSchedulers.length === 0) continue
    const allowedSchedulerSet = new Set(allowedSchedulers.map((entry) => entry.name))
    const schedulerCandidates = toUniqueNonEmptyStrings([
      opts.scheduler,
      ...(opts.preferredSchedulers ?? []),
      samplerSpec.default_scheduler,
      ...allowedSchedulers.map((entry) => entry.name),
    ])
    const scheduler = schedulerCandidates.find((candidate) => allowedSchedulerSet.has(candidate))
    if (!scheduler) continue
    return {
      sampler: samplerSpec.name,
      scheduler,
    }
  }
  return null
}

function asEngineDependencyCheckRow(
  value: unknown,
  context: { index: number; engine: string },
): EngineDependencyCheckRow {
  const { index, engine } = context
  if (value === null || typeof value !== 'object') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row #${index + 1} for '${engine}' must be an object.`)
  }
  const row = value as Record<string, unknown>
  const id = typeof row.id === 'string' ? row.id.trim() : ''
  const label = typeof row.label === 'string' ? row.label.trim() : ''
  const message = typeof row.message === 'string' ? row.message.trim() : ''
  if (!id) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row #${index + 1} for '${engine}' has missing 'id'.`)
  }
  if (!label) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has missing 'label'.`)
  }
  if (typeof row.ok !== 'boolean') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has non-boolean 'ok'.`)
  }
  if (!message) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency check row '${id}' for '${engine}' has missing 'message'.`)
  }
  return { id, label, ok: row.ok, message }
}

function asEngineDependencyStatus(
  value: unknown,
  context: { engine: string },
): EngineDependencyStatus {
  const { engine } = context
  if (value === null || typeof value !== 'object') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' must be an object.`)
  }
  const status = value as Record<string, unknown>
  if (typeof status.ready !== 'boolean') {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' has non-boolean 'ready'.`)
  }
  if (!Array.isArray(status.checks)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' has missing 'checks' array.`)
  }
  const checks = status.checks.map((row, index) => asEngineDependencyCheckRow(row, { index, engine }))
  const derivedReady = checks.every((row) => row.ok)
  if (derivedReady !== status.ready) {
    throw new Error(
      `${CAPABILITIES_CONTRACT_ERROR_PREFIX} dependency status for '${engine}' is inconsistent (ready=${String(status.ready)} but checks imply ready=${String(derivedReady)}).`,
    )
  }
  return { ready: status.ready, checks }
}

function parseDependencyChecks(payload: unknown): Record<string, EngineDependencyStatus> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'dependency_checks' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, EngineDependencyStatus> = {}
  for (const [engine, status] of Object.entries(raw)) {
    out[engine] = asEngineDependencyStatus(status, { engine })
  }
  return out
}

function parseEngineIdToSemanticMap(payload: unknown): Record<string, string> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'engine_id_to_semantic_engine' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, string> = {}
  for (const [keyRaw, valueRaw] of Object.entries(raw)) {
    const engineId = String(keyRaw || '').trim().toLowerCase()
    const semanticRaw = String(valueRaw || '').trim()
    if (!engineId) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} engine_id_to_semantic_engine has an empty key.`)
    }
    const semantic = normalizeSemanticEngine(semanticRaw)
    if (!semantic) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} invalid semantic engine '${semanticRaw}' for engine id '${engineId}'.`)
    }
    out[engineId] = semantic
  }
  for (const engineId of KNOWN_ENGINE_IDS) {
    resolveSemanticEngineForEngineId(engineId, out)
  }
  return out
}

function parseFamilyCapabilities(payload: unknown): Record<string, FamilyCapabilities> {
  if (payload === null || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'families' object.`)
  }
  const raw = payload as Record<string, unknown>
  const out: Record<string, FamilyCapabilities> = {}
  for (const [family, value] of Object.entries(raw)) {
    if (value === null || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' must be an object.`)
    }
    const row = value as Record<string, unknown>
    const supportsNegative = row.supports_negative_prompt
    const showsClipSkip = row.shows_clip_skip
    const supportedSamplers = parseFamilySamplingList(row, family, 'supported_samplers')
    const supportedSchedulers = parseFamilySamplingList(row, family, 'supported_schedulers')
    const excludedSamplers = parseFamilySamplingList(row, family, 'excluded_samplers')
    const excludedSchedulers = parseFamilySamplingList(row, family, 'excluded_schedulers')
    if (typeof supportsNegative !== 'boolean') {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-boolean 'supports_negative_prompt'.`,
      )
    }
    if (typeof showsClipSkip !== 'boolean') {
      throw new Error(
        `${CAPABILITIES_CONTRACT_ERROR_PREFIX} family capability '${family}' has non-boolean 'shows_clip_skip'.`,
      )
    }
    out[family] = {
      supports_negative_prompt: supportsNegative,
      shows_clip_skip: showsClipSkip,
      supported_samplers: supportedSamplers,
      supported_schedulers: supportedSchedulers,
      excluded_samplers: excludedSamplers,
      excluded_schedulers: excludedSchedulers,
    }
  }
  return out
}

export const useEngineCapabilitiesStore = defineStore('engineCapabilities', () => {
  const engines = ref<Record<string, EngineCapabilities>>({})
  const families = ref<Record<string, FamilyCapabilities>>({})
  const assetContracts = ref<Record<string, EngineAssetContractVariants>>({})
  const dependencyChecks = ref<Record<string, EngineDependencyStatus>>({})
  const engineIdToSemanticEngine = ref<Record<string, string>>({})
  const loaded = ref(false)
  const loading = ref(false)
  const error = ref<string | null>(null)
  let initPromise: Promise<void> | null = null

  async function init(opts: { force?: boolean } = {}): Promise<void> {
    const force = Boolean(opts.force)
    if (!force && loaded.value) return
    if (initPromise) return initPromise

    initPromise = (async () => {
      loading.value = true
      error.value = null
      try {
        const res: EngineCapabilitiesResponse = await fetchEngineCapabilities()
        if (!res || typeof res !== 'object') {
          throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} payload must be an object.`)
        }
        if (!res.engines || typeof res.engines !== 'object' || Array.isArray(res.engines)) {
          throw new Error(`${CAPABILITIES_CONTRACT_ERROR_PREFIX} missing 'engines' object.`)
        }
        const nextDependencyChecks = parseDependencyChecks(res.dependency_checks)
        const nextEngineMap = parseEngineIdToSemanticMap(res.engine_id_to_semantic_engine)
        const nextFamilies = parseFamilyCapabilities(res.families)

        engines.value = res.engines
        families.value = nextFamilies
        assetContracts.value = res.asset_contracts ?? {}
        dependencyChecks.value = nextDependencyChecks
        engineIdToSemanticEngine.value = nextEngineMap
        loaded.value = true
      } catch (e: unknown) {
        const message = e instanceof Error ? e.message : String(e)
        error.value = message
        loaded.value = false
        throw e
      } finally {
        loading.value = false
      }
    })()

    try {
      await initPromise
    } finally {
      initPromise = null
      }
    }

  function semanticEngineForId(engineId: string | null | undefined): string | null {
    if (!engineId) return null
    return resolveSemanticEngineForEngineId(engineId, engineIdToSemanticEngine.value)
  }

  function get(engine: string | null | undefined): EngineCapabilities | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return engines.value[semantic] ?? null
  }

  function getAssetVariants(engine: string | null | undefined): EngineAssetContractVariants | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return assetContracts.value[semantic] ?? null
  }

  function getDependencyStatus(engine: string | null | undefined): EngineDependencyStatus | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return dependencyChecks.value[semantic] ?? null
  }

  function getFamily(family: string | null | undefined): FamilyCapabilities | null {
    const key = String(family || '').trim().toLowerCase()
    if (!key) return null
    return families.value[key] ?? null
  }

  function getFamilyForEngine(engine: string | null | undefined): FamilyCapabilities | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return getFamily(semantic)
  }

  function firstDependencyError(engine: string | null | undefined): string {
    const status = getDependencyStatus(engine)
    if (!status) return "Dependency checks are not available for this engine."
    const first = status.checks.find((row) => !row.ok)
    return first?.message || ''
  }

  function getAssetContract(
    engine: string | null | undefined,
    opts: { checkpointCoreOnly: boolean }
  ): EngineAssetContract | null {
    const variants = getAssetVariants(engine)
    if (!variants) return null
    return opts?.checkpointCoreOnly ? variants.core_only : variants.base
  }

  function resolveSamplingDefaults(
    engineId: string | null | undefined,
    opts: { fallbackSampler: string; fallbackScheduler: string },
  ): SamplingDefaults {
    const surface = get(engineId)
    const sampler = String(surface?.default_sampler || '').trim() || opts.fallbackSampler
    const scheduler = String(surface?.default_scheduler || '').trim() || opts.fallbackScheduler
    return { sampler, scheduler }
  }

  function getLtxExecutionSurface(engine: string | null | undefined): LtxExecutionSurface | null {
    const semantic = semanticEngineForId(engine)
    if (!semantic) return null
    return asLtxExecutionSurface(engines.value[semantic]?.ltx_execution_surface, semantic)
  }

  const knownEngines = computed(() => Object.keys(engines.value))
  const notReadyEngines = computed(() =>
    Object.entries(dependencyChecks.value)
      .filter(([, status]) => !status.ready)
      .map(([engine]) => engine),
  )

  return {
    engines,
    families,
    assetContracts,
    dependencyChecks,
    engineIdToSemanticEngine,
    knownEngines,
    notReadyEngines,
    loaded,
    loading,
    error,
    init,
    semanticEngineForId,
    get,
    getFamily,
    getFamilyForEngine,
    getAssetVariants,
    getDependencyStatus,
    firstDependencyError,
    getAssetContract,
    getLtxExecutionSurface,
    resolveSamplingDefaults,
  }
})
