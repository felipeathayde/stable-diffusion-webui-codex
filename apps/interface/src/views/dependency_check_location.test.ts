/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Regression tests for single Home dependency-check panel placement and tab gating wiring.
Guards the UI contract that dependency checks render only once on Home while Image/WAN tabs keep readiness gating logic.

Symbols (top-level; keep in sync; no ghosts):
- `countOccurrences` (function): Counts non-overlapping token occurrences in a source string.
*/

import { describe, expect, it } from 'vitest'
import homeSource from './Home.vue?raw'
import imageSource from './ImageModelTab.vue?raw'
import wanSource from './WANTab.vue?raw'

function countOccurrences(source: string, token: string): number {
  if (!token) return 0
  let count = 0
  let index = 0
  while (index >= 0) {
    index = source.indexOf(token, index)
    if (index < 0) break
    count += 1
    index += token.length
  }
  return count
}

describe('dependency check panel location', () => {
  it('renders one dependency check panel in Home only', () => {
    expect(countOccurrences(homeSource, '<DependencyCheckPanel')).toBe(1)
    expect(homeSource).toContain("import DependencyCheckPanel from '../components/DependencyCheckPanel.vue'")
    expect(homeSource).toContain(':error="dependencyError"')

    expect(imageSource).not.toContain('<DependencyCheckPanel')
    expect(imageSource).not.toContain("import DependencyCheckPanel from '../components/DependencyCheckPanel.vue'")

    expect(wanSource).not.toContain('<DependencyCheckPanel')
    expect(wanSource).not.toContain("import DependencyCheckPanel from '../components/DependencyCheckPanel.vue'")
  })

  it('keeps dependency readiness gating in image and WAN tabs', () => {
    expect(imageSource).toContain('const dependencyReady = computed(() => Boolean(dependencyStatus.value?.ready))')
    expect(imageSource).toContain(
      'dependencyReady.value && (params.value.useInitImage ? supportsImg2Img.value : supportsTxt2Img.value)',
    )
    expect(imageSource).toContain('if (!dependencyStatus.value) return')
    expect(imageSource).toContain('if (!dependencyStatus.value.ready) return')

    expect(wanSource).toContain('const wanDependencyReady = computed(() => Boolean(wanDependencyStatus.value?.ready))')
    expect(wanSource).toContain('const canRunGeneration = computed(() => wanDependencyReady.value && canGenerate.value)')
    expect(wanSource).toContain('if (!wanDependencyReady.value)')
  })

  it('keeps dependency loading state independent from tab creation error state', () => {
    expect(homeSource).toContain('const dependencyError = ref(\'\')')
    expect(homeSource).toContain('const dependencyLoading = computed(() => !engineCaps.loaded && !dependencyError.value)')
    expect(homeSource).not.toContain('!engineCaps.loaded && !createError.value')
  })
})
