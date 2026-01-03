/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Local presets store backed by localStorage.
Persists named parameter snapshots for txt2img/img2img (mode-scoped) and exposes helpers to list, upsert, and retrieve presets.

Symbols (top-level; keep in sync; no ghosts):
- `PresetDef` (interface): Preset definition (name + values object).
- `usePresetsStore` (const): Pinia store for listing/saving/retrieving presets.
- `loadList` (function): Loads a preset list from localStorage for a mode.
- `saveList` (function): Persists a preset list to localStorage for a mode.
*/

import { defineStore } from 'pinia'

type Mode = 'txt2img' | 'img2img'

export interface PresetDef {
  name: string
  values: Record<string, unknown>
}

function loadList(mode: Mode): PresetDef[] {
  const raw = localStorage.getItem(`presets:${mode}`)
  if (!raw) return []
  try { return JSON.parse(raw) as PresetDef[] } catch { return [] }
}

function saveList(mode: Mode, list: PresetDef[]): void {
  localStorage.setItem(`presets:${mode}`, JSON.stringify(list))
}

export const usePresetsStore = defineStore('presets', () => {
  function names(mode: Mode): string[] {
    return loadList(mode).map(p => p.name)
  }
  function upsert(mode: Mode, name: string, values: Record<string, unknown>): void {
    const list = loadList(mode)
    const idx = list.findIndex(p => p.name === name)
    if (idx >= 0) list[idx] = { name, values }
    else list.push({ name, values })
    saveList(mode, list)
  }
  function get(mode: Mode, name: string): Record<string, unknown> | null {
    const list = loadList(mode)
    const it = list.find(p => p.name === name)
    return it ? it.values : null
  }
  return { names, upsert, get }
})
