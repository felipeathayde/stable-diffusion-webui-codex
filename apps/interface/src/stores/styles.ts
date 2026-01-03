/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Prompt styles store backed by localStorage.
Persists named prompt/negative prompt style snippets and exposes helpers to list, retrieve, and create/update styles.

Symbols (top-level; keep in sync; no ghosts):
- `StyleDef` (interface): Style definition (name + prompt + negative).
- `useStylesStore` (const): Pinia store for prompt styles.
- `load` (function): Loads styles list from localStorage.
- `save` (function): Persists styles list to localStorage.
*/

import { defineStore } from 'pinia'

export interface StyleDef { name: string; prompt: string; negative: string }

function load(): StyleDef[] {
  const raw = localStorage.getItem('styles:list')
  if (!raw) return []
  try { return JSON.parse(raw) as StyleDef[] } catch { return [] }
}
function save(list: StyleDef[]): void {
  localStorage.setItem('styles:list', JSON.stringify(list))
}

export const useStylesStore = defineStore('styles', {
  state: () => ({ list: load() as StyleDef[] }),
  actions: {
    names(): string[] { return this.list.map(s => s.name) },
    create(def: StyleDef): void {
      const idx = this.list.findIndex(s => s.name === def.name)
      if (idx >= 0) this.list[idx] = def; else this.list.push(def)
      save(this.list)
    },
    get(name: string): StyleDef | undefined { return this.list.find(s => s.name === name) },
  },
})
