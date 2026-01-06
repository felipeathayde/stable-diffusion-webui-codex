<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: QuickSettings path-like option editor.
Surfaces a small set of backend path-like options (temp/output dirs) and persists changes immediately via `/api/options`.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsPaths` (component): Path option editor shown in QuickSettings areas.
- `init` (function): Loads current values from `/api/options`.
- `onChange` (function): Persists an updated key via `updateOptions`.
-->

<template>
  <section class="quicksettings">
    <div class="quicksettings-group" v-for="item in items" :key="item.key">
      <label class="label-muted">{{ item.label }}</label>
      <div class="qs-row">
        <input class="ui-input" type="text" :value="values[item.key] || ''" @change="onChange($event, item.key)" />
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { reactive } from 'vue'
import { fetchOptions, updateOptions } from '../api/client'

interface PathItem { key: string; label: string }

// Minimal set of path-like options exposed by backend; extend as needed
const items: PathItem[] = [
  { key: 'temp_dir', label: 'Temp Dir' },
  { key: 'outdir_samples', label: 'Samples Dir' },
  { key: 'outdir_grids', label: 'Grids Dir' },
]

const values = reactive<Record<string, string>>({})

async function init(): Promise<void> {
  const res = await fetchOptions()
  for (const it of items) {
    if (typeof res.values[it.key] === 'string') values[it.key] = res.values[it.key]
  }
}
void init()

async function onChange(event: Event, key: string): Promise<void> {
  const val = String((event.target as HTMLInputElement).value || '')
  values[key] = val
  await updateOptions({ [key]: val })
}
</script>
