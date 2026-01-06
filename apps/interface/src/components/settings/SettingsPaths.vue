<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Settings panel for model search paths (`/api/paths`).
Edits engine-specific checkpoint/VAE/LoRA/text-encoder roots and persists them via the backend paths API, using `PathList` to manage per-key lists.

Symbols (top-level; keep in sync; no ghosts):
- `SettingsPaths` (component): Settings panel for model and asset path roots.
- `getList` (function): Normalizes a raw key list from the backend paths payload.
- `reload` (function): Fetches and populates current paths from the backend.
- `save` (function): Persists the edited paths back to the backend.
-->

<template>
  <div class="space-y-4">
    <div class="panel-section">
      <h3 class="label-muted">SD 1.5</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.sd15.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.sd15.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.sd15.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.sd15.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">SDXL</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.sdxl.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.sdxl.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.sdxl.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.sdxl.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">FLUX.1</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.flux1.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.flux1.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.flux1.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.flux1.tenc" />
        </div>
      </div>
    </div>

    <div class="panel-section">
      <h3 class="label-muted">WAN22</h3>
      <div class="space-y-2">
        <div>
          <label class="label-muted">Checkpoints</label>
          <PathList v-model="paths.wan22.ckpt" />
        </div>
        <div>
          <label class="label-muted">VAE</label>
          <PathList v-model="paths.wan22.vae" />
        </div>
        <div>
          <label class="label-muted">LoRA</label>
          <PathList v-model="paths.wan22.loras" />
        </div>
        <div>
          <label class="label-muted">Text Encoders</label>
          <PathList v-model="paths.wan22.tenc" />
        </div>
      </div>
    </div>

    <div class="settings-paths-actions">
      <button class="btn btn-md btn-outline" type="button" @click="reload">Reload</button>
      <button class="btn btn-md btn-primary" type="button" @click="save">Save</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive } from 'vue'
import { fetchPaths, updatePaths } from '../../api/client'
import PathList from './widgets/PathList.vue'

type EngineId = 'sd15' | 'sdxl' | 'flux1' | 'wan22'
type EnginePaths = { ckpt: string[]; vae: string[]; loras: string[]; tenc: string[] }
type EnginePathsState = Record<EngineId, EnginePaths>
type RawPaths = Record<string, string[]>

const paths = reactive<EnginePathsState>({
  sd15: { ckpt: [], vae: [], loras: [], tenc: [] },
  sdxl: { ckpt: [], vae: [], loras: [], tenc: [] },
  flux1: { ckpt: [], vae: [], loras: [], tenc: [] },
  wan22: { ckpt: [], vae: [], loras: [], tenc: [] },
})

const rawPaths = reactive<RawPaths>({})

function getList(raw: RawPaths, key: string): string[] {
  const value = raw[key]
  return Array.isArray(value) ? [...value] : []
}

async function reload(): Promise<void> {
  try {
    const res = await fetchPaths()
    const loaded = (res.paths || {}) as RawPaths

    // Reset rawPaths and repopulate.
    for (const key of Object.keys(rawPaths)) {
      delete rawPaths[key]
    }
    for (const [key, value] of Object.entries(loaded)) {
      rawPaths[key] = Array.isArray(value) ? [...value] : []
    }

    paths.sd15.ckpt = getList(loaded, 'sd15_ckpt')
    paths.sd15.vae = getList(loaded, 'sd15_vae')
    paths.sd15.loras = getList(loaded, 'sd15_loras')
    paths.sd15.tenc = getList(loaded, 'sd15_tenc')

    paths.sdxl.ckpt = getList(loaded, 'sdxl_ckpt')
    paths.sdxl.vae = getList(loaded, 'sdxl_vae')
    paths.sdxl.loras = getList(loaded, 'sdxl_loras')
    paths.sdxl.tenc = getList(loaded, 'sdxl_tenc')

    paths.flux1.ckpt = getList(loaded, 'flux1_ckpt')
    paths.flux1.vae = getList(loaded, 'flux1_vae')
    paths.flux1.loras = getList(loaded, 'flux1_loras')
    paths.flux1.tenc = getList(loaded, 'flux1_tenc')

    paths.wan22.ckpt = getList(loaded, 'wan22_ckpt')
    paths.wan22.vae = getList(loaded, 'wan22_vae')
    paths.wan22.loras = getList(loaded, 'wan22_loras')
    paths.wan22.tenc = getList(loaded, 'wan22_tenc')
  } catch {
    // Keep existing state on failure; errors are surfaced elsewhere.
  }
}

async function save(): Promise<void> {
  const next: RawPaths = {}

  // Preserve non-aggregated keys not managed explicitly here.
  for (const [key, value] of Object.entries(rawPaths)) {
    if (key === 'checkpoints' || key === 'vae' || key === 'lora' || key === 'text_encoders') continue
    next[key] = Array.isArray(value) ? [...value] : []
  }

  next.sd15_ckpt = [...paths.sd15.ckpt]
  next.sd15_vae = [...paths.sd15.vae]
  next.sd15_loras = [...paths.sd15.loras]
  next.sd15_tenc = [...paths.sd15.tenc]

  next.sdxl_ckpt = [...paths.sdxl.ckpt]
  next.sdxl_vae = [...paths.sdxl.vae]
  next.sdxl_loras = [...paths.sdxl.loras]
  next.sdxl_tenc = [...paths.sdxl.tenc]

  next.flux1_ckpt = [...paths.flux1.ckpt]
  next.flux1_vae = [...paths.flux1.vae]
  next.flux1_loras = [...paths.flux1.loras]
  next.flux1_tenc = [...paths.flux1.tenc]

  next.wan22_ckpt = [...paths.wan22.ckpt]
  next.wan22_vae = [...paths.wan22.vae]
  next.wan22_loras = [...paths.wan22.loras]
  next.wan22_tenc = [...paths.wan22.tenc]

  await updatePaths(next)
}

onMounted(() => {
  void reload()
})
</script>
