<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: PNG info inspection view.
Prototype view for inspecting PNG metadata and managing simple presets for parsing parameters.

Symbols (top-level; keep in sync; no ghosts):
- `PngInfo` (component): PNG info route view component.
-->

<template>
  <section class="panels">
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">PNG Info</div>
        <div class="panel-body">
          <div class="panel-section">
            <label class="label-muted">Upload Image</label>
            <input type="file" class="ui-input" accept="image/png" />
          </div>
          <div class="panel-section">
            <label class="label-muted">Infotext</label>
            <textarea class="ui-textarea h-prompt-sm" placeholder="Infotext will appear here after reading PNG metadata."></textarea>
          </div>
        </div>
      </div>
    </div>

    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header three-cols results-sticky">Results
          <div class="header-center"><button class="btn btn-md btn-primary results-generate">Analyze</button></div>
          <div class="header-right results-actions">
            <input class="ui-input" list="pnginfo-preset-list" v-model="presetName" placeholder="Preset" />
            <datalist id="pnginfo-preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="savePreset(presetName)">Save</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyPreset(presetName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <div class="viewer-card">
            <div class="viewer-empty">Metadata and parsed fields will appear here.</div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { usePresetsStore } from '../stores/presets'

const presets = usePresetsStore()
const presetName = ref('')
const presetNames = computed(() => presets.names('pnginfo'))
function snapshotParams(): Record<string, unknown> { return {} }
function applyParams(_v: Record<string, unknown>): void {}
function savePreset(name: string): void { presets.upsert('pnginfo', name, snapshotParams()) }
function applyPreset(name: string): void { const v = presets.get('pnginfo', name); if (v) applyParams(v) }
</script>
