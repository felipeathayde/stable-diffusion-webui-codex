<template>
  <section class="panels">
    <!-- Left column: basic controls (stub) -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Upscale</div>
        <div class="panel-body">
          <div class="panel-section">
            <label class="label-muted">Source Image</label>
            <input type="file" class="ui-input" accept="image/*" />
          </div>
          <div class="panel-section">
            <label class="label-muted">Upscaler</label>
            <select class="select-md">
              <option>ESRGAN 4x</option>
              <option>R-ESRGAN 4x+</option>
              <option>Latent</option>
            </select>
          </div>
          <div class="panel-section">
            <label class="label-muted">Scale</label>
            <div class="toolbar">
              <button class="btn btn-sm btn-outline">2x</button>
              <button class="btn btn-sm btn-outline">3x</button>
              <button class="btn btn-sm btn-outline">4x</button>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right column: Results (sticky header like txt2img) -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header three-cols results-sticky">Results
          <div class="header-center"><button class="btn btn-md btn-primary results-generate">Generate</button></div>
          <div class="header-right results-actions">
            <input class="ui-input" list="upscale-preset-list" v-model="presetName" placeholder="Preset" />
            <datalist id="upscale-preset-list"><option v-for="p in presetNames" :key="p" :value="p" /></datalist>
            <button class="btn btn-sm btn-secondary" type="button" @click="savePreset(presetName)">Save</button>
            <button class="btn btn-sm btn-outline" type="button" @click="applyPreset(presetName)">Apply</button>
          </div>
        </div>
        <div class="panel-body">
          <ResultViewer mode="image" :images="[]" emptyText="Upscaled image(s) will appear here." />
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import ResultViewer from '../components/ResultViewer.vue'
import { usePresetsStore } from '../stores/presets'

const presets = usePresetsStore()
const presetName = ref('')
const presetNames = computed(() => presets.names('upscale'))
function snapshotParams(): Record<string, unknown> { return {} }
function applyParams(_v: Record<string, unknown>): void {}
function savePreset(name: string): void { presets.upsert('upscale', name, snapshotParams()) }
function applyPreset(name: string): void { const v = presets.get('upscale', name); if (v) applyParams(v) }
</script>
