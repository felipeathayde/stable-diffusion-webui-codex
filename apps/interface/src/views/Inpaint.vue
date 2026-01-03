<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Inpaint route view.
Prototype inpaint workspace wiring an init image picker and a preview panel (mask editing and generation are TODO).

Symbols (top-level; keep in sync; no ghosts):
- `Inpaint` (component): Inpaint route view component.
- `onFileSet` (function): Handles init image file selection and stores it in the inpaint store.
-->

<template>
  <section class="panels">
    <!-- Left column: Init image + mask tools placeholder -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Inpaint</div>
        <div class="panel-body">
          <InitialImageCard
            label="Initial Image"
            :src="store.initImagePreview"
            :has-image="store.hasInitImage"
            @set="onFileSet"
            @clear="store.clearInitImage"
          >
            <template #footer>
              <p v-if="store.initImageName" class="caption">{{ store.initImageName }}</p>
            </template>
          </InitialImageCard>
          <div class="panel-section">
            <label class="label-muted">Mask</label>
            <div class="card text-xs opacity-70">Mask editor coming soon.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right column: placeholder -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Preview</div>
        <div class="panel-body">
          <ResultViewer mode="image" :images="[]" emptyText="Preview/output will appear here." />
        </div>
      </div>
    </div>
  </section>
  
</template>

<script setup lang="ts">
import { useInpaintStore } from '../stores/inpaint'
import InitialImageCard from '../components/InitialImageCard.vue'
import ResultViewer from '../components/ResultViewer.vue'

const store = useInpaintStore()

async function onFileSet(file: File): Promise<void> { await store.setInitImage(file) }
</script>
