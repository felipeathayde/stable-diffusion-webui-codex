<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Dynamic model tab view (`/models/:tabId`).
Loads the selected tab from the tabs store and mounts `WANTab`, `LTXTab`, or `ImageModelTab` based on tab type.

Symbols (top-level; keep in sync; no ghosts):
- `ModelTabView` (component): Route view that mounts the correct model tab workspace.
- `ImageTabType` (type): Non-video tab types supported by `ImageModelTab`.
- `imageTabType` (computed): Normalized non-video type passed to `ImageModelTab`.
-->

<template>
  <section v-if="tab">
    <WANTab v-if="tab.type === 'wan'" :tab-id="tab.id" :key="tab.id" />
    <LTXTab v-else-if="tab.type === 'ltx2'" :tab-id="tab.id" :key="tab.id" />
    <ImageModelTab v-else-if="imageTabType" :tab-id="tab.id" :key="tab.id" :type="imageTabType" />
    <div v-else class="panel">
      <div class="panel-body">Unsupported tab type: {{ tab.type }}</div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab não encontrada.</div></div>
  </section>
</template>

<script setup lang="ts">
import { computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import WANTab from './WANTab.vue'
import LTXTab from './LTXTab.vue'
import ImageModelTab from './ImageModelTab.vue'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'

const route = useRoute()
const store = useModelTabsStore()

const id = computed(() => String(route.params.tabId || ''))
const tab = computed(() => store.tabs.find(t => t.id === id.value) || null)

type ImageTabType = Exclude<BaseTabType, 'wan' | 'ltx2'>

const imageTabType = computed<ImageTabType | null>(() => {
  const t = tab.value?.type
  if (!t || t === 'wan' || t === 'ltx2') return null
  return t
})

watch(id, (nextId) => {
  if (nextId) store.setActive(nextId)
}, { immediate: true })
</script>
