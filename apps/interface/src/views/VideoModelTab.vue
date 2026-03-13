<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Canonical route-level owner for video model tabs under `/models/:tabId`.
Delegates `wan` and `ltx2` tabs to their family-owned workspace components and fails loud on impossible tab-type drift.

Symbols (top-level; keep in sync; no ghosts):
- `VideoModelTab` (component): Route owner for video-family model tabs.
- `VideoTabType` (type): Supported video tab types handled here.
- `videoTabType` (computed): Current video tab type when the selected tab belongs to the video lane.
-->

<template>
  <section v-if="tab">
    <WanVideoWorkspace v-if="videoTabType === 'wan'" :tab-id="tab.id" :key="tab.id" />
    <LtxVideoWorkspace v-else-if="videoTabType === 'ltx2'" :tab-id="tab.id" :key="tab.id" />
    <div v-else class="panel">
      <div class="panel-body">Unsupported video tab type: {{ tab.type }}</div>
    </div>
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab não encontrada.</div></div>
  </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'

import LtxVideoWorkspace from '../components/model-tabs/LtxVideoWorkspace.vue'
import WanVideoWorkspace from '../components/model-tabs/WanVideoWorkspace.vue'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'

const props = defineProps<{
  tabId: string
}>()

const store = useModelTabsStore()

type VideoTabType = Extract<BaseTabType, 'wan' | 'ltx2'>

const tab = computed(() => store.tabs.find((entry) => entry.id === props.tabId) || null)
const videoTabType = computed<VideoTabType | null>(() => {
  const value = tab.value?.type
  if (value === 'wan' || value === 'ltx2') return value
  return null
})
</script>
