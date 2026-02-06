<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Model tabs management view.
Lists model tabs and provides actions to create/open/duplicate/remove tabs (including Anima tabs when supported).

Symbols (top-level; keep in sync; no ghosts):
- `ModelsList` (component): View for managing model tabs.
-->

<template>
  <section class="panel-stack">
    <div class="panel">
      <div class="panel-header">Model Tabs
        <div class="toolbar">
          <select class="select-md" v-model="newType" aria-label="New tab type">
            <option value="sd15">SD 1.5</option>
            <option value="sdxl">SDXL</option>
            <option value="flux1">FLUX.1</option>
            <option value="zimage">Z Image</option>
            <option v-if="showAnimaOption" value="anima">Anima</option>
            <option value="wan">WAN 2.2</option>
          </select>
          <button class="btn btn-sm btn-primary" type="button" @click="createTab">New Tab</button>
        </div>
      </div>
      <div class="panel-body">
        <p v-if="!tabs.length" class="caption">No tabs yet. Create one above.</p>
        <ul v-else class="cdx-list">
          <li v-for="t in tabs" :key="t.id" class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">{{ t.title }}</div>
              <div class="cdx-list-meta">{{ t.type.toUpperCase() }} · {{ t.id }}</div>
            </div>
            <div class="cdx-list-actions">
              <RouterLink class="btn btn-sm btn-outline" :to="`/models/${t.id}`">Open</RouterLink>
              <button class="btn btn-sm btn-secondary" type="button" @click="dup(t.id)">Duplicate</button>
              <button class="btn btn-sm btn-destructive" type="button" @click="remove(t.id)">Remove</button>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
	import { ref, onMounted, computed } from 'vue'
	import { useRouter } from 'vue-router'
	import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'
	import { useEngineCapabilitiesStore } from '../stores/engine_capabilities'

	const router = useRouter()
	const store = useModelTabsStore()
	const engineCaps = useEngineCapabilitiesStore()
	const newType = ref<BaseTabType>('wan')

	onMounted(async () => {
	  await engineCaps.init()
	  await store.load()
	})

const tabs = computed(() => store.orderedTabs)
const showAnimaOption = computed(() => Boolean(engineCaps.get('anima')))

async function createTab(): Promise<void> {
  if (newType.value === 'anima' && !showAnimaOption.value) {
    const msg = "Cannot create Anima tab: '/api/engines/capabilities' does not expose 'anima'."
    console.error(`[ModelsList] ${msg}`)
    throw new Error(msg)
  }
  const id = await store.create(newType.value)
  if (id) void router.push(`/models/${id}`)
}

function dup(id: string): void { void store.duplicate(id) }
function remove(id: string): void { void store.remove(id) }
</script>
