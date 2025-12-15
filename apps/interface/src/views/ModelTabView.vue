<template>
  <section v-if="tab">
    <BaseTabHeader
      :title="tab.title"
      :enabled="tab.enabled"
      @rename="onRename"
      @duplicate="onDuplicate"
      @remove="onRemove"
      @set-enabled="onSetEnabled"
      @load="onLoad"
      @unload="onUnload"
      @generate="onGenerate"
      @send-to-workflows="onSendToWorkflows"
    />

    <WANTab v-if="tab.type === 'wan'" :tab-id="tab.id" :key="tab.id" ref="wanRef" />
    <ImageModelTab v-else :tab-id="tab.id" :key="tab.id" :type="tab.type as any" ref="imgRef" />
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab não encontrada.</div></div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import BaseTabHeader from '../components/BaseTabHeader.vue'
import WANTab from './WANTab.vue'
import ImageModelTab from './ImageModelTab.vue'
import { useModelTabsStore } from '../stores/model_tabs'
import { loadModelsForTab, unloadModelsForTab } from '../api/client'
import { createWorkflow } from '../api/client'

const route = useRoute()
const router = useRouter()
const store = useModelTabsStore()
const wanRef = ref<InstanceType<typeof WANTab> | null>(null)
const imgRef = ref<InstanceType<typeof ImageModelTab> | null>(null)

const id = computed(() => String(route.params.tabId || ''))
const tab = computed(() => store.tabs.find(t => t.id === id.value) || null)

onMounted(async () => {
  await store.load()
  if (id.value) store.setActive(id.value)
})

watch(id, (nextId) => {
  if (nextId) store.setActive(nextId)
})

function onRename(title: string): void { if (tab.value) store.rename(tab.value.id, title) }
function onDuplicate(): void { if (tab.value) store.duplicate(tab.value.id) }
function onRemove(): void {
  if (!tab.value) return
  store.remove(tab.value.id)
  void router.replace('/models')
}
function onSetEnabled(v: boolean): void { if (tab.value) store.setEnabled(tab.value.id, v) }
async function onLoad(): Promise<void> { if (tab.value) await loadModelsForTab(tab.value.id) }
async function onUnload(): Promise<void> { if (tab.value) await unloadModelsForTab(tab.value.id) }
function onGenerate(): void {
  if (tab.value?.type === 'wan') wanRef.value?.generate?.()
  else imgRef.value?.generate?.()
}
async function onSendToWorkflows(): Promise<void> {
  if (!tab.value) return
  try {
    await createWorkflow({
      name: `${tab.value.title} — ${new Date().toLocaleString()}`,
      source_tab_id: tab.value.id,
      type: tab.value.type,
      engine_semantics: tab.value.type === 'wan' ? 'wan22' : tab.value.type,
      params_snapshot: tab.value.params as Record<string, unknown>,
    })
    // eslint-disable-next-line no-alert
    alert('Workflow created successfully.')
  } catch (e) {
    // eslint-disable-next-line no-alert
    alert(`Failed to create workflow: ${e instanceof Error ? e.message : String(e)}`)
  }
}
</script>
