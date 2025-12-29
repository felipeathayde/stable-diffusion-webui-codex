<template>
  <section v-if="tab">
    <WANTab v-if="tab.type === 'wan'" :tab-id="tab.id" :key="tab.id" />
    <ImageModelTab v-else :tab-id="tab.id" :key="tab.id" :type="tab.type as any" />
  </section>
  <section v-else>
    <div class="panel"><div class="panel-body">Tab não encontrada.</div></div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, computed, watch } from 'vue'
import { useRoute } from 'vue-router'
import WANTab from './WANTab.vue'
import ImageModelTab from './ImageModelTab.vue'
import { useModelTabsStore } from '../stores/model_tabs'

const route = useRoute()
const store = useModelTabsStore()

const id = computed(() => String(route.params.tabId || ''))
const tab = computed(() => store.tabs.find(t => t.id === id.value) || null)

watch(id, (nextId) => {
  if (nextId) store.setActive(nextId)
}, { immediate: true })

onMounted(() => {
  void store.load()
})
</script>
