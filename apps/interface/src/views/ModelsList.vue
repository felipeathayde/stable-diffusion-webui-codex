<template>
  <section class="panel-stack">
    <div class="panel">
      <div class="panel-header">
        <span>Model Tabs</span>
        <div class="toolbar">
          <select class="select-md" v-model="newType" aria-label="New tab type">
            <option value="sd15">SD 1.5</option>
            <option value="sdxl">SDXL</option>
            <option value="flux">FLUX</option>
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

const router = useRouter()
const store = useModelTabsStore()
const newType = ref<BaseTabType>('wan')

onMounted(async () => {
  await store.load()
  // Auto-jump to first tab in the new UX
  const first = store.orderedTabs[0]
  if (first) void router.replace(`/models/${first.id}`)
})

const tabs = computed(() => store.orderedTabs)

async function createTab(): Promise<void> {
  const id = await store.create(newType.value)
  if (id) void router.push(`/models/${id}`)
}

function dup(id: string): void { void store.duplicate(id) }
function remove(id: string): void { void store.remove(id) }
</script>
