<template>
  <section>
    <div class="panel">
      <div class="panel-header">
        <h2 class="h3">Model Tabs</h2>
        <div class="panel-actions">
          <select class="select-md" v-model="newType" aria-label="New tab type">
            <option value="sd15">SD 1.5</option>
            <option value="sdxl">SDXL</option>
            <option value="flux">FLUX</option>
            <option value="wan">WAN 2.2</option>
          </select>
          <button class="btn btn-primary" style="margin-left: .5rem" @click="createTab">New Tab</button>
        </div>
      </div>
      <div class="panel-body">
        <div v-if="!tabs.length" class="muted">No tabs yet. Create one above.</div>
        <ul class="list">
          <li v-for="t in tabs" :key="t.id" class="list-row">
            <div class="list-col grow">
              <RouterLink class="link" :to="`/models/${t.id}`">{{ t.title }}</RouterLink>
              <span class="muted" style="margin-left:.5rem">{{ t.type.toUpperCase() }}</span>
            </div>
            <div class="list-col">
              <button class="btn btn-sm" @click="dup(t.id)">Duplicate</button>
              <button class="btn btn-sm btn-destructive" style="margin-left:.5rem" @click="remove(t.id)">Remove</button>
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
