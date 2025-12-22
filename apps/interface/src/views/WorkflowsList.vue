<template>
  <section class="panel-stack">
    <div class="panel">
      <div class="panel-header">
        <span>Workflows</span>
      </div>
      <div class="panel-body">
        <div v-if="workflows.error" class="panel-error">{{ workflows.error }}</div>
        <p v-else-if="!items.length" class="caption">No workflows yet. Use “Save snapshot” from a model tab.</p>
        <ul v-else class="cdx-list">
          <li v-for="wf in items" :key="wf.id" class="cdx-list-item">
            <div class="cdx-list-main">
              <div class="cdx-list-title">{{ wf.name }}</div>
              <div class="cdx-list-meta">Type: {{ wf.type.toUpperCase() }} · Created: {{ new Date(wf.created_at).toLocaleString() }}</div>
            </div>
            <div class="cdx-list-actions">
              <RouterLink class="btn btn-sm btn-outline" :to="`/models/${wf.source_tab_id}`">Open source tab</RouterLink>
              <button class="btn btn-sm btn-destructive" type="button" @click="remove(wf.id)">Delete</button>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useWorkflowsStore } from '../stores/workflows'
import { useModelTabsStore } from '../stores/model_tabs'

const router = useRouter()
const tabs = useModelTabsStore()
const workflows = useWorkflowsStore()
const items = computed(() => workflows.items)

onMounted(() => { void workflows.refresh() })

async function remove(id: string): Promise<void> {
  await workflows.remove(id)
}

async function loadIntoBase(itemId: string): Promise<void> {
  const wf = items.value.find(w => w.id === itemId)
  if (!wf) return
  await tabs.load()
  await tabs.updateParams(wf.source_tab_id, wf.params_snapshot)
  await router.push(`/models/${wf.source_tab_id}`)
}
</script>
