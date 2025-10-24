<template>
  <section>
    <div class="panel">
      <div class="panel-header">
        <h2 class="h3">Workflows</h2>
      </div>
      <div class="panel-body">
        <div v-if="!items.length" class="muted">No workflows yet. Use “Send to Workflows” from a model tab.</div>
        <ul class="list" v-else>
          <li v-for="wf in items" :key="wf.id" class="list-row">
            <div class="list-col grow">
              <div class="strong">{{ wf.name }}</div>
              <div class="muted small">Type: {{ wf.type.toUpperCase() }} • Created: {{ new Date(wf.createdAt).toLocaleString() }}</div>
            </div>
            <div class="list-col">
              <RouterLink class="btn btn-sm" :to="`/models/${wf.sourceTabId}`">Open Source Tab</RouterLink>
              <button class="btn btn-sm btn-destructive" style="margin-left:.5rem" @click="remove(wf.id)">Delete</button>
            </div>
          </li>
        </ul>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { fetchWorkflows, deleteWorkflow } from '../api/client'
import { useModelTabsStore } from '../stores/model_tabs'

const router = useRouter()
const tabs = useModelTabsStore()
const items = ref<Array<{ id: string; name: string; source_tab_id: string; type: string; created_at: string; engine_semantics: string; params_snapshot: Record<string, unknown> }>>([])

async function refresh(): Promise<void> {
  const res = await fetchWorkflows()
  items.value = res.workflows as any
}

onMounted(() => { void refresh() })

async function remove(id: string): Promise<void> {
  await deleteWorkflow(id)
  await refresh()
}

async function loadIntoBase(itemId: string): Promise<void> {
  const wf = items.value.find(w => w.id === itemId)
  if (!wf) return
  await tabs.load()
  await tabs.updateParams(wf.source_tab_id, wf.params_snapshot)
  await router.push(`/models/${wf.source_tab_id}`)
}
</script>
