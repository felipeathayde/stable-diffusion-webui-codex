<template>
  <section>
    <div class="panel">
      <div class="panel-header">
        <h2 class="h3">Workflows</h2>
      </div>
      <div class="panel-body">
        <div v-if="workflows.error" class="panel-error">{{ workflows.error }}</div>
        <div v-else-if="!items.length" class="muted">No workflows yet. Use “Save snapshot” from a model tab.</div>
        <ul class="list" v-else>
          <li v-for="wf in items" :key="wf.id" class="list-row">
            <div class="list-col grow">
              <div class="strong">{{ wf.name }}</div>
              <div class="muted small">Type: {{ wf.type.toUpperCase() }} • Created: {{ new Date(wf.created_at).toLocaleString() }}</div>
            </div>
            <div class="list-col">
              <RouterLink class="btn btn-sm" :to="`/models/${wf.source_tab_id}`">Open Source Tab</RouterLink>
              <button class="btn btn-sm btn-destructive" style="margin-left:.5rem" @click="remove(wf.id)">Delete</button>
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
const items = computed(() => workflows.items.value)

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
