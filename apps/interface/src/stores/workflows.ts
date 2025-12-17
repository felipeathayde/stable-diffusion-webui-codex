import { defineStore } from 'pinia'
import { ref } from 'vue'

import { createWorkflow, deleteWorkflow, fetchWorkflows } from '../api/client'
import type { WorkflowsResponse } from '../api/types'

type WorkflowItem = WorkflowsResponse['workflows'][number]

export const useWorkflowsStore = defineStore('workflows', () => {
  const items = ref<WorkflowItem[]>([])
  const isLoading = ref(false)
  const error = ref('')

  async function refresh(): Promise<void> {
    isLoading.value = true
    error.value = ''
    try {
      const res = await fetchWorkflows()
      items.value = (res.workflows || []) as WorkflowItem[]
    } catch (err) {
      error.value = err instanceof Error ? err.message : String(err)
    } finally {
      isLoading.value = false
    }
  }

  async function createSnapshot(payload: { name: string; source_tab_id: string; type: string; engine_semantics?: string; params_snapshot: Record<string, unknown> }): Promise<void> {
    await createWorkflow(payload)
    await refresh()
  }

  async function remove(id: string): Promise<void> {
    await deleteWorkflow(id)
    await refresh()
  }

  return { items, isLoading, error, refresh, createSnapshot, remove }
})
