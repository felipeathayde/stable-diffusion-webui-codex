/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Workflow snapshots store for the WebUI.
Fetches, creates, and deletes workflow snapshots via the backend API and keeps the workflows list reactive for the Workflows view.

Symbols (top-level; keep in sync; no ghosts):
- `useWorkflowsStore` (store): Pinia store for listing and mutating workflows (refresh/createSnapshot/remove).
*/

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
