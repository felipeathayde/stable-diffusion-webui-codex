/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Navigation helper for model tabs (`/models/:tabId`).
Ensures a tab exists for a given type, optionally applies init-image params, and navigates to the tab route.

Symbols (top-level; keep in sync; no ghosts):
- `useModelTabNavigation` (function): Provides `openModelTab(...)` helper for routing to a model tab.
*/

import { useRouter } from 'vue-router'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'

export function useModelTabNavigation(): {
  openModelTab: (type: BaseTabType, options?: { initImage?: { dataUrl: string; name: string } }) => Promise<void>
} {
  const router = useRouter()
  const tabs = useModelTabsStore()

  async function openModelTab(type: BaseTabType, options?: { initImage?: { dataUrl: string; name: string } }): Promise<void> {
    await tabs.load()
    const existing = tabs.orderedTabs.find(t => t.type === type)
    const id = existing?.id || (await tabs.create(type))
    if (!id) throw new Error('failed to resolve a model tab id')

    if (options?.initImage) {
      await tabs.updateParams(id, {
        useInitImage: true,
        initImageData: options.initImage.dataUrl,
        initImageName: options.initImage.name,
      })
    }

    await router.push(`/models/${id}`)
  }

  return { openModelTab }
}
