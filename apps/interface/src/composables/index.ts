/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Composables barrel export for the frontend.
Re-exports key composables and types so views/components can import from a single module.

Symbols (top-level; keep in sync; no ghosts):
- `useGeneration` (function): Generation composable export (re-export).
- `GenerationState` (type): Generation state type exported by `useGeneration` (re-export).
*/

export { useGeneration } from './useGeneration'
export type { GenerationState } from './useGeneration'
