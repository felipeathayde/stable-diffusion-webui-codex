/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Ambient TypeScript declarations for the Vue 3 WebUI.
Defines the `*.vue` module typing so `vue-tsc` and editors can typecheck SFC imports.

Symbols (top-level; keep in sync; no ghosts):
- (ambient) `*.vue` module declaration: Types default exports as a `DefineComponent`.
*/

declare module '*.vue' {
  import type { DefineComponent } from 'vue'

  const component: DefineComponent<{}, {}, any>
  export default component
}
