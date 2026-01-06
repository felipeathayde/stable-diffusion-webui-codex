/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WebUI router configuration.
Defines the SPA route table and lazy-loads page-level views (model tabs, tools, settings, etc.).

Symbols (top-level; keep in sync; no ghosts):
- `router` (const): Vue Router instance configured with history + route table.
- `default` (const): Default export of the configured router instance.
*/

import { createRouter, createWebHistory } from 'vue-router'

const Upscale = () => import('./views/Upscale.vue')
const PngInfo = () => import('./views/PngInfo.vue')
const Extensions = () => import('./views/Extensions.vue')
const Settings = () => import('./views/Settings.vue')
const ModelsList = () => import('./views/ModelsList.vue')
const ModelTabView = () => import('./views/ModelTabView.vue')
const WorkflowsList = () => import('./views/WorkflowsList.vue')
const XyzPlot = () => import('./views/XyzPlot.vue')
const Home = () => import('./views/Home.vue')
const ToolsTab = () => import('./views/ToolsTab.vue')

const router = createRouter({
  history: createWebHistory(),
  routes: [
    // Default landing: engine-agnostic home workspace
    { path: '/', component: Home },
    // Model tabs - dynamic engine tabs
    { path: '/models', component: ModelsList },
    { path: '/models/:tabId', component: ModelTabView },
    { path: '/xyz', component: XyzPlot },
    // Utilities
    { path: '/tools', component: ToolsTab },
    { path: '/workflows', component: WorkflowsList },
    { path: '/upscale', component: Upscale },
    { path: '/pnginfo', component: PngInfo },
    { path: '/extensions', component: Extensions },
    { path: '/settings', component: Settings },
  ],
})

export default router
