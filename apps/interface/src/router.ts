import { createRouter, createWebHistory } from 'vue-router'

const Upscale = () => import('./views/Upscale.vue')
const PngInfo = () => import('./views/PngInfo.vue')
const Extensions = () => import('./views/Extensions.vue')
const Settings = () => import('./views/Settings.vue')
const ModelsList = () => import('./views/ModelsList.vue')
const ModelTabView = () => import('./views/ModelTabView.vue')
const WorkflowsList = () => import('./views/WorkflowsList.vue')
const Sdxl = () => import('./views/Sdxl.vue')

const router = createRouter({
  history: createWebHistory(),
  routes: [
    // Default landing: SDXL txt2img
    { path: '/', redirect: '/sdxl' },
    // Legacy routes redirect into the single SDXL entrypoint or related tools
    { path: '/img2img', redirect: '/sdxl' },
    { path: '/inpaint', redirect: '/sdxl' },
    { path: '/txt2vid', redirect: '/workflows' },
    { path: '/img2vid', redirect: '/workflows' },
    // Model tabs remain accessible but are no longer surfaced as top-level inference tabs
    { path: '/models', component: ModelsList },
    { path: '/models/:tabId', component: ModelTabView },
    // Single canonical inference surface
    { path: '/sdxl', component: Sdxl },
    // Utilities
    { path: '/workflows', component: WorkflowsList },
    { path: '/upscale', component: Upscale },
    { path: '/pnginfo', component: PngInfo },
    { path: '/extensions', component: Extensions },
    { path: '/settings', component: Settings }
  ]
})

export default router
