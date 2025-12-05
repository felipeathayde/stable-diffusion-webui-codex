import { createRouter, createWebHistory } from 'vue-router'

const Upscale = () => import('./views/Upscale.vue')
const PngInfo = () => import('./views/PngInfo.vue')
const Extensions = () => import('./views/Extensions.vue')
const Settings = () => import('./views/Settings.vue')
const ModelsList = () => import('./views/ModelsList.vue')
const ModelTabView = () => import('./views/ModelTabView.vue')
const WorkflowsList = () => import('./views/WorkflowsList.vue')
const Sdxl = () => import('./views/Sdxl.vue')
const Flux = () => import('./views/Flux.vue')
const XyzPlot = () => import('./views/XyzPlot.vue')
const Home = () => import('./views/Home.vue')

const router = createRouter({
  history: createWebHistory(),
  routes: [
    // Default landing: engine-agnostic home workspace
    { path: '/', component: Home },
    // Model tabs remain accessible but are no longer surfaced as top-level inference tabs,
    // and WAN22 video flows live exclusively under model tabs (no standalone /txt2vid/img2vid routes).
    { path: '/models', component: ModelsList },
    { path: '/models/:tabId', component: ModelTabView },
    // Single canonical inference surface
    { path: '/sdxl', component: Sdxl },
    { path: '/flux', component: Flux },
    { path: '/xyz', component: XyzPlot },
    // Utilities
    { path: '/workflows', component: WorkflowsList },
    { path: '/upscale', component: Upscale },
    { path: '/pnginfo', component: PngInfo },
    { path: '/extensions', component: Extensions },
    { path: '/settings', component: Settings }
  ]
})

export default router
