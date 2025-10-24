import { createRouter, createWebHistory } from 'vue-router'

const Upscale = () => import('./views/Upscale.vue')
const PngInfo = () => import('./views/PngInfo.vue')
const Extensions = () => import('./views/Extensions.vue')
const Settings = () => import('./views/Settings.vue')
const ModelsList = () => import('./views/ModelsList.vue')
const ModelTabView = () => import('./views/ModelTabView.vue')
const WorkflowsList = () => import('./views/WorkflowsList.vue')
const Txt2Img = () => import('./views/Txt2Img.vue')
const Test = () => import('./views/Test.vue')

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', redirect: '/models' },
    // legacy routes -> models (except txt2img which is restored)
    { path: '/img2img', redirect: '/models' },
    { path: '/inpaint', redirect: '/models' },
    { path: '/txt2vid', redirect: '/models' },
    { path: '/img2vid', redirect: '/models' },
    // active routes
    { path: '/models', component: ModelsList },
    { path: '/models/:tabId', component: ModelTabView },
    { path: '/txt2img', component: Txt2Img },
    { path: '/test', component: Test },
    { path: '/workflows', component: WorkflowsList },
    { path: '/upscale', component: Upscale },
    { path: '/pnginfo', component: PngInfo },
    { path: '/extensions', component: Extensions },
    { path: '/settings', component: Settings }
  ]
})

export default router
