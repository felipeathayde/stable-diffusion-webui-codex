/*
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WebUI application entrypoint.
Creates and mounts the Vue app, installing Pinia + router, and applies the default theme class.

Symbols (top-level; keep in sync; no ghosts):
- `app` (const): Vue application instance created from `App`.
*/

import { createApp } from 'vue'
import { createPinia } from 'pinia'
import App from './App.vue'
import router from './router'
import './styles.css'

const app = createApp(App)
app.use(createPinia())
app.use(router)

document.documentElement.classList.add('dark')

app.mount('#app')
