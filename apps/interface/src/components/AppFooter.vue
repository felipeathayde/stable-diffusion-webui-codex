<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: WebUI footer with version info.
Renders footer links and fetches backend version metadata (commit/Python/PyTorch/CUDA) for display.

Symbols (top-level; keep in sync; no ghosts):
- `AppFooter` (component): Footer component rendered below the main layout.
-->

<template>
  <footer class="app-footer">
    <span>Stable Diffusion WebUI Codex</span>
    <a v-if="sourceUrl" class="link" :href="sourceUrl" target="_blank" rel="noopener">Source</a>
    <span v-if="commit">Commit: {{ commit }}</span>
    <span v-if="py">Python: {{ py }}</span>
    <span v-if="torch">PyTorch: {{ torch }}<template v-if="cuda"> (CUDA {{ cuda }})</template></span>
    <span>© {{ year }}</span>
  </footer>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { fetchVersion } from '../api/client'
const year = new Date().getFullYear()
// removed GPU mode label per request
const sourceUrl = computed(() => (import.meta.env.VITE_SOURCE_URL as string | undefined) || '')

const commit = ref<string>('')
const py = ref<string>('')
const torch = ref<string>('')
const cuda = ref<string>('')

onMounted(async () => {
  try {
    const v = await fetchVersion()
    commit.value = v.git_commit ?? ''
    py.value = v.python_version
    torch.value = v.torch_version ?? ''
    cuda.value = v.cuda_version ?? ''
  } catch (e) {
    // non-fatal
    console.warn('[footer] version fetch failed', e)
  }
})
</script>

<!-- footer styles are in src/styles.css (.app-footer) -->
