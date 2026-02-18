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
- `repoUrl` (const): Canonical repository URL used for footer links.
- `xProfileUrl` (const): Canonical X/Twitter profile URL used in footer attribution.
- `commitHref` (const): Computed commit URL for the currently reported backend commit.
- `commitLabel` (const): Short commit label used in the footer (`10` chars).
-->

<template>
  <footer class="app-footer">
    <a class="link" :href="repoUrl" target="_blank" rel="noopener noreferrer">Stable Diffusion WebUI Codex</a>
    <span v-if="commit">
      Commit:
      <a class="link" :href="commitHref" target="_blank" rel="noopener noreferrer">{{ commitLabel }}</a>
    </span>
    <span v-if="py">Python: {{ py }}</span>
    <span v-if="torch">PyTorch: {{ torch }}<template v-if="cuda"> (CUDA {{ cuda }})</template></span>
    <a class="link" :href="xProfileUrl" target="_blank" rel="noopener noreferrer">by @lucas_sangoi</a>
  </footer>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { fetchVersion } from '../api/client'

const repoUrl = 'https://github.com/sangoi-exe/stable-diffusion-webui-codex'
const xProfileUrl = 'https://x.com/lucas_sangoi'

const commit = ref<string>('')
const py = ref<string>('')
const torch = ref<string>('')
const cuda = ref<string>('')
const commitHref = computed(() => `${repoUrl}/commit/${encodeURIComponent(commit.value)}`)
const commitLabel = computed(() => commit.value.slice(0, 10))

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
