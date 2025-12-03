<template>
  <div class="layout">
    <div class="main-shell">
      <header class="main-header" ref="headerRef">
        <div class="main-header-top">
          <h1 class="h2">Stable Diffusion WebUI Codex</h1>
        </div>
        <div class="main-header-qs">
          <QuickSettingsBar />
        </div>
      </header>
      <nav class="tabs-nav">
        <!-- Single canonical inference tab -->
        <RouterLink class="tab-link" to="/sdxl">sdxl</RouterLink>
        <!-- Utilities on the right -->
        <RouterLink class="tab-link" to="/workflows">workflows</RouterLink>
        <RouterLink class="tab-link" to="/upscale">upscale</RouterLink>
        <RouterLink class="tab-link" to="/pnginfo">png info</RouterLink>
        <RouterLink class="tab-link" to="/extensions">extensions</RouterLink>
        <RouterLink class="tab-link" to="/settings">settings</RouterLink>
      </nav>
      <main class="main-content">
        <RouterView />
      </main>
      <AppFooter />
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: layout, navigation
import QuickSettingsBar from './components/QuickSettingsBar.vue'
import AppFooter from './components/AppFooter.vue'
import { onMounted, onBeforeUnmount, ref } from 'vue'

const headerRef = ref<HTMLElement | null>(null)
let headerRO: ResizeObserver | null = null
function setStickyOffset(): void {
  const h = headerRef.value?.offsetHeight ?? 0
  const mc = document.querySelector('.main-content') as HTMLElement | null
  const pt = mc ? parseFloat(getComputedStyle(mc).paddingTop || '0') : 0
  // Tabs não são sticky: o offset deve ser exatamente a altura do header.
  // Não subtrair padding do conteúdo para evitar subposição.
  const offset = Math.max(0, h)
  document.documentElement.style.setProperty('--sticky-offset', offset + 'px')
}

onMounted(() => {
  setStickyOffset()
  headerRO = new ResizeObserver(setStickyOffset)
  if (headerRef.value) headerRO.observe(headerRef.value)
  window.addEventListener('resize', setStickyOffset)
})

onBeforeUnmount(() => {
  if (headerRO && headerRef.value) headerRO.unobserve(headerRef.value)
  window.removeEventListener('resize', setStickyOffset)
})
</script>
