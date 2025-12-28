<template>
  <section>
    <!-- Route to engine-specific views that use their individual stores -->
    <component :is="engineComponent" v-if="engineComponent" />
    <div v-else class="panel">
      <div class="panel-header">Engine not implemented</div>
      <div class="panel-body">
        <div class="panel-error">No UI view is registered for engine: <code>{{ props.engine }}</code></div>
        <div class="mt-2">
          <RouterLink class="btn btn-sm btn-primary" to="/">Home</RouterLink>
          <RouterLink class="btn btn-sm btn-outline" to="/models">Model Tabs</RouterLink>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
/**
 * EngineView - Routes to the correct engine-specific view.
 * Each engine has its own store with specific behavior.
 */
import { computed, defineAsyncComponent, watch } from 'vue'
import type { EngineType } from '../stores/engine_config'
import { useQuicksettingsStore } from '../stores/quicksettings'

const props = defineProps<{ engine: EngineType }>()
const quicksettings = useQuicksettingsStore()

// Map engine types to their view components
const engineViews: Partial<Record<EngineType, ReturnType<typeof defineAsyncComponent>>> = {
  sd15: defineAsyncComponent(() => import('./Txt2Img.vue')),
  sdxl: defineAsyncComponent(() => import('./Sdxl.vue')),
  flux: defineAsyncComponent(() => import('./Flux.vue')),
  zimage: defineAsyncComponent(() => import('./ZImage.vue')),
  // chroma: defineAsyncComponent(() => import('./Chroma.vue')),  // Add when created
  // wan22_14b and wan22_5b use WANTab via ModelTabView
}

watch(
  () => props.engine,
  (engine) => {
    if (!engineViews[engine]) return
    void quicksettings.setEngine(engine)
  },
  { immediate: true },
)

const engineComponent = computed(() => engineViews[props.engine] ?? null)
</script>
