<template>
  <section>
    <!-- Route to engine-specific views that use their individual stores -->
    <component :is="engineComponent" v-if="engineComponent" />
    <div v-else class="panel"><div class="panel-body">Loading engine...</div></div>
  </section>
</template>

<script setup lang="ts">
/**
 * EngineView - Routes to the correct engine-specific view.
 * Each engine has its own store with specific behavior.
 */
import { computed, defineAsyncComponent } from 'vue'
import type { EngineType } from '../stores/engine_config'

const props = defineProps<{ engine: EngineType }>()

// Map engine types to their view components
const engineViews: Partial<Record<EngineType, ReturnType<typeof defineAsyncComponent>>> = {
  sd15: defineAsyncComponent(() => import('./Txt2Img.vue')),
  sdxl: defineAsyncComponent(() => import('./Sdxl.vue')),
  flux: defineAsyncComponent(() => import('./Flux.vue')),
  zimage: defineAsyncComponent(() => import('./ZImage.vue')),
  // chroma: defineAsyncComponent(() => import('./Chroma.vue')),  // Add when created
  // wan22_14b and wan22_5b use WANTab via ModelTabView
}

const engineComponent = computed(() => engineViews[props.engine] ?? null)
</script>
