<template>
  <section class="panels">
    <!-- Left column: Init image + mask tools placeholder -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Inpaint</div>
        <div class="panel-body">
          <InitialImageCard
            label="Initial Image"
            :src="store.initImagePreview"
            :has-image="store.hasInitImage"
            @set="onFileSet"
            @clear="store.clearInitImage"
          >
            <template #footer>
              <p v-if="store.initImageName" class="caption">{{ store.initImageName }}</p>
            </template>
          </InitialImageCard>
          <div class="panel-section">
            <label class="label-muted">Mask</label>
            <div class="card text-xs opacity-70">Mask editor coming soon.</div>
          </div>
        </div>
      </div>
    </div>

    <!-- Right column: placeholder -->
    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">Preview</div>
        <div class="panel-body">
          <ResultViewer mode="image" :images="[]" emptyText="Preview/output will appear here." />
        </div>
      </div>
    </div>
  </section>
  
</template>

<script setup lang="ts">
import { useInpaintStore } from '../stores/inpaint'
import InitialImageCard from '../components/InitialImageCard.vue'
import ResultViewer from '../components/ResultViewer.vue'

const store = useInpaintStore()

async function onFileSet(file: File): Promise<void> { await store.setInitImage(file) }
</script>
