<template>
  <section class="panels">
    <div class="panel-stack" ref="leftStack">
      <div class="panel">
        <div class="panel-header">
          <span>Prompt</span>
          <div class="toolbar prompt-toolbar">
            <button class="btn btn-sm btn-secondary" type="button" @click="showCkpt = true">Checkpoints</button>
            <button class="btn btn-sm btn-secondary" type="button" @click="showTI = true">Textual Inversion</button>
          </div>
        </div>
        <div class="panel-body panel-body-scroll">
          <PromptFields v-model:prompt="store.prompt" v-model:negative="store.negativePrompt" />
          <div class="grid grid-2" style="margin-top: .75rem">
            <div>
              <label class="label">Steps</label>
              <input
                v-model.number="store.steps"
                type="number"
                min="1"
                max="60"
                class="ui-input"
              >
            </div>
            <div>
              <label class="label">CFG (Flux)</label>
              <input
                v-model.number="store.cfgScale"
                type="number"
                step="0.1"
                class="ui-input"
              >
            </div>
          </div>
          <div class="grid grid-3" style="margin-top: .75rem">
            <div>
              <label class="label">Seed</label>
              <input
                v-model.number="store.seed"
                type="number"
                class="ui-input"
              >
              <div class="mt-1 flex gap-1">
                <button class="btn btn-xs btn-secondary" type="button" @click="store.randomizeSeed">Random</button>
                <button class="btn btn-xs btn-secondary" type="button" @click="store.reuseSeed">Reuse</button>
              </div>
            </div>
            <div>
              <label class="label">Batch size</label>
              <input
                v-model.number="store.batchSize"
                type="number"
                min="1"
                max="16"
                class="ui-input"
              >
            </div>
            <div>
              <label class="label">Batch count</label>
              <input
                v-model.number="store.batchCount"
                type="number"
                min="1"
                max="16"
                class="ui-input"
              >
            </div>
          </div>
          <div class="grid grid-3" style="margin-top: .75rem">
            <div>
              <label class="label">Width</label>
              <input
                v-model.number="store.width"
                type="number"
                min="64"
                step="64"
                class="ui-input"
              >
            </div>
            <div>
              <label class="label">Height</label>
              <input
                v-model.number="store.height"
                type="number"
                min="64"
                step="64"
                class="ui-input"
              >
            </div>
            <div>
              <label class="label">Aspect</label>
              <div class="aspect-row">
                <span>{{ store.aspectLabel }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-header">
          <span>Model & Sampler</span>
        </div>
        <div class="panel-body panel-body-scroll">
          <ModelSelector
            :models="store.models"
            v-model="store.selectedModel"
            @change="store.updateModel"
          />
          <SamplerSelector
            :samplers="store.samplers"
            v-model="store.selectedSampler"
          />
          <SchedulerSelector
            :schedulers="store.schedulers"
            v-model="store.selectedScheduler"
          />
          <PresetSelector
            label="Presets"
            :names="presetNames"
            @apply="applyPreset"
          />
          <StyleSelector
            label="Styles"
            :names="styleNames"
            @apply="applyStyle"
            @saved="onStyleSaved"
          />
          <div class="mt-2">
            <button class="btn btn-sm btn-secondary" type="button" @click="store.saveProfile">
              Save profile
            </button>
            <span class="text-muted ms-2">{{ store.profileMessage }}</span>
          </div>
        </div>
      </div>
      <CheckpointModal v-if="showCkpt" @close="showCkpt = false" />
      <TextualInversionModal v-if="showTI" @close="showTI = false" />
    </div>

    <div class="panel-stack">
      <div class="panel">
        <div class="panel-header">
          <span>Preview</span>
          <GentimeBadge :gentime-ms="store.gentimeMs" />
          <button
            class="btn btn-primary"
            type="button"
            :disabled="store.isRunning"
            @click="onGenerate"
          >
            {{ store.isRunning ? 'Generating…' : 'Generate' }}
          </button>
        </div>
        <div class="panel-body panel-body-scroll">
          <ResultViewer
            :images="store.gallery"
            :info="store.info"
            @save="onSave"
          />
          <div v-if="store.errorMessage" class="alert alert-danger mt-2">
            {{ store.errorMessage }}
          </div>
        </div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import CheckpointModal from '../components/modals/CheckpointModal.vue'
import TextualInversionModal from '../components/modals/TextualInversionModal.vue'
import PromptFields from '../components/prompt/PromptFields.vue'
import ModelSelector from '../components/ModelSelector.vue'
import SamplerSelector from '../components/SamplerSelector.vue'
import SchedulerSelector from '../components/SchedulerSelector.vue'
import PresetSelector from '../components/PresetsSelector.vue'
import StyleSelector from '../components/StyleSelector.vue'
import ResultViewer from '../components/ResultViewer.vue'
import GentimeBadge from '../components/GentimeBadge.vue'
import { useFluxStore } from '../stores/flux'
import { usePresetsStore } from '../stores/presets'
import { useStylesStore } from '../stores/styles'
import type { GeneratedImage } from '../api/types'

const store = useFluxStore()
const presetsStore = usePresetsStore()
const stylesStore = useStylesStore()
const router = useRouter()

const showCkpt = ref(false)
const showTI = ref(false)
const leftStack = ref<HTMLElement | null>(null)

onMounted(() => {
  void store.init()
})

async function onGenerate(): Promise<void> {
  await store.generate()
}

function onSave(image: GeneratedImage): void {
  // Reuse SDXL saver semantics for now (single-image download)
  try {
    const bytes = atob(image.data)
    const buf = new Uint8Array(bytes.length)
    for (let i = 0; i < bytes.length; i++) buf[i] = bytes.charCodeAt(i)
    const blob = new Blob([buf], { type: `image/${image.format}` })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = image.name || 'flux.png'
    link.click()
    URL.revokeObjectURL(url)
  } catch (e) {
    console.error('Failed to save FLUX image', e)
  }
}

const presetNames = computed(() => presetsStore.names('flux'))
const styleNames = computed(() => stylesStore.names())

function applyPreset(name: string): void {
  const v = presetsStore.get('flux', name)
  if (!v) return
  // For now, apply only basic fields; deeper mapping can be added later.
  if (typeof v.prompt === 'string') store.prompt = v.prompt
  if (typeof v.negativePrompt === 'string') store.negativePrompt = v.negativePrompt
}

function applyStyle(name: string): void {
  const d = stylesStore.get(name)
  if (!d) return
  if (d.prompt) store.prompt += (store.prompt ? ' ' : '') + d.prompt
  if (d.negative) store.negativePrompt += (store.negativePrompt ? ' ' : '') + d.negative
}

function onStyleSaved(): void {
  // reactive; no-op hook for now
}
</script>
