<template>
  <div :class="embedded ? '' : 'gen-card'">
    <div v-if="!embedded" class="wan22-toggle-head">
      <span class="label-muted">{{ title }}</span>
    </div>

    <div class="wan22-grid">
      <SamplerSelector
        :samplers="samplers"
        :modelValue="stage.sampler"
        :label="samplerLabel"
        :allow-empty="true"
        @update:modelValue="(v) => updateStage({ sampler: v })"
      />
      <SchedulerSelector
        :schedulers="schedulers"
        :modelValue="stage.scheduler"
        :label="schedulerLabel"
        :allow-empty="true"
        @update:modelValue="(v) => updateStage({ scheduler: v })"
      />
      <div>
        <label class="label-muted">Steps</label>
        <input
          class="ui-input"
          type="number"
          min="1"
          :disabled="disabled"
          :value="stage.steps"
          @change="updateStage({ steps: toInt($event, stage.steps) })"
        />
      </div>
      <div>
        <label class="label-muted">CFG</label>
        <input
          class="ui-input"
          type="number"
          step="0.5"
          :disabled="disabled"
          :value="stage.cfgScale"
          @change="updateStage({ cfgScale: toFloat($event, stage.cfgScale) })"
        />
      </div>
      <div>
        <label class="label-muted">Seed</label>
        <div class="flex flex-wrap gap-2 items-center">
          <input
            class="ui-input"
            type="number"
            :disabled="disabled"
            :value="stage.seed"
            @change="updateStage({ seed: toInt($event, stage.seed) })"
          />
          <button class="btn-icon" type="button" :disabled="disabled" title="Random seed" @click="randomizeSeed">🎲</button>
          <button class="btn-icon" type="button" :disabled="disabled || lastSeed === null" title="Reuse seed" @click="reuseSeed">↺</button>
        </div>
      </div>
      <div v-if="showModelDir">
        <label class="label-muted">Model Dir</label>
        <input
          class="ui-input"
          type="text"
          :disabled="disabled"
          :value="stage.modelDir"
          @change="updateStage({ modelDir: ($event.target as HTMLInputElement).value })"
          placeholder="/path/to/high-or-low"
        />
      </div>

      <div v-if="lightx2v">
        <label class="label-muted">LoRA (wan22-loras)</label>
        <select class="select-md" :disabled="disabled" :value="stage.loraPath" @change="updateStage({ loraPath: ($event.target as HTMLSelectElement).value })">
          <option value="">None</option>
          <option v-for="opt in loraChoices" :key="opt.path" :value="opt.path">{{ opt.name }}</option>
        </select>
      </div>
      <div v-if="lightx2v && stage.loraPath">
        <label class="label-muted">LoRA weight</label>
        <input
          class="ui-input"
          type="number"
          step="0.05"
          :disabled="disabled"
          :value="stage.loraWeight"
          @change="updateStage({ loraWeight: toFloat($event, stage.loraWeight) })"
        />
      </div>
    </div>

    <div v-if="showModelDir && !stage.modelDir" class="panel-error">{{ title }}: model directory is empty.</div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'

import type { SamplerInfo, SchedulerInfo } from '../../api/types'
import type { WanStageParams } from '../../stores/model_tabs'

import SamplerSelector from '../SamplerSelector.vue'
import SchedulerSelector from '../SchedulerSelector.vue'

const props = withDefaults(defineProps<{
  title: string
  stage: WanStageParams
  samplers: SamplerInfo[]
  schedulers: SchedulerInfo[]
  loraChoices?: Array<{ name: string; path: string }>
  showModelDir?: boolean
  embedded?: boolean
  disabled?: boolean
  lightx2v?: boolean
  samplerLabel?: string
  schedulerLabel?: string
}>(), {
  loraChoices: () => [],
  showModelDir: false,
  embedded: false,
  disabled: false,
  lightx2v: false,
  samplerLabel: 'Sampler',
  schedulerLabel: 'Scheduler',
})

const emit = defineEmits<{
  (e: 'update:stage', patch: Partial<WanStageParams>): void
}>()

const lastSeed = ref<number | null>(null)

const samplerLabel = computed(() => props.samplerLabel)
const schedulerLabel = computed(() => props.schedulerLabel)
const loraChoices = computed(() => props.loraChoices ?? [])
const lightx2v = computed(() => Boolean(props.lightx2v))

function updateStage(patch: Partial<WanStageParams>): void {
  emit('update:stage', patch)
}

function toInt(e: Event, fallback: number): number {
  const v = Number((e.target as HTMLInputElement).value)
  return Number.isFinite(v) ? Math.trunc(v) : fallback
}

function toFloat(e: Event, fallback: number): number {
  const v = Number((e.target as HTMLInputElement).value)
  return Number.isFinite(v) ? v : fallback
}

function randomizeSeed(): void {
  if (props.stage.seed !== -1) lastSeed.value = props.stage.seed
  updateStage({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) updateStage({ seed: lastSeed.value })
}
</script>
