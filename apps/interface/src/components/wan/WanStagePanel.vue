<template>
  <div :class="['gen-card', { 'gen-card--embedded': embedded }]">
    <div v-if="!embedded" class="row-split">
      <span class="label-muted">{{ title }}</span>
    </div>

    <div class="gc-row">
      <SamplerSelector
        class="gc-col"
        :samplers="samplers"
        :modelValue="stage.sampler"
        :label="samplerLabel"
        :allow-empty="true"
        @update:modelValue="(v) => updateStage({ sampler: v })"
      />
      <SchedulerSelector
        class="gc-col"
        :schedulers="schedulers"
        :modelValue="stage.scheduler"
        :label="schedulerLabel"
        :allow-empty="true"
        @update:modelValue="(v) => updateStage({ scheduler: v })"
      />
    </div>
    <div class="gc-row">
      <div class="gc-col gc-col--wide field">
        <label class="label-muted">Steps</label>
        <div class="row-inline">
          <input class="slider slider-grow" type="range" min="1" max="150" step="1" :disabled="disabled" :value="stage.steps" @input="updateStage({ steps: toInt($event, stage.steps) })" />
          <div class="number-with-controls ml-steps">
            <input class="ui-input ui-input-sm w-step pad-right" type="number" min="1" max="150" step="1" :disabled="disabled" :value="stage.steps" @change="updateStage({ steps: toInt($event, stage.steps) })" />
            <div class="stepper">
              <button class="step-btn" type="button" title="Increase" :disabled="disabled" @click="stepsInc">+</button>
              <button class="step-btn" type="button" title="Decrease" :disabled="disabled" @click="stepsDec">−</button>
            </div>
          </div>
        </div>
      </div>
      <div class="gc-col gc-col--wide field">
        <label class="label-muted">CFG</label>
        <div class="row-inline">
          <input class="slider slider-grow" type="range" min="0" max="30" step="0.5" :disabled="disabled" :value="stage.cfgScale" @input="updateStage({ cfgScale: toFloat($event, stage.cfgScale) })" />
          <div class="number-with-controls ml-steps">
            <input class="ui-input ui-input-sm w-cfg pad-right" type="number" min="0" max="30" step="0.5" :disabled="disabled" :value="stage.cfgScale" @change="updateStage({ cfgScale: toFloat($event, stage.cfgScale) })" />
            <div class="stepper">
              <button class="step-btn" type="button" title="Increase" :disabled="disabled" @click="cfgInc">+</button>
              <button class="step-btn" type="button" title="Decrease" :disabled="disabled" @click="cfgDec">−</button>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="gc-row">
      <div class="gc-col gc-col--wide field">
        <label class="label-muted">Seed</label>
        <div class="number-with-controls w-full">
          <input class="ui-input ui-input-sm pad-right" type="number" :disabled="disabled" :value="stage.seed" @change="updateStage({ seed: toInt($event, stage.seed) })" />
          <div class="stepper">
            <button class="step-btn" type="button" :disabled="disabled" title="Random seed" @click="randomizeSeed">🎲</button>
            <button class="step-btn" type="button" :disabled="disabled || lastSeed === null" title="Reuse seed" @click="reuseSeed">↺</button>
          </div>
        </div>
      </div>
    </div>
    <div v-if="showModelDir" class="gc-row">
      <div class="gc-col field">
        <label class="label-muted">Model Dir</label>
        <input class="ui-input" type="text" :disabled="disabled" :value="stage.modelDir" @change="updateStage({ modelDir: ($event.target as HTMLInputElement).value })" placeholder="/path/to/high-or-low" />
      </div>
    </div>

    <div v-if="lightx2v" class="gc-row">
      <div class="gc-col gc-col--wide field">
        <label class="label-muted">LoRA (wan22-loras)</label>
        <select class="select-md" :disabled="disabled" :value="stage.loraPath" @change="updateStage({ loraPath: ($event.target as HTMLSelectElement).value })">
          <option value="">None</option>
          <option v-for="opt in loraChoices" :key="opt.path" :value="opt.path">{{ opt.name }}</option>
        </select>
      </div>
      <div v-if="stage.loraPath" class="gc-col field">
        <label class="label-muted">LoRA weight</label>
        <input class="ui-input" type="number" step="0.05" :disabled="disabled" :value="stage.loraWeight" @change="updateStage({ loraWeight: toFloat($event, stage.loraWeight) })" />
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

function stepsInc(): void {
  const v = Math.min(150, Math.max(1, Number(props.stage.steps) + 1))
  updateStage({ steps: v })
}

function stepsDec(): void {
  const v = Math.min(150, Math.max(1, Number(props.stage.steps) - 1))
  updateStage({ steps: v })
}

function cfgInc(): void {
  const step = 0.5
  const v = Math.min(30, Math.max(0, Number(props.stage.cfgScale) + step))
  updateStage({ cfgScale: Number(v.toFixed(2)) })
}

function cfgDec(): void {
  const step = 0.5
  const v = Math.min(30, Math.max(0, Number(props.stage.cfgScale) - step))
  updateStage({ cfgScale: Number(v.toFixed(2)) })
}

function randomizeSeed(): void {
  if (props.stage.seed !== -1) lastSeed.value = props.stage.seed
  updateStage({ seed: -1 })
}

function reuseSeed(): void {
  if (lastSeed.value !== null) updateStage({ seed: lastSeed.value })
}
</script>
