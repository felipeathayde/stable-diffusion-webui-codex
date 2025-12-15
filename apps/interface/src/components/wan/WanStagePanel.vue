<template>
  <div class="panel">
    <div class="panel-header">
      <h3 class="h4">{{ title }}</h3>
    </div>
    <div class="panel-body">
      <div class="grid grid-3">
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
          <label class="label">Steps</label>
          <input
            class="ui-input"
            type="number"
            min="1"
            :disabled="disabled"
            :value="stage.steps"
            @change="updateStage({ steps: toInt($event, stage.steps) })"
          />
        </div>
      </div>

      <div class="grid grid-3 mt-2">
        <div>
          <label class="label">CFG</label>
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
          <label class="label">Seed</label>
          <div class="grid grid-3">
            <input
              class="ui-input"
              type="number"
              :disabled="disabled"
              :value="stage.seed"
              @change="updateStage({ seed: toInt($event, stage.seed) })"
            />
            <button class="btn btn-sm" type="button" :disabled="disabled" @click="randomizeSeed">Random</button>
            <button class="btn btn-sm" type="button" :disabled="disabled || lastSeed === null" @click="reuseSeed">Reuse</button>
          </div>
        </div>
        <div v-if="showModelDir">
          <label class="label">Model Dir</label>
          <input
            class="ui-input"
            type="text"
            :disabled="disabled"
            :value="stage.modelDir"
            @change="updateStage({ modelDir: ($event.target as HTMLInputElement).value })"
            placeholder="/path/to/high-or-low"
          />
        </div>
      </div>

      <div class="panel-sub mt-2">
        <label class="switch-label">
          <input type="checkbox" :disabled="disabled" :checked="stage.lightning" @change="onLightning" />
          <span>Lightning</span>
        </label>
        <label class="switch-label ml-4">
          <input type="checkbox" :disabled="disabled" :checked="stage.loraEnabled" @change="onLoraEnabled" />
          <span>Use LoRA</span>
        </label>
        <div v-if="stage.loraEnabled" class="grid grid-2 mt-2">
          <div>
            <label class="label">LoRA Path</label>
            <input
              class="ui-input"
              type="text"
              :disabled="disabled"
              :value="stage.loraPath"
              @change="updateStage({ loraPath: ($event.target as HTMLInputElement).value })"
            />
          </div>
          <div>
            <label class="label">Weight</label>
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
      </div>

      <div v-if="showModelDir && !stage.modelDir" class="error mt-2">{{ title }}: model directory is empty.</div>
      <div v-if="stage.loraEnabled && !stage.loraPath" class="error mt-1">{{ title }}: LoRA enabled but path is empty.</div>
    </div>
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
  showModelDir?: boolean
  disabled?: boolean
  samplerLabel?: string
  schedulerLabel?: string
}>(), {
  showModelDir: false,
  disabled: false,
  samplerLabel: 'Sampler',
  schedulerLabel: 'Scheduler',
})

const emit = defineEmits<{
  (e: 'update:stage', patch: Partial<WanStageParams>): void
}>()

const lastSeed = ref<number | null>(null)

const samplerLabel = computed(() => props.samplerLabel)
const schedulerLabel = computed(() => props.schedulerLabel)

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

function onLightning(e: Event): void {
  updateStage({ lightning: (e.target as HTMLInputElement).checked })
}

function onLoraEnabled(e: Event): void {
  updateStage({ loraEnabled: (e.target as HTMLInputElement).checked })
}
</script>
