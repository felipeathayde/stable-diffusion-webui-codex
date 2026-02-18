<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared run-progress status block.
Renders Stage/Progress/Step/ETA in a single reusable block for Run cards across views.

Symbols (top-level; keep in sync; no ghosts):
- `RunProgressStatus` (component): Shared status block for run progress details.
- `normalizedPercent` (const): Safe normalized percent value for display/progress bar.
-->

<template>
  <div class="panel-progress run-progress-status">
    <p><strong>Stage:</strong> {{ stageLabel }}</p>
    <p v-if="normalizedPercent !== null">Progress: {{ normalizedPercent.toFixed(1) }}%</p>
    <progress v-if="showProgressBar && normalizedPercent !== null" class="run-progress-status__bar" :value="normalizedPercent" max="100"></progress>
    <p v-if="step !== null && totalSteps !== null">Step {{ step }} / {{ totalSteps }}</p>
    <p v-if="etaSeconds !== null" class="caption">ETA ~ {{ etaSeconds.toFixed(0) }}s</p>
    <div v-if="queueLabel || $slots.extra" class="run-progress-status__extra">
      <span v-if="queueLabel" class="caption">{{ queueLabel }}</span>
      <slot name="extra" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = withDefaults(defineProps<{
  stage?: string | null
  percent?: number | null
  step?: number | null
  totalSteps?: number | null
  etaSeconds?: number | null
  queueLabel?: string
  showProgressBar?: boolean
}>(), {
  stage: 'running',
  percent: null,
  step: null,
  totalSteps: null,
  etaSeconds: null,
  queueLabel: '',
  showProgressBar: false,
})

const stageLabel = computed(() => String(props.stage || 'running'))
const normalizedPercent = computed(() => {
  if (props.percent === null || props.percent === undefined) return null
  if (!Number.isFinite(props.percent)) return null
  return Math.max(0, Math.min(100, props.percent))
})
const step = computed(() => props.step)
const totalSteps = computed(() => props.totalSteps)
const etaSeconds = computed(() => props.etaSeconds)
const queueLabel = computed(() => String(props.queueLabel || '').trim())
const showProgressBar = computed(() => Boolean(props.showProgressBar))
</script>

