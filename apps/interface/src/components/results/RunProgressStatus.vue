<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Shared run-status panel block.
Renders progress and non-progress run statuses (error/warning/info/success) in a single reusable panel for Run cards across views,
including severity-aware visuals and animated status icons.

Symbols (top-level; keep in sync; no ghosts):
- `RunProgressStatus` (component): Shared status panel for run progress and run notices/errors.
- `normalizeStatusVariant` (function): Normalizes incoming status variants into the supported union.
- `resolvedVariant` (const): Normalized status variant that drives panel styling and icon selection.
- `normalizedPercent` (const): Safe normalized percent value for display/progress bar.
-->

<template>
  <div
    :class="[
      'panel-status',
      'run-progress-status',
      `run-progress-status--${resolvedVariant}`,
    ]"
    :data-variant="resolvedVariant"
    :role="ariaRole"
    :aria-live="ariaLive"
  >
    <div class="run-progress-status__header">
      <span class="run-progress-status__icon" aria-hidden="true">
        <svg
          v-if="resolvedVariant === 'progress'"
          class="run-progress-status__icon-svg run-progress-status__icon-svg--spinner"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle class="run-progress-status__spinner-track" cx="12" cy="12" r="9"></circle>
          <circle class="run-progress-status__spinner-head" cx="12" cy="12" r="9"></circle>
        </svg>
        <svg
          v-else-if="resolvedVariant === 'error'"
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path d="M12 3L21 19H3L12 3Z"></path>
          <path d="M12 9V13"></path>
          <circle cx="12" cy="17" r="1"></circle>
        </svg>
        <svg
          v-else-if="resolvedVariant === 'warning'"
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <path d="M12 3L21 19H3L12 3Z"></path>
          <path d="M12 9V14"></path>
          <circle cx="12" cy="17.25" r="1"></circle>
        </svg>
        <svg
          v-else-if="resolvedVariant === 'success'"
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle cx="12" cy="12" r="9"></circle>
          <path d="M8.5 12.5L10.75 14.75L15.5 10"></path>
        </svg>
        <svg
          v-else
          class="run-progress-status__icon-svg"
          viewBox="0 0 24 24"
          fill="none"
        >
          <circle cx="12" cy="12" r="9"></circle>
          <path d="M12 10V16"></path>
          <circle cx="12" cy="7.5" r="1"></circle>
        </svg>
      </span>

      <div class="run-progress-status__headline">
        <p class="run-progress-status__title">{{ resolvedTitle }}</p>
        <p v-if="messageText" class="run-progress-status__message">{{ messageText }}</p>
        <p v-else-if="isProgressVariant" class="run-progress-status__message"><strong>Stage:</strong> {{ stageLabel }}</p>
      </div>

      <div v-if="isProgressVariant && normalizedPercent !== null" class="run-progress-status__percent">
        {{ normalizedPercent.toFixed(1) }}%
      </div>
    </div>

    <progress
      v-if="isProgressVariant && showProgressBar && normalizedPercent !== null"
      class="run-progress-status__bar"
      :value="normalizedPercent"
      max="100"
    ></progress>

    <div
      v-if="isProgressVariant && (step !== null && totalSteps !== null || etaSeconds !== null || queueLabel || $slots.extra)"
      class="run-progress-status__meta"
    >
      <span v-if="step !== null && totalSteps !== null" class="run-progress-status__meta-item">Step {{ step }} / {{ totalSteps }}</span>
      <span v-if="etaSeconds !== null" class="run-progress-status__meta-item">ETA ~ {{ etaSeconds.toFixed(0) }}s</span>
      <span v-if="queueLabel" class="run-progress-status__meta-item">{{ queueLabel }}</span>
      <slot name="extra" />
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

type RunStatusVariant = 'progress' | 'error' | 'warning' | 'info' | 'success'

const props = withDefaults(defineProps<{
  variant?: RunStatusVariant | string
  title?: string
  message?: string
  stage?: string | null
  percent?: number | null
  step?: number | null
  totalSteps?: number | null
  etaSeconds?: number | null
  queueLabel?: string
  showProgressBar?: boolean
}>(), {
  variant: 'progress',
  title: '',
  message: '',
  stage: 'running',
  percent: null,
  step: null,
  totalSteps: null,
  etaSeconds: null,
  queueLabel: '',
  showProgressBar: true,
})

function normalizeStatusVariant(rawVariant: string): RunStatusVariant {
  const variant = String(rawVariant || '').trim().toLowerCase()
  if (variant === 'error' || variant === 'warning' || variant === 'info' || variant === 'success') return variant
  return 'progress'
}

const resolvedVariant = computed<RunStatusVariant>(() => normalizeStatusVariant(String(props.variant || 'progress')))
const isProgressVariant = computed(() => resolvedVariant.value === 'progress')
const messageText = computed(() => String(props.message || '').trim())
const stageLabel = computed(() => String(props.stage || 'running'))
const resolvedTitle = computed(() => {
  const customTitle = String(props.title || '').trim()
  if (customTitle) return customTitle
  if (resolvedVariant.value === 'error') return 'Run failed'
  if (resolvedVariant.value === 'warning') return 'Warning'
  if (resolvedVariant.value === 'info') return 'Info'
  if (resolvedVariant.value === 'success') return 'Success'
  return 'Running'
})
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
const ariaRole = computed(() => (resolvedVariant.value === 'error' ? 'alert' : 'status'))
const ariaLive = computed(() => (resolvedVariant.value === 'error' ? 'assertive' : 'polite'))
</script>
