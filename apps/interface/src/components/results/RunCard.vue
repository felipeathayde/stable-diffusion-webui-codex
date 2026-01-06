<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Results run card wrapper (Generate button + batch controls popover).
Renders a `ResultsCard` with a Generate CTA and an optional batch settings panel (count/size) that can be reused across image/video views.

Symbols (top-level; keep in sync; no ghosts):
- `RunCard` (component): Run/results card with generate CTA and optional batch controls.
- `setBatchCount` (function): Emits a clamped batch-count update.
- `setBatchSize` (function): Emits a clamped batch-size update.
- `toggleBatchMenu` (function): Toggles the batch settings popover.
- `openBatchMenu` (function): Opens the batch settings popover and schedules positioning.
- `closeBatchMenu` (function): Closes the batch settings popover and clears handlers.
- `isEventInsideBatchMenu` (function): Checks whether a DOM event target is inside the menu panel/toggle.
- `onDocumentPointerDown` (function): Outside-click handler that closes the popover.
- `onDocumentKeyDown` (function): Keydown handler (Escape closes the popover).
- `updateBatchMenuPosition` (function): Computes and applies the popover position style.
- `scheduleBatchMenuPositionUpdate` (function): Debounced/nextTick positioning update helper.
- `clampInt` (function): Clamps and truncates numeric values to an integer range.
-->

<template>
  <ResultsCard
    :title="props.title"
    headerClass="three-cols results-sticky run-sticky"
    headerRightClass="run-controls"
    :showGenerate="true"
    :generateId="props.generateId"
    :generateButtonClass="props.generateButtonClass"
    :generateLabel="props.generateLabel"
    :runningLabel="props.runningLabel"
    :generateDisabled="props.generateDisabled"
    :generateTitle="props.generateTitle"
    :isRunning="props.isRunning"
    @generate="emit('generate')"
  >
    <template #header-right>
      <template v-if="props.showBatchControls">
        <div class="run-control run-batch-menu" :class="{ 'is-open': isBatchMenuOpen }">
          <button
            ref="batchMenuToggleEl"
            class="btn btn-sm btn-outline run-batch-menu__toggle"
            type="button"
            :disabled="inputsDisabled"
            :aria-expanded="isBatchMenuOpen ? 'true' : 'false'"
            aria-haspopup="dialog"
            title="Batch settings"
            @click="toggleBatchMenu"
          >
            Batch {{ props.batchCount }}×{{ props.batchSize }}
          </button>

          <Teleport to="body">
            <div
              v-if="isBatchMenuOpen"
              ref="batchMenuPanelEl"
              class="run-batch-menu__panel panel"
              :style="batchMenuStyle"
              role="dialog"
              aria-label="Batch settings"
            >
              <div class="run-batch-menu__rows">
                <div class="run-batch-menu__row">
                  <span class="caption">Batch count</span>
                  <NumberStepperInput
                    :modelValue="props.batchCount"
                    :min="minBatchCount"
                    :max="maxBatchCount"
                    :step="1"
                    :nudgeStep="1"
                    size="sm"
                    inputClass="cdx-input-w-xs"
                    :disabled="inputsDisabled"
                    updateOnInput
                    @update:modelValue="setBatchCount"
                  />
                </div>
                <div class="run-batch-menu__row">
                  <span class="caption">Batch size</span>
                  <NumberStepperInput
                    :modelValue="props.batchSize"
                    :min="minBatchSize"
                    :max="maxBatchSize"
                    :step="1"
                    :nudgeStep="1"
                    size="sm"
                    inputClass="cdx-input-w-xs"
                    :disabled="inputsDisabled"
                    updateOnInput
                    @update:modelValue="setBatchSize"
                  />
                </div>
              </div>

              <div class="run-batch-menu__actions">
                <button class="btn btn-sm btn-primary" type="button" :disabled="inputsDisabled" @click="closeBatchMenu">OK</button>
              </div>
            </div>
          </Teleport>
        </div>
      </template>
      <slot name="header-right" />
    </template>

    <slot />
  </ResultsCard>
</template>

<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import type { StyleValue } from 'vue'
import NumberStepperInput from '../ui/NumberStepperInput.vue'
import ResultsCard from './ResultsCard.vue'

const props = withDefaults(defineProps<{
  title?: string
  generateId?: string
  generateButtonClass?: string
  generateLabel?: string
  runningLabel?: string
  generateDisabled?: boolean
  generateTitle?: string
  isRunning?: boolean
  showBatchControls?: boolean
  batchCount?: number
  batchSize?: number
  disabled?: boolean
  minBatchCount?: number
  maxBatchCount?: number
  minBatchSize?: number
  maxBatchSize?: number
}>(), {
  title: 'Run',
  generateId: '',
  generateButtonClass: 'btn btn-md btn-primary results-generate',
  generateLabel: 'Generate',
  runningLabel: 'Running…',
  generateDisabled: false,
  generateTitle: '',
  isRunning: false,
  showBatchControls: true,
  batchCount: 1,
  batchSize: 1,
  disabled: false,
  minBatchCount: 1,
  maxBatchCount: 999,
  minBatchSize: 1,
  maxBatchSize: 999,
})

const emit = defineEmits<{
  (e: 'generate'): void
  (e: 'update:batchCount', value: number): void
  (e: 'update:batchSize', value: number): void
}>()

const inputsDisabled = computed(() => Boolean(props.disabled || props.generateDisabled))

const minBatchCount = computed(() => Number.isFinite(props.minBatchCount) ? Math.trunc(Number(props.minBatchCount)) : 1)
const maxBatchCount = computed(() => Number.isFinite(props.maxBatchCount) ? Math.trunc(Number(props.maxBatchCount)) : 999)
const minBatchSize = computed(() => Number.isFinite(props.minBatchSize) ? Math.trunc(Number(props.minBatchSize)) : 1)
const maxBatchSize = computed(() => Number.isFinite(props.maxBatchSize) ? Math.trunc(Number(props.maxBatchSize)) : 999)

const batchMenuToggleEl = ref<HTMLElement | null>(null)
const batchMenuPanelEl = ref<HTMLElement | null>(null)
const isBatchMenuOpen = ref(false)
const batchMenuStyle = ref<StyleValue | undefined>(undefined)
let batchMenuRAF: number | null = null

watch(() => props.showBatchControls, (show) => {
  if (!show) closeBatchMenu()
})

watch(inputsDisabled, (disabled) => {
  if (disabled) closeBatchMenu()
})

function setBatchCount(value: number): void {
  emit('update:batchCount', clampInt(value, minBatchCount.value, maxBatchCount.value))
}

function setBatchSize(value: number): void {
  emit('update:batchSize', clampInt(value, minBatchSize.value, maxBatchSize.value))
}

function toggleBatchMenu(): void {
  if (isBatchMenuOpen.value) {
    closeBatchMenu()
    return
  }

  openBatchMenu()
}

function openBatchMenu(): void {
  if (inputsDisabled.value) return

  isBatchMenuOpen.value = true

  void nextTick(() => {
    updateBatchMenuPosition()
    const firstInput = batchMenuPanelEl.value?.querySelector<HTMLInputElement>('input')
    firstInput?.focus()
  })
}

function closeBatchMenu(): void {
  isBatchMenuOpen.value = false
}

function isEventInsideBatchMenu(event: Event): boolean {
  const target = event.target
  if (!(target instanceof Node)) return false

  const toggle = batchMenuToggleEl.value
  const panel = batchMenuPanelEl.value
  return Boolean((toggle && toggle.contains(target)) || (panel && panel.contains(target)))
}

function onDocumentPointerDown(event: PointerEvent): void {
  if (!isBatchMenuOpen.value) return
  if (isEventInsideBatchMenu(event)) return
  closeBatchMenu()
}

function onDocumentKeyDown(event: KeyboardEvent): void {
  if (!isBatchMenuOpen.value) return
  if (event.key !== 'Escape') return
  event.preventDefault()
  closeBatchMenu()
}

function updateBatchMenuPosition(): void {
  const toggle = batchMenuToggleEl.value
  if (!toggle) return

  const rect = toggle.getBoundingClientRect()
  const viewportPadding = 8
  const gap = 6
  const top = rect.bottom + gap
  const right = Math.max(viewportPadding, window.innerWidth - rect.right)
  const maxHeight = Math.max(160, window.innerHeight - top - viewportPadding)

  batchMenuStyle.value = {
    position: 'fixed',
    top: `${top}px`,
    right: `${right}px`,
    maxHeight: `${maxHeight}px`,
  }
}

function scheduleBatchMenuPositionUpdate(): void {
  if (!isBatchMenuOpen.value) return
  if (batchMenuRAF !== null) return

  batchMenuRAF = window.requestAnimationFrame(() => {
    batchMenuRAF = null
    updateBatchMenuPosition()
  })
}

onMounted(() => {
  document.addEventListener('pointerdown', onDocumentPointerDown)
  document.addEventListener('keydown', onDocumentKeyDown)
  window.addEventListener('resize', scheduleBatchMenuPositionUpdate)
  window.addEventListener('scroll', scheduleBatchMenuPositionUpdate, true)
})

onBeforeUnmount(() => {
  document.removeEventListener('pointerdown', onDocumentPointerDown)
  document.removeEventListener('keydown', onDocumentKeyDown)
  window.removeEventListener('resize', scheduleBatchMenuPositionUpdate)
  window.removeEventListener('scroll', scheduleBatchMenuPositionUpdate, true)
  if (batchMenuRAF !== null) window.cancelAnimationFrame(batchMenuRAF)
})

function clampInt(value: number, min: number, max: number): number {
  const n = Number.isFinite(value) ? Math.trunc(value) : min
  return Math.min(max, Math.max(min, n))
}
</script>
