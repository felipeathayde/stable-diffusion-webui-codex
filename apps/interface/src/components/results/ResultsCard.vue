<template>
  <div class="panel">
    <div class="panel-header" :class="props.headerClass">
      <slot name="header-left">
        {{ props.title }}
      </slot>

      <div class="header-center" :class="props.headerCenterClass">
        <slot name="header-center">
          <button
            v-if="props.showGenerate"
            :id="props.generateId || undefined"
            :class="props.generateButtonClass"
            type="button"
            :disabled="props.generateDisabled"
            :title="props.generateTitle"
            @click="emit('generate')"
          >
            {{ props.isRunning ? props.runningLabel : props.generateLabel }}
          </button>
        </slot>
      </div>

      <div class="header-right" :class="props.headerRightClass">
        <slot name="header-right" />
      </div>
    </div>

    <div class="panel-body" :class="props.bodyClass" :style="props.bodyStyle">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
import type { StyleValue } from 'vue'

const props = withDefaults(defineProps<{
  title?: string
  headerClass?: string
  headerCenterClass?: string
  headerRightClass?: string
  bodyClass?: string
  bodyStyle?: StyleValue
  showGenerate?: boolean
  generateId?: string
  generateButtonClass?: string
  generateLabel?: string
  runningLabel?: string
  generateDisabled?: boolean
  generateTitle?: string
  isRunning?: boolean
}>(), {
  title: 'Results',
  headerClass: 'three-cols',
  headerCenterClass: '',
  headerRightClass: '',
  bodyClass: '',
  bodyStyle: undefined,
  showGenerate: true,
  generateId: '',
  generateButtonClass: 'btn btn-md btn-primary results-generate',
  generateLabel: 'Generate',
  runningLabel: 'Running…',
  generateDisabled: false,
  generateTitle: '',
  isRunning: false,
})

const emit = defineEmits<{
  (e: 'generate'): void
}>()
</script>
