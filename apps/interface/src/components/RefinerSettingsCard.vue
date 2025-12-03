<template>
  <div class="refiner-card">
    <div class="rf-header">
      <span class="label-muted">Refiner</span>
      <button class="rf-switch" type="button" @click="toggle">
        <span class="rf-switch-track" :data-on="enabled ? '1' : '0'">
          <span class="rf-switch-thumb" />
        </span>
      </button>
    </div>
    <div v-if="enabled" class="rf-grid">
      <div class="rf-cell">
        <label class="label-muted">Model</label>
        <input class="ui-input ui-input-sm" type="text" :value="model" @change="onModelChange" />
      </div>
      <div class="rf-cell">
        <label class="label-muted">VAE</label>
        <input class="ui-input ui-input-sm" type="text" :value="vae" @change="onVaeChange" />
      </div>
      <div class="rf-cell">
        <label class="label-muted">Steps</label>
        <input class="ui-input ui-input-sm" type="number" min="0" :value="steps" @change="onStepsChange" />
      </div>
      <div class="rf-cell">
        <label class="label-muted">CFG</label>
        <input class="ui-input ui-input-sm" type="number" step="0.1" :value="cfg" @change="onCfgChange" />
      </div>
      <div class="rf-cell">
        <label class="label-muted">Seed</label>
        <input class="ui-input ui-input-sm" type="number" :value="seed" @change="onSeedChange" />
        <p class="rf-hint">Use -1 for random</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
// tags: refiner, settings, grid
const props = defineProps<{
  enabled: boolean
  steps: number
  cfg: number
  seed: number
  model?: string
  vae?: string
}>()

const emit = defineEmits<{
  (e: 'update:enabled', value: boolean): void
  (e: 'update:steps', value: number): void
  (e: 'update:cfg', value: number): void
  (e: 'update:seed', value: number): void
  (e: 'update:model', value: string): void
  (e: 'update:vae', value: string): void
}>()

function toggle(): void {
  emit('update:enabled', !props.enabled)
}

function onStepsChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:steps', Number.isNaN(v) || v < 0 ? 0 : v)
}

function onCfgChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:cfg', Number.isNaN(v) ? props.cfg : v)
}

function onSeedChange(event: Event): void {
  const v = Number((event.target as HTMLInputElement).value)
  emit('update:seed', Number.isNaN(v) ? props.seed : v)
}

function onModelChange(event: Event): void {
  emit('update:model', (event.target as HTMLInputElement).value)
}

function onVaeChange(event: Event): void {
  emit('update:vae', (event.target as HTMLInputElement).value)
}
</script>

<!-- styles in styles/components/refiner-settings-card.css -->

