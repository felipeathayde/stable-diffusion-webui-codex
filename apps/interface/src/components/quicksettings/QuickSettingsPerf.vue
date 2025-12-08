<template>
  <!-- GPU VRAM -->
  <div class="quicksettings-group qs-group-perf qs-group-perf-vram">
    <label class="label-muted">GPU VRAM (MB)</label>
    <div class="qs-row">
      <input
        class="ui-input"
        type="number"
        :min="0"
        :max="gpuTotalMb"
        :value="gpuWeightsMb"
        @change="$emit('update:gpuWeightsMb', Number(($event.target as HTMLInputElement).value))"
      />
    </div>
  </div>

  <!-- Smart Offload -->
  <div class="quicksettings-group qs-group-perf qs-group-perf-offload">
    <label class="label-muted">Smart Offload</label>
    <div class="qs-row">
      <label class="qs-switch" title="Unload TE/UNet/VAE between stages to save VRAM">
        <input
          type="checkbox"
          :checked="smartOffload"
          @change="$emit('update:smartOffload', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
      </label>
    </div>
  </div>

  <!-- Smart Fallback -->
  <div class="quicksettings-group qs-group-perf qs-group-perf-fallback">
    <label class="label-muted">Smart Fallback</label>
    <div class="qs-row">
      <label class="qs-switch" title="Fallback to CPU when GPU runs out of memory">
        <input
          type="checkbox"
          :checked="smartFallback"
          @change="$emit('update:smartFallback', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
      </label>
    </div>
  </div>

  <!-- Smart Cache -->
  <div class="quicksettings-group qs-group-perf qs-group-perf-cache">
    <label class="label-muted">Smart Cache</label>
    <div class="qs-row">
      <label class="qs-switch" title="Cache text encoder embeddings for faster subsequent generations">
        <input
          type="checkbox"
          :checked="smartCache"
          @change="$emit('update:smartCache', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
      </label>
    </div>
  </div>

  <!-- Core Streaming -->
  <div class="quicksettings-group qs-group-perf qs-group-perf-streaming">
    <label class="label-muted">Core Streaming</label>
    <div class="qs-row">
      <label class="qs-switch" title="Stream model blocks from RAM for large quantized models (GGUF)">
        <input
          type="checkbox"
          :checked="coreStreaming"
          @change="$emit('update:coreStreaming', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
      </label>
    </div>
  </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  unetDtype: string
  unetDtypeChoices: string[]
  gpuWeightsMb: number
  gpuTotalMb: number
  smartOffload: boolean
  smartFallback: boolean
  smartCache: boolean
  coreStreaming: boolean
}>()

defineEmits<{
  (e: 'update:unetDtype', value: string): void
  (e: 'update:gpuWeightsMb', value: number): void
  (e: 'update:smartOffload', value: boolean): void
  (e: 'update:smartFallback', value: boolean): void
  (e: 'update:smartCache', value: boolean): void
  (e: 'update:coreStreaming', value: boolean): void
}>()
</script>
