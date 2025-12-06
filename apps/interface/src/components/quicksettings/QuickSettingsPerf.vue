<template>
  <div class="quicksettings-group">
    <label class="label-muted">Diffusion in Low Bits</label>
    <div class="qs-row">
      <select class="select-md" :value="unetDtype" @change="$emit('update:unetDtype', ($event.target as HTMLSelectElement).value)">
        <option v-for="opt in unetDtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
      </select>
    </div>
  </div>

  <div class="quicksettings-group">
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

  <div class="quicksettings-group">
    <label class="label-muted">Smart Offload</label>
    <div class="qs-row">
      <label class="qs-switch">
        <input
          type="checkbox"
          :checked="smartOffload"
          @change="$emit('update:smartOffload', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="qs-switch-label">Unload TE/UNet/VAE between stages</span>
      </label>
    </div>
  </div>

  <div class="quicksettings-group">
    <label class="label-muted">Smart Fallback</label>
    <div class="qs-row">
      <label class="qs-switch">
        <input
          type="checkbox"
          :checked="smartFallback"
          @change="$emit('update:smartFallback', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="qs-switch-label">Fallback to CPU on OOM</span>
      </label>
    </div>
  </div>

  <div class="quicksettings-group">
    <label class="label-muted">Smart Cache</label>
    <div class="qs-row">
      <label class="qs-switch">
        <input
          type="checkbox"
          :checked="smartCache"
          @change="$emit('update:smartCache', ($event.target as HTMLInputElement).checked)"
        />
        <span class="qs-switch-track"><span class="qs-switch-thumb" /></span>
        <span class="qs-switch-label">Cache TEnc/embeds (SDXL)</span>
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
}>()

defineEmits<{
  (e: 'update:unetDtype', value: string): void
  (e: 'update:gpuWeightsMb', value: number): void
  (e: 'update:smartOffload', value: boolean): void
  (e: 'update:smartFallback', value: boolean): void
  (e: 'update:smartCache', value: boolean): void
}>()
</script>
*** End Patch***  )->assistant to=functions.apply_patch ಸಂದ ***!
