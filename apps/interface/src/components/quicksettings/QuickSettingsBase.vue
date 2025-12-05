<template>
    <div class="quicksettings-group">
      <label class="label-muted">Mode</label>
      <div class="qs-row">
        <select class="select-md" :value="mode" @change="$emit('update:mode', ($event.target as HTMLSelectElement).value)">
          <option v-for="m in modeChoices" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>
    </div>
    <div class="quicksettings-group" v-if="!hideCheckpoint">
      <label class="label-muted">Checkpoint</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
            <option v-for="model in checkpoints" :key="model" :value="model">
              {{ model }}
            </option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" @click="$emit('addCheckpointPath')">+</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">VAE</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
            <option v-for="v in vaeChoices" :key="v" :value="v">
              {{ v === 'Automatic' ? 'Built-in' : v }}
            </option>
          </select>
          <button class="btn btn-outline qs-inline-btn" type="button" @click="$emit('addVaePath')">+</button>
        </div>
      </div>
    </div>

    <div v-if="showTextEncoder !== false" class="quicksettings-group">
      <label class="label-muted">Text Encoder</label>
      <div class="qs-row">
        <select class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ textEncoderAutomaticLabel }}</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ te }}</option>
        </select>
      </div>
    </div>

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
        <label class="qs-toggle">
          <input
            type="checkbox"
            :checked="smartOffload"
            @change="$emit('update:smartOffload', ($event.target as HTMLInputElement).checked)"
          />
          <span>Unload TE/UNet/VAE between stages</span>
        </label>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Smart Fallback</label>
      <div class="qs-row">
        <label class="qs-toggle">
          <input
            type="checkbox"
            :checked="smartFallback"
            @change="$emit('update:smartFallback', ($event.target as HTMLInputElement).checked)"
          />
          <span>Fallback to CPU on OOM</span>
        </label>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Smart Cache</label>
      <div class="qs-row">
        <label class="qs-toggle">
          <input
            type="checkbox"
            :checked="smartCache"
            @change="$emit('update:smartCache', ($event.target as HTMLInputElement).checked)"
          />
          <span>Cache TEnc/embeds (SDXL)</span>
        </label>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Attention Backend</label>
      <div class="qs-row">
        <select class="select-md" :value="attentionBackend" @change="$emit('update:attentionBackend', ($event.target as HTMLSelectElement).value)">
          <option v-for="opt in attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">Overrides</label>
      <div class="qs-row">
        <button class="btn btn-secondary qs-overrides-btn" type="button" @click="$emit('openOverrides')">
          Set overrides
        </button>
      </div>
    </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  mode: string
  modeChoices: string[]
  checkpoint: string
  checkpoints: string[]
  hideCheckpoint: boolean
  vae: string
  vaeChoices: string[]
  textEncoder: string
  textEncoderChoices: any
  unetDtype: string
  unetDtypeChoices: string[]
  gpuWeightsMb: number
  gpuTotalMb: number
  attentionBackend: string
  attentionChoices: Array<{ value: string; label: string }>
  textEncoderAutomaticLabel?: string
  showTextEncoder?: boolean
  smartOffload: boolean
  smartFallback: boolean
  smartCache: boolean
}>()

defineEmits<{
  (e: 'update:mode', value: string): void
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'update:unetDtype', value: string): void
  (e: 'update:gpuWeightsMb', value: number): void
  (e: 'update:attentionBackend', value: string): void
  (e: 'update:smartOffload', value: boolean): void
  (e: 'update:smartFallback', value: boolean): void
  (e: 'update:smartCache', value: boolean): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'openOverrides'): void
}>()

const textEncoderAutomaticLabel = props.textEncoderAutomaticLabel ?? 'Built-in'
</script>
