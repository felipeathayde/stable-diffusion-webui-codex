<template>
    <div class="quicksettings-group">
      <label class="label-muted">WAN High model</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="highModel" @change="$emit('update:highModel', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="m in highChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
          </select>
          <button class="btn btn-sm btn-outline" type="button" @click="$emit('browseHigh')">Browse…</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">WAN Low model</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="lowModel" @change="$emit('update:lowModel', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="m in lowChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
          </select>
          <button class="btn btn-sm btn-outline" type="button" @click="$emit('browseLow')">Browse…</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">WAN Text Encoder</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ te }}</option>
          </select>
          <button class="btn btn-sm btn-outline" type="button" @click="$emit('browseTe')">Browse…</button>
        </div>
      </div>
    </div>

    <div class="quicksettings-group">
      <label class="label-muted">WAN VAE</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
            <option value="">{{ builtInLabel }}</option>
            <option v-for="v in vaeChoices" :key="v" :value="v">{{ v }}</option>
          </select>
          <button class="btn btn-sm btn-outline" type="button" @click="$emit('browseVae')">Browse…</button>
        </div>
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
        <button class="btn btn-sm btn-outline" type="button" @click="$emit('openOverrides')">
          Set overrides
        </button>
      </div>
    </div>
</template>

<script setup lang="ts">
const props = defineProps<{
  highModel: string
  highChoices: string[]
  lowModel: string
  lowChoices: string[]
  textEncoder: string
  textEncoderChoices: string[]
  vae: string
  vaeChoices: string[]
  unetDtype: string
  unetDtypeChoices: string[]
  gpuWeightsMb: number
  gpuTotalMb: number
  attentionBackend: string
  attentionChoices: Array<{ value: string; label: string }>
}>()

const builtInLabel = 'Built-in'

function dirLabel(path: string): string {
  const norm = path.replace(/\\/g, '/')
  if (!norm) return ''
  const idx = norm.lastIndexOf('/')
  return idx >= 0 ? norm.slice(idx + 1) || norm : norm
}
</script>
