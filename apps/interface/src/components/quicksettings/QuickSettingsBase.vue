<template>
	    <div class="quicksettings-group qs-group-mode">
	      <label class="label-muted">Mode</label>
	      <div class="qs-row">
	        <select class="select-md" :value="mode" @change="$emit('update:mode', ($event.target as HTMLSelectElement).value)">
	          <option v-for="m in modeChoices" :key="m" :value="m">{{ m }}</option>
	        </select>
	      </div>
	    </div>
	    <div class="quicksettings-group qs-group-checkpoint">
	      <label class="label-muted">Checkpoint</label>
	      <div class="qs-row">
	        <div class="qs-pair">
	          <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
	            <option v-for="model in checkpoints" :key="model" :value="model">
	              {{ model }}
	            </option>
	          </select>
	          <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addCheckpointPath')">+</button>
	        </div>
	      </div>
	    </div>

    <div class="quicksettings-group qs-group-vae">
      <label class="label-muted">VAE</label>
      <div class="qs-row">
        <div class="qs-pair">
          <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
            <option v-for="v in vaeChoices" :key="v" :value="v">
              {{ v === 'Automatic' ? 'Built-in' : v }}
            </option>
          </select>
          <button class="btn qs-btn-outline qs-inline-btn" type="button" @click="$emit('addVaePath')">+</button>
        </div>
      </div>
    </div>

    <div v-if="showTextEncoder !== false" class="quicksettings-group qs-group-text-encoder">
      <label class="label-muted">Text Encoder</label>
      <div class="qs-row">
        <select class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
          <option value="">{{ textEncoderAutomaticLabel }}</option>
          <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ textEncoderLabel(te) }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group qs-group-attention">
      <label class="label-muted">Attention Backend</label>
      <div class="qs-row">
        <select class="select-md" :value="attentionBackend" @change="$emit('update:attentionBackend', ($event.target as HTMLSelectElement).value)">
          <option v-for="opt in attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>

    <div class="quicksettings-group qs-group-overrides">
      <label class="label-muted">Overrides</label>
      <div class="qs-row">
        <button class="btn qs-btn-secondary qs-overrides-btn" type="button" @click="$emit('openOverrides')">
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
	  vae: string
	  vaeChoices: string[]
	  textEncoder: string
	  textEncoderChoices: any
  attentionBackend: string
  attentionChoices: Array<{ value: string; label: string }>
  textEncoderAutomaticLabel?: string
  showTextEncoder?: boolean
}>()

defineEmits<{
  (e: 'update:mode', value: string): void
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoder', value: string): void
  (e: 'update:attentionBackend', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'openOverrides'): void
}>()

const textEncoderAutomaticLabel = props.textEncoderAutomaticLabel ?? 'Built-in'

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const [family, ...rest] = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (!family || rest.length === 0) return value
  const basename = rest[rest.length - 1] || rest[0]
  return `${family}/${basename}`
}
</script>
