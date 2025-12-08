<template>
  <!-- Flux-specific quicksettings row -->
  <div class="quicksettings-group qs-group-checkpoint">
    <label class="label-muted">Checkpoint</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="checkpoint" @change="$emit('update:checkpoint', ($event.target as HTMLSelectElement).value)">
          <option v-for="model in checkpoints" :key="model" :value="model">{{ truncatePath(model) }}</option>
        </select>
        <button class="btn btn-outline qs-inline-btn" type="button" @click="$emit('addCheckpointPath')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-vae">
    <label class="label-muted">VAE</label>
    <div class="qs-row">
      <div class="qs-pair">
        <select class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
          <option v-for="v in vaeChoices" :key="v" :value="v">{{ v === 'Automatic' ? 'Built-in' : truncatePath(v) }}</option>
        </select>
        <button class="btn btn-outline qs-inline-btn" type="button" @click="$emit('addVaePath')">+</button>
      </div>
    </div>
  </div>

  <div class="quicksettings-group qs-group-flux-tenc">
    <label class="label-muted">Text Encoders</label>
    <div class="qs-row">
      <select class="select-md" :value="textEncoderPrimary" @change="$emit('update:textEncoderPrimary', ($event.target as HTMLSelectElement).value)">
        <option value="">Built-in T5</option>
        <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ textEncoderLabel(te) }}</option>
      </select>
      <select class="select-md" :value="textEncoderSecondary" @change="$emit('update:textEncoderSecondary', ($event.target as HTMLSelectElement).value)">
        <option value="">CLIP (optional)</option>
        <option v-for="te in textEncoderChoices" :key="`sec-${te}`" :value="te">{{ textEncoderLabel(te) }}</option>
      </select>
    </div>
  </div>

  <div class="quicksettings-group qs-group-unet-dtype">
    <label class="label-muted">Unet Dtype</label>
    <div class="qs-row">
      <select class="select-md" :value="unetDtype" @change="$emit('update:unetDtype', ($event.target as HTMLSelectElement).value)">
        <option v-for="dt in unetDtypeChoices" :key="dt" :value="dt">{{ dt }}</option>
      </select>
    </div>
  </div>

  <div class="quicksettings-group qs-group-attention">
    <label class="label-muted">Attention</label>
    <div class="qs-row">
      <select class="select-md" :value="attentionBackend" @change="$emit('update:attentionBackend', ($event.target as HTMLSelectElement).value)">
        <option v-for="opt in attentionChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
      </select>
    </div>
  </div>

  <div class="quicksettings-group qs-group-overrides">
    <label class="label-muted">Overrides</label>
    <div class="qs-row">
      <button class="btn btn-secondary qs-overrides-btn" type="button" @click="$emit('openOverrides')">Set overrides</button>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  checkpoint: string
  checkpoints: string[]
  vae: string
  vaeChoices: string[]
  textEncoderPrimary: string
  textEncoderSecondary: string
  textEncoderChoices: string[]
  unetDtype: string
  unetDtypeChoices: string[]
  attentionBackend: string
  attentionChoices: Array<{ value: string; label: string }>
}>()

defineEmits<{
  (e: 'update:checkpoint', value: string): void
  (e: 'update:vae', value: string): void
  (e: 'update:textEncoderPrimary', value: string): void
  (e: 'update:textEncoderSecondary', value: string): void
  (e: 'update:unetDtype', value: string): void
  (e: 'update:attentionBackend', value: string): void
  (e: 'addCheckpointPath'): void
  (e: 'addVaePath'): void
  (e: 'openOverrides'): void
}>()

function truncatePath(path: string, maxLen = 40): string {
  if (!path || path.length <= maxLen) return path
  const parts = path.replace(/\\/g, '/').split('/')
  const name = parts[parts.length - 1] || path
  return name.length > maxLen ? `...${name.slice(-maxLen)}` : name
}

function textEncoderLabel(raw: unknown): string {
  const value = String(raw ?? '')
  if (!value.includes('/')) return value
  const parts = value.replace(/\\/g, '/').split('/').filter(Boolean)
  if (parts.length < 2) return value
  return `${parts[0]}/${parts[parts.length - 1]}`
}
</script>
