<template>
  <Modal v-model="open" title="WAN Assets">
    <div class="wan22-grid">
      <div class="quicksettings-group">
        <label class="label-muted">WAN Metadata</label>
        <div class="qs-row">
          <div class="qs-pair">
            <select id="qs-wan-metadata" class="select-md" :value="metadataDir" @change="$emit('update:metadataDir', ($event.target as HTMLSelectElement).value)">
              <option value="">{{ builtInLabel }}</option>
              <option v-for="m in metadataChoices" :key="m" :value="m">{{ dirLabel(m) }}</option>
            </select>
            <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseMetadata')">+</button>
          </div>
        </div>
      </div>

      <div class="quicksettings-group">
        <label class="label-muted">WAN Text Encoder</label>
        <div class="qs-row">
          <div class="qs-pair">
            <select id="qs-wan-text-encoder" class="select-md" :value="textEncoder" @change="$emit('update:textEncoder', ($event.target as HTMLSelectElement).value)">
              <option value="">{{ builtInLabel }}</option>
              <option v-for="te in textEncoderChoices" :key="te" :value="te">{{ encoderLabel(te) }}</option>
            </select>
            <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseTe')">+</button>
          </div>
        </div>
      </div>

      <div class="quicksettings-group">
        <label class="label-muted">WAN VAE</label>
        <div class="qs-row">
          <div class="qs-pair">
            <select id="qs-wan-vae" class="select-md" :value="vae" @change="$emit('update:vae', ($event.target as HTMLSelectElement).value)">
              <option value="">{{ builtInLabel }}</option>
              <option v-for="v in vaeChoices" :key="v" :value="v">{{ dirLabel(v) }}</option>
            </select>
            <button class="btn btn-outline qs-inline-btn" type="button" title="Browse…" aria-label="Browse…" @click="$emit('browseVae')">+</button>
          </div>
        </div>
      </div>
    </div>
    <template #footer>
      <button class="btn btn-md btn-primary" type="button" @click="close">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import Modal from '../ui/Modal.vue'

const props = defineProps<{
  modelValue: boolean
  metadataDir: string
  metadataChoices: string[]
  textEncoder: string
  textEncoderChoices: string[]
  vae: string
  vaeChoices: string[]
}>()

const emit = defineEmits<{
  (e: 'update:modelValue', v: boolean): void
  (e: 'update:metadataDir', v: string): void
  (e: 'update:textEncoder', v: string): void
  (e: 'update:vae', v: string): void
  (e: 'browseMetadata'): void
  (e: 'browseTe'): void
  (e: 'browseVae'): void
}>()

const open = computed({
  get: () => props.modelValue,
  set: (v: boolean) => emit('update:modelValue', v),
})

const builtInLabel = 'Built-in'

function close(): void {
  emit('update:modelValue', false)
}

function dirLabel(path: string): string {
  const norm = path.replace(/\\/g, '/')
  if (!norm) return ''
  const idx = norm.lastIndexOf('/')
  return idx >= 0 ? norm.slice(idx + 1) || norm : norm
}

function encoderLabel(value: string): string {
  const norm = String(value || '').replace(/\\/g, '/')
  if (!norm) return ''
  if (!norm.includes('/')) return norm
  const [family, ...rest] = norm.split('/').filter(Boolean)
  if (!family || rest.length === 0) return norm
  const tail = rest[rest.length - 1] || rest[0]
  return `${family}/${tail}`
}
</script>
