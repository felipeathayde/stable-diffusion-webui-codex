<template>
  <Modal v-model="open" title="Component overrides">
    <p class="muted" style="margin-bottom:.75rem">
      Configure global device and per-component device/dtype overrides. Leave values as <code>auto</code> to let the memory manager choose.
    </p>
    <div class="quicksettings-group" style="margin-bottom:.75rem">
      <label class="label-muted">Global device</label>
      <div class="qs-row">
        <select class="select-md" :value="store.currentDevice" @change="onGlobalDeviceChange">
          <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
        </select>
      </div>
    </div>
    <div class="grid grid-3">
      <div>
        <h4 class="h5">Core</h4>
        <label class="label-muted">Core dtype</label>
        <div class="qs-row">
          <select class="select-md" :value="store.coreDtype" @change="onCoreDtypeChange">
            <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </div>
        <label class="label-muted" style="margin-top:.25rem">Core device</label>
        <div class="qs-row">
          <select class="select-md" :value="store.coreDevice" @change="onCoreDeviceChange">
            <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
            <option value="auto">auto</option>
          </select>
        </div>
      </div>
      <div>
        <h4 class="h5">Text Encoder</h4>
        <label class="label-muted">TE dtype</label>
        <div class="qs-row">
          <select class="select-md" :value="store.teDtype" @change="onTeDtypeChange">
            <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </div>
        <label class="label-muted" style="margin-top:.25rem">TE device</label>
        <div class="qs-row">
          <select class="select-md" :value="store.teDevice" @change="onTeDeviceChange">
            <option value="auto">auto</option>
            <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
          </select>
        </div>
      </div>
      <div>
        <h4 class="h5">VAE</h4>
        <label class="label-muted">VAE dtype</label>
        <div class="qs-row">
          <select class="select-md" :value="store.vaeDtype" @change="onVaeDtypeChange">
            <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </div>
        <label class="label-muted" style="margin-top:.25rem">VAE device</label>
        <div class="qs-row">
          <select class="select-md" :value="store.vaeDevice" @change="onVaeDeviceChange">
            <option value="auto">auto</option>
            <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
          </select>
        </div>
      </div>
    </div>
    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="resetAll">Reset to auto</button>
      <button class="btn btn-md btn-primary" type="button" @click="close">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import Modal from '../ui/Modal.vue'
import { useQuicksettingsStore } from '../../stores/quicksettings'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void }>()

const open = computed({
  get: () => props.modelValue,
  set: (v: boolean) => emit('update:modelValue', v),
})

const store = useQuicksettingsStore()

function onCoreDtypeChange(e: Event): void {
  void store.setCoreDtype((e.target as HTMLSelectElement).value)
}
function onCoreDeviceChange(e: Event): void {
  void store.setCoreDevice((e.target as HTMLSelectElement).value)
}
function onTeDtypeChange(e: Event): void {
  void store.setTeDtype((e.target as HTMLSelectElement).value)
}
function onTeDeviceChange(e: Event): void {
  void store.setTeDevice((e.target as HTMLSelectElement).value)
}
function onVaeDtypeChange(e: Event): void {
  void store.setVaeDtype((e.target as HTMLSelectElement).value)
}
function onVaeDeviceChange(e: Event): void {
  void store.setVaeDevice((e.target as HTMLSelectElement).value)
}
function onGlobalDeviceChange(e: Event): void {
  void store.setDevice((e.target as HTMLSelectElement).value)
}

function resetAll(): void {
  void store.setCoreDtype('auto')
  void store.setTeDtype('auto')
  void store.setVaeDtype('auto')
  void store.setCoreDevice('auto')
  void store.setTeDevice('auto')
  void store.setVaeDevice('auto')
}

function close(): void {
  open.value = false
}
</script>
