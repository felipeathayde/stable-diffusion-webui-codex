<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Modal for quicksettings device/dtype overrides.
Provides global device selection and per-component device/dtype overrides (core/TE/VAE) backed by the quicksettings store, and reflects backend
apply metadata (`restart_required[]`) so restart warnings are shown only when required.

Symbols (top-level; keep in sync; no ghosts):
- `QuickSettingsOverridesModal` (component): Quicksettings overrides modal for device/dtype settings.
- `onCoreDtypeChange` (function): Store update handler for Core storage dtype selection.
- `onCoreComputeDtypeChange` (function): Store update handler for Core compute dtype selection.
- `onCoreDeviceChange` (function): Store update handler for Core device selection.
- `onTeDtypeChange` (function): Store update handler for TE storage dtype selection.
- `onTeComputeDtypeChange` (function): Store update handler for TE compute dtype selection.
- `onTeDeviceChange` (function): Store update handler for TE device selection.
- `onVaeDtypeChange` (function): Store update handler for VAE storage dtype selection.
- `onVaeComputeDtypeChange` (function): Store update handler for VAE compute dtype selection.
- `onVaeDeviceChange` (function): Store update handler for VAE device selection.
- `onGlobalDeviceChange` (function): Store update handler for global device selection.
- `resetAll` (function): Resets overrides back to `auto` for all components.
- `close` (function): Closes the modal.
-->

<template>
  <Modal v-model="open" title="Component overrides">
    <p class="subtitle">
      Configure global device and per-component overrides (device + storage/compute dtype). Leave values as <code>Default</code> to clear overrides.
    </p>
    <p v-if="store.lastRestartRequiredMessages.length > 0" class="cdx-qs-overrides-restart-note" role="note">
      Some settings require API restart before they take effect.
    </p>
    <ul v-if="store.lastRestartRequiredMessages.length > 0" class="cdx-qs-overrides-restart-list">
      <li v-for="message in store.lastRestartRequiredMessages" :key="message">{{ message }}</li>
    </ul>
    <p v-else class="cdx-qs-overrides-hot-note" role="note">
      Overrides are hot-applied for the next generation request.
    </p>

    <div class="gen-card">
      <div class="field">
        <label class="label-muted">Global device</label>
        <div class="qs-row">
          <select class="select-md" :value="store.currentDevice" @change="onGlobalDeviceChange">
            <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
          </select>
        </div>
      </div>
    </div>

    <div class="gen-card">
      <div class="panel-section-title">Per-component overrides</div>
      <div class="cdx-qs-overrides-grid">
        <div class="cdx-qs-overrides-col">
          <div class="panel-section-title">Core</div>
          <div class="field">
            <label class="label-muted">Core storage dtype</label>
            <div class="qs-row">
              <select class="select-md" :value="store.coreDtype" @change="onCoreDtypeChange">
                <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt === 'auto' ? 'Default' : opt }}</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="label-muted">Core compute dtype</label>
            <div class="qs-row">
              <select class="select-md" :value="store.coreComputeDtype" @change="onCoreComputeDtypeChange">
                <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt === 'auto' ? 'Default' : opt }}</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="label-muted">Core device</label>
            <div class="qs-row">
              <select class="select-md" :value="store.coreDevice" @change="onCoreDeviceChange">
                <option value="auto">Default</option>
                <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
              </select>
            </div>
          </div>
        </div>

        <div class="cdx-qs-overrides-col">
          <div class="panel-section-title">Text Encoder</div>
          <div class="field">
            <label class="label-muted">TE storage dtype</label>
            <div class="qs-row">
              <select class="select-md" :value="store.teDtype" @change="onTeDtypeChange">
                <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt === 'auto' ? 'Default' : opt }}</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="label-muted">TE compute dtype</label>
            <div class="qs-row">
              <select class="select-md" :value="store.teComputeDtype" @change="onTeComputeDtypeChange">
                <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt === 'auto' ? 'Default' : opt }}</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="label-muted">TE device</label>
            <div class="qs-row">
              <select class="select-md" :value="store.teDevice" @change="onTeDeviceChange">
                <option value="auto">Default</option>
                <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
              </select>
            </div>
          </div>
        </div>

        <div class="cdx-qs-overrides-col">
          <div class="panel-section-title">VAE</div>
          <div class="field">
            <label class="label-muted">VAE storage dtype</label>
            <div class="qs-row">
              <select class="select-md" :value="store.vaeDtype" @change="onVaeDtypeChange">
                <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt === 'auto' ? 'Default' : opt }}</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="label-muted">VAE compute dtype</label>
            <div class="qs-row">
              <select class="select-md" :value="store.vaeComputeDtype" @change="onVaeComputeDtypeChange">
                <option v-for="opt in store.dtypeChoices" :key="opt" :value="opt">{{ opt === 'auto' ? 'Default' : opt }}</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="label-muted">VAE device</label>
            <div class="qs-row">
              <select class="select-md" :value="store.vaeDevice" @change="onVaeDeviceChange">
                <option value="auto">Default</option>
                <option v-for="opt in store.deviceChoices" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
              </select>
            </div>
          </div>
        </div>
      </div>
    </div>
    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="resetAll">Reset to Default</button>
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
function onCoreComputeDtypeChange(e: Event): void {
  void store.setCoreComputeDtype((e.target as HTMLSelectElement).value)
}
function onCoreDeviceChange(e: Event): void {
  void store.setCoreDevice((e.target as HTMLSelectElement).value)
}
function onTeDtypeChange(e: Event): void {
  void store.setTeDtype((e.target as HTMLSelectElement).value)
}
function onTeComputeDtypeChange(e: Event): void {
  void store.setTeComputeDtype((e.target as HTMLSelectElement).value)
}
function onTeDeviceChange(e: Event): void {
  void store.setTeDevice((e.target as HTMLSelectElement).value)
}
function onVaeDtypeChange(e: Event): void {
  void store.setVaeDtype((e.target as HTMLSelectElement).value)
}
function onVaeComputeDtypeChange(e: Event): void {
  void store.setVaeComputeDtype((e.target as HTMLSelectElement).value)
}
function onVaeDeviceChange(e: Event): void {
  void store.setVaeDevice((e.target as HTMLSelectElement).value)
}
function onGlobalDeviceChange(e: Event): void {
  void store.setCoreDevice((e.target as HTMLSelectElement).value)
}

function resetAll(): void {
  void store.setCoreDtype('auto')
  void store.setCoreComputeDtype('auto')
  void store.setTeDtype('auto')
  void store.setTeComputeDtype('auto')
  void store.setVaeDtype('auto')
  void store.setVaeComputeDtype('auto')
  void store.setCoreDevice('auto')
  void store.setTeDevice('auto')
  void store.setVaeDevice('auto')
}

function close(): void {
  open.value = false
}
</script>
