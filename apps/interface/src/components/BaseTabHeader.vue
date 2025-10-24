<template>
  <div class="panel">
    <div class="panel-header">
      <input
        class="ui-input"
        type="text"
        :value="title"
        @input="onTitleInput"
        placeholder="Tab title"
        aria-label="Tab title"
      />
      <div class="panel-actions">
        <button class="btn btn-sm" type="button" @click="$emit('duplicate')">Duplicate</button>
        <button class="btn btn-sm btn-destructive" type="button" @click="$emit('remove')">Remove</button>
        <label class="switch-label">
          <input type="checkbox" :checked="enabled" @change="onToggle" />
          <span>Enabled</span>
        </label>
      </div>
    </div>
    <div class="panel-body">
      <div class="grid grid-3">
        <div>
          <button class="btn btn-secondary" type="button" @click="$emit('load')">Load</button>
          <button class="btn" type="button" style="margin-left: 0.5rem" @click="$emit('unload')">Unload</button>
        </div>
        <div style="text-align:center">
          <button class="btn" type="button" @click="$emit('send-to-workflows')">Send to Workflows</button>
        </div>
        <div style="text-align: right">
          <button class="btn btn-primary" type="button" @click="$emit('generate')">Generate</button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{ title: string; enabled: boolean }>()
const emit = defineEmits(['rename', 'duplicate', 'remove', 'set-enabled', 'load', 'unload', 'generate'])

function onTitleInput(e: Event): void {
  emit('rename', (e.target as HTMLInputElement).value)
}

function onToggle(e: Event): void {
  emit('set-enabled', (e.target as HTMLInputElement).checked)
}
</script>
