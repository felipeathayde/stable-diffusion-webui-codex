<template>
  <Teleport to="body">
    <div v-if="modelValue" class="modal-backdrop" @click.self="close">
      <div class="modal-panel" role="dialog" aria-modal="true">
        <header class="modal-header">
          <h3 class="h3">{{ title }}</h3>
          <button class="btn-icon" type="button" @click="close" aria-label="Close">✕</button>
        </header>
        <div class="modal-body">
          <slot />
        </div>
        <footer class="modal-footer">
          <slot name="footer">
            <button class="btn btn-md btn-outline" type="button" @click="close">Close</button>
          </slot>
        </footer>
      </div>
    </div>
  </Teleport>
  
</template>

<script setup lang="ts">
const props = defineProps<{ title: string; modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void }>()

function close(): void {
  emit('update:modelValue', false)
}
</script>

