<template>
  <div class="wan-subheader">
    <div class="wan-subheader-left">
      <span class="wan-subheader-title">{{ title }}</span>
      <slot name="subtitle" />
    </div>
    <div v-if="$slots.default" class="wan-subheader-right">
      <slot />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  title: string
}>()
</script>

