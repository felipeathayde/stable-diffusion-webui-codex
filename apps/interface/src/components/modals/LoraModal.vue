<template>
  <Modal v-model="open" title="LoRA Selector">
    <div class="form-grid">
      <div>
        <label class="label-muted">Search</label>
        <input class="ui-input" v-model="q" placeholder="type to filter..." />
      </div>
      <div>
        <label class="label-muted">Weight</label>
        <input class="ui-input" type="number" step="0.1" min="0" v-model.number="weight" />
      </div>
    </div>
    <div class="panel-section" style="margin-top:.5rem">
      <ul class="list" role="listbox">
        <li v-for="item in filtered" :key="item.name" class="list-item clickable" @click="insert(item.name)">
          {{ item.name }}
        </li>
      </ul>
    </div>
    <template #footer>
      <button class="btn btn-md btn-outline" type="button" @click="open=false">Close</button>
    </template>
  </Modal>
</template>

<script setup lang="ts">
import { computed, ref, onMounted } from 'vue'
import Modal from '../ui/Modal.vue'
import { fetchLoras } from '../../api/client'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void; (e:'insert', token: string): void }>()
const open = computed({ get: () => props.modelValue, set: (v: boolean) => emit('update:modelValue', v) })

interface LoraItem { name: string; path: string }
const items = ref<LoraItem[]>([])
const q = ref('')
const weight = ref(0.8)

const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  return items.value.filter(n => n.name.toLowerCase().includes(query))
})

onMounted(async () => {
  try {
    const res = await fetchLoras()
    items.value = res.loras || []
  } catch (e) {
    items.value = []
  }
})

function insert(name: string): void {
  const t = `<lora:${name}:${(weight.value ?? 1.0).toFixed(2)}>`
  emit('insert', t)
}
</script>
