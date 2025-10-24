<template>
  <Modal v-model="open" title="Textual Inversion">
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
        <li v-for="name in filtered" :key="name" class="list-item clickable" @click="insert(name)">
          {{ name }}
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
import { fetchEmbeddings } from '../../api/client'

const props = defineProps<{ modelValue: boolean }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: boolean): void; (e:'insert', token: string): void }>()
const open = computed({ get: () => props.modelValue, set: (v: boolean) => emit('update:modelValue', v) })

const names = ref<string[]>([])
const q = ref('')
const weight = ref(1.0)

const filtered = computed(() => {
  const query = q.value.toLowerCase().trim()
  return names.value.filter(n => n.toLowerCase().includes(query))
})

onMounted(async () => {
  try {
    const res = await fetchEmbeddings()
    names.value = Object.keys(res.loaded || {}).sort((a,b)=>a.localeCompare(b))
  } catch (e) {
    names.value = []
  }
})

function insert(name: string): void {
  const t = weight.value && weight.value !== 1.0 ? `(${name}:${weight.value.toFixed(2)})` : name
  emit('insert', t)
}
</script>
