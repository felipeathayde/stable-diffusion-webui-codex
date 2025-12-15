<template>
  <section>
    <div class="panel">
      <div class="panel-body">
        <div class="caption">Redirecionando…</div>
        <div v-if="error" class="panel-error mt-2">{{ error }}</div>
      </div>
    </div>
  </section>
</template>

<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { useModelTabsStore, type BaseTabType } from '../stores/model_tabs'

const props = defineProps<{ type: BaseTabType }>()
const router = useRouter()
const store = useModelTabsStore()
const error = ref<string>('')

onMounted(async () => {
  try {
    await store.load()
    const existing = store.orderedTabs.find(t => t.type === props.type)
    const id = existing?.id || (await store.create(props.type))
    if (!id) throw new Error('failed to resolve a model tab id')
    await router.replace(`/models/${id}`)
  } catch (e) {
    error.value = e instanceof Error ? e.message : String(e)
    void router.replace('/models')
  }
})
</script>
