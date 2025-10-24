<template>
  <div class="space-y-4">
    <div class="panel-section">
      <label class="label-muted">Checkpoints</label>
      <PathList v-model="paths.checkpoints" />
    </div>
    <div class="panel-section">
      <label class="label-muted">VAE</label>
      <PathList v-model="paths.vae" />
    </div>
    <div class="panel-section">
      <label class="label-muted">LoRA</label>
      <PathList v-model="paths.lora" />
    </div>
    <div class="panel-section">
      <label class="label-muted">Text Encoders</label>
      <PathList v-model="paths.text_encoders" />
    </div>
    <div class="settings-paths-actions">
      <button class="btn btn-md btn-outline" type="button" @click="reload">Reload</button>
      <button class="btn btn-md btn-primary" type="button" @click="save">Save</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { onMounted, reactive } from 'vue'
import { fetchPaths, updatePaths } from '../../api/client'
import PathList from './widgets/PathList.vue'

type Paths = { checkpoints: string[]; vae: string[]; lora: string[]; text_encoders: string[] }
const paths = reactive<Paths>({ checkpoints: [], vae: [], lora: [], text_encoders: [] })

async function reload(): Promise<void> {
  try {
    const res = await fetchPaths()
    const p = (res.paths || {}) as Partial<Paths>
    paths.checkpoints = [...(p.checkpoints || [])]
    paths.vae = [...(p.vae || [])]
    paths.lora = [...(p.lora || [])]
    paths.text_encoders = [...(p.text_encoders || [])]
  } catch {}
}

async function save(): Promise<void> {
  await updatePaths({
    checkpoints: paths.checkpoints,
    vae: paths.vae,
    lora: paths.lora,
    text_encoders: paths.text_encoders,
  })
}

onMounted(() => { void reload() })
</script>
