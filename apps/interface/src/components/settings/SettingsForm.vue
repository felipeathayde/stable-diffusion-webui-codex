<template>
  <div class="settings-form">
    <div v-if="fields.length === 0" class="card text-xs opacity-70">No settings in this section.</div>
    <div v-else class="grid-col">
      <div v-for="f in fields" :key="f.key" class="form-row">
        <label class="form-label">{{ f.label }}</label>
        <div class="form-control">
          <template v-if="f.type === 'checkbox'">
            <input type="checkbox" class="ui-checkbox" :checked="asBool(model[f.key])" @change="onChange(f.key, ($event.target as HTMLInputElement).checked)" />
          </template>
          <template v-else-if="f.type === 'slider'">
            <input type="range" class="ui-range" :min="f.min ?? 0" :max="f.max ?? 100" :step="f.step ?? 1" :value="asNumber(model[f.key], f.default)" @input="onChange(f.key, asNumber(($event.target as HTMLInputElement).value))" />
            <input type="number" class="ui-input w-24" :min="f.min ?? undefined" :max="f.max ?? undefined" :step="f.step ?? 1" :value="asNumber(model[f.key], f.default)" @input="onChange(f.key, asNumber(($event.target as HTMLInputElement).value))" />
          </template>
          <template v-else-if="f.type === 'radio' && f.choices && f.choices.length">
            <div class="radio-group">
              <label v-for="opt in f.choices" :key="String(opt)" class="radio-item">
                <input type="radio" :name="'rad-'+f.key" class="ui-radio" :checked="String(model[f.key])===String(opt)" @change="onChange(f.key, opt)" />
                <span>{{ String(opt) }}</span>
              </label>
            </div>
          </template>
          <template v-else-if="f.type === 'dropdown'">
            <select class="ui-select" :value="String(model[f.key] ?? '')" @change="onChange(f.key, ($event.target as HTMLSelectElement).value)">
              <option v-for="opt in (f.choices ?? [])" :key="String(opt)" :value="String(opt)">{{ String(opt) }}</option>
            </select>
          </template>
          <template v-else-if="f.type === 'number'">
            <input type="number" class="ui-input" :min="f.min ?? undefined" :max="f.max ?? undefined" :step="f.step ?? 1" :value="asNumber(model[f.key], f.default)" @input="onChange(f.key, asNumber(($event.target as HTMLInputElement).value))" />
          </template>
          <template v-else-if="f.type === 'color'">
            <input type="color" class="ui-input" :value="String(model[f.key] ?? f.default ?? '#000000')" @input="onChange(f.key, ($event.target as HTMLInputElement).value)" />
          </template>
          <template v-else-if="f.type === 'html'">
            <div class="card text-xs opacity-70" v-html="String(f.default ?? '')" />
          </template>
          <template v-else>
            <input type="text" class="ui-input" :value="String(model[f.key] ?? f.default ?? '')" @input="onChange(f.key, ($event.target as HTMLInputElement).value)" />
          </template>
        </div>
      </div>
      <div class="form-actions">
        <button class="btn btn-sm btn-primary" :disabled="pending || changedCount===0" @click="applyChanges">Apply</button>
        <span class="text-xs opacity-60" v-if="changedCount>0">{{ changedCount }} change(s) pending</span>
      </div>
    </div>
  </div>
  
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { SettingsField } from '../../api/types'
import { updateOptions } from '../../api/client'

const props = defineProps<{ fields: SettingsField[]; values: Record<string, unknown> }>()

const model = ref<Record<string, unknown>>({})
const dirty = ref<Record<string, unknown>>({})
const pending = ref(false)

watch(
  () => props.values,
  (v) => {
    model.value = { ...(v || {}) }
    dirty.value = {}
  },
  { immediate: true, deep: true },
)

function onChange(key: string, value: unknown) {
  model.value[key] = value
  dirty.value[key] = value
}

const changedCount = computed(() => Object.keys(dirty.value).length)

async function applyChanges() {
  if (pending.value || changedCount.value === 0) return
  pending.value = true
  try {
    await updateOptions(dirty.value)
    dirty.value = {}
  } finally {
    pending.value = false
  }
}

function asBool(v: unknown, fallback = false) {
  if (typeof v === 'boolean') return v
  if (typeof v === 'string') return v === 'true' || v === '1'
  if (typeof v === 'number') return v !== 0
  return fallback
}

function asNumber(v: unknown, def?: unknown) {
  if (typeof v === 'number') return v
  if (typeof v === 'string' && v !== '') return Number(v)
  if (typeof def === 'number') return def
  return 0
}
</script>

<style scoped>
.settings-form { display: block; }
.grid-col { display: flex; flex-direction: column; gap: .75rem; }
.form-row { display: grid; grid-template-columns: 1fr; gap: .35rem; }
.form-label { font-size: .85rem; opacity: .85; }
.form-control { display: flex; gap: .5rem; align-items: center; flex-wrap: wrap; }
.radio-group { display: flex; gap: .75rem; flex-wrap: wrap; }
.radio-item { display: inline-flex; align-items: center; gap: .35rem; }
.form-actions { margin-top: .75rem; display: flex; gap: .5rem; align-items: center; }
.ui-input.w-24{ width:6rem; }
.btn.sm{ padding: .3rem .6rem; font-size: .85rem; }
</style>
