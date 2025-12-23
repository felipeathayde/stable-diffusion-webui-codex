<template>
  <div class="param-blocks">
    <div v-for="blk in blocks" :key="blk.id" class="panel-section">
      <div class="param-grid" :data-cols="gridCols(blk)">
        <template v-for="field in blk.fields" :key="field.key">
          <div v-if="isVisible(field, blk)" class="field">
            <template v-if="field.type==='slider'">
              <SliderField
                :label="field.label"
                :modelValue="Number(readField(field))"
                :min="field.min ?? 0"
                :max="field.max ?? 100"
                :step="field.step ?? 1"
                :inputStep="field.step ?? 1"
                :showButtons="false"
                :numberUpdateOnInput="true"
                inputClass="cdx-input-w-sm"
                @update:modelValue="(v) => writeField(field, v)"
              />
            </template>
            <template v-else>
              <label class="label-muted">{{ field.label }}</label>
              <!-- text -->
              <input v-if="field.type==='text'" class="ui-input" type="text" :placeholder="String(field.default ?? '')" :value="readField(field)" @change="onChangeText($event, field)" />
              <!-- number -->
              <input v-else-if="field.type==='number'" class="ui-input" type="number" :min="field.min" :max="field.max" :step="field.step || 1" :value="readField(field)" @change="onChangeNumber($event, field)" />
              <!-- checkbox -->
              <label v-else-if="field.type==='checkbox'" class="toggle inline">
                <input type="checkbox" :checked="Boolean(readField(field))" @change="onChangeCheckbox($event, field)" />
                <span>{{ field.help || '' }}</span>
              </label>
              <!-- select -->
              <select v-else-if="field.type==='select'" class="select-md" :value="readField(field)" @change="onChangeSelect($event, field)">
                <option v-for="opt in (field.options||[])" :key="String(opt)" :value="String(opt)">{{ String(opt) }}</option>
              </select>
              <!-- textarea -->
              <textarea v-else-if="field.type==='textarea'" class="ui-textarea" rows="3" :value="readField(field)" @change="onChangeText($event, field)"></textarea>
            </template>
          </div>
        </template>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import type { UiBlock, UiField } from '../api/types'
import SliderField from './ui/SliderField.vue'

const props = defineProps<{ blocks: UiBlock[]; store: any; tab?: 'txt2img' | 'img2img' | 'txt2vid' | 'img2vid' }>()
const store = props.store as Record<string, any>

function bindingPath(field: UiField): string | null {
  const bind = field.bind || {}
  const t = props.tab
  if (t === 'txt2vid' && bind.txt2vid) return bind.txt2vid
  if (t === 'img2vid' && bind.img2vid) return bind.img2vid
  return null
}

function readField(field: UiField): any {
  const p = bindingPath(field)
  if (!p) return field.default ?? ''
  return (store as any)[p]
}

function writeField(field: UiField, value: any): void {
  const p = bindingPath(field)
  if (!p) return
  ;(store as any)[p] = value
}

function onChangeText(e: Event, f: UiField): void {
  writeField(f, (e.target as HTMLInputElement).value)
}
function onChangeNumber(e: Event, f: UiField): void {
  const v = Number((e.target as HTMLInputElement).value)
  writeField(f, Number.isFinite(v) ? v : 0)
}
function onChangeCheckbox(e: Event, f: UiField): void {
  writeField(f, Boolean((e.target as HTMLInputElement).checked))
}
function onChangeSelect(e: Event, f: UiField): void {
  writeField(f, (e.target as HTMLSelectElement).value)
}

function gridCols(blk: UiBlock): number {
  const raw = Math.trunc(Number(blk.layout?.columns || 2))
  const cols = Number.isFinite(raw) ? raw : 2
  return Math.min(4, Math.max(1, cols))
}

function isVisible(field: UiField, blk: UiBlock): boolean {
  const cond = field.visibleIf
  if (!cond) return true
  // visibleIf keys refer to other field keys within the same block; resolve to their bindings
  for (const k of Object.keys(cond)) {
    const target = blk.fields.find((f) => f.key === k)
    if (!target) return true
    const val = readField(target)
    if (val !== (cond as any)[k]) return false
  }
  return true
}
</script>
