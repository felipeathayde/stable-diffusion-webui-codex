<template>
  <div class="prompt-editor">
    <EditorContent :editor="editor" />
  </div>
</template>

<script setup lang="ts">
import { onMounted, onBeforeUnmount, ref, watch, nextTick } from 'vue'
import { EditorContent, useEditor } from '@tiptap/vue-3'
import StarterKit from '@tiptap/starter-kit'
import { PromptToken, parsePromptToTiptap, serializePrompt } from './PromptToken'

const props = defineProps<{ modelValue: string }>()
const emit = defineEmits<{ (e:'update:modelValue', v:string): void }>()

const editor = useEditor({
  extensions: [StarterKit.configure({ history: true }), PromptToken],
  content: parsePromptToTiptap(props.modelValue || ''),
  onUpdate: ({ editor }) => {
    const json = editor.getJSON()
    emit('update:modelValue', serializePrompt(json))
  },
})

watch(() => props.modelValue, (v) => {
  const ed = editor.value
  if (!ed) return
  const current = serializePrompt(ed.getJSON())
  if ((v || '') !== current) ed.commands.setContent(parsePromptToTiptap(v || ''))
})

onBeforeUnmount(() => { editor.value?.destroy() })

function insertToken(kind: 'lora'|'ti'|'style', name: string, weight = 1.0): void {
  const ed = editor.value
  if (!ed) return
  ed.chain().focus().insertContent({ type: 'promptToken', attrs: { kind, name, weight, enabled: true } }).run()
}

defineExpose({ insertToken })
</script>

