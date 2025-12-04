<template>
  <NodeViewWrapper as="span" class="chip" :data-enabled="node.attrs.enabled ? '1' : '0'" contenteditable="false">
    <span class="chip-name">{{ node.attrs.name }}</span>
    <span class="chip-weight">{{ node.attrs.weight.toFixed(2) }}</span>
    <button class="chip-btn" type="button" @click.stop="toggle" :title="node.attrs.enabled ? 'Disable' : 'Enable'">⛔</button>
    <button class="chip-btn" type="button" @click.stop="dec" title="Weight -" >-</button>
    <button class="chip-btn" type="button" @click.stop="inc" title="Weight +" >+</button>
    <button class="chip-btn" type="button" @click.stop="remove" title="Remove">✕</button>
  </NodeViewWrapper>
</template>

<script setup lang="ts">
import { NodeViewWrapper, type NodeViewProps } from '@tiptap/vue-3'
const props = defineProps<NodeViewProps>()
const editor = props.editor
const node = props.node
const getPos = props.getPos

function update(attrs: Record<string, unknown>): void {
  const pos = getPos()
  if (typeof pos !== 'number') return
  editor.commands.command(({ tr, dispatch }) => {
    const trNext = tr.setNodeMarkup(pos, undefined, { ...node.attrs, ...attrs })
    if (dispatch) dispatch(trNext)
    return true
  })
}
function remove(): void {
  const pos = getPos()
  if (typeof pos !== 'number') return
  editor.commands.command(({ tr, dispatch }) => {
    const trNext = tr.delete(pos, pos + node.nodeSize)
    if (dispatch) dispatch(trNext)
    return true
  })
}
function toggle(): void { update({ enabled: !node.attrs.enabled }) }
function inc(): void { update({ weight: Math.min(Number(node.attrs.weight) + 0.1, 2.0) }) }
function dec(): void { update({ weight: Math.max(Number(node.attrs.weight) - 0.1, 0.0) }) }
</script>

<!-- styles moved to styles/components/prompt-chip.css -->
