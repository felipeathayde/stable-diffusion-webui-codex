<!--
Repository: stable-diffusion-webui-codex
Repository URL: https://github.com/sangoi-exe/stable-diffusion-webui-codex
Author: Lucas Freire Sangoi
License: PolyForm Noncommercial 1.0.0
SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
Required Notice: see NOTICE

Purpose: Tiptap node-view chip for prompt tokens.
Renders a compact chip for `PromptToken` nodes with controls to enable/disable, adjust weight, and remove the token.

Symbols (top-level; keep in sync; no ghosts):
- `PromptTokenChip` (component): Node-view component for prompt token chips.
- `update` (function): Updates node attributes via a Tiptap transaction after resolving the current node from `tr.doc` at `getPos()`.
- `remove` (function): Deletes the token node from the document after resolving the live node size from `tr.doc` at `getPos()`.
- `toggle` (function): Toggles the token enabled state.
- `inc` (function): Increments token weight (clamped).
- `dec` (function): Decrements token weight (clamped).
-->

<template>
  <NodeViewWrapper as="span" class="chip" :data-enabled="props.node.attrs.enabled ? '1' : '0'" contenteditable="false">
    <span class="chip-name">{{ props.node.attrs.name }}</span>
    <span class="chip-weight">{{ props.node.attrs.weight.toFixed(2) }}</span>
    <button class="chip-btn" type="button" @click.stop="toggle" :title="props.node.attrs.enabled ? 'Disable' : 'Enable'">⛔</button>
    <button class="chip-btn" type="button" @click.stop="dec" title="Weight -" >-</button>
    <button class="chip-btn" type="button" @click.stop="inc" title="Weight +" >+</button>
    <button class="chip-btn" type="button" @click.stop="remove" title="Remove">✕</button>
  </NodeViewWrapper>
</template>

<script setup lang="ts">
import { NodeViewWrapper, type NodeViewProps } from '@tiptap/vue-3'
const props = defineProps<NodeViewProps>()
const editor = props.editor
const getPos = props.getPos

function update(
  next:
    | Record<string, unknown>
    | ((currentAttrs: Record<string, unknown>) => Record<string, unknown>),
): void {
  const pos = getPos()
  if (typeof pos !== 'number') return
  editor.commands.command(({ tr, dispatch }) => {
    const currentNode = tr.doc.nodeAt(pos)
    if (!currentNode) return false
    const currentAttrs = currentNode.attrs as Record<string, unknown>
    const attrs = typeof next === 'function' ? next(currentAttrs) : next
    const trNext = tr.setNodeMarkup(pos, undefined, { ...currentAttrs, ...attrs })
    if (dispatch) dispatch(trNext)
    return true
  })
}
function remove(): void {
  const pos = getPos()
  if (typeof pos !== 'number') return
  editor.commands.command(({ tr, dispatch }) => {
    const currentNode = tr.doc.nodeAt(pos)
    if (!currentNode) return false
    const trNext = tr.delete(pos, pos + currentNode.nodeSize)
    if (dispatch) dispatch(trNext)
    return true
  })
}
function toggle(): void {
  update((currentAttrs) => ({ enabled: !Boolean(currentAttrs.enabled) }))
}
function inc(): void {
  update((currentAttrs) => ({
    weight: Math.min(Number(currentAttrs.weight ?? 0) + 0.1, 2.0),
  }))
}
function dec(): void {
  update((currentAttrs) => ({
    weight: Math.max(Number(currentAttrs.weight ?? 0) - 0.1, 0.0),
  }))
}
</script>

<!-- styles moved to styles/components/prompt-chip.css -->
