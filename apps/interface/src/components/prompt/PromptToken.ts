import { Node, mergeAttributes } from '@tiptap/core'
import { VueNodeViewRenderer } from '@tiptap/vue-3'
import PromptTokenChip from './PromptTokenChip.vue'

export interface PromptTokenAttrs {
  kind: 'lora' | 'ti' | 'style'
  name: string
  weight: number
  enabled: boolean
}

export const PromptToken = Node.create({
  name: 'promptToken',
  inline: true,
  group: 'inline',
  atom: true,

  addAttributes() {
    return {
      kind: { default: 'lora' },
      name: { default: '' },
      weight: { default: 1.0 },
      enabled: { default: true },
    }
  },

  parseHTML() {
    return [
      { tag: 'span[data-token]' },
    ]
  },

  renderHTML({ HTMLAttributes }) {
    return ['span', mergeAttributes(HTMLAttributes, { 'data-token': '1' }), 0]
  },

  addNodeView() {
    return VueNodeViewRenderer(PromptTokenChip)
  },
})

export function serializePrompt(doc: any): string {
  // Walk the ProseMirror document and build the legacy prompt string
  const parts: string[] = []
  function walk(node: any) {
    if (node.type.name === 'text') {
      parts.push(node.text || '')
    } else if (node.type.name === 'promptToken') {
      const { kind, name, weight, enabled } = node.attrs as PromptTokenAttrs
      if (!enabled) return
      if (kind === 'lora') parts.push(`<lora:${name}:${Number(weight).toFixed(2)}>`)
      else if (kind === 'ti') parts.push(`(${name}:${Number(weight).toFixed(2)})`)
      else parts.push(name)
    }
    if (node.content) node.content.forEach(walk)
  }
  walk(doc)
  return parts.join('')
}

export function parsePromptToTiptap(prompt: string) {
  // Very small parser for <lora:name:w> and (name:w); leave everything else as text
  const nodes: any[] = []
  let i = 0
  const pushText = (s: string) => { if (s) nodes.push({ type: 'text', text: s }) }
  while (i < prompt.length) {
    // lora pattern
    if (prompt[i] === '<' && prompt.slice(i, i + 6) === '<lora:') {
      const end = prompt.indexOf('>', i + 6)
      if (end > -1) {
        const body = prompt.slice(i + 6, end)
        const [name, w] = body.split(':')
        nodes.push({ type: 'promptToken', attrs: { kind: 'lora', name, weight: parseFloat(w ?? '1'), enabled: true } })
        i = end + 1
        continue
      }
    }
    // TI pattern
    if (prompt[i] === '(') {
      const end = prompt.indexOf(')', i + 1)
      const colon = prompt.indexOf(':', i + 1)
      if (end > -1 && colon > -1 && colon < end) {
        const name = prompt.slice(i + 1, colon)
        const w = prompt.slice(colon + 1, end)
        nodes.push({ type: 'promptToken', attrs: { kind: 'ti', name, weight: parseFloat(w || '1'), enabled: true } })
        i = end + 1
        continue
      }
    }
    // default: add single char and continue (coalesce later)
    pushText(prompt[i])
    i++
  }
  return { type: 'doc', content: [{ type: 'paragraph', content: nodes }] }
}

