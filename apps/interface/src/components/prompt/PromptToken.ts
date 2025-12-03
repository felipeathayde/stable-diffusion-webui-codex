// tags: prompt, serialization, tiptap
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

function nodeTypeName(node: any): string | undefined {
  if (!node || typeof node !== 'object') return undefined
  if (typeof node.type === 'string') return node.type
  if (typeof node.type?.name === 'string') return node.type.name
  return undefined
}

function forEachChild(node: any, visit: (child: any) => void): void {
  const content = (node as any)?.content
  if (!content) return
  if (Array.isArray(content)) {
    content.forEach(visit)
  } else if (typeof content.forEach === 'function') {
    content.forEach(visit)
  }
}

export function serializePrompt(doc: any): string {
  // Walk the ProseMirror document and build the legacy prompt string
  const parts: string[] = []
  function walk(node: any) {
    const type = nodeTypeName(node)
    if (type === 'text') {
      const text = typeof node.text === 'string' ? node.text : typeof node.textContent === 'string' ? node.textContent : ''
      if (text) parts.push(text)
    } else if (type === 'promptToken') {
      const { kind, name, weight, enabled } = (node as { attrs?: Partial<PromptTokenAttrs> }).attrs ?? {}
      if (enabled === false) return
      const safeWeight = Number.isFinite(Number(weight)) ? Number(weight) : 1
      if (kind === 'lora') parts.push(`<lora:${name}:${safeWeight.toFixed(2)}>`)
      else if (kind === 'ti') parts.push(`(${name}:${safeWeight.toFixed(2)})`)
      else if (name) parts.push(String(name))
    }
    forEachChild(node, walk)
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
