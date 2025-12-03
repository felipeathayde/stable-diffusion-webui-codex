import { describe, expect, it } from 'vitest'
import { parsePromptToTiptap, serializePrompt } from './PromptToken'

describe('serializePrompt', () => {
  it('serializes plain text from ProseMirror JSON', () => {
    const doc = {
      type: 'doc',
      content: [
        { type: 'paragraph', content: [
          { type: 'text', text: 'hello ' },
          { type: 'text', text: 'world' },
        ] },
      ],
    }

    expect(serializePrompt(doc)).toBe('hello world')
  })

  it('serializes prompt tokens and skips disabled nodes', () => {
    const doc = {
      type: 'doc',
      content: [
        { type: 'paragraph', content: [
          { type: 'text', text: 'a ' },
          { type: 'promptToken', attrs: { kind: 'lora', name: 'cool', weight: 0.75, enabled: true } },
          { type: 'text', text: ' and ' },
          { type: 'promptToken', attrs: { kind: 'ti', name: 'style', weight: 1.2, enabled: true } },
          { type: 'text', text: ' tokens' },
          { type: 'promptToken', attrs: { kind: 'lora', name: 'skip', weight: 1.1, enabled: false } },
        ] },
      ],
    }

    expect(serializePrompt(doc)).toBe('a <lora:cool:0.75> and (style:1.20) tokens')
  })

  it('supports ProseMirror Node-like shapes with type.name', () => {
    const textNode = { type: { name: 'text' }, text: 'node path' }
    const paragraph = { type: { name: 'paragraph' }, content: { forEach: (fn: (child: any) => void) => fn(textNode) } }
    const doc = { type: { name: 'doc' }, content: { forEach: (fn: (child: any) => void) => fn(paragraph) } }

    expect(serializePrompt(doc)).toBe('node path')
  })
})

describe('parsePromptToTiptap', () => {
  it('round-trips prompts with LoRA and TI tokens', () => {
    const prompt = '<lora:cat:0.75> (style:1.10) tail'
    const doc = parsePromptToTiptap(prompt)

    expect(serializePrompt(doc)).toBe(prompt)
  })
})
