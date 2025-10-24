import { defineStore } from 'pinia'

export interface StyleDef { name: string; prompt: string; negative: string }

function load(): StyleDef[] {
  const raw = localStorage.getItem('styles:list')
  if (!raw) return []
  try { return JSON.parse(raw) as StyleDef[] } catch { return [] }
}
function save(list: StyleDef[]): void {
  localStorage.setItem('styles:list', JSON.stringify(list))
}

export const useStylesStore = defineStore('styles', {
  state: () => ({ list: load() as StyleDef[] }),
  actions: {
    names(): string[] { return this.list.map(s => s.name) },
    create(def: StyleDef): void {
      const idx = this.list.findIndex(s => s.name === def.name)
      if (idx >= 0) this.list[idx] = def; else this.list.push(def)
      save(this.list)
    },
    get(name: string): StyleDef | undefined { return this.list.find(s => s.name === name) },
  },
})

