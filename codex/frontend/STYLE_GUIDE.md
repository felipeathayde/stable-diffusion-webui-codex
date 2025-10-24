# Guia de Estilo da Interface (Tailwind + shadcn‑vue)

Objetivo: padronizar o front para ser legível, previsível e fácil de theming, sem estilos inline e sem dependência de classes geradas do Gradio.

## Regra de ouro
- Sem estilos inline. Nada de `style="..."`, `el.style.foo = ...` ou `<component :style="...">`.
- Estilização só via classes utilitárias (Tailwind v4) + tokens CSS e classes semânticas.
- Estados visuais devem ser expressos por classes (`.is-open`, `.is-active`) ou atributos (`data-state="open"`).

## Stack
- Tailwind v4 com `@tailwindcss/vite` (já configurado em `apps/interface`).
- Tokens CSS (modo dark‑first) em `src/styles.css`.
- Componentes headless/shadcn‑vue opcionais; quando usados, mantenha o styling via utilitários/tokens.

## Tokens (base)
- Defina os tokens em `:root` e `.dark` (ex.: `--background`, `--foreground`, `--primary`, `--border`, `--radius`, etc.).
- Mapeie para Tailwind com `@theme` (ex.: `--color-background: hsl(var(--background));`).
- Nunca referencie cores literais em componentes. Use utilitários que já apontam para tokens (`bg-background`, `text-foreground`, `border-border`).

## Padrões de layout
- Use Flex/Grid; evite hacks como `fit-content` indiscriminado.
- Em elementos “alvo” (ex.: barras, linhas de controles) use classes semânticas curtas (`.form-row`, `.toolbar`) definidas em `@layer components` de `styles.css`.
- Nada de classes Svelte geradas (`.svelte-xxxx`).

## Estados e interações
- Nada de tocar `element.style`. Em Vue, mude estado → aplique classe com `:class`.
- Para valores dinâmicos (ex.: largura), prefira variantes utilitárias (`w-80`, `w-[24rem]`) escolhidas por estado. Evite CSS variables em linha.

## Componentes de formulário
- Dropdowns/combobox próprios (HTML + JS) devem receber só classes utilitárias. Sem `style` ou propriedades injetadas inline.
- Sliders, inputs, botões: use as utilidades definidas (ex.: `.ui-input`, `.btn`, `.btn-md`) em `@layer components`.

## Exemplos
- Ruim: `<div style="padding:.5rem;color:#999">...`  → Bom: `<div class="px-2 py-2 text-muted-foreground">`.
- Ruim: `el.style.display='none'` → Bom: `el.classList.toggle('hidden', true)`.

## Nomes e ids
- IDs estáveis (via backend/elem_id) para hooks JS/CSS. Use classes semânticas para estilo.

## Lint (recomendado)
- Adicionar ESLint com `eslint-plugin-vue` e habilitar `vue/no-static-inline-styles: error` (TODO).
- Verificação simples com ripgrep em CI: `rg -n "style=\"" apps/interface/src || true`.

## Onde editar
- Tokens e utilidades: `apps/interface/src/styles.css`.
- Componentes Vue: `apps/interface/src/**` (não usar `<style>` inline com CSS literal; prefira utilitários globais).

## Nota
- Este guia vale para toda UI nova (Vite/Vue). Código legado Gradio permanecerá em fase de remoção.
