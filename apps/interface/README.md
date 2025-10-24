# Interface (Vite + Vue)

Regra essencial: sem estilos inline. Use apenas utilitários Tailwind + tokens documentados em `codex/frontend/STYLE_GUIDE.md`.

## Dev
- `python tools/services_tui.py` – inicia TUI para subir/monitorar API e Vite juntos (setas ↑↓ ou números para interagir).
- Ou manualmente: `cd apps/interface && npm i && npm run dev -- --host`.
- Proxy `/api` aponta para `API_HOST:API_PORT` (configurado via TUI ou env vars).

## Estrutura
- `src/styles.css`: tokens (dark‑first) + utilitários semânticos.
- `src/views`: telas (txt2img, img2img, extras, settings).
- `src/router.ts`: rotas simples.

## Padrões de estilo
- Sem `style="..."` e sem `:style`. Classes somente.
- Estados por classes/atributos (`data-state`), não por estilos.
- Não use classes internas de terceiros (ex.: `.svelte-xxxx`).

Mais detalhes em `codex/frontend/STYLE_GUIDE.md`.
