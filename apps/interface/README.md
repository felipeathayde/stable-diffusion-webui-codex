# Interface (Vite + Vue)

Regra essencial: sem estilos inline. Use apenas utilitários Tailwind + tokens documentados em `.sangoi/frontend/guidelines/frontend-style-guide.md`.

## Dev
- WSL/Linux: `./run-webui.sh` (na raiz do repo) — sobe API + Vite juntos usando `~/.venv`.
- Ou manualmente:
  - API: `API_PORT_OVERRIDE=7850 ~/.venv/bin/python apps/backend/interfaces/api/run_api.py`
  - UI: `cd apps/interface && npm install && npm run dev -- --host`
- Proxy `/api` aponta para `API_HOST:API_PORT` (env vars).

## Estrutura
- `src/styles.css`: tokens (dark-first) + utilitários semânticos.
- `src/views`: telas (txt2img, img2img, extras, settings).
- `src/router.ts`: rotas simples.

## Padrões de estilo
- Sem `style="..."` e sem `:style`. Classes somente.
- Estados por classes/atributos (`data-state`), não por estilos.
- Não use classes internas de terceiros (ex.: `.svelte-xxxx`).

Mais detalhes em `.sangoi/frontend/guidelines/frontend-style-guide.md`.
