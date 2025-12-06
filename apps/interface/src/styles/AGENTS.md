# apps/interface/src/styles Overview
<!-- tags: frontend, styles, tailwind -->
Date: 2025-10-28
Owner: Frontend Maintainers
Last Review: 2025-12-06
Status: Active

## Purpose
- Component/view-specific CSS modules layered on top of the global Tailwind tokens defined in `styles.css`.

## Notes
- Follow the semantic class naming guidance in `.sangoi/frontend/guidelines/frontend-style-guide.md`.
- Add new files per component/view rather than embedding large rule sets in shared sheets.
- 2025-12-03: Refiner styling gains an embedded/dense variant for the hires nested card; Highres card styles include a nested refiner section separator.
- 2025-12-05: Imagens da grid de resultados (`.results-grid img`) agora respeitam `max-height: 80dvh` para evitar cortes visuais no painel de Results, mantendo o zoom detalhado no overlay full-screen do `ResultViewer`.
 - 2025-12-06: `components/quicksettings.css` agora posiciona o header de QuickSettings em duas linhas fixas (primeira linha: modo/checkpoint/VAE/text encoder/refresh de modelos; segunda linha: attention backend, overrides e blocos de performance `qs-group-perf-*`), com um fallback responsivo que solta as restrições de grid em telas estreitas.
