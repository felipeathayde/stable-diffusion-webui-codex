# apps/backend/use_cases Overview
Date: 2025-10-28
Owner: Backend Use-Case Maintainers
Last Review: 2025-10-28
Status: Active

## Purpose
- Defines high-level orchestration flows for each supported task (txt2img, img2img, txt2vid, img2vid). Each module prepares inputs, invokes the appropriate engine, and handles post-processing.

## Key Files
- `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py` — Task-specific pipelines that bind request parameters, engines, runtimes, and services.
- `__init__.py` — Exposes helpers to orchestrator modules in `apps/backend/core`.

## Notes
- Introduza novos use cases sempre que uma combinação de tarefa + modo precisar de orquestração própria; mantenha a lógica focalizada em preparar entradas, chamar engines e relatar progresso, delegando detalhes de modelo para `engines/` ou `runtime/`.
- Quando adicionar um novo use case, espelhe o padrão existente e registre com o orquestrador e os contratos de API.
