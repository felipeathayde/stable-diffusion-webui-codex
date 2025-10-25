Engine Authoring Cookbook
Date: 2025-10-25

Goal
Guidar a criação de engines (familias de modelos) sem sair da arquitetura.

1) Esqueleto
- Copie `tools/templates/new_engine/` para `apps/server/backend/engines/diffusion/<engine_name>/`
- Crie `__init__.py` se necessário exportando `register_<engine_name>(registry, replace=False)`

2) Registro
- Em `engines/registration.py`, adicione uma função `register_<engine_name>` que registra as classes específicas
- Use keys minúsculas e sem espaços; adicionar aliases se útil

3) Tarefas
- Em `txt2img.py`/`img2img.py`/`txt2vid.py`/`img2vid.py`:
  - Valide o payload; converta em dataclasses do runtime (se existirem)
  - Prepare texto: `runtime/text_processing/*` (CLIP/T5 etc.)
  - Selecione scheduler via `engines/util/schedulers.SamplerKind`
  - Chame NN do runtime (`runtime/nn/...`) e produza `InferenceEvent`s
  - Informe progresso com `ProgressEvent`, resultados com `ResultEvent`

4) Assets
- Exija explicitamente os diretórios/arquivos mínimos; se faltar, lance `EngineLoadError`/`EngineExecutionError`
- Não caia em fallback silencioso para outro pipeline; deixe claro no erro

5) Logs e erros
- `apps/server/backend/runtime/exception_hook.py` gera dumps automáticos; não capture exceção para “esconder”
- Use logs informativos (init de portas, decisão de scheduler, shapes/dtypes)

6) Teste local
- Start API pelo TUI, observe os logs, acione uma chamada de cada tarefa
- Verifique que a UI de blocos reflete os samplers/schedulers expostos

Pitfalls comuns
- Evitar imports para `legacy/` e `DEPRECATED/`
- Não usar `backend.*` namespace antigo
- Não criar engines/video/wan/*; WAN 2.2 é `runtime/nn/wan22.py`

