Contributor Guide — Doing Things the Codex Way
Date: 2025-10-25

Read First
- Canonical structure: codex/architecture/repo-structure.md
- Rules: codex/architecture/architecture-rules.md
- Pipelines: codex/architecture/model-pipelines-bible.md

Principles
- Small, composable modules; no monolitos
- Public imports only via `apps.server.backend.*` façade (when applicable)
- Prefer PyTorch SDPA; não criar kernels custom se o PyTorch já tem
- Erros explícitos; sem fallbacks silenciosos
- Evite nomes/integrações "Forge": use `Codex*` (ex.: `CodexDiffusionEngine`, `CodexObjects`) e APIs sob `apps.server.backend.codex.*`. Não introduza `modules_forge.*` em código novo.

Common Tasks

1) Adicionar um novo engine (modelo)
- Crie uma pasta em `apps/server/backend/engines/diffusion/` se necessário
- Implemente por tarefa: `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py`
- Em cada tarefa, orquestre chamadas ao runtime (`runtime/nn/*`, `text_processing/*`)
- Mapeie sampler/scheduler via `engines/util/schedulers.py` (`SamplerKind`)
- Registre em `engines/registration.py` (sem condicionais profundas)
- Use `tools/templates/new_engine/` como referência
- Se precisar alterar seleção de módulos/modelos: use `apps.server.backend.codex.main` em vez de `modules_forge.main_entry`.

2) Adicionar/estender um modelo no runtime
- Coloque classes em `apps/server/backend/runtime/nn/`
- Helpers genéricos em `runtime/ops/` (somente quando o PyTorch não supre)
- Tokenizers/encoders: `runtime/text_processing/*`
- Não importar de `legacy/` ou `DEPRECATED/`
- Se houver dependência transitória de `modules.*`, encapsule em `engines/util/a1111_bridge.py`. Engines/serviços não devem importar `modules.*` diretamente.

3) Carregar assets do Hugging Face
- Prefira layout Diffusers padrão (model_index + subpastas)
- WAN 2.2: `*.gguf` (High/Low), `tokenizer/` (+spiece), `text_encoder/` (config + pesos), VAE (dir ou arquivo único)
- Documentos: ver a “bíblia” de pipelines

Checklists de PR
- Estrutura: engine por tarefa? runtime sem ciclos? sem imports proibidos?
- Logs: mensagens claras, sem mascarar erros; dumps de exceção mantidos
- Docs: linkou ou atualizou docs relevantes?
- Staged set: somente arquivos do escopo
