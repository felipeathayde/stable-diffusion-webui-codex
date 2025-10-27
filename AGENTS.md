⚠️ IMPORTANT: **DO NOT** use git clean under any circumstances. **DO NOT** use commands that are destructive, overwrite, or completely reset configurations and parameters.

- When in doubt, **RESEARCH** or **ASK**.
- **PRIME DIRECTIVE**: **DO NOT** write ad-hoc code fixated on output. Results emerge from code crafted with quality, resilience, and clarity.
- **IMPORTANT**: **FIRST** enumerate at least five solution paths; then select the most robust and non-lazy option, optionally integrating useful parts from others.
- **ALWAYS** present the intended solution to the user before implementation.
- **NEVER** rush. Speed kills quality. Take the time required to write it right.
- When proposing or shipping a solution, **DO NOT REINVENT THE WHEEL**. Fix root causes; skip quick fixes, hacks, and throwaway workarounds.
- **NEVER** remove, disable, or narrow existing features to hide errors; preserve functional parity and user-facing behavior.
- **NÃO CRIE FALLBACK**, PORRA DO CARALHO, ERRO TEM QUE RETORNAR EXCEÇÃO COM A PORRA DA CAUSA DO ERRO. ISSO INCLUI IMPORT ERROR, FILHO DA PUTA.
- **DO NOT** add catch-all helpers or duplicate checks.
- **ENSURE** verbose, actionable logs to support debugging.
- Rename variables or functions **ONLY** when strictly necessary.
- Strong, reliable code with zero fluff.
- Choose clear, descriptive names for variables and functions.
- Update or add documentation when behaviour or configuration surfaces change.
- Any time you consult web.run, include a concise record of all pertinent findings in your output before you finish.
- Use progress bars in Python for time-consuming operations.
- No WebUI support in this sandbox; the user handles testing in an external environment.
- After edits, overwrite .git/codex-changed with the exact repo-relative paths you modified, separated by a single NUL character (\0) and terminated by a final NUL (no newlines or extra whitespace).

## Pipeline implementation reference
- See `/.refs/ComfyUI` for pipeline references.
- The default core to use for calculation is **Pytorch SDPA**.
## PyTorch-first directive*
- Prefer existing PyTorch implementations for all math/ops. Do not plan or build custom CUDA kernels when PyTorch already provides the functionality. Custom kernels are only acceptable for missing features and only after the WebUI is stable. If the user asks for an implementation that PyTorch already offers, warn them and reconfirm before proceeding.

## Git Workflow & Hygiene
- Prefer GitHub CLI `gh` for remote actions (PRs, issues, merges, branch mgmt). Keep raw `git` for local work.
- Commit scope: commit exactly and only the files you changed for the task.
- Verify staged set: `git diff --cached --name-only` must match your intent.
- Reject conflicts: `rg -n "<<<<<<<|=======|>>>>>>>"` must return empty.
- Clean tree: before ending a task, ensure a clean working tree. Keep artifacts like `outputs/`, caches, and large data/model dirs untracked (see gitignore.md).
- Deps/config changes: update the proper manifest/lock and mention impact briefly.
- JS/TS: update `package.json` (+ lockfile).
- Python: update `requirements*.txt` (or tool‑specific files) under VCS when appropriate.
- Documentation: author docs in English by default; link new docs from the nearest README.
- Ignore/attributes: see gitignore.md for the full policy.

## Documentation Index
- Docs home: `.sangoi/docs/` — architecture, design, research, tasks, roadmaps, reports, and sprint logs.
  - Consolidated directives: `.sangoi/docs/architecture/CONSOLIDATED_DIRECTIVES.md`
  - Architecture (video): `.sangoi/docs/backend-video-architecture.md`.
  - Canonical structure: `.sangoi/docs/architecture/repo-structure.md`
  - Architecture rules: `.sangoi/docs/architecture/architecture-rules.md`
  - Pipelines bible: `.sangoi/docs/architecture/model-pipelines-bible.md`
  - Repository inventory: `.sangoi/docs/architecture/repo-inventory.md`
- Cleanup checklists: `.sangoi/docs/architecture/repo-cleanup-checklists.md`
- Native LoRA (registry + apply): see `.sangoi/docs/architecture/CONSOLIDATED_DIRECTIVES.md` (LoRA section)
  - Research (loading/efficiency): `.sangoi/docs/research/`.
  - Design/UX: `.sangoi/docs/design/`.
- Operational logs: `.sangoi/` — `task-logs/`, `handoffs/`, and machine-readable inventories.
- Backend inventories: `apps/backend/SHIM_INVENTORY.md` (shim status/history).

## Porting Protocol (Quick Checklist)
- Do not call legacy code (`modules.*`, `modules_forge.*`, `.legacy/`).
- Read the source thoroughly; list risks/side effects/globals.
- Enumerate 5+ viable approaches; choose the most robust (non‑lazy). You may combine useful parts across options.
- Re‑design to Codex style: dataclasses/enums, small modules, explicit errors, clear names.
- Add validation points (logs, invariants, device/dtype/shape checks) and a clean migration path (no shims).
- Acceptance: no legacy imports, clear API, explicit errors, rationale documented (include the five options summary), docs updated.

Tip — Native LoRA usage
- Discover adapters via `GET /api/loras` (backend registry). Set selections with `POST /api/loras/apply`.
- Engines apply LoRA natively at generation time using `codex.lora` selections and `patchers.lora_apply`.

### QUANDO FOR COMMITAR, USA A MERDA DESSA SEQUÊNCIA, PORRA:
1. `git status -sb`
2. `git fetch -p`
3. `git pull --rebase --autostash`
4. `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
5. `git diff --staged --check`
6. `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 rg -n '^(<<<<<<<|=======|>>>>>>>)' || true`
7. `git commit -m "type(scope): concise summary"`
8. `git push -u origin HEAD || git push --force-with-lease`


### QUANDO FOR DAR REVERT, USA ESSA MERDA DE SEQUÊNCIA:
1) `git status`
2) `git switch -c safety/backup-$(date +%Y%m%d-%H%M%S)`
3) `git pull --rebase`
4) `git revert --no-commit <SHA>`
5) `git add -p`
6) `git commit -m "revert: undo <SHA>"`
7) `rg -n '<<<<<<<|=======|>>>>>>>' || true`
8) `git push -u origin $(git branch --show-current)`

## End-of-task documentation:
- Log each task under `.sangoi/task-logs/` (create if missing). If present, follow `.sangoi/task-guidelines.md`. Summarize user-visible highlights in `.sangoi/CHANGELOG.md`.
- QUANDO TU TERMINAR O CARALHO DE UMA TASK, FAZ A PORRA DE UM COMMIT ATÔMICO E DÁ PUSH NESSA MERDA. NÃO USA `git add -A` OU SIMILARES, CARALHO.

## Global python enviroment
- python / pip / py (user-local): wrappers live in `~/.codextools/bin`.
	- `python --version` (points to your preferred interpreter)
	- `pip list` or `python -m pip install <pkg>`
	- `py` alias behaves like `python`

## Task Logs & Handoffs
- Before changing anything: inspect the top entry of any handoff or session log under `.sangoi/` (e.g., `.sangoi/task-logs/` or `.sangoi/handoffs/`). If absent, create a new log entry.
- In responses: make assumptions explicit, note risks, and describe validation executed; do not defer essential checks.
- At completion: write a brief handoff entry under `.sangoi/handoffs/` with:
  - Summary of work and rationale
  - Exact files/paths touched
  - Commands run and validations performed (tests, linters, snapshots)
  - Next steps / open risks / TODOs
- Keep entries concise and action-oriented; prefer file paths and commands over prose. Link user-facing changes in `.sangoi/CHANGELOG.md`.

## Model Loading (Research Reference)
- For efficient and safe model loading guidance (PyTorch 2.9, Diffusers/Accelerate, SafeTensors, GGUF), see:
  - .sangoi/docs/research/model-loading-efficient-2025-10.md
  - Apply those practices for new loaders/engines (e.g., WAN GGUF path): SafeTensors preferred, `torch.load(..., weights_only=True, mmap=True)`, Diffusers with `low_cpu_mem_usage`/`device_map`, and GGUF bake/dequantize once before sampling.

## Frontend CSS Guidelines (Semantic, No Utility Dump)
- Prefer semantic, per-component class names over generic utility helpers.
- Each view/component should own its styles under `src/styles/components/` or `src/styles/views/`.
- Do not add ad‑hoc helpers like `.ml-sm`, `.w-220`, `.btn-generate`. If a pattern is reused, define a semantic class that describes intent in its context (e.g., `.prompt-toolbar`, `.styles-input`, `.results-actions`).
- No inline styles or `<style scoped>` in Vue SFCs. Move rules into the appropriate CSS file and import via `src/styles.css` using `@layer components`.
- Units: use `rem` for all measurements (sizes, radii, borders, shadows, spacing). Coherent exceptions are tolerable.
 
## Repository Map
- `apps/backend/` – server-side services, request handling, runtime/nn/ops, engines
- `modules/` and `modules_forge/` – core Stable Diffusion pipelines, attention, schedulers, localization files.
- `scripts/` – repeatable maintenance tasks, diagnostics, and automation helpers (prefer these over bespoke shell snippets).
- `models/` – local weights and checkpoints (never commit contents); use symlinks or `.gitignore` patterns to avoid tracking.
- `.sangoi/docs/` – internal documentation, roadmaps, and operational logs; update these whenever behaviour, contracts, or processes evolve.
- `legacy/` – snapshot of legacy WebUI code, for REFERENCE only (read-only).
- `.legacy/` – historical code moved out of the active tree (do not import).

## Legacy Code Policy (read-only)
- `legacy/` is a historical reference. DO NOT modify, move, or remove files under `legacy/`.
- Do not introduce new dependencies from active code to modules in `legacy/`. If you need logic from there, port it to the active directories (`modules/`, `extensions-builtin/`, etc.), with validation and documentation.
- Use `legacy/` only for behavior lookups/diffs. Fixes must go into non-legacy code.

## Backend Engine Structure (Required)
- See also: `.sangoi/docs/backend-video-architecture.md` (architecture rationale and next steps).
- Per‑task runtimes under `apps/backend/engines/diffusion/`:
  - Always implement by task, not by monolithic engine. Files: `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py`.
  - Engines de modelo (ex.: `wan22_14b`, `wan22_5b`) devem delegar para esses módulos por tarefa.
- Vídeo helpers
  - `engines/diffusion/base_video.py` contém utilitários mínimos (serialize/export hooks). Não criar helpers dispersos.
- Registro canônico
  - Registro de engines vive em `apps/backend/engines/registration.py`. Não referenciar registradores antigos.
- WAN 2.2 (vídeo)
  - Engines: `wan22_14b` e `wan22_5b` em `engines/diffusion/`. Não usar `engines/video/wan/*`.
  - GGUF: usar apenas o pacote genérico `apps.server.backend.gguf` e os ops de `apps.server.backend.runtime.ops`. Proibido criar “core” próprio (`wan_gguf*`).
  - Runtime específico (GGUF) vive em `apps/backend/runtime/nn/wan22.py`. Sem kernels custom; SDPA do PyTorch é o padrão.
- Dataclasses & Enum (Obrigatório)
  - Parâmetros/estruturas públicas devem ser `@dataclass` (ex.: `VideoExportOptions`, `VideoInterpolationOptions`, `EngineOpts`, `WanComponents`, `WanStageOptions`).
  - Mapeamentos de sampler/scheduler devem usar `SamplerKind` (enum) via `engines/util/schedulers.py`.
- UI blocks
  - Fonte única em `apps/interface/blocks.json` (+ overrides em `apps/interface/blocks.d`). Não usar `apps/ui`.
- Nomes de engines
  - ‘wan22’ é semântico; o backend resolve para `wan22_14b` ou `wan22_5b`.

## Do/Don’t (WAN e vídeo)
- Do: Diffusers‑first; cair para GGUF apenas quando não houver pipeline Diffusers disponível localmente.
- Do: Erros explícitos para partes não implementadas no GGUF (sem “saídas falsas”).
- Don’t: Criar novos diretórios sob `engines/video/wan/` ou pacotes `wan_gguf*`.
- Don’t: Codificar lógica por-engine com condicionais profundas; preferir runtimes por tarefa + utilitários dedicados.

## Deprecations
- Completed: `backend.*` namespace
  - Removed on 2025-10-24. Use `apps.server.backend.*` diretamente.
  - Import redirector não existe mais; imports antigos quebram.

- Proibido: qualquer “bridge/shim/compat” para A1111/Forge (`modules.*`, `modules_forge.*`).
  - Se o backend precisar de uma peça existente fora de `apps/`, portar o código como módulo nativo sob `apps/backend/**` com nomes claros (dataclasses/enums quando fizer sentido).
  - Erros devem ser explícitos (sem fallbacks silenciosos). Se a funcionalidade ainda não foi portada, lance `NotImplementedError`.

## Roadmap — Backend Consolidation (apps/backend‑first)

Goal
- Refatorar e organizar tudo o que é importante para dentro de `apps/backend`, mantendo paridade funcional e sem criar acoplamentos novos ao legado.

Principles (always on)
- Código ativo vive em `apps/backend` (backend) e `apps/interface` (UI nova).
- Nada de novos imports a partir de `.legacy/`. Sem `backend.*` (namespace antigo).
- Preferir PyTorch SDPA; erros explícitos (sem fallbacks silenciosos).

Phases
- P0 (now)
  - Fonte de verdade: seguir `.sangoi/docs/architecture/*` (repo‑structure, rules, pipelines, inventory, checklists).
- `backend/` (raiz) substituído pela nova estrutura em `apps/backend`. WAN `wan_gguf*` vive sob `apps/backend/runtime/wan22/`.
  - Sem novos códigos fora de `apps/`. Exceções estritas habilitadas (dumps em logs/exceptions‑YYYYmmdd‑pid.log).

- P1 (next sprint)
  - Consolidar entradas das tasks: engines/diffusion/{txt2img,img2img,txt2vid,img2vid}.py como única orquestração.
- Implementar nativamente tudo que o backend precisa; nada de `modules/`/`modules_forge/` no backend ativo.
  - Samplers/Schedulers via `engines/util/schedulers.SamplerKind` (única fonte); UI recebe a lista pelo backend.
  - Presets: alinhar `configs/presets` com presets servidos pelo backend (sem duplicidade futura).

- P2 (2–4 semanas)
  - Migrar loaders/modelos necessários de `modules/` para `runtime/models` e `runtime/nn` (sem quebrar extensões).
  - Texto/Tokenizers: consolidar em `runtime/text_processing/*` (CLIP/T5/LLAMA/Qwen que usamos).
  - Limpar dependências optativas em `extensions-builtin/` (listar o que é realmente usado em startup).

- P3 (backlog)
  - UI legacy: aposentar `javascript*/html` quando a UI nova cobrir os fluxos.
  - Presets legados: consolidar em endpoints do backend; arquivar `configs/presets`.
- Mover ferramentas/inspeções de `scripts/` para `tools/` ou `.legacy/` quando não forem mais úteis.

Acceptance Criteria por fase
- P0: Sem novos arquivos fora de `apps/`. Importar `backend.*` quebra com mensagem clara. Documentação publicada.
- P1: Todas as tasks entram por engines/diffusion/*; lista de samplers/schedulers sai do backend; presets servidos pelo backend; zero `modules_forge.*` em `apps/backend/**`.
- P2: Carregadores/NN críticos no `runtime/*`; texto/tokenizers unificados; extensões opcionais isoladas.
- P3: UI nova cobre fluxos; legados marcados como referência; repositório limpo de duplicidades.
