⚠️ IMPORTANT: **DO NOT** use git clean under any circumstances. **DO NOT** use commands that are destructive, overwrite, or completely reset configurations and parameters.

- When in doubt, **RESEARCH** or **ASK**.
- **PRIME DIRECTIVE**: **DO NOT** write ad-hoc code fixated on output. Results emerge from code crafted with quality, resilience, and clarity.
- **IMPORTANT**: **FIRST** enumerate at least five solution paths; then select the most robust and non-lazy option, optionally integrating useful parts from others.
- **ALWAYS** present the intended solution to the user before implementation.
- **NEVER** rush. Speed kills quality. Take the time required to write it right.
- When proposing or shipping a solution, **DO NOT REINVENT THE WHEEL**. Fix root causes; skip quick fixes, hacks, and throwaway workarounds.
- **NEVER** remove, disable, or narrow existing features to hide errors; preserve functional parity and user-facing behavior.
- Handle errors explicitly; **DO NOT** hide them behind fallbacks.
- **DO NOT** add catch-all helpers or duplicate checks.
- **ENSURE** verbose, actionable logs to support debugging.
- Rename variables or functions **ONLY** when strictly necessary.
- Strong, reliable code with zero fluff.
- Choose clear, descriptive names for variables and functions.
- Update or add documentation when behaviour or configuration surfaces change.
- Any time you consult web.run, include a concise record of all pertinent findings in your output before you finish.
- Use progress bars in Python for time-consuming operations.
- No WebUI support in this sandbox; the user handles testing in an external environment.
- **PyTorch-first directive**: Prefer existing PyTorch implementations for all math/ops. Do not plan or build custom CUDA kernels when PyTorch already provides the functionality. Custom kernels are only acceptable for missing features and only after the WebUI is stable. If the user asks for an implementation that PyTorch already offers, warn them and reconfirm before proceeding.

## Pipeline implementation reference
- See `/.refs/ComfyUI` for pipeline references.
- The default core to use for calculation is **Pytorch SDPA**.

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
- Docs home: `codex/` — architecture, design, research, tasks, roadmaps, reports, and sprint logs.
  - Architecture (video): `codex/backend-video-architecture.md`.
  - Research (loading/efficiency): `codex/research/`.
  - Design/UX: `codex/design/`.
- Operational logs: `.sangoi/` — `task-logs/`, `handoffs/`, and machine-readable inventories.
- Backend inventories: `apps/server/backend/SHIM_INVENTORY.md` (shim status/history).

### Git commit guide
1. `git status`
2. `git branch safety/backup-$(date +%Y%m%d-%H%M%S)`
3. `git pull --rebase`
4. `git add -p`
5. `git diff --cached --name-only`
6. `rg -n '<<<<<<<|=======|>>>>>>>'`
7. `git commit -m "type(scope): concise summary"`
8. `git push -u origin $(git branch --show-current)`

### Git revert guide
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
- **YOU MUST DO** an atomic commit and push; no uncommitted leftovers.

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
  - codex/research/model-loading-efficient-2025-10.md
  - Apply those practices for new loaders/engines (e.g., WAN GGUF path): SafeTensors preferred, `torch.load(..., weights_only=True, mmap=True)`, Diffusers with `low_cpu_mem_usage`/`device_map`, and GGUF bake/dequantize once before sampling.

## Frontend CSS Guidelines (Semantic, No Utility Dump)
- Prefer semantic, per-component class names over generic utility helpers.
- Each view/component should own its styles under `src/styles/components/` or `src/styles/views/`.
- Do not add ad‑hoc helpers like `.ml-sm`, `.w-220`, `.btn-generate`. If a pattern is reused, define a semantic class that describes intent in its context (e.g., `.prompt-toolbar`, `.styles-input`, `.results-actions`).
- No inline styles or `<style scoped>` in Vue SFCs. Move rules into the appropriate CSS file and import via `src/styles.css` using `@layer components`.
- Units: use `rem` for all measurements (sizes, radii, borders, shadows, spacing). Coherent exceptions are tolerable.
 
## Repository Map
- `codex/` – internal documentation, roadmaps, and operational logs; update these whenever behaviour, contracts, or processes evolve.
- `legacy/` – snapshot of legacy WebUI code, for REFERENCE only (read-only).

## Legacy Code Policy (read-only)
- `legacy/` is a historical reference. DO NOT modify, move, or remove files under `legacy/`.
- Do not introduce new dependencies from active code to modules in `legacy/`. If you need logic from there, port it to the active directories (`modules/`, `extensions-builtin/`, etc.), with validation and documentation.
- Use `legacy/` only for behavior lookups/diffs. Fixes must go into non-legacy code.

## MVP Scope
- Supported engines for the MVP:
  - WAN 2.2 (video) — txt2vid and img2vid.
  - SDXL (image) — txt2img and img2img.
  - FLUX (image) — txt2img (img2img optional later).

- Engine semantics
  - The `engine` setting gates UI visibility and defaults. Model type is ultimately determined by the loaded weights and pipeline graph.
- Friendly key `wan22` maps to concrete engines:
    - `wan22_14b` and `wan22_5b` (apps/server/backend/engines/registration.py)

- Server‑driven parameter blocks
  - All engine‑specific UI blocks are served by the backend at `/api/ui/blocks`.
  - Source of truth: `apps/interface/blocks.json` (overrides in `apps/interface/blocks.d/*.json`). No frontend fallback.

- Error handling: invalid params return explicit errors; no silent fallbacks.

## Backend Engine Structure (Required)
- See also: `codex/backend-video-architecture.md` (architecture rationale and next steps).
- Per‑task runtimes under `apps/server/backend/engines/diffusion/`:
  - Always implement by task, not by monolithic engine. Files: `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py`.
  - Engines de modelo (ex.: `wan22_14b`, `wan22_5b`) devem delegar para esses módulos por tarefa.
- Vídeo helpers
  - `engines/diffusion/base_video.py` contém utilitários mínimos (serialize/export hooks). Não criar helpers dispersos.
- Registro canônico
  - Registro de engines vive em `apps/server/backend/engines/registration.py`. Não referenciar registradores antigos.
- WAN 2.2 (vídeo)
  - Engines: `wan22_14b` e `wan22_5b` em `engines/diffusion/`. Não usar `engines/video/wan/*`.
  - GGUF: usar apenas o pacote genérico `apps.server.backend.gguf` e os ops de `apps.server.backend.runtime.ops`. Proibido criar “core” próprio (`wan_gguf*`).
  - Runtime específico (GGUF) vive em `apps/server/backend/runtime/nn/wan22.py`. Sem kernels custom; SDPA do PyTorch é o padrão.
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
  - Removed on 2025-10-24. Use the façade `apps.server.backend.*`.
  - See: `codex/deprecations/2025-10-backend-namespace-deprecation.md` and `apps/server/backend/SHIM_INVENTORY.md`.
  - Import redirector no longer exists; old imports raise `ModuleNotFoundError`.
