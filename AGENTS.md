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
- Make an atomic commit and push; no uncommitted leftovers.

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
- `backend/` – server-side services, request handling, memory management helpers.
- `modules/` and `modules_forge/` – core Stable Diffusion pipelines, attention, schedulers, localization files.
- `scripts/` – repeatable maintenance tasks, diagnostics, and automation helpers (prefer these over bespoke shell snippets).
- `models/` – local weights and checkpoints (never commit contents); use symlinks or `.gitignore` patterns to avoid tracking.
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
  - Friendly key `wan22` maps in the backend to concrete engines:
    - txt2vid → `wan_t2v_14b` (apps/server/run_api.py)
    - img2vid → `wan_i2v_14b` (apps/server/run_api.py)

- Server‑driven parameter blocks
  - All engine‑specific UI blocks are served by the backend at `/api/ui/blocks`.
  - Source of truth: `apps/interface/blocks.json` (overrides in `apps/interface/blocks.d/*.json`). No frontend fallback.

- Error handling: invalid params return explicit errors; no silent fallbacks.

## Deprecations
- Completed: `backend.*` namespace
  - Removed on 2025-10-24. Use the façade `apps.server.backend.*`.
  - See: `codex/deprecations/2025-10-backend-namespace-deprecation.md` and `apps/server/backend/SHIM_INVENTORY.md`.
  - Import redirector no longer exists; old imports raise `ModuleNotFoundError`.
