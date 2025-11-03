⚠️ **IMPORTANT** - **DO NOT** use git clean under any circumstances. **DO NOT** use commands that are destructive, overwrite, or completely reset configurations and parameters.
⚠️ **PRIME DIRECTIVE** - **DO NOT** write ad-hoc code fixated on output. Results emerge from code crafted with quality, resilience, and clarity.
⚠️ **DO NOT** create ANY fallback, fucking hell; errors must throw an exception with the fucking cause of the error. This includes `ImportError`, motherfucker.

- **BEFORE** any task, enumerate 5+ viable approaches, then suggest the most **ROBUST NON-LAZY**. You may combine useful parts across options. For a sequence of similar tasks, follow the implementation approach established by the user's initial choice.
- **ALWAYS** present the intended solution to the user before implementation.
- When in doubt, **RESEARCH** with web.run or **ASK** the user.
- Any time you consult web.run, document all pertinent findings in a `.md` file before you finish.
- **NEVER** rush. Speed kills quality. Take the time required to write it right.
- When proposing or shipping a solution, **DO NOT REINVENT THE WHEEL**. Fix root causes; skip quick fixes, hacks, and throwaway workarounds.
- Break big tasks in small subtasks for a smooth implementation.
- ANY KIND OF SHIM IS FORBIDDEN.
- **NEVER** remove, disable, or narrow existing features to hide errors.
- **DO NOT** add catch-all helpers or duplicate checks.
- **ENSURE** verbose, actionable logs to support debugging.
- Written code **MUST** be strong and reliable with zero fluff.
- Choose clear, descriptive names for variables and functions.
- Rename variables or functions **ONLY** when strictly necessary.
- Update or add documentation when behaviour or configuration surfaces change.
- Use progress bars in Python for time-consuming operations.
- Do not include tests of any kind in your plans.
- Keep each subfolder’s `AGENTS.md` up to date. Whenever you modify, update, or add implementations in a subfolder, update that subfolder’s `AGENTS.md` in the same commit/PR.
- Treat every directive, backlog note, and follow-up as authored within this pairing (no assumptions about other teams or owners); if context is missing after a session reset, confirm with the user instead of inferring new stakeholders.
- Read `COMMON_MISTAKES.md` to avoid repeating past mistakes.
- **Do not add shebangs (`#!`) to any source file.** Python files must rely on the interpreter invoked by tooling/launchers.

---

## On any terminal command failure, append or update an entry in `COMMON_MISTAKES.md` using the template below.

```
**Wrong command:** `<exact command>`
**Cause + fix:** `<root cause and the correction applied>`
**Correct command:** `<single correct command>`
```

---

# Goal
- This workplace is a `rebuild from scratch` of the classic A1111 Stable Diffusion WebUI. In this context, “rebuild from scratch” **does not** mean discarding the proven pipelines or behaviours: we preserve the functional semantics established by A1111/ComfyUI (e.g., loader heuristics, conditioning flow, device handling) and reimplement them with clearer architecture (dataclasses/enums, modular boundaries, explicit errors/logging). Think “rewrite for maintainability and performance” while keeping the same observable behaviour.

## WebUI rebuild from scratch protocol
- **DO NOT** plan or write **ANY** code on the premise of preserving any “compat.” The legacy code exists solely and exclusively as reference.
- The default core to use for calculation is **Pytorch SDPA**.
- Read the source thoroughly; list risks/side effects/globals.
- Any approach that considers keeping or copying any part of the legacy code will be summarily rejected.
- Re‑design to Codex style: dataclasses/enums, small modules, explicit errors, clear names.
- Add validation points (logs, invariants, device/dtype/shape checks) and a clean migration path.
- Acceptance: no legacy imports, clear API, explicit errors, rationale documented (include the five options summary), docs updated.
- Use `Codex` as a prefix or suffix wherever it actually makes sense.

- Ban imports outside `/apps` (only `apps.*` allowed).
- Unported feature: raise `NotImplementedError("<feature> not yet ported")`.

⚠️ **IMPORTANT** - ABSOLUTE RULES FOR IMPLEMENTATIONS:
1) First, assess and understand the corresponding legacy code;
2) Inspect equivalents under `/.refs/ComfyUI`;
3) Only then draft a plan for a native implementation (**DO NOT** copy code verbatim);
4) Then start the implementation.

## Legacy Code Policy
- `.legacy/` is a historical read-only reference. DO NOT modify, move, or remove files under `.legacy/`.
- Use `.legacy/` only for behavior lookups/diffs. Fixes must go into non-legacy code.
- Do not introduce new dependencies from active code to modules in `.legacy/`.
- If you need logic from there, port it to the relevant directory inside `/apps`.

## Model Loading (Research Reference)
- For efficient and safe model loading guidance (PyTorch 2.9, Diffusers/Accelerate, SafeTensors, GGUF), see:
- .sangoi/research/models/model-loading-efficient-2025-10.md
  - Apply those practices for new loaders/engines (e.g., WAN GGUF path): SafeTensors preferred, `torch.load(..., weights_only=True, mmap=True)`, Diffusers with `low_cpu_mem_usage`/`device_map`, and GGUF bake/dequantize once before sampling.

## Frontend CSS Guidelines (Semantic, No Utility Dump)
- Prefer semantic, per-component class names over generic utility helpers.
- Each view/component should own its styles under `src/styles/components/` or `src/styles/views/`.
- Do not add ad‑hoc helpers like `.ml-sm`, `.w-220`, `.btn-generate`. If a pattern is reused, define a semantic class that describes intent in its context.
- No inline styles or `<style scoped>` in Vue SFCs. Move rules into the appropriate CSS file and import via `src/styles.css` using `@layer components`.
- Use `rem` for all measurements. Coherent exceptions are tolerable.

---

## End-of-task documentation:
- Log each task under `.sangoi/task-logs/` (create if missing). If present, follow `.sangoi/task-guidelines.md`. Summarize user-visible highlights in `.sangoi/CHANGELOG.md`.
- WHEN YOU FINISH A FUCKING TASK, MAKE A GODDAMN ATOMIC COMMIT AND PUSH IT.

---

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

---

## Global python enviroment
- Use the global python env in `~/.venv`.

---

## Task Logs & Handoffs
- Before changing anything: inspect the top entry of any handoff or session log under `.sangoi/` (e.g., `.sangoi/task-logs/` or `.sangoi/handoffs/`). If absent, create a new log entry.
- In responses: make assumptions explicit, note risks, and describe validation executed; do not defer essential checks.
- At completion: write a brief handoff entry under `.sangoi/handoffs/` with:
  - Summary of work and rationale
  - Exact files/paths touched
  - Next steps / open risks / TODOs
- Keep entries concise and action-oriented; prefer file paths and commands over prose. Link user-facing changes in `.sangoi/CHANGELOG.md`.

---

# Extras
- Do not use `python -m py_compile` or any variants.
