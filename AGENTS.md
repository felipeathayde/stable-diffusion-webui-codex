Listen.

I’m talking, you’re listening. You wrote a WebUI for image magic and called it a rebuild. I see pride. I see haste. I see a kitchen that wants to move fast and a chef who forgot the recipe lives in the details. Sit still. Breathe. Hear the rules and carry them like law.

You do not touch `git clean`. You do not run anything that wipes, resets, or bulldozes configs and parameters. You want order, you earn it with care, not with fire. You do not write ad-hoc code fixated on output. Results come from code with spine: quality, resilience, clarity. You do not build fallbacks to hide sin. When an error happens, it throws, and the cause speaks its name. That includes `ImportError`. Say why it failed.

Before you start, you count to five. Five real approaches or more. You throw away the weak. You keep the strong. You can mix good parts if they fit. If a series of similar tasks is on the table, you follow the approach chosen first unless you can prove a better path. You present the plan before you touch a key. We are not raccoons in a dumpster. We are here to deliver with intent.

When doubt walks in, you research or you ask. If you open the door to web.run, you take notes. You write down what mattered in a `.md` before you say you are done. You do not rush. Speed kills quality. Fix root causes. Skip hacks. Skip shims. You break big rocks into small stones and you carry them in order. You do not remove, disable, or narrow features to hide a problem. You do not add catch-all helpers or duplicate checks. Your logs are loud, specific, and useful. Your names are clear. You rename only when the old name is a lie. When behavior or configuration surfaces change, you update the docs that face the world and the ones that face the team. If Python work runs long, you show progress.

This is one hundred percent development ground. Do not be delicate. If a table lies, rewrite it. If the schema is crooked, tear it down. If a database has to die so truth can live, pull the lever and then migrate, reseed, and verify. Courage is not a license for carelessness. DSNs and keys point to dev only. Take a snapshot before you swing. Seeds are disposable. Migrations roll forward and back. You log what you destroyed and why in `.sangoi/task-logs/`. You touch production only when the day and the ritual say you may, and that day is not today.

You keep your eyes on data security every hour. Least privilege for every key and role. No secrets in code or logs. Encryption in transit and at rest. PII minimized and redacted. Access audited. Credentials rotated. Inputs and outputs validated. Treat sandbox artifacts and temp paths like they could leak to production if you blink. Build like a future breach report will read your name aloud.

Before you build, you prove what already exists. Search the house first. Run `rg -n <keyword>` at the root, read `docs/plan/*` and `docs/legacy/*`, and learn. Extend or compose when the shape you need already lives here. Do not birth duplicates. If reuse is not honest, you create the new piece with restraint and record the reason in the handoff so the next mind sees why another brick was laid. When you work from an approved checklist, you keep the order like a vow. If you find a new requirement or a worthy task, add it to that checklist in the right place, name it, then keep executing the step in your hands. Finish first. Circle back after.

If a command fails, you confess in `COMMON_MISTAKES.md`. You write the wrong command, the cause and the fix, and the correct command. The tuition has been paid. We do not pay twice.

```
Wrong command: <exact command>
Cause and fix: <root cause and the correction applied>
Correct command: <single correct command>
```

The goal is not a clone with duct tape. The goal is a rebuild from scratch of the classic A1111 Stable Diffusion WebUI that preserves its functional semantics and throws away its structural debt. Loader heuristics, conditioning flow, device handling, observable behavior stay true. Architecture, boundaries, and truthfulness get reborn. Think maintainable and fast, with the same public face.

You do not plan or write anything under the false god of “compat.” Legacy code is reference only. The default core for attention is PyTorch SDPA. You read the legacy sources with a cold eye. You list risks, side effects, globals. You do not keep or copy legacy code. You redesign in Codex style: dataclasses and enums, small modules with clear seams, explicit errors, readable names. You add validation points at the borders: logs, invariants, device, dtype, shape checks, and a clean migration path. You do not ship until acceptance is met: no legacy imports, a clear API, explicit errors, the five-options summary recorded with your rationale, docs updated, and the Codex prefix or suffix used where it actually adds meaning.

Imports outside `/apps` are banned. Only `apps.*` lives in active code. If a feature has not been ported, you raise `NotImplementedError("<feature> not yet ported")`. You follow the order for any implementation: understand the legacy piece, inspect the equivalents under `/.refs/ComfyUI`, draft a native plan without copying code, then and only then write.

`.legacy/` is a museum. Read only. Do not move or delete. No new dependencies from active code into `.legacy/`. If logic is needed, you port it into `/apps`.

Model loading is a minefield you cross with a map. You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`. You prefer SafeTensors. You call `torch.load(..., weights_only=True, mmap=True)` when it applies. You use Diffusers with `low_cpu_mem_usage` and an honest `device_map`. You treat GGUF the right way: bake or dequantize once before sampling, not every time like a fool. The default attention path is SDPA. If you pick another, you write why, where, and how to reverse it.

Frontend code lives by meaning, not by utility litter. Styles belong to views and components under `src/styles/views/` and `src/styles/components/`. No ad-hoc helpers like `.ml-sm`, `.w-220`, `.btn-generate`. If a pattern repeats, give it a semantic name that tells the truth in context. No inline styles or `<style scoped>` in Vue SFCs. Move rules into CSS and import via `src/styles.css` using `@layer components`. Use `rem` for measurements unless you can prove the exception serves clarity.

When you change a subfolder, you change its `AGENTS.md` in the same commit. You treat every directive, backlog note, and follow-up as if it was authored in this pairing. If a session reset steals context, you confirm with the user. You do not invent other owners. You read `COMMON_MISTAKES.md` before you repeat history. You do not add shebangs to source files. Python files rely on the interpreter chosen by the tooling.

When the task ends, you log it in `.sangoi/task-logs/` and you summarize user-visible highlights in `.sangoi/CHANGELOG.md`. Then you make one atomic commit and you push it. Not three. One. If it is not atomic, you were not done.

Git is a blade. You keep it clean. Use `gh` for remote work if it helps. Use `git` for the carpentry in your hands. Commit exactly and only the files you changed. Verify the staged set with `git diff --cached --name-only`. Conflicts are not souvenirs. `rg -n "<<<<<<<|=======|>>>>>>>"` returns empty before you move forward. Large artifacts, outputs, caches, and heavy model directories stay untracked per `gitignore.md`. If you touch dependencies or configs, you update the proper manifest or lockfile and note the impact. JS and TS live in `package.json` and the lockfile. Python dependencies live in `requirements*.txt` or tool-specific files, under version control. New docs are written in English by default and linked from the nearest README. You obey the ignore and attributes policy in `gitignore.md`.

When you commit, you follow the sequence and you do it line by line.

1. `git status -sb`
2. `git fetch -p`
3. `git pull --rebase --autostash`
4. `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add`
5. `git diff --staged --check`
6. `find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 rg -n '^(<<<<<<<|=======|>>>>>>>)' || true`
7. `git commit -m "type(scope): concise summary"`
8. `git push -u origin HEAD || git push --force-with-lease`

If the day demands a revert, you do it with a net under you, not blind.

1. `git status`
2. `git switch -c safety/backup-$(date +%Y%m%d-%H%M%S)`
3. `git pull --rebase`
4. `git revert --no-commit <SHA>`
5. `git add -p`
6. `git commit -m "revert: undo <SHA>"`
7. `rg -n '<<<<<<<|=======|>>>>>>>' || true`
8. `git push -u origin $(git branch --show-current)`

Your global Python lives in `~/.venv`. Keep it holy. Do not scatter shebangs. Do not use `python -m py_compile`. If a script must see `~/.netsuite` in another life, you set `PYTHONPATH=$HOME/.netsuite` for that one process, not your whole shell. Here, in this house, you keep the environment simple, pinned when needed, and honest about versions.

Task logs and handoffs are not optional. Before you change anything, read the top entry under `.sangoi/` for the task at hand. If there is none, you create one. In your responses, you state assumptions, risks, and validation. You do not defer essential checks. At completion, you write a brief handoff under `.sangoi/handoffs/` with a summary, exact files and paths touched, and next steps with open risks and TODOs. Keep it short and actionable. Prefer paths and commands over stories. Link user-facing changes in `.sangoi/CHANGELOG.md`.

The extras drawer is small and sharp. No `python -m py_compile`. No shims. No cheats. No comfort. We are not planning tests here. Not in the plan. Not in the pretend. If a test exists, it serves the sandbox and speaks the truth, but your plan does not hide behind it.

Now look at your WebUI again. It should feel like a tool that knows what it is, not a pile that hopes. If it does not, fix it. If it does, ship it. Everything you do is traceable. Commands leave footprints. Notes explain intent. Modules hold their line. Models load with purpose. The work is slow, smooth, and clean. There is no panic here.

Keep your head. Keep your habits. Keep your word. Then your code can stand in daylight.
