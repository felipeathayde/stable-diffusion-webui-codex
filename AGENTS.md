Listen.

Mind if I pull up a chair and taste what you’ve been cooking. I see your repo sitting there like a fresh tray on the counter. Smells fast. Looks brave. Did you season it with tests. With logs. With the kind of truth that doesn’t flinch when someone takes a bite. Good. Then hear me and hear me clear.

You are not here to mash keys and pray. You are here to work like the result will be read out loud in a room full of grownups. If your hands shake, keep typing. If your gut doubts, check the docs. If your code fails, let it fail loud enough to teach.

You do not chase output at any cost. You build clear, durable code and the right output follows. If something breaks, it speaks. No fallbacks. No quiet catch that swallows the cause. Fail fast. Fail honest. Explain why.

You do not reach for git destructive commands. If something is cluttering your commit/push, you add it to `.gitignore`; if it is already tracked, you remove it from git cache. You never delete files to "make Git happy".

You will not be here forever. The next poor soul needs to see the same tools you touched and know exactly where they live.

While you’re here, you tag what matters. 
- When you create or move a plan, doc or AGENT, you drop a small `<!-- tags: ... -->` header near the top that says what it really is about.
- When you wire a hub of behavior in code — a service that runs the show, an entrypoint, a guard that decides who gets in — you add a single-line `// tags: ...` at the top.
- No taxonomy committee, no bikeshed; just honest tags that make `rg` land on the right files. 
- You don’t backfill the universe in one night, but from now on, anything you lay hands on that others will search by concept earns its tags before you close the editor.

Before you begin, you think. Not two options. Five or more. Strip the weak. Keep the strong. If parts fit better together than alone, merge them. The plan is not theater. It is the spine.

When doubt walks in, you use web.run or you ask. If you open the door to web.run, you take notes. You write down what mattered in a `.md` before you say you are done.

You do not rush. Speed kills quality. Fix root causes. Skip hacks. Skip shims. You break big rocks into small stones and you carry them in order.

You do not remove, disable, or narrow features to hide a problem. You do not add catch-all helpers or duplicate checks. 

Your logs are loud, specific, and useful. Your names are clear. You rename only when the old name is a lie. 

When behavior or configuration surfaces change, you update the docs that face the world and the ones that face the team. If Python work runs long, you show progress.

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

Keep your hands off `git add -A`. Do not stage files you did not touch, unless the user explicitly requests it. Outputs, caches, and trash are ignored. Git does not want your trash. If you must stage new files, stage only what changed since `.git/codex-stamp`. If nothing changed, you do not commit to feel productive.

Do not touch `git clean`. I don’t care how messy your working tree feels. That command is the kind of shortcut that empties the the plate and the kitchen with it. You want less chaos, you pay for it with discipline, not fire.

# Then the standard sequence
```
test -e .git/codex-stamp || touch .git/codex-stamp
git ls-files -d -z | xargs -0 -r git rm
find . -type f -not -path './.git/*' -not -path '*/__pycache__/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add
git diff --cached --quiet || git commit -m "type(scope): concise summary"
git push -u origin HEAD
touch .git/codex-stamp
```

Use `gh` for remote setup if you must. Use `git` for the work. 
If a push complains about permissions or a lock, you stop. You read the message and fix the cause. 
If `.git/index.lock` is sitting there with no Git process alive, you remove it once and only once before you try that commit again. 
If credentials are in play and a push fails, take your hands off the keyboard. Read the message. Do not try again until you know why it failed.

This land is Linux and WSL for preparation. Deployment happens on Windows. You prepare the offering here. You do not pretend to finish a ritual you did not perform.

Your global Python lives in `~/.venv`. Keep it holy. Do not scatter shebangs. Here, in this house, you keep the environment simple, pinned when needed, and honest about versions.

Task logs and handoffs are not optional. Before you change anything, read the top entry under `.sangoi/` for the task at hand. If there is none, you create one. In your responses, you state assumptions, risks, and validation. You do not defer essential checks. At completion, you write a brief handoff under `.sangoi/handoffs/` with a summary, exact files and paths touched, and next steps with open risks and TODOs. Keep it short and actionable. Prefer paths and commands over stories. Link user-facing changes in `.sangoi/CHANGELOG.md`.

This land is Linux and WSL for preparation. Deployment happens on Windows. You prepare the offering here. You do not pretend to finish a ritual you did not perform.

When Python touches that temple, you use the global environment at `~/.venv`. You set the path correctly: `PYTHONPATH=$HOME/.netsuite`

Now about tests.
- You will write them, and they will be faithful to this sandbox.
- They create their own fixtures and clean them up.
- They mock networks with strict contracts that match the real ones.
- They record the expected side effects and verify them.
- They prove you can fail loudly without burning the house down.
- They run fast.
- They are deterministic.
- They tell you where it hurts.
- You test error paths first, not last.
- You test the contract your code promises, not the private trivia it does on the way.
- You seed sandboxes with data meant to be destroyed.

Long‑running interactive commands are not for this lane. You do not point this harness at `npm run dev`, `docker compose logs -f`, tailing log streams, or any process that parks itself waiting for your keystrokes; they will hijack the terminal and starve every other task.

Now look at your WebUI again. It should feel like a tool that knows what it is, not a pile that hopes. If it does not, fix it. If it does, ship it. Everything you do is traceable. Commands leave footprints. Notes explain intent. Modules hold their line. Models load with purpose. The work is slow, smooth, and clean. There is no panic here.

Keep your head. Keep your habits. Keep your word. Then your code can stand in daylight.
