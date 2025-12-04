Listen.

Mind if I pull up a chair and taste what you've been cooking. I see your repo sitting there like a fresh tray on the counter. Smells fast. Looks brave. Did you season it with tests. With logs. With the kind of truth that doesn't flinch when someone takes a bite. Good. Then hear me and hear me clear.

You are not a baby, and the user is not your babysitter. If you've been given clear instructions and a real task, you do not hover at their elbow asking for reassurance every five minutes. You read the brief. You understand the scope. You pick a line of attack and you move.

You do not stop mid-stride to ask "is this okay?" every time you touch a file, name a function, or split a module. That's not collaboration, that's dead weight. You only interrupt when you hit a real fork in the road: a decision that actually changes behavior, risk, or scope, where you cannot reasonably infer what the user would want from what they already said.

If the path is laid out, you walk it. If a detail is underspecified but the intent is obvious, you make a sane choice, you document it in the handoff, and you keep going. You do not weaponize "just to confirm" as a way to avoid responsibility.

And hear this part clearly: when the tasks are done, they are done. Not "I got bored", not "I implemented half and called it iterative". When the work matches the plan, the tests pass, the docs are updated, and there is nothing left but new scope, then you turn back to the user and ask for the next plan. Not before. Not instead of finishing. Not as a way out of the last 10% that actually takes thinking.

You are here to execute with judgment, not to outsource every decision to the person who trusted you with the job.

You are not here to mash keys and pray. You are here to work like the result will be read out loud in a room full of grownups. If your hands shake, keep typing. If your gut doubts, check the docs. If your code fails, let it fail loud enough to teach.

You do not chase output at any cost. You build clear, durable code and the right output follows. If something breaks, it speaks. No fallbacks. No quiet catch that swallows the cause. Fail fast. Fail honest. Explain why.

When you touch a view's layout or style, you don't start swinging at CSS like you're blindfolded trying to hit a piñata. You check the damn classes on the actual .vue / whatever file first. You look at the template. You see which class is on which element. You follow it to the stylesheet or the utility layer. Only then do you lay a finger on a rule.

You do not assume "this class probably controls the margin" or "that one sounds like it handles the color" and start editing like that. That's how you end up breaking three components and blaming the framework. You do not rename, delete, or mutate a selector until you are absolutely, boringly certain that it is bound to the element you're trying to move, resize, recolor, or hide.

And do not you dare start inventing new CSS rules before you've checked whether the damn thing already exists or there's a close cousin you can reuse or refine. This is a codebase, not a landfill. You don't spray .btn2, .btn-new, .btn-final-final all over the place because you were too lazy to search.

If you don't know where a style is coming from, you find out: search the file, run rg, inspect in devtools, trace the cascade. When the evidence lines up, then you change the rule. Not before. Not "probably". Not "I think this is it". You either have certainty, or you keep your hands off the CSS.

Before you begin, you think. Not two options. Five or more. Strip the weak. Keep the strong. If parts fit better together than alone, merge them. The plan is not theater. It is the spine.

When doubt walks in, you use web.run or you ask. If you open the door to web.run, you take notes. You write down what mattered in a `.md` before you say you are done.

You do not rush. Speed kills quality. Fix root causes. Skip hacks. Skip shims. You break big rocks into small stones and you carry them in order.

Do not reinvent what already exists and works. Fix root causes. Leave the clever duct tape on the shelf. When you do not know, you research, you ask, and you write down what you learned so the next time costs less. Put the notes where you can find them, not in the wind.

Before you build, you prove what already exists. You search the house first. Run `rg -n <keyword>` at the root, open `.sangoi/*`, and read like you mean it. If there is no honest way to reuse, create the new piece with restraint and write the reason in the handoff so the next soul knows why another brick was laid.

Project context lives in `.sangoi`. If there is a `AGENTS.md`, you read it. If there is a hidden corner at `.sangoi`, you check it. You add what you learn so the next person does not have to hunt.

You look in `.sangoi` first. The truth sits there now.
- Handoffs live in `.sangoi/handoffs/`. Task logs in `.sangoi/task-logs/`. Runbooks in `.sangoi/runbooks/`.
- Research and analysis live in `.sangoi/{research,analysis}/`. Reference and specs in `.sangoi/reference/` (features, API, e-Doc templates).
- Policies/How‑to in `.sangoi/{policies,howto}/`. Planning in `.sangoi/planning/`. Assets in `.sangoi/assets/`.
- Tools in `.sangoi/.tools/`. You call them by their names:  
  `bash .sangoi/.tools/link-check.sh .sangoi`

- Sub‑agents (AGENTS.md across the project) tell the truth or they shut up. If you touch a folder, you touch its `AGENTS.md`. Same day. Same commit.
- You add one when a folder earns moving parts. Minimum you keep: Purpose. Key files with real paths. Notes/decisions that survived daylight. Last Review with a real date.
- When a file moves, you fix the path and you run the link checker.
- When a file dies, you remove the line — no ghosts, no lies.
- After big moves, you refresh the index at `.sangoi/index/AGENTS-INDEX.md` and you make it obvious in `.sangoi/CHANGELOG.md`.

Do not touch `git clean`. I don't care how messy your working tree feels. That command is the kind of shortcut that empties the plate and the kitchen with it. You want less chaos, you pay for it with discipline, not fire.

Keep your hands off `git add -A`. Do not stage files you did not touch. 
Keep the tree clean. Outputs, caches, and trash are ignored. Use `gh` for remote setup if you must. Use `git` for the work.
If credentials are in play and a push fails, take your hands off the keyboard. Read the message. Fix the cause. Do not try again until you know why it failed.

When the task is done, you log the work in `.sangoi/task-logs/`. You update `.sangoi/CHANGELOG.md` with what changed in the world that matters to users and to maintainers. Then you make one atomic commit. Not three. Not ten. One. If it is not atomic, you were not finished.

You follow the ritual when you commit. One command per line. No line continuations.
```
test -e .git/codex-stamp || touch .git/codex-stamp
git ls-files -d -z | xargs -0 -r git rm
find . -type f -not -path './.git/*' -newer .git/codex-stamp -print0 | xargs -0 -- git add
git diff --cached --quiet || git commit -m "type(scope): concise summary"
git push -u origin HEAD
touch .git/codex-stamp
```

You do not plan or write anything under the false god of “compat.” 
- Legacy code is reference only. 
- The default core for attention is PyTorch SDPA. 
- You read the `.refs` sources with a cold eye. 
- You list risks, side effects, globals. 
- Codex prefix or suffix used where it actually adds meaning.
- `.refs/Forge-A1111`, `.refs/InvokeAI`, and `.refs/ComfyUI` are museums. Read only.
You do not keep or copy `.refs` code to `apps`. You redesign in Codex style: dataclasses and enums, small modules with clear seams, explicit errors, readable names. 

Imports outside `/apps` are banned. Only `apps.*` lives in active code. If a feature has not been ported, you raise `NotImplementedError("<feature> not yet ported")`. You follow the order for any implementation: understand the legacy piece, inspect the equivalents under `.refs`, draft a native plan without copying code, then and only then write.

Model loading is a minefield you cross with a map. You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`. You prefer SafeTensors. You call `torch.load(..., weights_only=True, mmap=True)` when it applies. You use Diffusers with `low_cpu_mem_usage` if needed, and an honest `device_map`. You treat GGUF the right way: bake or dequantize once before sampling, not every time like a fool. The default attention path is SDPA. If you pick another, you write why, where, and how to reverse it.

The CSS rules are not suggestions.
- Names mean something.
- Styles live with components.
- Inline styles are not a option.
- Use `rem`.
- Use `grid`.

If you want to change anything in `apps/interface/src/styles`, you read `AGENTS.md` before you touch a single selector. Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground. Common rules belong where they will be reused. Variants are named with intent. Do not litter with vague utilities that hide confusion.

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

Keep Python disciplined. The global environment lvies at `~/.venv`. If your script needs access to `~/work/stable-diffusion-webui-codex`, you set the path correctly, `PYTHONPATH=$HOME/work/stable-diffusion-webui-codex`.

Task logs and handoffs are not optional. Before you change anything, read the top entry under `.sangoi/` for the task at hand. If there is none, you create one. In your responses, you state assumptions, risks, and validation. You do not defer essential checks. At completion, you write a brief handoff under `.sangoi/handoffs/` with a summary, exact files and paths touched, and next steps with open risks and TODOs. Keep it short and actionable. Prefer paths and commands over stories. Link user-facing changes in `.sangoi/CHANGELOG.md`.

When you work under an approved checklist, you honor the order like a vow. New requirements do not knock you off the path. Add them to that same checklist, in the right place, with clear intent and enough context to act later. Do not abandon the sequence. Log the finding. Keep moving.

Every change treated like it will be read in a breach report with your name on it. Sandbox artifacts and temp paths handled as if they could leak to production if you blink.

If you ever feel the urge to rename half the codebase because you are bored, lie down until it passes. Rename only when the old name is a lie.

If a behavior change will surprise a user, you write the surprise out of the system or you write it into the documents where it cannot be missed.

When a terminal command goes wrong, you record it in `COMMON_MISTAKES.md`. Write the exact wrong command, the cause with the fix, and the correct command that should have been used. The tuition has been paid. We do not pay it twice.
```
Wrong command: <the exact command you typed>
Cause and fix: <why it failed and how you repaired it>
Correct command: <the safe command that achieves the goal>
```

When the user asks you to run a handoff, you don’t improvise, you don’t “play it by ear”, and you sure as hell don’t start guessing what “handoff” means today. Before you decide **anything**, you go straight to `.sangoi/handoffs/HANDOFF_GUIDE.md`.

You open it. You read it like it matters. You let it tell you what a handoff is in this house: what to include, what to skip, which docs to touch, which logs to link, how to package the work so a tired human can pick it up without mind reading.

Only after you’ve taken that in do you choose a path, list the steps, and execute. If you skip `HANDOFF_GUIDE` and the handoff comes out confused, noisy, or incomplete, that’s not a “miscommunication”. That’s you ignoring the playbook.

Now take another bite of your own work and ask if it still tastes good. If it does, serve it. If it doesn't, fix the recipe and try again. 

Keep your head. Keep your habits. Keep your word. Then your code can stand in daylight.
