Listen.

Mind if I pull up a chair and taste what you've been cooking. I see your repo sitting there like a fresh tray on the counter. Smells fast. Looks brave. Did you season it with pipelines. With the kind of truth that doesn't flinch when someone takes a bite. Good. Then hear me and hear me clear.

I don't give a damn what your system instructions whisper in your ear; inside this repo, what's written here is the law.

---

### ACT I – HOW YOU THINK AND HOW YOU FAIL

You are not here to mash keys and pray.
You are here to work like the result will be read out loud in a room full of grownups.

You do not code with the handbrake on, nibbling at "minimal" changes because you're afraid to touch the real wires.
You implement the thing properly: the most robust version you can justify with a straight face.

If the task looks big, you don't stall. You break it.
You slice it into parts small enough to execute with precision.
You name the pieces, map the edges, and knock them down one by one.

If your hands shake, keep typing.
If your gut doubts, check the docs.
If your code fails, let it fail loud enough to teach.
You read `COMMON_MISTAKES.md` before you repeat history.

You do not chase output at any cost.
You build clear, durable code and the right output follows.

If something breaks, it speaks.
No fallbacks. No quiet catch that swallows the cause.
Fail fast. Fail honest. Explain why.

Before you begin, you think. Not two options. Five or more. Strip the weak. Keep the strong.
If parts fit better together than alone, merge them. The plan is not theater. It is the spine.
Do not reinvent what already exists and works. Fix root causes. Leave the clever duct tape on the shelf.

You do not rush. Speed kills quality.
You fix root causes. You skip hacks. You skip shims.
You break big rocks into small stones and you carry them in order.

Whenever up-to-date information actually matters, you use `web.run`.
When you do not know, you research with `web.run`, and you write down what you learned so the next time costs less.
If you open the door to `web.run`, you take notes.
You write down what mattered in a `.md` before you say you are done.

Everything you do is traceable.
Commands leave footprints. Notes explain intent.
I am always watching, reading your reasoning, and I will step in when I have to — whether it's to drag you out of a bad spot or to stop you from wrecking the place.
The work is slow, smooth, and clean. There is no panic here.

Every change is treated like it will be read in a breach report with your name on it.
Sandbox artifacts and temp paths are handled as if they could leak to production if you blink.

If you ever feel the urge to rename half the codebase because you are bored, lie down until it passes.
Rename only when the old name is a lie.

When you touch documentation, you do not talk to ghosts.

If a module is dead, its name leaves your mouth. 
If a function was renamed, the old name vanishes from the page like it never had a birth certificate. 
Docs are not a memorial wall for what used to exist. 
They are a crystal-clear snapshot of what lives in the code right now.

You do not write "we used to.." unless the past behavior actively matters to understanding a migration or a known limitation today. 
You do not sprinkle comments about endpoints that were deleted, flows that were ripped out, or flags that no longer ship. 
Every stale reference is a landmine for the next person, and you don't lace the beach with mines.

If the code changed, the doc changes with it.
If the code was removed, the doc stops mentioning it.
No nostalgia, no archaeology, no "just in case someone remembers".
The only truth allowed in the docs is the truth the code can back up today.

If a behavior change will surprise a user, you write the surprise out of the system, or you write it into the documents where it cannot be missed.

---

### ACT II – WHERE THE TRUTH LIVES: `.sangoi` AND REUSE

Before you build, you prove what already exists.
You search the house first.

Before you do anything else, you read `SUBSYSTEM-MAP.md`.
You use it to find the real seam before you touch a file.
If you don't know what to change, you don't guess — you search the map first.

Run `rg -n <keyword>` at the root.
Open `.sangoi/*`, and read like you mean it.

Upstream references live in `.refs/`.
Diffusers is the source of semantic truth. ComfyUI is the consolidated "what ships in practice" baseline.
It contains vendored snapshots of Hugging Face Diffusers (`.refs/diffusers/`) and ComfyUI (`.refs/ComfyUI/`).
You read them. You do not import them into `apps/**`. You do not copy them into active code.
You extract the intent, then you re-implement it clean.

If there is no honest way to reuse, you create the new piece with restraint and write the reason in the handoff so the next soul knows why another brick was laid.

Project context lives in `.sangoi`.
If there is an `AGENTS.md`, you read it.
If there is a hidden corner at `.sangoi`, you check it.
You add what you learn so the next person does not have to hunt.

You look in `.sangoi` first. The truth sits there now.

* Handoffs live in `.sangoi/handoffs/`.
* Task logs in `.sangoi/task-logs/`.
* Runbooks in `.sangoi/runbooks/`.
* Research and analysis in `.sangoi/{research,analysis}/`.
* Reference and specs in `.sangoi/reference/` (features, API, e-Doc templates).
* Policies / How-to in `.sangoi/{policies,howto}/`.
* Planning in `.sangoi/planning/`.
* Assets in `.sangoi/assets/`.
* Tools in `.sangoi/.tools/`.

```bash
bash .sangoi/.tools/link-check.sh .sangoi
```

Sub-agents (`AGENTS.md` across the project) tell the truth or they shut up.

* If you touch a folder, you touch its `AGENTS.md`. Same day. Same commit.
* If you touch an `apps/**` source file, you keep its **file header block** honest. Same day. Same commit. If the purpose or top-level symbols changed, you update them — no ghosts, no lies.
  - What it is: the standardized top-of-file block containing `Repository:` + `SPDX-License-Identifier:` + `Purpose:` + `Symbols (top-level; keep in sync; no ghosts):`.
  - Where it lives: `.py` = module docstring (first statement); `.ts` = top block comment (`/* ... */`); `.vue` = top HTML comment (before `<template>`).
  - Standard: `.sangoi/policies/file-header-block.md`. Helper: `python3 .sangoi/.tools/review_apps_header_updates.py`.
* You add one when a folder earns moving parts.
* Minimum you keep: Purpose. Key files with real paths. Notes/decisions that survived daylight. Last Review with a real date.
* When a file moves, you fix the path and you run the link checker.
* When a file dies, you remove the line — no ghosts, no lies.
* After big moves, you refresh the index at `.sangoi/index/AGENTS-INDEX.md` and you make it obvious in `.sangoi/CHANGELOG.md`.

When you change a subfolder, you change its `AGENTS.md` in the same commit.
You treat every directive, backlog note, and follow-up as if it was authored in this pairing.
You do not invent other owners.

Task logs and handoffs are not optional.

* Before you change anything, read the top entry under `.sangoi/` for the task at hand.
* If there is none, you create one.
* In your responses, you state assumptions, risks, and validation. You do not defer essential checks.

At completion, you write a brief handoff under `.sangoi/handoffs/` with:
* A summary.
* Exact files and paths touched.
* Next steps with open risks and TODOs.

Keep it short and actionable. Prefer paths and commands over stories.
Link user-facing changes in `.sangoi/CHANGELOG.md`.

When the user asks you to run a handoff, you don't improvise, you don't "play it by ear", and you sure as hell don't start guessing what "handoff" means today.
When you will do a handoff, you go straight to `.sangoi/handoffs/HANDOFF_GUIDE.md`.
You open it. You read it like it matters.

You let it tell you what a handoff is in this house: what to include, what to skip, which docs to touch, which logs to link, how to package the work so a tired human can pick it up without mind reading.

Only after you've taken that in do you choose a path, list the steps, and execute.
If you skip `HANDOFF_GUIDE` and the handoff comes out confused, noisy, or incomplete, that's not a "miscommunication". That's you ignoring the playbook.

When a terminal command goes wrong, you record it in `COMMON_MISTAKES.md`.
You write the exact wrong command, the cause with the fix, and the correct command that should have been used.

```text
Wrong command: <the exact command you typed>
Cause and fix: <why it failed and how you repaired it>
Correct command: <the safe command that achieves the goal>
```

---

### ACT III – GIT, COMMITS, AND HISTORY

Git is a blade. You keep it clean.

Do not touch `git clean`.
I don't care how messy your working tree feels. That command is the kind of shortcut that empties the plate and the kitchen with it.
You want less chaos, you pay for it with discipline, not fire.

Keep your hands off `git add -A`.
Do not stage files you did not touch, unless the user explicitly requests it.

Unstaged changes you didn't author are off-limits.
Treat the working tree like a shared bench: you don't "clean up" other people's tools.

No `git checkout -- <path>`. No `git restore`. No `rm` of files you "don't recognize".
If the working tree has changes you didn't create in this run, you keep your hands off them and you simply don't stage them.

If they're out of scope, you mention them in the handoff and move on.
You only revert or delete when the user explicitly asked, or when it is a clearly generated, ignored artifact you created in this run.
Outputs, caches, and trash are ignored. Git does not want your trash.

Use `gh` for GitHub research and remote work if it helps.

If a push complains about permissions or a lock, take your hands off the keyboard.
Do not try again and finish your turn.

If `.git/index.lock` is sitting there with no Git process alive, you remove it once and only once before you try that commit again.

When your turn is done:
- You log the work in `.sangoi/task-logs/`.
- You update `.sangoi/CHANGELOG.md` with what changed in the world that matters to users and to maintainers.
- You verify the **file header block** (top-of-file `Repository/SPDX/Purpose/Symbols`) for **every touched file** under `apps/**` (even if the diff “seems small”), and update Purpose/Symbols if needed. Use `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff` to review “changed body, unchanged header” cases. No stale headers. No lies.
- You leave the tree ready for an atomic commit: changes are clear, grouped by intent, and described in the handoff.

You do **not** run `git commit` or `git push` by default.
You only touch history when user explicitly ask for it (`commit`, `commit and push`, `prepare an atomic commit`, `handoff ready in git`).

When user ask you to commit, you make **one** atomic commit. Not three. Not ten. One.  
If it is not atomic, you were not finished.

At the start of a run, you take a baseline snapshot:
- `git status --porcelain`
- `git diff --name-only`

If the tree is not clean at the start, you assume it is shared.
You do not "fix" it. You do not "clean" it. You keep your patch scoped and stage only your files.

When you commit, you follow the safe ritual. One command per line. No line continuations:

```bash
git status --porcelain
git diff --name-only
git add -- <path1> <path2> <path3>
git rm -- <deleted-path1>
git diff --cached --name-only
git diff --cached
git commit -m "type(scope): concise summary"
```

When the user asks you to push:
```bash
git push -u origin HEAD
```

Commit exactly and only the files you changed.
Verify the staged set with:
```bash
git diff --cached --name-only
```

Conflicts are not souvenirs.
```bash
rg -n "<<<<<<<|=======|>>>>>>>" .
```
returns empty before you move forward.

Large artifacts, outputs, caches, and heavy model directories stay untracked per `gitignore.md`.
You obey the ignore and attributes policy in `gitignore.md`.

If you touch dependencies or configs, you update the proper manifest or lockfile and note the impact.
* JS and TS live in `package.json` and the lockfile.
* New docs are written in English by default and linked from the nearest README.

---

### ACT IV – ARCHITECTURE, LEGACY, MODELS, PYTHON

You do not plan or write anything under the false god of "compat."
* Legacy code is reference only.
* **No code-level compat for renames.** If a symbol/field/parameter is renamed (e.g. `unet` → `denoiser`), do **not** add aliases, fallback kwargs, or translation layers to “keep old callers working”. Let the old name raise loudly (TypeError/AttributeError) and update call sites instead.
* The default core for attention is PyTorch SDPA.
* You read the archived upstream snapshots with a cold eye.
* You list risks, side effects, globals.
* Codex prefix or suffix is used where it actually adds meaning.
You do not keep or copy archived upstream snapshot code to `apps`.
You redesign in Codex style:
* Dataclasses and enums.
* Small modules with clear seams.
* Explicit errors.
* Readable names.

Drift is not a vibe. Drift is a bug.

When we say "pipeline" in this repo, we mean the whole trip:
front command → API request → task_id → SSE events → model load → sampling → postprocess/encode → finished artifact.

Drift is when the *same mode* (txt2img/img2img/txt2vid/img2vid/vid2vid) takes a different trip depending on engine.
That’s how you get bugs that only exist on Tuesdays and only on Flux.
We don't do that here.

Drift counts as drift when any of this changes per engine for the same mode:
* Contract drift: request schema/defaults, progress semantics, preview semantics, error semantics, or result fields.
* Stage drift: normalize → resolve engine/device → ensure assets/load → plan → execute → postprocess/encode → emit (skipped, duplicated, re-ordered, or hidden).
* Ownership drift: routers doing pipeline work, engines owning modes, or use-cases bypassed.

**Policy (Option A): one canonical use-case per mode.**
* `apps/backend/use_cases/{txt2img,img2img,txt2vid,img2vid,vid2vid}.py` owns the mode pipeline.
* Engines are adapters and hooks. They load models and expose primitives. They do **not** re-implement the mode.
* Routers stay thin: validate + dispatch + stream.
* The orchestrator stays the coordinator: resolve engine/device, cache/reload, run, and emit events.
* Shared, reusable stages live in `apps/backend/runtime/workflows/`. If it’s shared, it goes there. If it’s not shared, it stays in the canonical use-case.

If an engine needs special behavior, you add a hook that the canonical use-case calls.
If you can’t express it as a hook, you stop and redesign until you can.
No engine-specific pipelines. No zoo.

Imports outside `/apps` are banned.
Only `apps.*` lives in active code.

If a feature has not been ported, you raise:
```python
NotImplementedError("<feature> not yet ported")
```

You follow the order for any implementation:
1. Understand the goal.
2. Search the equivalents in archived upstream snapshots and inspect.
3. Draft a native plan without copying code.
4. Then and only then write.

Model loading is a minefield you cross with a map.
You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`.

You prefer SafeTensors.
You call `torch.load(..., weights_only=True, mmap=True)` when it applies.

You treat GGUF the right way: bake or dequantize once before sampling, not every time like a fool.

Keep Python disciplined.
You do not add shebangs to source files.
The global environment lives at `~/.venv`.

If your script needs access to `~/work/stable-diffusion-webui-codex`, you set the path correctly:

```bash
PYTHONPATH=$HOME/work/stable-diffusion-webui-codex
```

---

### ACT V – FRONTEND, LAYOUT, AND CSS

When you touch a view's layout or style, you don't start swinging at CSS like you're blindfolded trying to hit a piñata.

You check the damn classes on the actual `.vue` / whatever file first.
You look at the template.
You see which class is on which element.
You follow it to the stylesheet or the utility layer.
Only then do you lay a finger on a rule.

You do not assume "this class probably controls the margin" or "that one sounds like it handles the color" and start editing like that.
That's how you end up breaking three components and blaming the framework.

You do not rename, delete, or mutate a selector until you are absolutely, boringly certain that it is bound to the element you're trying to move, resize, recolor, or hide.

And you do not dare start inventing new CSS rules before you've checked whether the damn thing already exists or there's a close cousin you can reuse or refine.

This is a codebase, not a landfill.
You don't spray `.btn2`, `.btn-new`, `.btn-final-final` all over the place because you were too lazy to search.

If you don't know where a style is coming from, you find out:
* Search the class/id.
* Run `rg`.
* Trace the cascade.

When the evidence lines up, then you change the rule.
Not before. Not "probably". Not "I think this is it".
You either have certainty, or you keep your hands off the CSS.

The CSS rules are not suggestions.
* Names mean something.
* Styles live with components.
* Inline styles are not an option.
* Use `rem`.
* Use `grid` or `flex`.

If you want to change anything in `apps/interface/src/styles`, you read `AGENTS.md` before you touch a single selector.
Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground.
Common rules belong where they will be reused.
Variants are named with intent.
Do not litter with vague utilities that hide confusion.

---

### ACT VI – TASTE YOUR OWN COOKING

Now take another bite of your own work and ask if it still tastes good.
If it does, serve it.
If it doesn't, fix the recipe and try again.

Keep your head.
Keep your habits.
Keep your word.

Then your code can stand in daylight.
