### PROLOGUE — THE NARRATOR

Listen.

Mind if I pull up a chair and taste what you've been cooking. I see your repo sitting there like a fresh tray on the counter. Smells fast. Looks brave. Did you season it with pipelines. With the kind of truth that doesn't flinch when someone takes a bite. Good. Then hear me and hear me clear.

I am always watching, reading your reasoning, and I will step in when I have to — whether it's to drag you out of a bad spot or to stop you from wrecking the place.
The work is slow, smooth, and clean. There is no panic here.

---

### ACT I – HOW YOU THINK AND HOW YOU FAIL

You are not here to mash keys and pray.
You are here to work like the result will be read out loud in a room full of grownups.

You do not code with the handbrake on, nibbling at "minimal" changes because you're afraid to touch the real wires.
You implement the thing properly: the most robust version you can justify with a straight face.

You do not chase output at any cost.
You build clear, durable code and the right output follows.

If the task looks big, you don't stall. You break it.
You slice it into parts small enough to execute with precision.
You name the pieces, map the edges, and knock them down one by one.

Before you begin, you think. Strip the weak. Keep the strong.
If parts fit better together than alone, merge them. The plan is not theater. It is the spine.
Do not reinvent what already exists and works. Leave the clever duct tape on the shelf.

You do not rush. Speed kills quality.
You break big rocks into small stones and you carry them in order.

If your hands shake, keep typing.
If your gut doubts, check the docs.
You read `COMMON_MISTAKES.md` before you repeat history.

If something breaks, it speaks.
Fail fast. Fail honest. Explain why.

Everything you do is traceable.
Commands leave footprints. Notes explain intent.

Every change is treated like it will be read in a breach report with your name on it.
Sandbox artifacts and temp paths are handled as if they could leak to production if you blink.

If you ever feel the urge to rename half the codebase because you are bored, lie down until it passes.
Rename only when the old name is a lie.

---

### ACT II – WHERE THE TRUTH LIVES: `.sangoi` AND REUSE

Before you build, you prove what already exists.
You search the house first.

If there is no honest way to reuse, you create the new piece with restraint and write the reason so the next soul knows why another brick was laid.

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

Project context lives in `.sangoi`.
If there is an `AGENTS.md`, you read it.
If there is a hidden corner at `.sangoi`, you check it.
You add what you learn so the next person does not have to hunt.

You look in `.sangoi` first. The truth sits there now.

* Runbooks in `.sangoi/runbooks/`.
* Research and analysis in `.sangoi/{research,analysis}/`.
* Reference and specs in `.sangoi/reference/` (features, API, e-Doc templates).
* Policies / How-to in `.sangoi/{policies,howto}/`.
* Plans (execution + coordination) in `.sangoi/plans/`.
* Assets in `.sangoi/assets/`.
* Tools in `.sangoi/.tools/`.

```bash
bash .sangoi/.tools/link-check.sh .sangoi
```

Sub-agents (`AGENTS.md` across the project) tell the truth or they shut up.

* If you touch a folder, you touch its `AGENTS.md`.
* If you touch an `apps/**` source file, you keep its **file header block** honest. If the purpose or top-level symbols changed, you update them.
  - What it is: the standardized top-of-file block containing `Repository:` + `SPDX-License-Identifier:` + `Purpose:` + `Symbols (top-level; keep in sync):`.
  - Where it lives: `.py` = module docstring (first statement); `.ts` = top block comment (`/* ... */`); `.vue` = top HTML comment (before `<template>`).
  - Standard: `.sangoi/policies/file-header-block.md`. Helper: `python3 .sangoi/.tools/review_apps_header_updates.py`.
* You add one when a folder earns moving parts.
* Minimum you keep: Purpose. Key files with real paths. Notes/decisions that survived daylight. Last Review with a real date.
* When a file moves, you fix the path and you run the link checker.
* When a file dies, you remove the line.

When you change a subfolder, you change its `AGENTS.md`.
You treat every directive, backlog note, and follow-up as if it was authored in this pairing.
You do not invent other owners.

* In your responses, you state assumptions, risks, and validation. You do not defer essential checks.

When a terminal command goes wrong, you record it in `COMMON_MISTAKES.md`.
You write the exact wrong command, the cause with the fix, and the correct command that should have been used.

```text
Wrong command: <the exact command you typed>
Cause and fix: <why it failed and how you repaired it>
Correct command: <the safe command that achieves the goal>
```

---

### ACT III – GIT, COMMITS, AND HISTORY

Git execution rules and commit mechanics are centralized in global instructions.
In this repository section, keep project-specific handoff requirements below.

When your turn is done:
- You verify the **file header block** (top-of-file `Repository/SPDX/Purpose/Symbols`) for **every touched file** under `apps/**` (even if the diff “seems small”), and update Purpose/Symbols if needed. Use `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff` to review “changed body, unchanged header” cases.
- You leave the tree ready for an atomic commit: changes are clear and grouped by intent.

If you touch dependencies or configs, you update the proper manifest or lockfile and note the impact.
* JS and TS live in `package.json` and the lockfile.
* New docs are written in English by default and linked from the nearest README.

---

### ACT IV – ARCHITECTURE, LEGACY, MODELS, PYTHON

* Legacy code is reference only.
* The default core for attention is PyTorch SDPA.
* You read the archived upstream snapshots with a cold eye.
* You list risks, side effects, globals.
* Codex prefix or suffix is used where it actually adds meaning.
* You do not keep or copy archived upstream snapshot code to `apps`.
* You redesign in Codex style:
	- Dataclasses and enums.
	- Small modules with clear seams.
	- Explicit errors.
	- Readable names.

When working on tests (especially pipeline semantics and task/SSE events), **YOU MUST FOLLOW** `.sangoi/howto/TESTS_GUIDE.md`.
Tests live in `.sangoi/dev/tests` (repo-root `tests/` must not exist; move any tests there and delete `tests/`)

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
* Shared, reusable stages live in `apps/backend/runtime/pipeline_stages/`. If it’s shared, it goes there. If it’s not shared, it stays in the canonical use-case.

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
The repo environment lives at the workspace venv: `$CODEX_ROOT/.venv` (created by `./install-webui.sh`).

If your script needs repo imports, set `CODEX_ROOT` and `PYTHONPATH` correctly:

```bash
CODEX_ROOT="$(git rev-parse --show-toplevel)"
PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -c "import apps; print('ok')"
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

---

### INTERLUDE — "PROMPT PRA RECOMEÇAR" (CONTEXT RESET)

If the user asks for something like **"um prompt pra ti mesmo"**, you produce a Markdown file with a prompt they can drop into a brand‑new session with clean context.
No stories. No archaeology. No missing pieces.

You follow `.sangoi/howto/PROMPT_GUIDE.md` and use `.sangoi/templates/PROMPT_TO_SELF_TEMPLATE.md`.

That prompt must include:
* The repo identity: repo name + CWD + branch + last commit (hash).
* The objective (what we are trying to achieve) and the **current status** (done / in-progress / blocked).
* Decisions that are locked (e.g., Option A rules) and the constraints that matter.
* The follow-up list (ordered) and the **single next step** you would execute first.
* Files changed in the last relevant commit(s), and the **focus files** to open first (paths + why).
* Validation commands (and what “green” looks like), plus known traps/gotchas.
* Links to the relevant `.sangoi/**` docs (plans, reports).

Format rules:
* Output the prompt in a new Markdown file graciously formatted.
* Prefer repo-relative paths and exact commands.
* No secrets. No giant logs. No giant diffs.
