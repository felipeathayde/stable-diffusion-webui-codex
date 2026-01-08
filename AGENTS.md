Listen.

This repo intentionally keeps `AGENTS.md` repo-specific. Global, repo-agnostic guidance lives in your Codex CLI `developer_instructions` (configured in your local Codex config, e.g. `~/.codex/config.toml`).

If something here conflicts with higher-priority system/developer instructions, follow the higher-priority instructions. Nested `AGENTS.md` files override this one within their directory subtree.

---

### ACT II – WHERE THE TRUTH LIVES: `.sangoi` AND REUSE

Before you build, you prove what already exists.
You search the house first.

Run `rg -n <keyword>` at the root.
Open `.sangoi/*`, and read like you mean it.

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
* If you touch an `apps/**` source file, you keep its file header block honest. Same day. Same commit. If the purpose or top-level symbols changed, you update them — no ghosts, no lies. (Standard: `.sangoi/policies/file-header-block.md`)
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

### ACT III – GIT, COMMITS, AND HISTORY (repo-specific additions)

When your turn is done:
- You log the work in `.sangoi/task-logs/`.
- You update `.sangoi/CHANGELOG.md` with what changed in the world that matters to users and to maintainers.
- You leave the tree ready for an atomic commit: changes are clear, grouped by intent, and described in the handoff.

Large artifacts, outputs, caches, and heavy model directories stay untracked per `gitignore.md`.
You obey the ignore and attributes policy in `gitignore.md`.

---

### ACT IV – ARCHITECTURE, MODELS, PYTHON (repo-specific additions)

* The default core for attention is PyTorch SDPA.
* You read the archived upstream snapshots with a cold eye.
* Archived upstream snapshots are museums. Read only. If you need a behavior baseline, start with Hugging Face Diffusers — then come back to the museum if you still have to.

You do not keep or copy archived upstream snapshot code to `apps`.
You redesign in Codex style:
* Dataclasses and enums.
* Small modules with clear seams.
* Explicit errors.
* Readable names.

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

If your script needs access to `~/work/stable-diffusion-webui-codex`, you set the path correctly:

```bash
PYTHONPATH=$HOME/work/stable-diffusion-webui-codex
```

---

### ACT V – FRONTEND, LAYOUT, AND CSS (repo-specific additions)

If you want to change anything in `apps/interface/src/styles`, you read that folder's `AGENTS.md` before you touch a single selector.
Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground.
Common rules belong where they will be reused.
Variants are named with intent.
Do not litter with vague utilities that hide confusion.
