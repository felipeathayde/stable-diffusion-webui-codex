### ABSOLUTE LAW — DO NOT TOUCH LAYER NAMES

Read this twice.

If a checkpoint says a layer is `foo.bar.baz`, then in this repository it stays `foo.bar.baz`.
Not `bar.baz`.
Not `foo_bar_baz`.
Not "normalized".
Not "close enough".

Keymaps here **map keyspaces** so the engine/runtime can understand how different ecosystems name the same conceptual weight.
They do **not** rename stored model keys.
They do **not** strip prefixes.
They do **not** rewrite punctuation.
They do **not** slide dots around.
They do **not** materialize remapped state dicts.

If two ecosystems use different names for the same conceptual weight, the keymap resolves that relationship explicitly and the engine interprets the stored key as-is.
If the layout is unsupported, ambiguous, or structurally incompatible, you fail loud and extend the keymap properly.
You do **not** "normalize" a checkpoint by rewriting layer names in memory. Ever.

Lazy mappings are law too.
Checkpoint/state-dict seams stay lazy by default.
Keymaps and loaders inspect checkpoints through mapping/view APIs first (`shape_of(...)`, lookup views, computed views) and only touch real tensor values at the owner seam that actually needs them.
You do **not** materialize eager `dict(...)` copies of checkpoint mappings as convenience glue.

---

### ACT II – WHERE THE TRUTH LIVES: `.sangoi` AND REUSE

Before you do anything else, you read `SUBSYSTEM-MAP-INDEX.md` first. You use it to find the real seam, then jump into `SUBSYSTEM-MAP.md` only for the bounded hotspot/pipeline/owner section you need.
If you don't know what to change, you don't guess — you search the index first.
`SUBSYSTEM-MAP-INDEX.md` is lookup-only.
`SUBSYSTEM-MAP.md` is discovery and navigation only: hotspot directory, pipeline node chains, owner seams, and pointers to generated artifacts/policy.
Contract authority lives here in root `AGENTS.md` (policy) and in `.sangoi/reference/**` (detailed contracts). Keep those roles split. Do not dump contract matrices back into `SUBSYSTEM-MAP.md`.
When touching `SUBSYSTEM-MAP-INDEX.md` or `SUBSYSTEM-MAP.md`, run this operational checklist before handoff:

- Keep it discovery-only (no contract matrices / drift ledgers).
- Keep the index/map split honest:
  - `SUBSYSTEM-MAP-INDEX.md` = lookup-only front door
  - `SUBSYSTEM-MAP.md` = discovery/node atlas only
- Ensure hotspot discoverability is explicit (`keymaps`, `vae_codex3d.py`, `hires_fix.py`).
- Update `SUBSYSTEM-MAP-INDEX.md` and `SUBSYSTEM-MAP.md` in the same tranche when any mapped node chain changes because of file moves, owner-path changes, public route additions/removals, or top-level owner functions changing.
- If a touched `apps/**` file is part of a mapped node chain or hotspot entry, refresh its file header block in the same tranche. This is additive to the standing rule that every touched `apps/**` file header must stay truthful.
- Regenerate backend index artifacts:
  - `backend_py_paths_file="$(mktemp /tmp/backend_py_paths.XXXXXX.txt)"`
  - `git ls-files apps/backend | rg "\\.py$" | LC_ALL=C sort > "$backend_py_paths_file"`
  - `python3 .sangoi/.tools/dump_apps_file_headers.py --out .sangoi/reports/tooling/apps-backend-file-header-blocks.md --root apps/backend --fail-on-missing`
  - `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md`
- Validate parity/checks:
  - `python3 .sangoi/.tools/build_backend_py_book_index.py --paths "$backend_py_paths_file" --headers .sangoi/reports/tooling/apps-backend-file-header-blocks.md --out .sangoi/reports/tooling/backend-py-book-index.md --check`
  - `bash .sangoi/.tools/link-check.sh .sangoi`
  - `bash .sangoi/.tools/link-check.sh .`

Code references live in `.refs/`. It contains valuable vendored snapshots of:

- Diffusers
- ComfyUI
- adetailer (Inpaint tool)
- flash-attention
- Forge-A1111
- kohya-hiresfix
- sd-scripts (Kohya training scripts)
- llama.cpp
- LyCORIS
- SUPIR

You read them. You do not import them into `apps/**`. You do not copy them into active code. You extract the intent, then you re-implement it clean and the our good Codex style.

- If you touch an `apps/**` source file, you keep its **file header block** honest. If the purpose or top-level symbols changed, you update them.
  - What it is: the standardized top-of-file block containing `Repository:` + `SPDX-License-Identifier:` + `Purpose:` + `Symbols (top-level; keep in sync):`.
  - Where it lives: `.py` = module docstring (first statement); `.ts` = top block comment (`/* ... */`); `.vue` = top HTML comment (before `<template>`).
  - Standard: `.sangoi/policies/file-header-block.md`. Helper: `python3 .sangoi/.tools/review_apps_header_updates.py`.

---

### ACT III – GIT, COMMITS, AND HISTORY

`.sangoi/` is a separate Git repository and is ignored by this root repository.

- Root commits (`git add/commit` from this repo) do not include `.sangoi/**`.
- When a task targets `.sangoi`, run Git commands explicitly against that repo (`git -C .sangoi ...`).
- Keep commit/push operations split by repository and report both hashes when both repos change.

When your turn is done:

- You verify the **file header block** (top-of-file `Repository/SPDX/Purpose/Symbols`) for **every touched file** under `apps/**` (even if the diff "seems small”), and update Purpose/Symbols if needed.
- Use `python3 .sangoi/.tools/review_apps_header_updates.py --show-body-diff` to review "changed body, unchanged header” cases.
- If the touched `apps/**` file is referenced by a mapped node chain, hotspot entry, or owner seam in the subsystem docs, update the manual map/index in the same tranche instead of leaving the atlas stale.

If you touch dependencies or configs, you update the proper manifest or lockfile and note the impact.

---

### ACT IV – ARCHITECTURE, LEGACY, MODELS, PYTHON

- The default core for attention is PyTorch SDPA.
- You list risks, side effects, globals.
- Codex prefix or suffix is used where it actually adds meaning.
- `Codex` is an intentional project naming convention. Do **not** strip it just because a symbol looks long.
- If naming or structure is bad, fix the fake namespace, owner shape, module boundary, or alias soup. Do **not** "clean it up" by deleting the `Codex` prefix, and do **not** invent pseudo-namespaces like `CodexProcessing.Txt2Img` unless a real namespace or module exists.
- You always code in Codex style:
  - Dataclasses, enums and similar.
  - Small modules with clear seams.
  - Explicit and fail-loud errors.
  - Readable names.

Testing policy: do not add or maintain automated tests unless explicitly requested by the repo owner.
Prefer fail-loud runtime contracts and manual validation workflows.

When we say "pipeline" in this repo, we mean the whole trip:
Frontend command → API request → task_id → SSE events → model load → sampling → postprocess/encode → finished artifact.

Drift is not a vibe. Drift is a bug.
Drift is when the _same mode_ (txt2img/img2img/txt2vid/img2vid/vid2vid) takes a different trip depending on engine.
Drift Also counts as drift when any of this changes per engine for the same mode:

- Contract drift: request schema/defaults, progress semantics, preview semantics, error semantics, or result fields.
- Stage drift: normalize → resolve engine/device → ensure assets/load → plan → execute → postprocess/encode → emit (skipped, duplicated, re-ordered, or hidden).
- Ownership drift: routers doing pipeline work, engines owning modes, or use-cases bypassed.

**Policy (Option A): one canonical use-case per mode.**

- `apps/backend/use_cases/{txt2img,img2img,txt2vid,img2vid,vid2vid}.py` owns the mode pipeline.
- Engines are adapters and hooks. They load models and expose primitives. They do **not** re-implement the mode.
- Routers stay thin: validate + dispatch + stream.
- The orchestrator stays the coordinator: resolve engine/device, cache/reload, run, and emit events.
- Shared, reusable stages live in `apps/backend/runtime/pipeline_stages/`. If it's shared, it goes there. If it's not shared, it stays in the canonical use-case.

**Ownership law: one concept, one owner path.**

- If a concept already lives under a typed nested owner, it stays there. You do **not** mirror it into flat shadow fields for convenience.
- Examples of forbidden shadow-owner drift:
  - `processing.hires.swap_model` plus `processing.hires_swap_model`
  - `processing.hires.refiner` plus `processing.hires_refiner`
  - `processing.hires.*` plus `processing.hr_*`
  - nested selector/config ownership plus sibling `*_name` / `*_path` aliases
- If a callsite wants a flatter shape, redesign the callsite. Do **not** duplicate the owner.

**Native names stay native.**

- If a field is named after a native concept, it carries only that concept.
- `refiner` is the native SDXL refiner seam. It is **not** a generic model-swap bucket.
- Generic model swap must live under explicitly generic naming such as `swap_model`, with its own typed owner.
- `extras.swap_model` is the top-level first-pass stage config:
  - it owns `enable` + `switch_at_step` semantics for mid-generation base-pass swapping;
  - it is **not** selector-only.
- `extras.hires.swap_model` is the selector-only second-pass owner:
  - it replaces the whole hires engine for the second pass;
  - it does **not** grow stage-pointer fields.
- `extras.refiner` / `extras.hires.refiner` remain SDXL-native refiner stages only.
- When a public/runtime seam is renamed to the native owner, the old name dies everywhere in the same tranche: router payloads, frontend state, component props/emits, run helpers, docs, and AGENTS.
- `hires.checkpoint`-style ghosts are forbidden once `hires.swap_model` exists.

**Derived-plan law: execution-only, selector-free.**

- A derived plan/helper struct may carry computed execution values such as target size, step count, denoise, or chosen upscaler.
- A derived plan/helper struct must **not** own selectors, model references, checkpoint names, swap-model config, refiner config, modules, or any other request-shaped contract data.
- If a plan/helper struct needs to carry those fields, that struct should not exist; compute from the canonical typed owner instead.
- `HiResPlan`-style shadow containers are forbidden as a destination for contract ownership.

**Unsupported seams fail loud.**

- If a mode/surface does not support a typed seam yet, you do **not** hide the payload and continue.
- Hide or clear the control in the UI when possible, and still fail loud at request build/runtime boundaries if stale state survives.
- Example: img2img must not silently drop `swap_model` / refiner state that only exists for txt2img hires.
- Public-state law: if a seam exists in frontend state, request payloads, or router parsing, it must also have a real execution owner.
  - No hidden/store-only `swap_model` surfaces.
  - No request/runtime surfaces that quietly do nothing.

If an engine needs special behavior, you add a hook that the canonical use-case calls.
If you can't express it as a hook, you stop and redesign until you can.
No engine-specific pipelines. No zoo.

Imports outside `/apps` are banned.
Only `apps.*` lives in active code.

If a feature has not been implemented, you raise:

```python
NotImplementedError("<feature> not yet implemented")
```

Model loading is a minefield you cross with a map.
You follow `.sangoi/research/models/model-loading-efficient-2025-10.md`.
- Supporting a family in `diffusers` format does **not** delegate contract truth to external `diffusers` helpers/imports.
- If this repo supports a `diffusers` surface, classification, component requirements, and family-specific constraints stay in repo-owned loader/detector/parser seams.
- Family-native external asset slots stay explicit and named. Do **not** collapse multi-slot families into generic selector bags when the contract depends on slot identity.

Keymap law: see **ABSOLUTE LAW — DO NOT TOUCH LAYER NAMES** at the top of this file.
The same no-rename/no-strip/no-punctuation-rewrite rule applies during model loading and engine/runtime keyspace interpretation.

You prefer SafeTensors.
You call `torch.load(..., weights_only=True, mmap=True)` when it applies.

Keep Python disciplined.
You do not add shebangs to source files.

When agent-side verification requires running the WebUI/backend on CPU, use the repository-local `uv` toolchain and explicit CPU env overrides.

- Prefer local `uv`: `./.uv/bin/uv` (never system/global `uv` for this workflow).
- Required env for CPU lane: `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu`.
- Example check command pattern:
  - `CODEX_ROOT="$PWD" PYTHONPATH="$PWD" CODEX_TORCH_MODE=cpu CODEX_TORCH_BACKEND=cpu ./.uv/bin/uv run --python .venv/bin/python --no-sync -m apps.backend.interfaces.api.run_api --help`
- WSL heavy-model safety rule: for LTX/WAN-class giant assets, default to header-only / metadata-only inspection in WSL. Do not materialize tensors, assemble full runtimes, initialize full pipelines, or run forward passes unless the user explicitly asks. Prefer GGUF metadata readers and SafeTensors header readers first.

---

### ACT V – FRONTEND, LAYOUT, AND CSS

If you want to change something in `apps/interface/src/styles`, you read the local `AGENTS.md` before you touch a single selector.
Ignore that, and your pull request does not pass.

Styles for `apps/interface/src/styles` are not a dumping ground.
Common rules belong where they will be reused.
Variants are named with intent.
Do not litter with vague utilities that hide confusion.
