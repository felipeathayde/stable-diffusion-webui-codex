# Prompt to self — resume UI fixes session (2026-02-08)

```text
You are working in the repo `stable-diffusion-webui-codex`.
Continue from:
- CWD: /home/lucas/work/stable-diffusion-webui-codex
- Branch: master
- Last commit: 8191d86b79192dda4032b1a334ca51bdee64c0f6
- Date (UTC-3): 2026-02-08T02:12:18-03:00

Objective (1 sentence)
- Finalize and hand off the UI fix set (a→m) for batch/prompt/quicksettings/capabilities with fail-loud contracts and clean validation.

State
- Done:
  - Added backend endpoint `POST /api/models/prompt-token-count` with offline tokenizer mapping and strict validation in `apps/backend/interfaces/api/routers/models.py`.
  - Added frontend prompt-token API types/client wrappers in `apps/interface/src/api/types.ts` and `apps/interface/src/api/client.ts`.
  - Added strict `families` capability parsing/getters in `apps/interface/src/stores/engine_capabilities.ts` + tests in `apps/interface/src/stores/engine_capabilities.test.ts`.
  - Switched generation negative-prompt gating to live family capabilities in `apps/interface/src/composables/useGeneration.ts`.
  - Implemented debounced params persistence (coalesced PATCH) in `apps/interface/src/stores/model_tabs.ts`.
  - Implemented UI fixes in prompt/quicksettings/image-tab/refiner/hires/result viewer/dependency panel/metadata modal/run card.
  - Added shared model filtering util + tests in `apps/interface/src/utils/model_family_filters.ts` and `apps/interface/src/utils/model_family_filters.test.ts`.
  - Validation already green once: `npm --prefix apps/interface run typecheck`, `npm --prefix apps/interface test`, backend `py_compile` for `models.py`.
- In progress:
  - Final diff audit for scope correctness in shared dirty tree.
  - Final docs pass (`.sangoi/task-logs`, `.sangoi/CHANGELOG.md`, plan status sync).
- Blocked / risks:
  - Working tree is shared with other agents (ER-SDE + Anima vectorization); avoid touching unrelated backend files.
  - `COMMON_MISTAKES.md` is intentionally user-owned dirty change; do not modify/revert.
  - Need to keep edits scoped to this task’s files only.

Decisions / constraints (locked)
- User-locked choices:
  - b) token counter = real backend token count.
  - e) replace all native `window.prompt` path inputs.
  - h) swap model = complete flow without VAE field.
- Negative prompt must be capability-driven visibility (no manual toggle in UI).
- No compat shims / silent fallbacks; fail loud when required contracts are missing.
- Shared tree discipline: do not revert or alter unrelated changes from parallel agents.

Follow-up (ordered)
1. Audit scoped diffs for only the intended a→m behavior and remove any accidental spillover.
2. Verify file-header symbol blocks are accurate for all touched `apps/**` files.
3. Re-run focused and broad validations (frontend + backend compile check).
4. Update `.sangoi/plans/2026-02-08-interface-ui-fixes-batch-token-capabilities.md` status/checklist to reflect actual progress.
5. Write task log for this session in `.sangoi/task-logs/` (summary, files touched, risks/TODOs).
6. Update `.sangoi/CHANGELOG.md` with user-visible behavior changes.

Next immediate step (do this first)
- Re-establish scoped truth from diffs, then run validations once more before docs updates.
Commands:
`git diff -- apps/backend/interfaces/api/routers/models.py apps/interface/src/api/client.ts apps/interface/src/api/types.ts apps/interface/src/stores/model_tabs.ts apps/interface/src/components/QuickSettingsBar.vue apps/interface/src/views/ImageModelTab.vue apps/interface/src/components/prompt/PromptBox.vue`
`npm --prefix apps/interface run typecheck && npm --prefix apps/interface test`
`CODEX_ROOT="$(git rev-parse --show-toplevel)" && PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m py_compile apps/backend/interfaces/api/routers/models.py`

Files
- Changed files (current uncommitted task scope):
  - apps/backend/interfaces/api/routers/models.py
  - apps/interface/src/api/client.ts
  - apps/interface/src/api/types.ts
  - apps/interface/src/api/payloads.ts
  - apps/interface/src/components/prompt/PromptBox.vue
  - apps/interface/src/components/prompt/PromptCard.vue
  - apps/interface/src/components/prompt/PromptFields.vue
  - apps/interface/src/composables/usePromptCard.ts
  - apps/interface/src/stores/engine_capabilities.ts
  - apps/interface/src/stores/engine_capabilities.test.ts
  - apps/interface/src/stores/model_tabs.ts
  - apps/interface/src/components/QuickSettingsBar.vue
  - apps/interface/src/views/ImageModelTab.vue
  - apps/interface/src/components/RefinerSettingsCard.vue
  - apps/interface/src/components/HiresSettingsCard.vue
  - apps/interface/src/components/ResultViewer.vue
  - apps/interface/src/components/DependencyCheckPanel.vue
  - apps/interface/src/components/modals/AssetMetadataModal.vue
  - apps/interface/src/components/results/RunCard.vue
  - apps/interface/src/composables/useGeneration.ts
  - apps/interface/src/views/WANTab.vue
  - apps/interface/src/styles/components/quicksettings.css
  - apps/interface/src/utils/model_family_filters.ts
  - apps/interface/src/utils/model_family_filters.test.ts
  - apps/interface/src/views/dependency_check_location.test.ts
- Focus files to open first:
  - apps/interface/src/components/QuickSettingsBar.vue — path modal replacement, use-init-image quick toggle, shared model filtering.
  - apps/interface/src/views/ImageModelTab.vue — capability-driven negative/clip, swap model wiring, prompt bindings.
  - apps/interface/src/stores/model_tabs.ts — debounced/coalesced persistence and rollback behavior.
  - apps/interface/src/components/prompt/PromptBox.vue — backend token counter and debounce behavior.
  - apps/backend/interfaces/api/routers/models.py — prompt-token endpoint contract.
  - apps/interface/src/stores/engine_capabilities.ts — strict family capability parsing/getters.

Validation (what “green” looks like)
- `npm --prefix apps/interface run typecheck`  # expected: no TS errors
- `npm --prefix apps/interface test`  # expected: all tests pass
- `CODEX_ROOT="$(git rev-parse --show-toplevel)" && PYTHONPATH="$CODEX_ROOT" "$CODEX_ROOT/.venv/bin/python" -m py_compile apps/backend/interfaces/api/routers/models.py`  # expected: no output/errors
- `rg -n "window\.prompt\(" apps/interface/src/components/QuickSettingsBar.vue`  # expected: no matches
- `rg -n "allowNegativeToggle|defaultShowNegative|toggleInitImage" apps/interface/src/views/ImageModelTab.vue apps/interface/src/components/prompt/PromptCard.vue`  # expected: no matches

References (read before coding)
- .sangoi/plans/2026-02-08-interface-ui-fixes-batch-token-capabilities.md
- .sangoi/reference/ui/model-tabs-and-quicksettings.md
- .sangoi/task-logs/2026-01-13-metadata-provenance-and-tools-surfaces.md
- .sangoi/task-logs/2026-02-07-home-single-dependency-check-all-engines.md
- .sangoi/policies/file-header-block.md
- .sangoi/howto/TESTS_GUIDE.md
```
