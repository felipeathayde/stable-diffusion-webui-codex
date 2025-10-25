Repo Cleanup Checklists by Priority (P0→P3)
Date: 2025-10-25

Priority legend
- P0: must do now (breakage risk, policy compliance, or immediate clarity)
- P1: next sprint (1–2 semanas)
- P2: soonish (2–4 semanas)
- P3: backlog / quando houver tempo

Format
- Path — Category (Active/Compat/Vendor/Dev/Data/Reference/Deprecated) → Priority
- Checklist (⊟ compact, expand as you work) with acceptance criteria

—

apps/ — Active → P0
- [ ] Freeze “source of truth” decision
  - AC: README at `apps/server/backend/` links structure/rules docs
- [ ] Run_API startup sanity quick pass (imports order, hooks)
  - AC: no circular imports; exception dumps working (CODEX_STRICT_EXIT)

codex/ — Active (docs) → P0
- [ ] Ensure all new docs linked from AGENTS.md
  - AC: links checked manually; no broken references
- [ ] Add short “How to navigate docs” section in codex/README (optional)

.sangoi/ — Active (logs) → P1
- [ ] Confirm CHANGELOG entry policy
  - AC: last 3 changes summarized; index kept lean

modules/ — Compat → P0
- [ ] Audit backend imports (only where necessary)
  - AC: list of allowed imports called out in `codex/architecture/architecture-rules.md`
- [ ] Mark as compat in docs (no new features)
  - AC: note exists in repo-inventory + rules

modules_forge/ — Compat → P0
- [ ] Same audit and “compat only” label
  - AC: note in repo-inventory; no new backend coupling added
 - [ ] Confirm backend only uses `apps.server.backend.codex.*` (zero `modules_forge.*` in apps/)
   - AC: grep em `apps/server/backend/**` retorna 0 para `modules_forge.*`

k_diffusion/ — Vendor/Compat → P1
- [ ] Pin version hash in docs (vendor note)
  - AC: commit hash noted in repo-inventory or a VENDOR file

extensions-builtin/ — Compat → P1
- [ ] Inventory which extensions are truly used on startup
  - AC: list in repo-inventory; unused marked “optional”

javascript/ — Compat (legacy UI) → P1
- [ ] List scripts still required by legacy flows
  - AC: small table in repo-inventory; anything else marked “retirable”

javascript-src/ — Dev → P3
- [ ] Keep TS shims minimal; no build hard deps
  - AC: `npm run build:ts` optional; build emits nothing by default

html/ — Compat → P3
- [ ] Confirm only static assets referenced remain
  - AC: grep shows only known images referenced

packages_3rdparty/gguf/ — Vendor → P1
- [ ] Backend sem dependências de A1111 (`modules.*`)
  - AC: grep zero em `apps/server/backend/**` para `\bmodules(\.|_forge)`, `forge_`

packages_3rdparty/* (collections) — Vendor (optional) → P3
- [ ] Tag as examples only
  - AC: README note; safe to prune later

configs/presets/ — Config → P1
- [ ] Cross-check with backend‑served presets/blocks
  - AC: doc table shows which preset is mirrored; plan to consolidate

repositories/ — Data → P2
- [ ] Verify any scripts depending on clones
  - AC: note which are required; others removable later

scripts/ — Dev → P2
- [ ] Classify each script (legacy utility vs dev tool vs WAN inspection)
  - AC: miniature index under `scripts/README.md` (optional)
- [ ] Move WAN inspection scripts to tools/ (optional)
  - AC: paths updated in docs only (no runtime use)

models/ — Data → P0
- [ ] Ensure `.gitignore` keeps it untracked
  - AC: `git status` clean after adding dummy files locally

cache/ — Data → P0
- [ ] `.gitignore` confirmed; no accidental files

artifacts/ — Data → P0
- [ ] `.gitignore` confirmed; no accidental files

legacy/ — Reference → P2
- [ ] “Read‑only” policy reiterated; no imports from backend
  - AC: no grep hits from `apps/server/**`

.refs/ — Reference (ComfyUI) → P2
- [ ] “Reference only” banner; no imports
  - AC: no grep hits from active code

backend/ — Deprecated → P0 (done)
- [x] Blocking `__init__` raises with guidance
  - AC: attempting `import backend` fails with ModuleNotFoundError

_ext_hg/ — Deprecated candidate → P1
- [ ] Move to `DEPRECATED/` (or remove) after confirming no usage
  - AC: grep returns no imports; move committed when convenient

DEPRECATED/ — Reference/Archive → P1
- [ ] Ensure all code here is unimported by active tree
  - AC: grep shows zero imports

—

How to use
- Treat this file as a live checklist. When you complete a line, add a short note (who/when) and optional pointer to a commit or doc.
