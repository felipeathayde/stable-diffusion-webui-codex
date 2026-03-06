# Codex Local Scratchpad (stable-diffusion-webui-codex)

Durable workspace notes that should change future behavior.

- Zero tolerance for shortcut/remap hacks in model loading paths.
- KEYMAP LAW: keymaps are maps of keyspaces/naming conventions, not runtime remappers.
- Required contract: engines must interpret model keys exactly as stored in the checkpoint; this repo never renames incoming model keys at runtime.
- Never create remapped state dicts, alias maps, translation glue, or compatibility shims for unsupported layouts.
- Fused/unfused conventions, including Comfy weirdness, belong to interpretation logic over the stored keys, not runtime key renaming.
- Unsupported layouts are invalid inputs: fail immediately, explicitly, and loudly.
- Required contract: engine must know canonical keys; if unknown keys appear, fail immediately and loudly.
- Do not add rename/remap glue that hides bad checkpoints or broken contracts.
- Redundant user invariant: this WebUI never does runtime remap, never will do runtime remap, and any code that reintroduces that idea is architecture drift and stop-ship.
- User severity signal: if this kind of "porquice" repeats, user will be **absolutely furious** ("absolutely putaço").
- STOP-SHIP RULE (GLOBAL, CONTUNDENT): never ship "hello world" technical solutions in any part of this codebase. No simplistic hacks, no ad-hoc glue, no compatibility band-aids, no toy-level shortcuts in advanced systems. Every implementation must match the technical level of the platform, be architecture-consistent, and fail loud on broken invariants. Any low-rigor shortcut anywhere is a critical quality breach and release blocker.
- HARD LAW (PERFORMANCE): if any load path is slower than WAN22 14B baseline load behavior, it is WRONG and stop-ship until fixed.
- Toolchain discipline: for workspace package/runtime operations, use local `./.uv/bin/uv` with `./.venv/bin/python`; do not use system `uv` or global `pip`.
- User path preference: when user provides Windows paths/links, access them through WSL mount paths using `/mnt/{drive-letter}/...` (e.g., `C:\\...` -> `/mnt/c/...`).
- User preference (career docs): produce lean, role-targeted resume/About content in English with direct relevance to target role; include IQ 130 mention only when explicitly requested.
- User style preference (career docs): prefer aggressive, high-ownership tone; avoid conservative/corporate phrasing unless explicitly requested.
- User preference (career format): create a denser resume variant closer to the provided dark multi-section layout and include concrete webui technical specifics for GPT-based screening.
