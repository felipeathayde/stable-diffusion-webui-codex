# Codex Local Scratchpad (stable-diffusion-webui-codex)

Durable workspace notes that should change future behavior.

- Zero tolerance for shortcut/remap hacks in model loading paths.
- Required contract: engine must know canonical keys; if unknown keys appear, fail immediately and loudly.
- Do not add rename/remap glue that hides bad checkpoints or broken contracts.
- User severity signal: if this kind of "porquice" repeats, user will be **absolutely furious** ("absolutely putaço").
- STOP-SHIP RULE (GLOBAL, CONTUNDENT): never ship "hello world" technical solutions in any part of this codebase. No simplistic hacks, no ad-hoc glue, no compatibility band-aids, no toy-level shortcuts in advanced systems. Every implementation must match the technical level of the platform, be architecture-consistent, and fail loud on broken invariants. Any low-rigor shortcut anywhere is a critical quality breach and release blocker.
- HARD LAW (PERFORMANCE): if any load path is slower than WAN22 14B baseline load behavior, it is WRONG and stop-ship until fixed.
