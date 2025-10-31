# apps/backend/patchers Overview
Date: 2025-10-30
Owner: Backend Runtime Maintainers
Last Review: 2025-10-31
Status: Active

## Purpose
- Hosts runtime patching utilities (LoRA injection, token merging, adapter application) that modify networks or inference behavior after models are loaded.

## Key Files
- `base.py` — Core `ModelPatcher` with typed registries (LoRA/object patches) and lifecycle hooks.
- `lora.py` — Implements `CodexLoraLoader` and variant-aware merge helpers backed by typed payloads, progress reporting, and strict validation.
- `lora_apply.py` — Applies native LoRA selections to loaded networks.
- `token_merging.py` — Implements token merging strategies consumed by engines/use cases.
- `unet.py` — Codex-native UNet patcher built on typed helpers (`SamplingReservation`, `ControlNetChain`) for deterministic sampling reservations, ControlNet chaining, and patch registration.
- `clipvision.py` — Adapter around `apps/backend/runtime/vision/clip/` providing legacy-facing APIs backed by the Codex encoder.
- Additional patch modules (e.g., adapters) live here as they are ported.

## Notes
- Patchers should operate on runtime objects provided by `runtime/` and `engines/` without duplicating loading logic.
- LoRA merges are transactional: loaders snapshot parameters, track deterministic patch order, surface tqdm progress, and raise on any mismatched tensor metadata.
- When introducing new patch behaviour, add explicit configuration flags/options and document them in `.sangoi/backend/`.
- Mutator methods must raise on invalid payloads (no fallbacks) and emit backend debug logs; `ModelPatcher` now centralises logging/telemetry for patch registration.
- ControlNet patching lives under `apps/backend/patchers/controlnet/`, with architecture-specific modules located in `architectures/` (SD today; Flux/Chroma placeholders ready). Use `apply_controlnet_advanced` or `UnetPatcher.add_control_node` to register controls.
- Clip vision patcher reuses the runtime encoder; avoid reintroducing preprocessing or state-dict manipulation in the patcher—extend the runtime if new variants arise.
- Extension-facing compatibility is preserved via the new graph-backed patcher—no linked lists remain, and `UnetPatcher.add_patched_controlnet` builds `ControlNode` instances directly.
