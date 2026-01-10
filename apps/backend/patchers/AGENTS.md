# apps/backend/patchers Overview
Date: 2025-10-30
Owner: Backend Runtime Maintainers
Last Review: 2026-01-05
Status: Active

## Purpose
- Hosts runtime patching utilities (LoRA injection, adapter application) that modify networks or inference behavior after models are loaded.

## Key Files
- `base.py` — Core `ModelPatcher` with typed registries (LoRA/object patches) and lifecycle hooks.
- `lora.py` — Public LoRA facade (re-exports loader/merge/state-dict helpers).
- `lora_loader.py` — `CodexLoraLoader` transactional applier (backups, GGUF re-quantization, tqdm progress).
- `lora_merge.py` — Variant-aware weight merge helpers (diff/set/lora/loha/lokr/glora) with strict validation.
- `lora_state_dict.py` — LoRA tensor parsing + target-key mapping wrappers (backed by `runtime.adapters.lora`).
- `lora_apply.py` — Applies native LoRA selections to loaded networks.
- `unet.py` — Codex-native UNet patcher built on typed helpers (`SamplingReservation`, `ControlNetChain`) for deterministic sampling reservations, ControlNet chaining, and patch registration.
- `denoiser.py` — Generic `DenoiserPatcher` wrapper (ControlNet-free) for non-UNet denoisers; wraps `KModel` and exposes the shared `ModelPatcher` surface.
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
- VAE patcher now respects the AUTO precision ladder: decode/encode paths inspect for NaNs and escalate fp16↔bf16↔fp32 via `memory_management.manager.report_precision_failure`; user-forced dtypes skip the ladder and surface explicit errors.
- 2025-11-03: Host pinning for offloaded models honours `RuntimeMemoryConfig.swap.pin_shared_memory`; disable the flag to avoid Windows pagefile pressure.
- 2025-11-22: VAE patcher unwraps diffusers `DecoderOutput`/`AutoencoderKLOutput` before `.to(...)`, preventing `'DecoderOutput' object has no attribute 'to'` when SDXL uses the standard diffusers VAE.
- 2025-12-05: VAE patcher gains a `smart_fallback` path that, when enabled, catches CUDA OOM during decode and performs a single full-image decode on CPU instead of repeatedly retrying GPU paths (regular + tiled). Encode now mirrors this behaviour: OOM during encode triggers a single full-image CPU encode when Smart Fallback is on, otherwise it falls back to tiled encode.
- 2026-01-02: Removed token merging patches; prompt token-merging tags are stripped but have no effect.
- 2026-01-02: Added standardized file header docstrings to patcher modules (doc-only change; part of rollout).
- 2026-01-04: Added `DenoiserPatcher` for Flux/Z-Image/WAN runtimes; `UnetPatcher` remains UNet/ControlNet-specific.

### unet.py notes
- `control_nodes` é uma propriedade somente leitura (retorna cópia). Acesse como `unet.control_nodes`, não `unet.control_nodes()`.
- `activate_control()` recompõe o composite (`build_composite`) sempre que os nós mudam (ex.: após `add_control_node`).
- 2025-11-02: removido `@property` duplicado em `control_nodes` que podia levar a `TypeError: 'property' object is not callable` em tempo de acesso.
