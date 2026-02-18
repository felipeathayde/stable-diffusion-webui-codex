# apps/backend/patchers Overview
Date: 2025-10-30
Last Review: 2026-02-18
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
- `denoiser.py` — Generic `DenoiserPatcher` wrapper (ControlNet-free) for non-UNet denoisers; wraps `SamplerModel` and exposes the shared `ModelPatcher` surface.
- `vae_normalization_policy.py` — Typed VAE normalization policy resolver (`enum` + `dataclasses`) with explicit per-family shift contracts.
- Additional patch modules (e.g., adapters) live here as they are ported.

## Notes
- Patchers should operate on runtime objects provided by `runtime/` and `engines/` without duplicating loading logic.
- LoRA merges are transactional: loaders snapshot parameters, track deterministic patch order, surface tqdm progress, and raise on any mismatched tensor metadata.
- When introducing new patch behaviour, add explicit configuration flags/options and document them in `.sangoi/backend/`.
- Mutator methods must raise on invalid payloads (no fallbacks) and emit backend debug logs; `ModelPatcher` now centralises logging/telemetry for patch registration.
- `patchers/__init__.py` is a package marker (no facade exports); import patcher APIs from their defining modules.
- ControlNet patching lives under `apps/backend/patchers/controlnet/`, with architecture-specific modules located in `architectures/` (SD today; Flux/Chroma placeholders ready). Use `apply_controlnet_advanced` or `UnetPatcher.add_control_node` to register controls.
- Extension-facing compatibility is preserved via the new graph-backed patcher—no linked lists remain, and `UnetPatcher.add_patched_controlnet` builds `ControlNode` instances directly.
- VAE patcher now respects the AUTO precision ladder: decode/encode paths inspect for NaNs and escalate fp16↔bf16↔fp32 via `memory_management.manager.report_precision_failure`; user-forced dtypes skip the ladder and surface explicit errors.
- 2025-11-03: Host pinning for offloaded models honours `RuntimeMemoryConfig.swap.pin_shared_memory`; disable the flag to avoid Windows pagefile pressure.
- 2025-11-22: VAE patcher unwraps diffusers `DecoderOutput`/`AutoencoderKLOutput` before `.to(...)`, preventing `'DecoderOutput' object has no attribute 'to'` when SDXL uses the standard diffusers VAE.
- 2025-12-05: VAE patcher gains a `smart_fallback` path that, when enabled, catches CUDA OOM during decode/encode and performs a single full-image CPU fallback. When smart fallback is off, regular decode/encode OOM retries tiled VAE; forced `vae_always_tiled` OOM now fails loud.
- 2026-01-02: Removed token merging patches; prompt token-merging tags are stripped but have no effect.
- 2026-01-02: Added standardized file header docstrings to patcher modules (doc-only change; part of rollout).
- 2026-01-04: Added `DenoiserPatcher` for Flux/Z-Image/WAN runtimes; `UnetPatcher` remains UNet/ControlNet-specific.
- 2026-01-20: Global LoRA apply mode now supports `merge` (default; merges into weights once) vs `online` (patch during forward) via `CODEX_LORA_APPLY_MODE` / `--lora-apply-mode`.
- 2026-02-18: VAE decode/encode runtime now resolves a compute-preferred forward dtype and casts VAE residency to that effective forward dtype before execution/memory sizing to avoid mixed-dtype forward failures.
- 2026-02-18: Tiled VAE fallback was rewritten to a native context-padding + center-crop stitching flow (SUPIR-inspired, no fast/approximate path, no external tiled-scale dependency).
- 2026-02-18: Tiled encode/decode crop/index math now uses deterministic integer mapping (decode multiply, encode floor-div) to avoid border mismatch on odd image dimensions.
- 2026-02-18: Phase-1 logging cleanup removed residual stdout prints in patcher runtime paths (`controlnet/architectures/sd/t2i_adapter.py` state-dict mismatch notices and `vae.py` regular OOM retry notices), replacing them with structured backend logger warnings/debug entries.
- 2026-02-18: `lora_loader.py` now supports strict runtime toggles for merge/signature behavior: `CODEX_LORA_MERGE_MODE` (`fast=float32`, `precise=float64`) and `CODEX_LORA_REFRESH_SIGNATURE` (`structural` vs `content_sha256`) for deterministic refresh invalidation policy.
- 2026-02-18: `ModelPatcher.codex_unpatch_model` now emits canonical smart-offload INFO audit events (`pin_host_memory`) via `log_smart_offload_action(...)` when host pinning actually happens on CPU offload.
- 2026-02-08: `_NormalizingFirstStage` now supports optional per-channel latent stats (`latents_mean`/`latents_std`) in addition to scalar `scaling_factor`/`shift_factor`; 4D/5D rank, channel count, and non-finite/invalid stats are fail-loud.
- 2026-02-08: VAE normalization now resolves scale/shift via `vae_normalization_policy.py` with explicit family shift contracts: no-shift families reject explicit numeric shifts; shift-required families fail loud on missing/`None` shift.

### unet.py notes
- `control_nodes` é uma propriedade somente leitura (retorna cópia). Acesse como `unet.control_nodes`, não `unet.control_nodes()`.
- `activate_control()` recompõe o composite (`build_composite`) sempre que os nós mudam (ex.: após `add_control_node`).
- 2025-11-02: removido `@property` duplicado em `control_nodes` que podia levar a `TypeError: 'property' object is not callable` em tempo de acesso.
- 2026-02-09: LoRA apply now clears stale patch state on empty selections and fails loud on partial patcher contracts.
- 2026-02-09: LoRA refresh/merge must run outside `torch.inference_mode()`; internal inference-mode disabling was removed to keep version-counter fixes scoped to correct request entrypoints.
- 2026-02-11: `lora_apply.py` empty-selection reset now clears/refreshes denoiser + any available text-encoder patchers (engine-key agnostic) instead of hardcoding `text_encoders['clip']`; non-empty selection path remains fail-loud when CLIP mapping prerequisites are absent.
- 2026-02-18: `lora_apply.py` now fails loud when a selected LoRA produces zero compatible patches (`no compatible layers` / `zero parameters touched`) so SDXL key-layout mismatches do not silently no-op.
