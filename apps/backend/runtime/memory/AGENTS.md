# apps/backend/runtime/memory Overview
Date: 2025-10-28
Last Review: 2026-02-09
Status: Active

## Purpose
- Memory management policies (VRAM/CPU balance, offload strategies) used by engines during execution.

## Notes
- `manager.py` hosts `CodexMemoryManager`; `memory_management.py` exposes the active singleton as `memory_management.manager` (call sites should use manager methods/properties directly).
- `config.py` defines typed configuration (devices, swap policies, attention backends). Update both docs and `apps/backend/infra/config/args.py` when adding new options.
- Keep policy changes centralized here to ensure consistent behaviour across tasks (engines, patchers, workflows).
- 2026-01-24: Attention backend can be switched at runtime via `memory_management.set_attention_backend(...)` (used by `POST /api/options` key `codex_attention_backend`); the attention dispatcher is runtime-config-driven (no import-time binding).
- AUTO precision is coordinated via a native ladder: VAE supports bf16→fp16→fp32, diffusion/text encoders bf16→fp16. Fallbacks are handled by `CodexMemoryManager.report_precision_failure` and will refuse to advance when users forced a dtype.
- 2025-11-04: `CodexMemoryManager` unwraps wrappers (e.g., VAE) to their `ModelPatcher`/`nn.Module` targets before allocation/load so engines can call `memory_management.manager.load_model(wrapper)` without tripping AttributeErrors.
- 2025-12-05: `smart_offload` and `smart_fallback` flags are driven via Codex options (`codex_smart_offload`, `codex_smart_fallback`); the VAE patcher now uses `smart_fallback` to reroute decode and encode to CPU após um CUDA OOM instead of repeatedly retrying on GPU (encode falls back to tiled mode when Smart Fallback is disabled).
- 2025-12-05: `CodexMemoryManager` expõe `memory_snapshot()` e `hardware_probe` para diagnósticos — use `apps.backend.runtime.memory.memory_management.memory_snapshot()` quando precisar inspecionar VRAM/CPU sem tocar em campos internos ou forçar `empty_cache`; o endpoint `/api/memory` é o consumidor público recomendado.
- 2025-12-30: GGUF-packed weights detection for state dicts lives in `apps.backend.runtime.model_parser.quantization.detect_state_dict_dtype` (returns `"gguf"` when `CodexParameter` markers are present).
- 2026-01-02: Added standardized file header docstrings to `__init__.py`, `config.py`, `exceptions.py`, `smart_offload.py`, and `stream.py` (doc-only change; part of rollout).
- 2026-01-26: Added smart-offload stage invariants (`smart_offload_invariants.py`) used by use-cases and the sampling driver to auto-unload forbidden accelerator residents (TE must be off before sampling; denoiser/VAE cleared before conditioning).
- 2026-02-09: Added post-decode smart-offload invariant (`enforce_smart_offload_post_decode_residency`) that enforces VAE-off after decode and applies cache-aware denoiser residency (`warm on cache hit`, `unload on miss`) for image wrappers.
