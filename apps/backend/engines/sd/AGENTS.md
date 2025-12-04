# apps/backend/engines/sd Overview
<!-- tags: backend, engines, sdxl -->
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2025-11-28
Status: Active

## Purpose
- Stable Diffusion engine implementations (txt2img/img2img) leveraging the SD runtime components.

## Notes
- Keep SD engine logic aligned with runtime helpers under `runtime/sd/`.
- Shared assembly helpers live in `spec.py` — define engine specs (dataclasses) there and assemble components via `assemble_engine_runtime` inside each engine’s `_build_components` implementation.
- Each engine must expose `EngineCapabilities` (txt2img/img2img) and rely on `_require_runtime()` style guards when touching assembled runtime state.
- Preference order: extend specs first, then consume them in `_build_components`; never reintroduce legacy component dictionaries or silent clip-skip fallbacks.
- 2025-11-14: SDXL conditioning now wraps prompts in metadata-aware `_SDXLPrompt` objects so ADM/time embeddings honor the requested `width`/`height`/targets and blank negative prompts collapse to zeros (parity with Forge-A1111/Comfy pipelines under `.refs/{Forge-A1111,ComfyUI}`).
- 2025-11-22: SDXL engine must refuse to run when a WAN VAE is loaded (signals misuse of SDXL checkpoints); loader now supplies diffusers `AutoencoderKL` for SD/SDXL families.
- 2025-11-23: SDXL `_build_components` now raises a `RuntimeError` when `AutoencoderKLWan` is wired into the runtime, instead of logging a warning and proceeding with corrupted decodes.
- 2025-11-28: SDXL `get_learned_conditioning` now validates cross-attn/ADM tensors (shapes, NaN/Inf) against UNet config and fails fast on mismatches to prevent “golesma” outputs.

### Event Emission
- Engines must emit `ProgressEvent` and a final `ResultEvent` for UI/services to render progress and images.
- SDXL `txt2img` decodes latents to RGB and emits a `ResultEvent` with `images` and a JSON `info` string.
- Progress streaming can be added by polling `apps.backend.core.state.state` while sampling or by converting the sampler into an event-yielding generator; keep the approach explicit per engine.

### Assembly Invariants (spec.py)
- Ao montar o runtime (`assemble_engine_runtime`):
  - UNet deve expor `diffusion_model` com `codex_config` (`UNetConfig`).
  - `codex_config.context_dim` não pode ser `None`.
  - Para `sdxl`, `sdxl_refiner`, `sd35`: `num_classes` do UNet não pode ser `None`; se for `'sequential'`, `adm_in_channels` deve ser definido (>0).
- Qualquer violação levanta `SDEngineConfigurationError` com causa explícita (sem fallbacks).
