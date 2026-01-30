# apps/backend/engines/sd Overview
<!-- tags: backend, engines, sdxl -->
Date: 2025-10-28
Owner: Engine Maintainers
Last Review: 2026-01-30
Status: Active

## Purpose
- Stable Diffusion engine implementations (txt2img/img2img) leveraging the SD runtime components.

## Notes
- Keep SD engine logic aligned with runtime helpers under `runtime/families/sd/`.
- Shared assembly helpers live in `spec.py`; SD-family engines assemble runtimes via `CodexSDFamilyFactory` (`apps/backend/engines/sd/factory.py`) which wraps `assemble_engine_runtime`.
- SD15 and SD20 share implementation via `CodexSDClassicEngineBase` (`apps/backend/engines/sd/classic_base.py`) to avoid duplicated TE/VAE helpers.
- `CodexObjects` stores the sampling core under `denoiser`; SD-family engines still assemble a true UNet denoiser as `SDEngineRuntime.unet`.
- Each engine must expose `EngineCapabilities` (txt2img/img2img) and rely on `_require_runtime()` style guards when touching assembled runtime state.
- Preference order: extend specs first, then consume them in `_build_components`; never reintroduce legacy component dictionaries or silent clip-skip fallbacks.
- 2025-11-14: SDXL conditioning now wraps prompts in metadata-aware `_SDXLPrompt` objects so ADM/time embeddings honor the requested `width`/`height`/targets and blank negative prompts collapse to zeros (parity with reference pipelines).
- 2025-11-22: SDXL engine must refuse to run when a WAN VAE is loaded (signals misuse of SDXL checkpoints); loader now supplies diffusers `AutoencoderKL` for SD/SDXL families.
- 2025-11-23: SDXL `_build_components` now raises a `RuntimeError` when `AutoencoderKLWan` is wired into the runtime, instead of logging a warning and proceeding with corrupted decodes.
- 2025-11-28: SDXL `get_learned_conditioning` now validates cross-attn/ADM tensors (shapes, NaN/Inf) against UNet config and fails fast on mismatches to prevent “golesma” outputs.
- 2025-12-05: SDXL base/refiner engines honor per-job Smart Cache (`smart_cache` on `Txt2ImgRequest`/`CodexProcessingTxt2Img`) via `_SDXLPrompt.smart_cache`, with fallback to the global option when unset. SDXL `txt2img` also preenche `info["timings_ms"]` com tempos aproximados de sampling/decode para apoiar profiling backend sem impactar o gentime da UI.
- 2026-01-02: Fixed SDXL refiner embed cache correctness (cache hits no longer crash) and aligned refiner time-id embedding with `_prompt_meta` + `_validate_conditioning_payload` (fail-fast on malformed conditioning).
- 2026-01-02: Added standardized file header docstrings to SD engine modules (doc-only change; part of rollout).
- 2026-01-06: SDXL generation info no longer defaults sampler/scheduler to `"Automatic"`; missing values serialize as null to reflect strict canonical inputs.
- 2026-01-25: `clip_skip=0` is now treated as an explicit “use default” sentinel across SD-family engines, resetting clip skip to the per-branch spec defaults to prevent state leaking across jobs.
- 2026-01-30: SD-family txt2img now consumes `GenerationResult` from the canonical staged runner (removed `_already_decoded` decode sentinels).

### Event Emission
- Engines must emit `ProgressEvent` and a final `ResultEvent` for UI/services to render progress and images.
- SDXL `txt2img` decodes latents to RGB and emits a `ResultEvent` with `images` and a JSON `info` string.
- Progress streaming can be added by polling `apps.backend.core.state.state` while sampling or by converting the sampler into an event-yielding generator; keep the approach explicit per engine.

### Assembly Invariants (spec.py)
- Ao montar o runtime (`assemble_engine_runtime`, via `CodexSDFamilyFactory`):
  - UNet deve expor `diffusion_model` com `codex_config` (`UNetConfig`).
  - `codex_config.context_dim` não pode ser `None`.
  - Para `sdxl`, `sdxl_refiner`, `sd35`: `num_classes` do UNet não pode ser `None`; se for `'sequential'`, `adm_in_channels` deve ser definido (>0).
- Qualquer violação levanta `SDEngineConfigurationError` com causa explícita (sem fallbacks).
