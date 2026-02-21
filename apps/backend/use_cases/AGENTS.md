# apps/backend/use_cases Overview
Date: 2025-10-30
Last Review: 2026-02-21
Status: Active

## Purpose
- Defines high-level orchestration flows for each supported task (txt2img, img2img, txt2vid, img2vid). Each module prepares inputs, invokes the appropriate engine, and handles post-processing.

## Key Files
- `txt2img.py`, `img2img.py`, `txt2vid.py`, `img2vid.py`, `vid2vid.py` — Task-specific pipelines that bind request parameters, engines, runtimes, and services.
- `upscale.py` — Standalone upscaling pipeline (Spandrel/builtin upscalers) used by `/api/upscale`.
- `txt2img_pipeline/` — Stage-based runner used by `txt2img.py` to prepare prompts, execute base sampling, and handle HiRes passes without monolithic functions.
- `__init__.py` — Exposes helpers to orchestrator modules in `apps/backend/core`.
- Shared orchestration logic now lives in `apps/backend/runtime/pipeline_stages/`; use cases should prefer these helpers over bespoke copies of sampler/prompt setup.

## Notes
- Introduza novos use cases sempre que uma combinação de tarefa + modo precisar de orquestração própria; mantenha a lógica focalizada em preparar entradas, chamar engines e relatar progresso, delegando detalhes de modelo para `engines/` ou `runtime/`.
- Quando adicionar um novo use case, espelhe o padrão existente e registre com o orquestrador e os contratos de API.
- 2026-01-22: `txt2img.py` now includes a canonical event wrapper (`run_txt2img`) used by engines to keep mode orchestration in the use-case layer (Option A).
- 2025-12-16: `vid2vid.py` implements WAN22 video-to-video orchestration (decode input video via ffmpeg, flow-guided chunking, optional VFI, export + metadata), plus a `vid2vid_method="wan_animate"` path that runs Diffusers `WanAnimatePipeline` from preprocessed pose/face videos + reference image.
- 2026-01-02: Added standardized file header docstrings to use case modules (doc-only change; part of rollout).
- 2026-01-03: Added standardized file header docstrings to remaining use case modules (`__init__.py`, `img2vid.py`, `txt2img.py`, `txt2vid.py`) (doc-only change; part of rollout).
- 2026-01-26: Smart offload pre-conditioning cleanup now ensures denoiser/VAE are not left resident when conditioning starts (txt2img runner + img2img).
- 2026-01-29: `img2img.py` now supports Codex-native masked img2img (“inpaint”) for SD-family engines (Forge-style full-res crop/paste-back + selectable enforcement: post-blend vs per-step clamp). Flux Kontext explicitly fails loud when a mask is provided (not yet supported).
- 2026-01-30: `txt2img` now consumes `GenerationResult` from the staged runner; removed the global `_already_decoded` decode sentinel.
- 2026-01-31: `img2img.py` now includes a canonical event wrapper (`run_img2img`) and shared image streaming helpers live in `apps/backend/use_cases/_image_streaming.py` (Option A: mode streaming stays in use-cases; engines delegate).
- 2026-02-03: Hires pass validation/errors now reference `hires.*` naming (contract cutover).
- 2026-01-27: WAN video results now support video-first payloads: txt2vid/img2vid export via ffmpeg and return `result.video { rel_path, mime }`; a per-tab `Return frames` knob controls whether frames are included (fallback returns frames when `saveOutput` is off or export fails). Vid2vid now honors the same knob for preview frames.
- 2026-02-12: txt2vid now runs the same shared interpolation stage used by img2vid/vid2vid (`runtime/pipeline_stages/video.py`) and records `video_interpolation` metadata consistently across all WAN video modes.
- 2026-02-13: WAN video use-cases now resolve output FPS from request FPS plus applied interpolation factor (`times`) to keep duration stable when VFI is enabled.
- 2026-02-13: txt2vid/img2vid export path now fails loud when `video_options.save_output=true` and export does not produce a saved artifact (no silent fallback on failed required runtime deps).
- 2026-02-13: vid2vid default FPS source is request FPS (`use_source_fps=false` by default unless explicitly enabled in `extras.vid2vid`), and export failures now propagate fail-loud when saving output is requested.
- 2026-02-08: `txt2img_pipeline/refiner.py` now interprets refiner swap as a pointer (`switch_at_step`) and starts the refiner pass at that step while preserving total diffusion steps.
- 2026-02-09: `_image_streaming._decode_generation_output(...)` now enforces post-decode smart-offload residency centrally for image wrappers; it consumes `GenerationResult.metadata["conditioning_cache_hit"]` (strict bool contract, fail-loud on malformed metadata), always executes cleanup in a `finally` path, and only prewarms denoiser when decode succeeds.
- 2026-02-18: `run_txt2img` and `run_img2img` execute sampling + decode + post-cleanup inside the same inner worker-thread envelope; `_image_streaming._run_inference_worker(...)` now owns smart-runtime override propagation into that worker (removing wrapper-local override blocks) and cleanup hooks run in `finally` so decode failures cannot skip unload paths.
- 2026-02-18: `_image_streaming._decode_generation_output(...)` now performs pre-decode GC/cache flush and requests `decode_latent_batch(..., target_device=cpu)` for final image decode, matching Forge-style final decode flow to reduce VRAM pressure in hires second-pass output.
- 2026-02-09: Image conditioning helpers (`txt2img`/`img2img`) now stage all registered text encoders (`codex_objects.text_encoders`, e.g. `clip`, `qwen3`) during conditioning and offload them after embeddings are produced. Smart Cache provides the embed-reuse signal (`conditioning_cache_hit`), while Smart Offload invariants execute the actual load/unload transitions (including cache-hit warm path vs cache-miss unload path).
- 2026-02-11: `img2img.py` now enforces init-image VAE encode before TE conditioning across classic unmasked, classic masked, and Flux Kontext variants (smart-offload order contract for img2img/img2vid-style flows).
- 2026-02-15: `txt2img_pipeline/runner.py` no longer embeds raw negative prompt text in zero-uncond fail-loud errors; message now keeps only technical context (`count`) to avoid prompt leakage through downstream task/log surfaces.
- 2026-02-16: img2img now emits truthful `GenerationResult.metadata["conditioning_cache_hit"]` across classic + Flux Kontext paths (derived from per-call Smart Cache bucket deltas when conditioning is computed, or `True` when conditioning is fully pre-supplied), so shared decode cleanup applies the same warm-vs-unload policy parity already used by txt2img.
- 2026-02-17: `img2vid.py` GGUF path now supports optional chunked generation with overlap stitching, anchor blending (`img2vid_anchor_alpha`), and deterministic/per-chunk seed policies (`img2vid_chunk_seed_mode=fixed|increment|random`) while preserving the legacy non-chunk execution path.
- 2026-02-20: `img2vid.py` chunk parser now fails loud when `img2vid_chunk_frames >= total_frames` (no silent chunk-disable); legacy per-chunk prepare progress was replaced by phase-batched runtime progress (`chunk.prepare`, `chunk.phase_high`, `chunk.phase_low`, `chunk.phase_decode`).
- 2026-02-21: `img2vid.py` chunked GGUF path is now delegated to `wan22.stream_img2vid_chunked(...)`; use-case no longer loops per chunk with full pipeline reruns. Runtime now performs one text-conditioning pass and chunk-major execution (per chunk: `high` -> `low` -> decode/stitch).
- 2026-02-21: `img2vid.py` chunk parser now accepts `img2vid_chunk_buffer_mode` (`hybrid|ram|ram+hd`) and also honors launcher env default `CODEX_WAN22_IMG2VID_CHUNK_BUFFER_MODE` when payload omits the field.
- 2026-02-21: `img2vid.py` now parses img2vid temporal controls via explicit `img2vid_mode` (`solo|chunk|sliding`) and dispatches GGUF runtime accordingly (`stream_img2vid`, `stream_img2vid_chunked`, `stream_img2vid_sliding_window`) with fail-loud mode-specific validation.
- 2026-02-21: `vid2vid.py` flow-chunks now sanitizes inner `Img2VidRequest.extras` by removing img2vid temporal controls (`img2vid_mode`, `img2vid_chunk_*`, `img2vid_window_*`) and fails loud if any survive, preventing nested chunk/sliding recursion inside each flow chunk call.
- 2026-02-21: WAN video Diffusers orchestration now requires explicit stage prompts in `extras.wan_high.prompt`/`extras.wan_low.prompt` (no fallback to request prompt inside use-cases); stage negatives preserve explicit empty string and only fallback to request negative when stage negative is missing.
- 2026-02-21: WAN video result/export booleans in `txt2vid.py`, `img2vid.py`, and `vid2vid.py` now use shared strict parsing (`core.strict_values.parse_bool_value`) for `video_return_frames`, `video_options.save_output`, and export/probe flags (no permissive `bool("false")==True` coercion).
- 2026-02-21: `txt2vid.py`/`img2vid.py` WAN22 GGUF result handling now avoids unconditional `list(...)` copies when `frames` is already a list, and Diffusers `img2vid` high→low transition now reuses the high-stage frame list instead of duplicating it before stage-2 seed extraction.
